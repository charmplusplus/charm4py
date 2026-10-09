"""Charm chares that implement Ray: _RayWorker (one per PE) runs tasks,
_ActorHost (one array element per actor) runs an actor, _RayControl (one, on
PE 0) owns jobs, actor names, node resources and placement groups, and _Gateway
(on PE 0) lets other processes join the job and, under charm4py.ray.launch, runs
the driver script in its own process.

With isolation 'process' (the default under charm4py.ray.launch) the user's code
-- tasks, actors and the driver -- runs in worker processes (see _proc.py and
_worker_main.py); these chares order the work, resolve arguments, store results
and serve the workers' ray API calls. With isolation 'none' tasks and actors run
right here, as coroutines of the PE.

Registered by charm._registerInternalChares on every PE at charm.start()."""

import collections
import itertools
import math
import os
import sys

from greenlet import getcurrent

from charm4py import Chare, coro
from . import api, _ipc
from ._errors import task_error as _task_error, crash_error as _crash_error
from ._options import ACCELERATOR_KEYS
from ._serialization import resolve, by_value
from .actor import ActorClass, _ActorExit, killed_message
from .remote_function import RemoteFunction
from .exceptions import ActorDiedError, RaySystemError, RayTaskError


def _host_api():
    """The ray API calls an actor worker process can make, run on its host."""
    import importlib
    # (ray.util.placement_group the module; the package exports a function of
    # the same name)
    pgm = importlib.import_module(__package__ + '.util.placement_group')
    return {
        '_init': _init_link,
        '_shutdown': _shutdown_link,
        '_get_stored': api._get_stored,
        'wait': api.wait,
        'put': api.put,
        'kill': api.kill,
        'get_actor': api.get_actor,
        'available_resources': api.available_resources,
        'available_resources_per_node': api.available_resources_per_node,
        '_submit_task': lambda desc, args, kwargs, options:
            RemoteFunction._from_spec(desc)._remote(args, kwargs, options),
        '_create_actor': lambda spec, args, kwargs, options:
            ActorClass._from_spec(spec)._remote(args, kwargs, options),
        '_invoke': lambda handle, method, args, kwargs, n: handle._invoke(method, args, kwargs, n),
        'placement_group': pgm.placement_group,
        'remove_placement_group': pgm.remove_placement_group,
        'placement_group_table': pgm.placement_group_table,
        'get_placement_group': pgm.get_placement_group,
        '_pg_ready': lambda pg: pg.ready(),
        '_object_counts': api._object_counts,
    }


_HOST_API = None

# API calls during which a worker may wait for other work to finish
_BLOCKING_CALLS = {'_get_stored', 'wait', '_pg_ready'}

# API calls that act on the calling process's link (its job)
_LINK_CALLS = {'_init', '_shutdown'}

_SHARED_GLOBALS = ('_initialized', '_nodes', '_session_id', '_isolation', '_address')


def _shared_globals():
    return {name: getattr(api, name) for name in _SHARED_GLOBALS}


def _init_link(link, **job_args):
    """ray.init() in the driver or in a connected process: begins its job."""
    if link.job is not None:
        raise RuntimeError('ray.init() was already called by this process')
    link.job = api._start_job(**job_args)
    return dict(_shared_globals(), _job=link.job)


def _shutdown_link(link):
    job, link.job = link.job, None
    if job is not None:
        api._end_job(job['job_id'])


def _resume_later(gr, value):
    """Resumes a suspended coroutine from the scheduler, whatever greenlet runs."""
    api._charm().scheduleCallableAfter(
        lambda: api._charm().threadMgr.resumeThread(gr, value), 0)


class _Link(object):
    """One worker process, owned by a chare (see _LinkOwner)."""

    def __init__(self, owner, link_id, kind, env=None, sock=None, **info):
        """``kind``: 'driver', 'actor' or 'tasks' starts a worker process; 'client'
        serves a process that connected through ``sock``."""
        from ._proc import WorkerProcess
        self.owner = owner
        self.id = link_id
        self.kind = kind
        self.job = None               # the job a driver or client began (ray.init)
        self.env_key = None
        self.side = _ipc.HostSide()
        self.replies = {}             # request id -> greenlet waiting for the reply
        self.rids = itertools.count()
        self.blocking = 0             # API calls in progress that may wait
        self.api_calls = {}           # rid -> decoded API call waiting for serve_api
        self.worker = WorkerProcess(env or {}, self._on_message, self._on_exit, sock=sock)
        charm = api._charm()
        shared = _shared_globals()
        if kind in ('driver', 'client'):
            shared['_initialized'] = False   # until this process calls ray.init
        info.update(kind=kind, sys_path=list(sys.path), main_script=api._main_script,
                    main_module=api._main_module,
                    globals=dict(shared, _client_pe=charm.myPe(),
                                 _client_host=charm.myHost(), _client_kind=kind))
        self.worker.send(_ipc.encode_header('init'), self.side.dumps(info))

    @property
    def alive(self):
        return not self.worker.closed

    @property
    def pid(self):
        return self.worker.pid

    def request(self, kind, payload):
        """Sends a request and suspends until the worker answers. Returns
        (status, value); status 'died' if the worker died first."""
        rid = next(self.rids)
        try:
            data = self.side.dumps(payload)
        except Exception as e:
            return ('error', _task_error('<arguments>', e))
        self.replies[rid] = getcurrent()
        if not self.worker.send(_ipc.encode_header(kind, rid), data, request=True):
            self.replies.pop(rid, None)
            return ('died', None)
        return api._charm().threadMgr.pauseThread()

    def send(self, kind, rid, payload):
        return self.worker.send(_ipc.encode_header(kind, rid), self.side.dumps(payload))

    def kill(self):
        self.worker.kill()
        self._wake_waiters()

    def _wake_waiters(self):
        replies, self.replies = self.replies, {}
        for gr in replies.values():
            _resume_later(gr, ('died', None))

    def _on_message(self, header, data):
        # runs on the scheduler's main greenlet, from the _proc poller. Payloads
        # are decoded here, in the order they came: user values in them are
        # Opaques, which decode without importing anything
        kind, rid = _ipc.decode_header(header)
        try:
            payload = self.side.loads(data)
        except Exception as e:
            payload = e
        if kind == 'done':
            self.worker.outstanding -= 1
            if isinstance(payload, Exception):
                payload = ('error', RaySystemError(
                    'the result of this task could not be decoded: ' + repr(payload)))
            gr = self.replies.pop(rid, None)
            if gr is not None:
                api._charm().threadMgr.resumeThread(gr, payload)
        elif kind == 'api':
            if isinstance(payload, Exception):
                self.send('api_reply', rid, (False, RaySystemError(
                    'this ray call could not be decoded: ' + repr(payload))))
                return
            self.api_calls[rid] = payload
            # run it as a coroutine: it may block (ray.get)
            owner = self.owner
            owner.thisProxy[owner.thisIndex].serve_api(self.id, rid)
        elif kind == 'release':
            if not isinstance(payload, Exception):
                self.side.release(payload)
        elif kind == 'driver_exit':
            self.owner._driver_exit(payload)

    def _on_exit(self, code):
        self._wake_waiters()
        self.owner._link_lost(self, code)


class _LinkOwner(object):
    """Serves the ray API calls of the owner's worker processes."""

    @coro
    def serve_api(self, link_id, rid):
        global _HOST_API
        if _HOST_API is None:
            _HOST_API = _host_api()
        link = self.links.get(link_id)
        if link is None:
            return
        name, args, kwargs, job = link.api_calls.pop(rid)
        if not link.alive:
            return
        blocking = name in _BLOCKING_CALLS
        if blocking:
            link.blocking += 1
            self._link_blocked(link)
        try:
            # the calling task's, actor's or process's job
            api._set_context(api._ExecContext(job=job))
            if name in _LINK_CALLS:
                args = (link,) + tuple(args)
            reply = (True, _HOST_API[name](*args, **kwargs))
        except Exception as e:
            reply = (False, e)
        finally:
            if blocking:
                link.blocking -= 1
        if not link.alive:
            return
        try:
            link.send('api_reply', rid, reply)
        except Exception as e:
            link.send('api_reply', rid, (False, RaySystemError(
                'the result of ray.' + name + ' could not be sent to the worker: ' +
                repr(e))))

    def _link_blocked(self, link):
        pass

    def _link_lost(self, link, code):
        pass

    def _driver_exit(self, code):
        pass


class _RayWorker(Chare, _LinkOwner):
    """Runs tasks on its PE: in a pool of task worker processes (isolation
    'process') or as coroutines of the PE (isolation 'none')."""

    def __init__(self):
        self.links = {}           # link id -> _Link
        self.idle = []            # task workers waiting for a task (newest last)
        self.busy = set()         # task workers running a task
        self.waiting = collections.deque()   # coroutines waiting for a worker
        self.link_ids = itertools.count()

    def node_info(self):
        return api._local_node_info()

    def release_refs(self, obj_ids):
        """Copies of references this PE sent are accounted for (see _refs.py)."""
        from . import _refs
        _refs.released(obj_ids)

    def object_counts(self):
        """(objects, locations) in this PE's object store branch (for tests)."""
        return api.object_store[api._charm().myPe()].ckLocalBranch().num_objects()

    def _limit(self):
        """Task workers that may run at once on this PE: the node's CPUs shared
        among its task PEs. A worker blocked in ray.get or ray.wait does not
        count, as in Ray, so nested tasks cannot deadlock the pool."""
        node = api._my_node()
        pes = len(api._worker_pes(node['_host']))
        return max(1, int(math.ceil(node['Resources'].get('CPU', 1.0) / pes)))

    def _running(self):
        return sum(1 for link in self.busy if link.blocking == 0)

    def _acquire(self, env):
        """A task worker whose environment has ``env`` (the job's env_vars)."""
        env_key = tuple(sorted(env.items()))
        while True:
            for i in range(len(self.idle) - 1, -1, -1):
                link = self.idle[i]
                if link.alive and link.env_key == env_key:
                    del self.idle[i]
                    self.busy.add(link)
                    return link
            if self._running() < self._limit():
                if len(self.idle) >= self._limit():
                    # idle workers of other jobs' environments: retire the oldest
                    self.idle.pop(0).kill()
                link = _Link(self, next(self.link_ids), 'tasks', env=env)
                link.env_key = env_key
                self.links[link.id] = link
                self.busy.add(link)
                return link
            self.waiting.append(getcurrent())
            api._charm().threadMgr.pauseThread()

    def _release(self, link):
        self.busy.discard(link)
        if link.alive:
            self.idle.append(link)
        self._wake_one()

    def _wake_one(self):
        if self.waiting:
            _resume_later(self.waiting.popleft(), None)

    def _link_blocked(self, link):
        # a blocked worker frees its slot for another task
        if link in self.busy:
            self._wake_one()

    def stop_task_workers(self):
        """ray.shutdown(): ends this PE's task worker processes."""
        for link in list(self.links.values()):
            link.kill()
        self.links.clear()
        self.idle = []
        self.busy.clear()

    def _link_lost(self, link, code):
        self.links.pop(link.id, None)
        self.busy.discard(link)
        if link in self.idle:
            self.idle.remove(link)
        self._wake_one()

    @coro
    def execute_task(self, fn_desc, args, kwargs, return_refs, name, pg_info, job,
                     runtime_env):
        ctx = api._ExecContext(
            task_id=api._new_task_id(), node_id=api._my_node()['NodeID'],
            pg_id=pg_info[0] if pg_info else None,
            pg_bundle_specs=pg_info[1] if pg_info else None, job=job,
            runtime_env=runtime_env)
        api._set_context(ctx)
        function_name = name or (fn_desc[2] if fn_desc[0] == 'ref' else 'task')
        if api._isolation == 'process':
            # the worker unwraps the arguments (and reports a failed dependency)
            args, kwargs = api._fetch_args(args, kwargs)
            status, value = self._run_in_worker(fn_desc, args, kwargs, ctx, name,
                                                len(return_refs))
            if status == 'ok':
                for ref, encoded in zip(return_refs, value):
                    api._store(ref, encoded)
                return
            if status == 'died':
                value = RayTaskError(function_name, '  The worker process died while '
                                     'running this task.', RaySystemError(
                                         'the task worker process died'),
                                     ip=api.get_node_ip_address())
            for ref in return_refs:
                api._store_error(ref, value)
            return
        try:
            args, kwargs = api._resolve_args(args, kwargs)
        except api._DependencyFailed as e:
            err = api._dependency_error(function_name, e.error)
            for ref in return_refs:
                api._store_error(ref, err)
            return
        status, value = self._run_here(fn_desc, args, kwargs, function_name, name)
        if status == 'ok':
            api._store_results(return_refs, value, function_name)
        else:
            for ref in return_refs:
                api._store_error(ref, value)

    def _run_in_worker(self, fn_desc, args, kwargs, ctx, name, num_returns):
        if fn_desc[0] == 'ref' and fn_desc[1] in ('__main__', '__mp_main__') and \
                not api._main_script:
            fn_desc = by_value(resolve(fn_desc))
        # task workers are pooled by environment (the task's and its job's env_vars)
        env_vars = (ctx.runtime_env or {}).get('env_vars') or {}
        link = self._acquire({k: str(v) for k, v in env_vars.items()})
        try:
            return link.request('task', (
                fn_desc, args, kwargs,
                {k: getattr(ctx, k) for k in api._ExecContext.__slots__}, name,
                num_returns))
        finally:
            self._release(link)

    def _run_here(self, fn_desc, args, kwargs, function_name, name):
        try:
            fn = resolve(fn_desc)
            if name is None:
                function_name = getattr(fn, '__qualname__', function_name)
            return 'ok', fn(*args, **kwargs)
        except SystemExit as e:
            return 'error', _crash_error(function_name, e.code)
        except Exception as e:
            return 'error', _task_error(function_name, e)


def _exit_error(spec):
    return ActorDiedError(actor_id=spec['actor_id'], error_msg=(
        'The actor died unexpectedly before finishing this task.\n\tclass_name: ' +
        spec['class_name'] + '\n\tactor_id: ' + spec['actor_id'] +
        '\nThe actor is dead because its worker process has died. Worker exit type: '
        'INTENDED_USER_EXIT Worker exit detail: Worker exits by an user request. '
        'exit_actor() is called.'))


def _crashed_error(spec, pid, code):
    return ActorDiedError(actor_id=spec['actor_id'], error_msg=(
        'The actor died unexpectedly before finishing this task.\n\tclass_name: ' +
        spec['class_name'] + '\n\tactor_id: ' + spec['actor_id'] + '\n\tpid: ' +
        str(pid) + '\nThe actor is dead because its worker process has died. Worker '
        'exit type: SYSTEM_ERROR Worker exit detail: the worker process exited with '
        'code ' + str(code) + '.'))


class _ActorHost(Chare, _LinkOwner):
    """The Charm side of one actor. It orders the calls (one at a time, in the
    order each handle made them, even when a call blocks in ray.get), resolves
    their arguments and stores their results. The actor's code runs in a worker
    process of its own (isolation 'process') or right here, as a coroutine of
    this PE (isolation 'none')."""

    def __init__(self, spec):
        self.spec = spec
        self.state = 'WAIT_CTOR'   # -> 'ALIVE' -> 'DEAD'
        self.instance = None
        self.death = None
        self.queue = collections.deque()
        self.expected = {}          # caller -> next sequence number to run
        self.held = {}              # caller -> {seq: call} that arrived early
        self.running = False
        self.link = None            # the actor's worker process (isolation 'process')
        self.links = {}
        self.setMigratable(False)
        self.ctx = api._ExecContext(
            actor_id=spec['actor_id'], actor_name=spec['name'],
            accelerator_ids=spec['accelerator_ids'], pg_id=spec['pg_id'],
            pg_bundle_specs=spec['pg_bundle_specs'], runtime_env=spec['runtime_env'],
            assigned_resources=spec['assigned_resources'], node_id=spec['node_id'],
            job=spec['job'])
        if spec.get('doa') is not None:
            self._die(spec['doa'], notify_control=False)
        elif api._isolation == 'process':
            self.link = _Link(self, 0, 'actor', env=self._worker_env(),
                              class_name=spec['class_name'],
                              context={k: getattr(self.ctx, k)
                                       for k in api._ExecContext.__slots__})
            self.links[0] = self.link

    def _worker_env(self):
        env_vars = (self.spec['runtime_env'] or {}).get('env_vars') or {}
        env = {k: str(v) for k, v in env_vars.items()}
        gpus = self.spec['accelerator_ids'].get('GPU')
        noset = env.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES',
                        os.environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'))
        if gpus and not noset:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        return env

    def _link_lost(self, link, code):
        # the worker process died without being asked to
        if self.state != 'DEAD':
            self._die(_crashed_error(self.spec, link.pid, code), notify_control=True)

    def _worker_desc(self, desc):
        """How to send a class to the worker: by name when the worker can import
        it, else by value."""
        if desc[0] == 'ref' and desc[1] in ('__main__', '__mp_main__') and \
                not api._main_script:
            return by_value(resolve(desc))
        return desc

    def _context(self):
        ctx = api._ExecContext(**{k: getattr(self.ctx, k)
                                  for k in api._ExecContext.__slots__})
        ctx.task_id = api._new_task_id()
        api._set_context(ctx)
        return ctx

    def _actor_repr(self):
        try:
            return repr(self.instance)
        except Exception:
            return None

    @coro
    def construct(self, cls_desc, args, kwargs):
        if self.state != 'WAIT_CTOR':
            return
        self._context()
        name = self.spec['class_name'] + '.__init__'
        if self.link is not None:
            # the worker unwraps the arguments (and reports a failed dependency)
            args, kwargs = api._fetch_args(args, kwargs)
            status, value = self.link.request('construct', (self._worker_desc(cls_desc),
                                                             args, kwargs))
            if self.state == 'DEAD':
                return
            if status != 'ok':
                if status == 'died':
                    self._die(_crashed_error(self.spec, self.link.pid, None),
                              notify_control=True)
                else:
                    # the worker made the actor's creation error
                    self._die(value, notify_control=True)
                return
            instance = None
        else:
            try:
                args, kwargs = api._resolve_args(args, kwargs)
            except api._DependencyFailed as e:
                self._die(self._creation_error(api._dependency_error(name, e.error)),
                          notify_control=True)
                return
            try:
                env_vars = (self.spec['runtime_env'] or {}).get('env_vars') or {}
                # NOTE: process-wide; in-process actors share os.environ
                os.environ.update({k: str(v) for k, v in env_vars.items()})
                instance = resolve(cls_desc)(*args, **kwargs)
            except Exception as e:
                err = _task_error(name, e, actor_id=self.spec['actor_id'])
                self._die(self._creation_error(err), notify_control=True)
                return
            if self.state == 'DEAD':   # killed while the constructor was blocked
                return
        self.instance = instance
        self.state = 'ALIVE'
        self._drain()

    def _creation_error(self, error):
        return api._creation_error(error, self.spec['actor_id'])

    @coro
    def invoke(self, caller, seq, method_name, args, kwargs, return_refs):
        call = (method_name, args, kwargs, return_refs)
        expected = self.expected.get(caller, 0)
        if seq != expected:
            self.held.setdefault(caller, {})[seq] = call
            return
        self._enqueue(call)
        expected += 1
        held = self.held.get(caller)
        while held and expected in held:
            self._enqueue(held.pop(expected))
            expected += 1
        if held is not None and not held:
            del self.held[caller]
        self.expected[caller] = expected
        self._drain()

    def _enqueue(self, call):
        if self.state == 'DEAD':
            self._fail(call)
        else:
            self.queue.append(call)

    def _fail(self, call):
        for ref in call[3]:
            api._store_error(ref, self.death)

    def _drain(self):
        if self.running or self.state != 'ALIVE':
            return
        self.running = True
        try:
            while self.queue and self.state == 'ALIVE':
                self._execute(self.queue.popleft())
        finally:
            self.running = False
        if self.state == 'DEAD':
            while self.queue:
                self._fail(self.queue.popleft())

    def _execute(self, call):
        method_name, args, kwargs, refs = call
        ctx = self._context()
        name = self.spec['class_name'] + '.' + method_name
        in_worker = self.link is not None and method_name not in (
            '__ray_ready__', '__ray_terminate__')
        try:
            if in_worker:
                # the worker unwraps the arguments (and reports a failed dependency)
                args, kwargs = api._fetch_args(args, kwargs)
            else:
                args, kwargs = api._resolve_args(args, kwargs)
        except api._DependencyFailed as e:
            err = api._dependency_error(name, e.error)
            for ref in refs:
                api._store_error(ref, err)
            return
        if method_name == '__ray_ready__':
            status, value = 'ok', True
        elif method_name == '__ray_terminate__':
            status, value = 'exit', None
        elif in_worker:
            status, value = self.link.request('call', (method_name, args, kwargs,
                                                        ctx.task_id, len(refs)))
        else:
            status, value = self._run_here(name, method_name, args, kwargs)
        if self.state == 'DEAD' or status == 'died':
            # killed or crashed while this call ran
            for ref in refs:
                api._store_error(ref, self.death)
        elif status == 'ok':
            if in_worker:
                # one serialized value per return
                for ref, encoded in zip(refs, value):
                    api._store(ref, encoded)
            else:
                api._store_results(refs, value, name)
        elif status == 'error':
            for ref in refs:
                api._store_error(ref, value)
        elif status == 'exit':
            if method_name == '__ray_terminate__':
                # graceful termination succeeds; later calls fail
                for ref in refs:
                    api._store(ref, None)
                refs = []
            self._die(_exit_error(self.spec), notify_control=True, current_refs=refs)

    def _run_here(self, name, method_name, args, kwargs):
        try:
            return 'ok', getattr(self.instance, method_name)(*args, **kwargs)
        except SystemExit:   # exit_actor() or sys.exit()
            return 'exit', None
        except Exception as e:
            return 'error', _task_error(name, e, actor_repr=self._actor_repr(),
                                        actor_id=self.spec['actor_id'])

    def kill(self):
        if self.state != 'DEAD':
            self._die(ActorDiedError(actor_id=self.spec['actor_id'], error_msg=killed_message(
                self.spec['class_name'], self.spec['actor_id'])), notify_control=False)

    def _die(self, error, notify_control, current_refs=()):
        self.state = 'DEAD'
        self.death = error
        if self.link is not None:
            self.link.kill()
        for ref in current_refs:
            api._store_error(ref, error)
        while self.queue:
            self._fail(self.queue.popleft())
        for calls in self.held.values():
            for call in calls.values():
                self._fail(call)
        self.held.clear()
        self.instance = None
        if notify_control:
            api._control.actor_died(self.spec['key'])


class _Gateway(Chare, _LinkOwner):
    """On PE 0: other processes of the job -- ones the driver starts, say --
    connect here to use Ray (see api._connect), each as a job of its own that ends
    when it disconnects. Under charm4py.ray.launch it also runs the driver script
    in a process of its own and serves its ray calls; the Charm4py job ends when
    the driver does."""

    def __init__(self, argv):
        from ._proc import Listener
        self.setMigratable(False)
        self.links = {}
        self.link_ids = itertools.count()
        self.exited = False
        self.listener = Listener(self._connected)
        api._address = api.ADDRESS_PREFIX + self.listener.path
        # every process started from here on inherits it
        os.environ[api.ADDRESS_ENV] = api._address
        if argv is not None:
            link = _Link(self, next(self.link_ids), 'driver', argv=argv)
            self.links[link.id] = link

    def address(self):
        return api._address

    def _connected(self, sock):
        link = _Link(self, next(self.link_ids), 'client', sock=sock)
        self.links[link.id] = link

    def _driver_exit(self, code):
        self.exited = True
        self.listener.close()
        api._charm().exit(code)

    def _link_lost(self, link, code):
        self.links.pop(link.id, None)
        if link.kind == 'driver':
            if not self.exited:
                print('charm4py.ray: the driver process exited with code ' + str(code) +
                      ' without finishing', file=sys.stderr)
                self.listener.close()
                api._charm().exit(code if code else 1)
        elif link.job is not None:
            # the process ended (or disconnected) without ray.shutdown()
            self.thisProxy.end_job(link.job['job_id'])
            link.job = None

    @coro
    def end_job(self, job_id):
        api._end_job(job_id)


def _fits(avail, request):
    return all(avail.get(k, 0.0) + 1e-9 >= v for k, v in request.items())


def _take(avail, ids_pool, request):
    for k, v in request.items():
        avail[k] = avail.get(k, 0.0) - v
    ids = {}
    for k in ACCELERATOR_KEYS:
        n = int(request.get(k, 0))
        if n:
            ids[k] = ids_pool[k][:n]
            del ids_pool[k][:n]
    return ids


def _give(avail, ids_pool, request, ids):
    for k, v in request.items():
        avail[k] = avail.get(k, 0.0) + v
    for k, taken in ids.items():
        ids_pool.setdefault(k, []).extend(taken)
        ids_pool[k].sort(key=_id_order)


def _id_order(device_id):
    return (0, int(device_id), '') if device_id.isdigit() else (1, 0, device_id)


class _RayControl(Chare):
    """The authority for names, resources and placement groups (on PE 0)."""

    def __init__(self):
        self.setMigratable(False)
        self.nodes = []
        self.actors_proxy = None
        self.job_counter = 0
        self._reset()

    def _reset(self):
        self.jobs = {}         # job id -> job record (see api._job)
        self.actors = {}       # actor key -> record
        self.names = {}        # (namespace, name) -> actor key
        self.pgs = {}          # placement group id (hex) -> record
        self.pending = []      # ('actor', key) / ('pg', id) waiting for resources
        self.pe_load = collections.Counter()
        self.killed_early = set()

    def configure(self, nodes, actors_proxy):
        self.actors_proxy = actors_proxy
        self.nodes = []
        for i, node in enumerate(nodes):
            total = dict(node['Resources'])
            self.nodes.append({
                'index': i, 'node_id': node['NodeID'], 'pes': list(node['_pes']),
                'avail': dict(total),
                'ids': {k: (list(node['_gpu_ids']) if k == 'GPU' else
                            [str(i) for i in range(int(total[k]))])
                        for k in ACCELERATOR_KEYS if total.get(k)}})
        self._reset()
        return True

    def new_job(self, namespace, runtime_env, sys_path):
        import struct
        import uuid
        self.job_counter += 1
        job = {'job_id': struct.pack('<I', self.job_counter).hex(),
               'namespace': namespace if namespace is not None else str(uuid.uuid4()),
               'runtime_env': dict(runtime_env or {}), 'sys_path': list(sys_path)}
        self.jobs[job['job_id']] = job
        return job

    def end_job(self, job_id):
        """Kills the job's actors and removes its placement groups. Returns how
        many jobs are left."""
        if self.jobs.pop(job_id, None) is not None:
            for rec in list(self.actors.values()):
                if rec['state'] != 'DEAD' and rec['meta']['job']['job_id'] == job_id:
                    self._kill(rec)
            for pg in list(self.pgs.values()):
                if pg['job_id'] == job_id and pg['state'] != 'REMOVED':
                    self.remove_pg(pg['id'])
            self._schedule_pending()
        return len(self.jobs)

    def shutdown(self):
        for key, rec in list(self.actors.items()):
            if rec['state'] != 'DEAD':
                self._kill(rec)
        for pg in self.pgs.values():
            pg['state'] = 'REMOVED'
        self._reset()
        return True

    # ---- queries

    def available_resources_per_node(self):
        return {n['node_id']: {k: v for k, v in n['avail'].items() if v > 1e-9}
                for n in self.nodes}

    def get_actor(self, name, namespace):
        key = self.names.get((namespace, name))
        if key is None:
            return None
        rec = self.actors[key]
        return rec['meta']['handle_state'] if rec['state'] != 'DEAD' else None

    # ---- actors

    def _driver_first_nodes(self):
        driver = api._charm().getPeHost(0)
        return sorted(self.nodes, key=lambda n: (n['index'] != driver, n['index']))

    def _choose_pe(self, node):
        pes = list(node['pes'])
        if len(pes) > 1 and 0 in pes:
            pes.remove(0)
        return min(pes, key=lambda pe: (self.pe_load[pe], pe))

    def create_actor(self, meta):
        key = meta['key']
        name = meta['name']
        if name is not None:
            name_key = (meta['namespace'], name)
            existing = self.names.get(name_key)
            if existing is not None and self.actors[existing]['state'] != 'DEAD':
                if meta['get_if_exists']:
                    return ('exists', self.actors[existing]['meta']['handle_state'])
                return ('taken',)
            self.names[name_key] = key
        rec = {'meta': meta, 'state': 'PENDING', 'node': None, 'pe': None, 'ids': {},
               'bundle': None}
        self.actors[key] = rec
        if key in self.killed_early:
            self.killed_early.discard(key)
            self._insert_dead(rec, killed_message(meta['class_name'], meta['actor_id']))
        elif not self._schedule_actor(rec):
            self.pending.append(('actor', key))
        return ('ok',)

    def _host_spec(self, rec, node_index, pg_id=None, bundle_specs=None, doa=None):
        meta = rec['meta']
        node_id = self.nodes[node_index]['node_id'] if node_index is not None else None
        return {'key': meta['key'], 'actor_id': meta['actor_id'],
                'class_name': meta['class_name'], 'name': meta['name'],
                'runtime_env': meta['runtime_env'], 'job': meta['job'],
                'accelerator_ids': {k: [str(i) for i in v] for k, v in rec['ids'].items()},
                'assigned_resources': dict(meta['request']), 'node_id': node_id,
                'pg_id': pg_id, 'pg_bundle_specs': bundle_specs, 'doa': doa}

    def _insert_dead(self, rec, message, pe=0):
        """Creates the actor's element only to fail the calls already sent to it."""
        meta = rec['meta']
        rec['state'] = 'DEAD'
        err = ActorDiedError(actor_id=meta['actor_id'], error_msg=message)
        self.actors_proxy.ckInsert(meta['key'], args=[self._host_spec(rec, None, doa=err)],
                                   onPE=pe)
        self._forget_name(rec)

    def _schedule_actor(self, rec):
        from .util.scheduling_strategies import (PlacementGroupSchedulingStrategy,
                                                 NodeAffinitySchedulingStrategy)
        meta = rec['meta']
        request = meta['request']
        strategy = meta['strategy']
        pg_id = bundle_specs = None
        if isinstance(strategy, PlacementGroupSchedulingStrategy):
            pg_id = strategy.placement_group.id.hex()
            pg = self.pgs.get(pg_id)
            if pg is None or pg['state'] == 'REMOVED':
                self._insert_dead(rec, 'The actor could not be scheduled: its placement '
                                  'group ' + pg_id + ' was removed or does not exist.')
                return True
            if pg['state'] != 'CREATED':
                return False
            index = strategy.placement_group_bundle_index
            indices = [index] if index >= 0 else range(len(pg['bundles']))
            for i in indices:
                bundle = pg['bundles'][i]
                if _fits(bundle['avail'], request):
                    rec['ids'] = _take(bundle['avail'], bundle['ids'], request)
                    rec['bundle'] = (pg_id, i)
                    node = self.nodes[bundle['node']]
                    bundle_specs = pg['specs']
                    break
            else:
                return False
        else:
            candidates = self._driver_first_nodes()
            if isinstance(strategy, NodeAffinitySchedulingStrategy):
                pinned = [n for n in self.nodes if n['node_id'] == strategy.node_id]
                candidates = pinned + (candidates if strategy.soft else [])
            for node in candidates:
                if _fits(node['avail'], request):
                    rec['ids'] = _take(node['avail'], node['ids'], request)
                    break
            else:
                return False
        pe = self._choose_pe(node)
        rec.update(state='ALIVE', node=node['index'], pe=pe)
        self.pe_load[pe] += 1
        self.actors_proxy.ckInsert(meta['key'], args=[self._host_spec(
            rec, node['index'], pg_id, bundle_specs)], onPE=pe)
        return True

    def _release(self, rec):
        request = rec['meta']['request']
        if rec['bundle'] is not None:
            pg = self.pgs.get(rec['bundle'][0])
            if pg is not None and pg['state'] == 'CREATED':
                bundle = pg['bundles'][rec['bundle'][1]]
                _give(bundle['avail'], bundle['ids'], request, rec['ids'])
        elif rec['node'] is not None:
            node = self.nodes[rec['node']]
            _give(node['avail'], node['ids'], request, rec['ids'])
        if rec['pe'] is not None:
            self.pe_load[rec['pe']] -= 1
        rec['ids'] = {}
        rec['bundle'] = None

    def _forget_name(self, rec):
        name = rec['meta']['name']
        if name is not None:
            name_key = (rec['meta']['namespace'], name)
            if self.names.get(name_key) == rec['meta']['key']:
                del self.names[name_key]

    def _kill(self, rec):
        if rec['state'] == 'PENDING':
            self.pending = [p for p in self.pending if p != ('actor', rec['meta']['key'])]
            self._insert_dead(rec, killed_message(rec['meta']['class_name'],
                                                  rec['meta']['actor_id']))
            return
        if rec['state'] == 'ALIVE':
            self.actors_proxy[rec['meta']['key']].kill()
            self._release(rec)
        rec['state'] = 'DEAD'
        self._forget_name(rec)

    def kill_actor(self, key):
        rec = self.actors.get(key)
        if rec is None:
            # the creation request has not arrived yet
            self.killed_early.add(key)
            return True
        self._kill(rec)
        self._schedule_pending()
        return True

    def actor_died(self, key):
        rec = self.actors.get(key)
        if rec is None or rec['state'] == 'DEAD':
            return
        self._release(rec)
        rec['state'] = 'DEAD'
        self._forget_name(rec)
        self._schedule_pending()

    def _schedule_pending(self):
        progress = True
        while progress and self.pending:
            progress = False
            for item in list(self.pending):
                kind, key = item
                if kind == 'actor':
                    done = self._schedule_actor(self.actors[key])
                else:
                    done = self._schedule_pg(self.pgs[key])
                if done:
                    self.pending.remove(item)
                    progress = True

    # ---- placement groups

    def create_pg(self, pg_id, specs, strategy, name, job_id):
        pg = {'id': pg_id, 'specs': specs, 'strategy': strategy, 'name': name,
              'state': 'PENDING', 'bundles': None, 'ready_refs': [], 'job_id': job_id}
        self.pgs[pg_id] = pg
        if not self._schedule_pg(pg):
            self.pending.append(('pg', pg_id))

    def _assign_bundles(self, specs, strategy):
        """Node index for each bundle, or None if they cannot all be placed now.
        Works on copies so a partial assignment never reserves anything."""
        avail = {n['index']: dict(n['avail']) for n in self.nodes}
        order = [n['index'] for n in self._driver_first_nodes()]

        def place(i, node):
            if _fits(avail[node], specs[i]):
                for k, v in specs[i].items():
                    avail[node][k] = avail[node].get(k, 0.0) - v
                return True
            return False

        if strategy in ('PACK', 'STRICT_PACK'):
            for node in order:
                saved = {k: dict(v) for k, v in avail.items()}
                if all(place(i, node) for i in range(len(specs))):
                    return [node] * len(specs)
                avail = saved
            if strategy == 'STRICT_PACK':
                return None
            assignment = []
            for i in range(len(specs)):
                preferred = ([assignment[-1]] if assignment else []) + order
                for node in preferred:
                    if place(i, node):
                        assignment.append(node)
                        break
                else:
                    return None
            return assignment
        if strategy == 'SPREAD':
            assignment = []
            for i in range(len(specs)):
                ranked = sorted(order, key=lambda n: (assignment.count(n), order.index(n)))
                for node in ranked:
                    if place(i, node):
                        assignment.append(node)
                        break
                else:
                    return None
            return assignment
        # STRICT_SPREAD
        assignment = []
        for i in range(len(specs)):
            for node in order:
                if node not in assignment and place(i, node):
                    assignment.append(node)
                    break
            else:
                return None
        return assignment

    def _schedule_pg(self, pg):
        if pg['state'] != 'PENDING':
            return True
        assignment = self._assign_bundles(pg['specs'], pg['strategy'])
        if assignment is None:
            return False
        bundles = []
        for spec, node_index in zip(pg['specs'], assignment):
            node = self.nodes[node_index]
            ids = _take(node['avail'], node['ids'], spec)
            bundles.append({'node': node_index, 'avail': dict(spec), 'ids': ids,
                            'all_ids': {k: list(v) for k, v in ids.items()}})
        pg['bundles'] = bundles
        pg['state'] = 'CREATED'
        for ref in pg['ready_refs']:
            api._store(ref, self._pg_object(pg))
        pg['ready_refs'] = []
        return True

    def _pg_object(self, pg):
        from .util.placement_group import PlacementGroup, PlacementGroupID
        return PlacementGroup(PlacementGroupID(pg['id']), pg['specs'])

    def pg_ready(self, pg_id, ref):
        pg = self.pgs.get(pg_id)
        if pg is None or pg['state'] == 'REMOVED':
            api._store_error(ref, RaySystemError('Placement group ' + pg_id +
                                                 ' was removed'))
        elif pg['state'] == 'CREATED':
            api._store(ref, self._pg_object(pg))
        else:
            pg['ready_refs'].append(ref)

    def remove_pg(self, pg_id):
        pg = self.pgs.get(pg_id)
        if pg is None or pg['state'] == 'REMOVED':
            return True
        if pg['state'] == 'PENDING':
            self.pending = [p for p in self.pending if p != ('pg', pg_id)]
        else:
            for rec in self.actors.values():
                if rec['state'] != 'DEAD' and rec['bundle'] is not None and \
                        rec['bundle'][0] == pg_id:
                    self._kill(rec)
            for spec, bundle in zip(pg['specs'], pg['bundles']):
                node = self.nodes[bundle['node']]
                _give(node['avail'], node['ids'], spec, bundle['all_ids'])
        for ref in pg['ready_refs']:
            api._store_error(ref, RaySystemError('Placement group ' + pg_id + ' was removed'))
        pg['ready_refs'] = []
        pg['state'] = 'REMOVED'
        self._schedule_pending()
        return True

    def placement_group_bundle_nodes(self, pg_id):
        return [b['node'] for b in self.pgs[pg_id]['bundles']]

    def _pg_table_entry(self, pg):
        to_node = {}
        if pg['bundles'] is not None and pg['state'] == 'CREATED':
            to_node = {i: self.nodes[b['node']]['node_id'] for i, b in enumerate(pg['bundles'])}
        return {'placement_group_id': pg['id'], 'name': pg['name'],
                'bundles': {i: dict(s) for i, s in enumerate(pg['specs'])},
                'bundles_to_node_id': to_node, 'strategy': pg['strategy'],
                'state': pg['state'],
                'stats': {'scheduling_state': 'FINISHED' if pg['state'] == 'CREATED'
                          else 'PENDING'}}

    def pg_table(self, pg_id):
        if pg_id is None:
            return {i: self._pg_table_entry(pg) for i, pg in self.pgs.items()}
        pg = self.pgs.get(pg_id)
        return self._pg_table_entry(pg) if pg is not None else {}

    def pg_by_name(self, name):
        for pg in self.pgs.values():
            if pg['name'] == name and pg['state'] != 'REMOVED':
                return (pg['id'], pg['specs'])
        return None
