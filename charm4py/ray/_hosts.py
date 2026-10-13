"""Charm chares that implement Ray: _RayWorker (one per PE) runs tasks,
_ActorHost (one array element per actor) runs an actor, _JobRegistry (one, on
PE 0) keeps the job table, and _Gateway
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
        self.worker = WorkerProcess(env or {}, self._on_message, self._on_exit, sock=sock,
                                    own_session=kind in ('actor', 'tasks'))
        charm = api._charm()
        shared = _shared_globals()
        if kind in ('driver', 'client'):
            shared['_initialized'] = False   # until this process calls ray.init
        info.update(kind=kind, sys_path=list(sys.path), main_script=api._main_script,
                    main_module=api._main_module,
                    globals=dict(shared, _client_pe=charm.myPe(),
                                 _client_host=api._my_host(), _client_kind=kind))
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
        self._drop_tokens()

    def _drop_tokens(self):
        # the process is gone: so are its references (now, not whenever the
        # cyclic garbage collector gets to this link)
        self.side.refs.clear()
        self.side.sent.clear()
        self.side.handles.clear()
        self.api_calls.clear()

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
        elif kind in ('api', 'api_async'):
            reply = kind == 'api'
            if isinstance(payload, Exception):
                if reply:
                    self.send('api_reply', rid, (False, RaySystemError(
                        'this ray call could not be decoded: ' + repr(payload))))
                else:
                    print('charm4py.ray: a ray call could not be decoded: ' + repr(payload),
                          file=sys.stderr, flush=True)
                return
            self.api_calls[rid] = payload
            # run it as a coroutine: it may block (ray.get)
            owner = self.owner
            owner.thisProxy[owner.thisIndex].serve_api(self.id, rid, reply)
        elif kind == 'release':
            if not isinstance(payload, Exception):
                self.side.release(payload)
        elif kind == 'driver_exit':
            self.owner._driver_exit(payload)

    def _on_exit(self, code):
        self._wake_waiters()
        self._drop_tokens()
        self.owner._link_lost(self, code)


class _LinkOwner(object):
    """Serves the ray API calls of the owner's worker processes."""

    @coro
    def serve_api(self, link_id, rid, reply_wanted=True):
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
        if not reply_wanted:
            # nobody waits for it (call_async): an error can only be reported here
            if not reply[0]:
                print('charm4py.ray: ray.' + name + ' failed: ' + repr(reply[1]),
                      file=sys.stderr, flush=True)
            return
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
            # the actor's coordinator (see _control.py)
            api._ctl[self.spec['key'][0]].actor_died(self.spec['key'])


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


class _JobRegistry(Chare):
    """The job table (on PE 0): job IDs and the jobs still running. Jobs begin
    and end rarely; everything else about them is kept where it is used (see
    _control.py)."""

    def __init__(self):
        self.setMigratable(False)
        self.job_counter = 0
        self.jobs = {}

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
        """Returns how many jobs are left."""
        self.jobs.pop(job_id, None)
        return len(self.jobs)

    def reset(self):
        self.jobs = {}
        return True
