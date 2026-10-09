"""Core of the Ray Core API on Charm4py: runtime state, references (get, wait,
put), cluster topology and the per-task execution context.

Model: ``ray.init()`` runs on the driver (PE 0, inside ``charm.start``). It creates
the Charm collections that implement Ray -- one object store branch, one task
worker per PE, a 2-D array holding one element per actor, and a control chare on
PE 0 that owns names, resources and placement groups -- and broadcasts them to
every PE. A Ray *node* is a physical host as Charm++ sees it, not a PE."""

import copy
import hashlib
import itertools
import os
import socket
import sys
import uuid

from greenlet import getcurrent

import charm4py
from .object_ref import ObjectRef, wrap_value, unwrap_value, _ErrorValue
from . import exceptions

# ---- state broadcast to every PE by init() and shutdown() (updateGlobals) ----
_initialized = False
object_store = None   # Group(ObjectStore)
_workers = None       # Group(_RayWorker): runs tasks; element 0 is not special
_actors = None        # Array(_ActorHost, ndims=2): one element per actor
_control = None       # _RayControl singleton chare on PE 0
_session_id = None
_namespace = None
_nodes = None         # list of node records, indexed by Charm physical node
_job_runtime_env = None
_isolation = 'none'   # 'process': tasks, actors (and the driver) run in worker processes

JOB_ID = '01000000'

# ---- set in actor worker processes (see _worker_main.py) ----
_client = None        # forwards API calls to the actor's host
_client_pe = None     # the host's PE
_client_host = None   # the host's node index
_client_kind = None   # 'driver', 'actor' or 'tasks'

# set by charm4py.ray.launch: the script, which worker processes can import
_main_script = None
_default_isolation = None

# ---- per-process state ----
_collections_created = False
_task_counter = itertools.count()
_actor_counter = itertools.count()
_caller_counter = itertools.count()
_rr_counter = itertools.count()
_waiters = {}         # store_id -> list of _Waiter blocked on that object
_pg_bundle_nodes = {}  # pg id hex -> list of node index per bundle (immutable once created)


def _charm():
    return charm4py.charm


def get_object_store():
    return object_store


def is_initialized():
    return _initialized


def _ensure_initialized():
    if _client is not None:
        if not _initialized and _client_kind == 'driver':
            init()
        return
    if not _initialized:
        charm = _charm()
        if charm.started and charm.myPe() == 0 and not charm.threadMgr.isMainThread():
            init()
        else:
            raise exceptions.RaySystemError(
                'Ray has not been started yet. Call ray.init() from the driver first.')


class _RayContext(dict):
    """Returned by ray.init(), like Ray's RayContext it is also a dict."""

    def __init__(self, address_info):
        super().__init__(address_info)
        self.address_info = address_info
        self.dashboard_url = None

    def disconnect(self):
        shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutdown()


_INIT_ACCEPTED = {
    '_isolation',
    'include_dashboard', 'dashboard_host', 'dashboard_port', 'log_to_driver',
    'logging_level', 'logging_format', 'logging_config', 'configure_logging',
    'object_store_memory', 'local_mode', 'job_config', 'storage', 'labels',
    'enable_resource_isolation', 'system_reserved_cpu', 'system_reserved_memory',
    'object_spilling_directory',
}


def init(address=None, *, num_cpus=None, num_gpus=None, resources=None, namespace=None,
         runtime_env=None, ignore_reinit_error=False, **kwargs):
    """Start the Ray layer on the running Charm4py job.

    ``address`` is accepted for source compatibility and ignored: the "cluster" is
    always the set of processes started with this Charm4py job. ``num_cpus``,
    ``num_gpus`` and ``resources`` describe each node (each physical host), since
    every host is started together; by default a node has one CPU per PE.

    ``_isolation`` (charm4py only): 'process' runs tasks and actors in worker
    processes, like Ray; 'none' runs them inside the Charm4py processes, as
    coroutines. The default is 'process' for programs started with
    charm4py.ray.launch and 'none' for Charm4py programs (charm.start), whose
    tasks and actors may use Charm4py."""
    global _collections_created
    if _initialized:
        if ignore_reinit_error:
            return _RayContext(_address_info())
        raise RuntimeError(
            "Maybe you called ray.init twice by accident? This error can be suppressed "
            "by passing in 'ignore_reinit_error=True' or by calling 'ray.shutdown()' "
            "prior to 'ray.init()'.")
    for key in kwargs:
        if key not in _INIT_ACCEPTED and not key.startswith('_'):
            raise RuntimeError('Unknown keyword argument(s): ' + key)
    if _client is not None:
        # the driver process: Charm4py's PE 0 initializes, and sends back the state
        # every process shares
        shared = _client.call('_init', address, num_cpus=num_cpus, num_gpus=num_gpus,
                              resources=resources, namespace=namespace,
                              runtime_env=runtime_env, **kwargs)
        globals().update(shared)
        return _RayContext(_address_info())
    charm = _charm()
    isolation = kwargs.get('_isolation') or _default_isolation or \
        ('process' if _main_script else 'none')
    if isolation not in ('process', 'none'):
        raise ValueError("_isolation must be 'process' or 'none'")
    if not charm.started:
        raise RuntimeError('charm4py.ray.init() must be called after charm.start(), '
                           'from the driver (the main function on PE 0)')
    if charm.myPe() != 0:
        raise RuntimeError('ray.init() can only be called from the driver (PE 0)')
    if runtime_env is not None:
        from ._options import validate, ACTOR_OPTIONS
        validate({'runtime_env': runtime_env}, ACTOR_OPTIONS, 'actors')
    if resources:
        for key in resources:
            if key in ('CPU', 'GPU'):
                raise ValueError("Use the 'num_cpus' and 'num_gpus' keyword instead of "
                                 "'CPU' and 'GPU' in 'resources' keyword")

    from ._hosts import _RayWorker, _ActorHost, _RayControl
    globals_update = {}
    if not _collections_created:
        store = charm4py.Group(charm4py.ObjectStore)
        workers = charm4py.Group(_RayWorker)
        actors = charm4py.Array(_ActorHost, ndims=2)
        control = charm4py.Chare(_RayControl, onPE=0)
        globals_update.update(object_store=store, _workers=workers, _actors=actors,
                              _control=control)
        _collections_created = True
    else:
        workers, actors, control = _workers, _actors, _control

    session_id = uuid.uuid4().hex
    nodes = _discover_nodes(workers, session_id, num_cpus, num_gpus, resources)
    control.configure(nodes, actors, ret=True).get()
    globals_update.update(_initialized=True, _session_id=session_id,
                          _isolation=isolation,
                          _namespace=namespace or str(uuid.uuid4()), _nodes=nodes,
                          _job_runtime_env=dict(runtime_env or {}))
    charm.thisProxy.updateGlobals(globals_update, awaitable=True,
                                  module_name=__name__).get()
    return _RayContext(_address_info())


def _address_info():
    ip = get_node_ip_address()
    return {'node_ip_address': ip, 'address': ip + ':0', 'gcs_address': ip + ':0',
            'session_dir': None, 'node_id': _my_node()['NodeID'] if _nodes else None,
            'webui_url': None}


def _node_id_hex(session_id, host):
    # Ray node IDs are 28 bytes (56 hex digits); sha224 has exactly that length
    return hashlib.sha224((session_id + ':node:' + str(host)).encode()).hexdigest()


def _discover_nodes(workers, session_id, num_cpus, num_gpus, resources):
    charm = _charm()
    num_hosts = charm.numHosts()
    replies = [workers[charm.getHostFirstPe(h)].node_info(ret=True) for h in range(num_hosts)]
    nodes = []
    for host, reply in enumerate(replies):
        hostname, ip, memory = reply.get()
        if num_hosts == 1:
            ip = '127.0.0.1'
        pes = list(charm.getHostPes(host))
        res = {'CPU': float(len(pes) if num_cpus is None else num_cpus),
               'memory': float(memory),
               'object_store_memory': float(memory) * 0.3,
               'node:' + ip: 1.0}
        if num_gpus:
            res['GPU'] = float(num_gpus)
        for key, amount in (resources or {}).items():
            res[key] = float(amount)
        if host == charm.myHost():
            res['node:__internal_head__'] = 1.0
        node_id = _node_id_hex(session_id, host)
        nodes.append({'NodeID': node_id, 'Alive': True, 'alive': True,
                      'NodeManagerAddress': ip, 'NodeManagerHostname': hostname,
                      'NodeName': ip, 'NodeManagerPort': 0, 'ObjectManagerPort': 0,
                      'ObjectStoreSocketName': '', 'RayletSocketName': '',
                      'MetricsExportPort': 0, 'MetricsAgentPort': 0,
                      'DashboardAgentListenPort': 0, 'RuntimeEnvAgentPort': 0,
                      'Resources': res, 'Labels': {'ray.io/node_id': node_id},
                      'DeathReason': 0, 'DeathReasonMessage': '',
                      # charm4py-specific: not part of Ray's node table
                      '_host': host, '_pes': pes})
    return nodes


def _local_node_info():
    """(hostname, routable IP, memory bytes) of this host."""
    hostname = socket.gethostname()
    ip = os.environ.get('CHARM4PY_RAY_NODE_IP') or os.environ.get('VLLM_HOST_IP')
    if not ip:
        try:
            # no packet is sent; this selects the interface of the default route
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('10.255.255.255', 1))
                ip = s.getsockname()[0]
            finally:
                s.close()
        except OSError:
            ip = '127.0.0.1'
    try:
        memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        memory = 8 * 1024 ** 3
    return hostname, ip, memory


def shutdown(_exiting_interpreter=False):
    """Kill the actors and placement groups of this job and mark Ray as not
    initialized on every PE. The Charm4py program keeps running, and ray.init()
    may be called again."""
    if not _initialized:
        return
    if _client is not None:
        if _client_kind == 'driver':
            _client.call('_shutdown')
            globals()['_initialized'] = False
        return
    charm = _charm()
    if charm.myPe() != 0:
        raise RuntimeError('ray.shutdown() can only be called from the driver (PE 0)')
    _control.shutdown(ret=True).get()
    _pg_bundle_nodes.clear()
    _workers.stop_task_workers(awaitable=True).get()
    charm.thisProxy.updateGlobals({'_initialized': False}, awaitable=True,
                                  module_name=__name__).get()


# ---------------------------------------------------------------- references

def _new_ref():
    return _charm().threadMgr.createFuture(store=True)


def _store(ref, value):
    """Stores ``value`` (or an _ErrorValue) as the object ``ref`` names. The first
    value stored for a reference wins (see ObjectStore.register_object)."""
    ref.create_object(value if isinstance(value, _ErrorValue) else wrap_value(value))


def _store_error(ref, error):
    _store(ref, _ErrorValue(error))


class _Waiter(object):
    """A coroutine blocked until ``needed`` of the objects in ``pending`` are local,
    or until its timeout fires."""

    __slots__ = ('gr', 'pending', 'needed', 'done')

    def __init__(self, ids, needed):
        self.gr = getcurrent()
        self.pending = set(ids)
        self.needed = needed
        self.done = False

    def object_local(self, obj_id):
        if self.done or obj_id not in self.pending:
            return
        self.pending.discard(obj_id)
        self.needed -= 1
        if self.needed <= 0:
            self._finish()

    def timed_out(self):
        if not self.done:
            self._finish()

    def _finish(self):
        self.done = True
        _charm().threadMgr.resumeThread(self.gr, None)


def _object_local(obj_id):
    """Called (on the scheduler's main greenlet) whenever an object arrives in this
    PE's object store branch."""
    waiters = _waiters.get(obj_id)
    if waiters:
        for w in list(waiters):
            w.object_local(obj_id)


def _fetch(refs, needed, timeout):
    """Requests every non-local object in ``refs`` (unique) and blocks the calling
    coroutine until at least ``needed`` of them are local or ``timeout`` seconds
    pass (``None`` waits forever)."""
    charm = _charm()
    missing = []
    for ref in refs:
        if ref.lookup_object() is None:
            ref.request_object()
            missing.append(ref.store_id)
    have = len(refs) - len(missing)
    if have >= needed or (timeout is not None and timeout <= 0):
        return
    if charm.threadMgr.isMainThread():
        raise RuntimeError('ray.get() and ray.wait() must be called from the driver, a '
                           'remote function or an actor method, not from a non-coroutine '
                           'Charm4py entry method')
    waiter = _Waiter(missing, needed - have)
    for obj_id in missing:
        _waiters.setdefault(obj_id, []).append(waiter)
    if timeout is not None:
        charm.scheduleCallableAfter(waiter.timed_out, timeout)
    try:
        charm.threadMgr.pauseThread()
    finally:
        for obj_id in missing:
            lst = _waiters.get(obj_id)
            if lst is not None:
                try:
                    lst.remove(waiter)
                except ValueError:
                    pass
                if not lst:
                    del _waiters[obj_id]


def _local_value(ref):
    return ref.lookup_object()


def get(object_refs, *, timeout=None):
    """Returns the value of an ObjectRef or the values of a list of ObjectRefs,
    blocking until they are available. Raises the error of a failed task, or
    GetTimeoutError if ``timeout`` seconds pass first."""
    _ensure_initialized()
    single = isinstance(object_refs, ObjectRef)
    if single:
        refs = [object_refs]
    elif isinstance(object_refs, list):
        refs = object_refs
        for ref in refs:
            if not isinstance(ref, ObjectRef):
                raise TypeError("'object_refs' must either be an ObjectRef or a list of "
                                "ObjectRefs. Got " + str(type(ref)) + ' in the list.')
    else:
        raise ValueError('Invalid type of object refs, ' + str(type(object_refs)) +
                         ", is given. 'object_refs' must either be an ObjectRef or a "
                         'list of ObjectRefs. ')
    if not refs:
        return []
    if _client is not None:
        stored = _client.call('_get_stored', refs, timeout)
    else:
        stored = _get_stored(refs, timeout)
    values = [unwrap_value(value) for value in stored]
    return values[0] if single else values


def _get_stored(refs, timeout):
    """The stored values (still wrapped) of ``refs``, fetching them first."""
    unique = list({ref.store_id: ref for ref in refs}.values())
    _fetch(unique, len(unique), timeout)
    stored = [_local_value(ref) for ref in refs]
    if any(value is None for value in stored):
        raise exceptions.GetTimeoutError('Get timed out: some object(s) not ready.')
    return stored


def wait(ray_waitables, *, num_returns=1, timeout=None, fetch_local=True):
    """Returns ``(ready, not_ready)``: the first ``num_returns`` references (in
    input order) whose objects are available, and the rest, also in input order."""
    _ensure_initialized()
    if isinstance(ray_waitables, ObjectRef):
        raise TypeError('wait() expected a list of ray.ObjectRef or '
                        'ray.ObjectRefGenerator, got a single ray.ObjectRef or '
                        'ray.ObjectRefGenerator ' + repr(ray_waitables))
    if not isinstance(ray_waitables, list):
        raise TypeError('wait() expected a list of ray.ObjectRef or '
                        'ray.ObjectRefGenerator, got ' + str(type(ray_waitables)))
    for ref in ray_waitables:
        if not isinstance(ref, ObjectRef):
            raise TypeError('wait() expected a list of ray.ObjectRef or '
                            'ray.ObjectRefGenerator, got list containing ' +
                            str(type(ref)))
    if timeout is not None and timeout < 0:
        raise ValueError("The 'timeout' argument must be nonnegative. Received " +
                         str(timeout))
    if not ray_waitables:
        return [], []
    if len(set(ref.store_id for ref in ray_waitables)) != len(ray_waitables):
        raise ValueError('Wait requires a list of unique ray_waitables.')
    if num_returns <= 0:
        raise ValueError('Invalid number of objects to return ' + str(num_returns) + '.')
    if num_returns > len(ray_waitables):
        raise ValueError('num_returns cannot be greater than the number of '
                         'ray_waitables provided to ray.wait.')
    if _client is not None:
        return _client.call('wait', ray_waitables, num_returns=num_returns,
                            timeout=timeout, fetch_local=fetch_local)
    _fetch(ray_waitables, num_returns, timeout)
    ready, not_ready = [], []
    for ref in ray_waitables:
        if len(ready) < num_returns and _local_value(ref) is not None:
            ready.append(ref)
        else:
            not_ready.append(ref)
    return ready, not_ready


def put(value, *, _owner=None):
    """Stores ``value`` in the object store and returns a reference to it."""
    _ensure_initialized()
    if isinstance(value, ObjectRef):
        raise TypeError("Calling 'put' on an ray.ObjectRef is not allowed. If you really "
                        "want to do this, you can wrap the ray.ObjectRef in a list and "
                        "call 'put' on it.")
    if _client is not None:
        return _client.call('put', value)
    ref = _new_ref()
    stored = wrap_value(value)
    ref.create_object(stored)
    # the owner of a put object has it locally at once, as in Ray (small objects
    # otherwise live only on their home PE)
    object_store[_charm().myPe()].ckLocalBranch()._object_store.receive_remote_object(
        ref.store_id, stored)
    return ref


def _resolve_args(args, kwargs):
    """Replaces top-level ObjectRef arguments with their values, like Ray does.
    Raises _DependencyFailed if any of them holds an error."""
    refs = [a for a in args if isinstance(a, ObjectRef)]
    refs += [v for v in kwargs.values() if isinstance(v, ObjectRef)]
    if not refs:
        return args, kwargs
    unique = list({ref.store_id: ref for ref in refs}.values())
    _fetch(unique, len(unique), None)

    def value_of(arg):
        if not isinstance(arg, ObjectRef):
            return arg
        stored = _local_value(arg)
        if isinstance(stored, _ErrorValue):
            raise _DependencyFailed(stored.error)
        return unwrap_value(stored)

    return ([value_of(a) for a in args], {k: value_of(v) for k, v in kwargs.items()})


class _DependencyFailed(Exception):
    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


def _dependency_error(function_name, error):
    if isinstance(error, exceptions.RayTaskError):
        cause = error.cause
        text = exceptions.RayTaskError.__str__(error)
        prefix = 'ray.exceptions.RayTaskError: '
    else:
        cause = error
        text = str(error)
        prefix = 'ray.exceptions.' + type(error).__name__ + ': '
    return exceptions.RayTaskError(
        function_name,
        '  At least one of the input arguments for this task could not be computed:\n' +
        prefix + text, cause, ip=get_node_ip_address())


def _store_results(return_refs, result, function_name):
    n = len(return_refs)
    if n == 0:
        return
    if n == 1:
        _store(return_refs[0], result)
        return
    try:
        values = list(result)
    except TypeError:
        values = None
    if values is None or len(values) != n:
        got = 'a non-iterable' if values is None else str(len(values)) + ' objects'
        err = exceptions.RayTaskError(
            function_name, '  ValueError: Task returned ' + got + ', but num_returns=' +
            str(n) + '.', ValueError('Task returned ' + got + ', but num_returns=' +
                                     str(n) + '.'), ip=get_node_ip_address())
        for ref in return_refs:
            _store_error(ref, err)
        return
    for ref, value in zip(return_refs, values):
        _store(ref, value)


def _copy_if_local(pe, args, kwargs):
    """Ray passes arguments by value. Charm4py delivers messages to the same PE
    without serializing them, so copy them when the receiver is local."""
    if pe == _charm().myPe():
        return copy.deepcopy(args), copy.deepcopy(kwargs)
    return args, kwargs


# ---------------------------------------------------------------- topology

def _my_node():
    if _client is not None:
        return _nodes[_client_host]
    return _nodes[_charm().myHost()]


def _node_index(node_id):
    for i, node in enumerate(_nodes):
        if node['NodeID'] == node_id:
            return i
    return None


def _public_node(node):
    return {k: copy.deepcopy(v) for k, v in node.items() if not k.startswith('_')}


def nodes():
    """The node table: one entry per physical host in the Charm4py job."""
    _ensure_initialized()
    return [_public_node(node) for node in _nodes]


def cluster_resources():
    _ensure_initialized()
    total = {}
    for node in _nodes:
        for key, amount in node['Resources'].items():
            total[key] = total.get(key, 0.0) + amount
    return total


def available_resources():
    _ensure_initialized()
    if _client is not None:
        return _client.call('available_resources')
    total = {}
    for avail in _control.available_resources_per_node(ret=True).get().values():
        for key, amount in avail.items():
            total[key] = total.get(key, 0.0) + amount
    return {k: v for k, v in total.items() if v > 0}


def available_resources_per_node():
    _ensure_initialized()
    if _client is not None:
        return _client.call('available_resources_per_node')
    return _control.available_resources_per_node(ret=True).get()


def total_resources_per_node():
    _ensure_initialized()
    return {node['NodeID']: dict(node['Resources']) for node in _nodes}


_node_ip = None


def get_node_ip_address():
    global _node_ip
    if _initialized and _nodes:
        return _my_node()['NodeManagerAddress']
    if _node_ip is None:
        charm = None if _client is not None else _charm()
        if charm is not None and charm.started and charm.numHosts() == 1:
            _node_ip = '127.0.0.1'
        else:
            _node_ip = _local_node_info()[1]
    return _node_ip


def _worker_pes(node_index=None):
    """PEs that run tasks: those of a node (or of every node), leaving out PE 0,
    which runs the driver, whenever there is another choice."""
    charm = _charm()
    if node_index is None:
        pes = list(range(charm.numPes()))
    else:
        pes = list(_nodes[node_index]['_pes'])
    if len(pes) > 1 and 0 in pes:
        pes.remove(0)
    return pes


def _choose_task_pe(options):
    from .util.scheduling_strategies import (PlacementGroupSchedulingStrategy,
                                             NodeAffinitySchedulingStrategy)
    from . import _options
    strategy = _options.scheduling_strategy(options)
    node_index = None
    if isinstance(strategy, PlacementGroupSchedulingStrategy):
        bundle_nodes = _placement_group_bundle_nodes(strategy.placement_group)
        index = strategy.placement_group_bundle_index
        node_index = bundle_nodes[index if index >= 0 else 0]
    elif isinstance(strategy, NodeAffinitySchedulingStrategy):
        node_index = _node_index(strategy.node_id)
        if node_index is None and not strategy.soft:
            raise ValueError('No node with node_id ' + str(strategy.node_id))
    pes = _worker_pes(node_index)
    return pes[(next(_rr_counter) + _charm().myPe()) % len(pes)]


def _placement_group_bundle_nodes(pg):
    """Node index of every bundle of a placement group, waiting for it to be
    created if necessary."""
    key = pg.id.hex()
    if key not in _pg_bundle_nodes:
        get(pg.ready())
        _pg_bundle_nodes[key] = _control.placement_group_bundle_nodes(key, ret=True).get()
    return _pg_bundle_nodes[key]


# ---------------------------------------------------------------- context

class _ExecContext(object):
    """What a running task or actor knows about itself. Stored on the greenlet that
    runs it, because tasks and actors of one PE interleave on one Python thread."""

    __slots__ = ('task_id', 'actor_id', 'actor_name', 'accelerator_ids', 'pg_id',
                 'pg_bundle_specs', 'runtime_env', 'assigned_resources', 'node_id')

    def __init__(self, task_id=None, actor_id=None, actor_name=None,
                 accelerator_ids=None, pg_id=None, pg_bundle_specs=None,
                 runtime_env=None, assigned_resources=None, node_id=None):
        self.task_id = task_id
        self.actor_id = actor_id
        self.actor_name = actor_name
        self.accelerator_ids = accelerator_ids or {}
        self.pg_id = pg_id
        self.pg_bundle_specs = pg_bundle_specs
        self.runtime_env = runtime_env
        self.assigned_resources = assigned_resources or {}
        self.node_id = node_id


def _current_context():
    return getattr(getcurrent(), '_ray_ctx', None)


def _set_context(ctx):
    getcurrent()._ray_ctx = ctx


def _my_pe():
    return _client_pe if _client is not None else _charm().myPe()


def _new_task_id():
    if _client is not None:
        # unique across the worker processes of one PE
        return format((_client_pe << 40) + ((os.getpid() & 0xffff) << 24) +
                      next(_task_counter), '048x')
    return format((_charm().myPe() << 40) + next(_task_counter), '048x')


def _worker_id():
    who = str(_my_pe()) + (':' + str(os.getpid()) if _client is not None else '')
    return hashlib.sha224(((_session_id or '') + ':worker:' + who).encode()).hexdigest()


# ---------------------------------------------------------------- actors

def kill(actor, *, no_restart=True):
    """Forcefully stops an actor. Its pending and future method calls fail with
    ActorDiedError."""
    from .actor import ActorHandle
    if not isinstance(actor, ActorHandle):
        raise ValueError('ray.kill() only supported for actors. For tasks, try '
                         'ray.cancel(). Got: ' + str(type(actor)) + '.')
    _ensure_initialized()
    if _client is not None:
        return _client.call('kill', actor, no_restart=no_restart)
    actor._mark_killed()
    _control.kill_actor(actor._key, ret=True).get()


def get_actor(name, namespace=None):
    """Returns a handle to the live actor called ``name``."""
    _ensure_initialized()
    if not name:
        raise ValueError('Please supply a non-empty value to get_actor')
    if _client is not None:
        return _client.call('get_actor', name, namespace)
    ns = namespace if namespace is not None else _namespace
    state = _control.get_actor(name, ns, ret=True).get()
    if state is None:
        raise ValueError(
            "Failed to look up actor with name '" + name + "'. This could because 1. "
            "You are trying to look up a named actor you didn't create. 2. The named "
            "actor died. 3. You did not use a namespace matching the namespace of the "
            "actor.")
    from .actor import ActorHandle
    return ActorHandle._from_state(state)


def cancel(ref, *, force=False, recursive=True):
    raise NotImplementedError('ray.cancel() is not supported by charm4py.ray yet')
