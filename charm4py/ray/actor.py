"""``@ray.remote`` classes (Ray actors): ActorClass, ActorHandle, ActorMethod."""

import collections
import functools
import inspect
import itertools
import weakref

from . import api, _options
from ._serialization import describe
from .exceptions import ActorAlreadyExistsError, ActorDiedError
from .remote_function import _returns

# How many calls per handle are remembered so ray.kill() can fail them at once
# from the caller's side (the actor's own PE may be busy and unable to).
_INFLIGHT_LIMIT = 10000

_SPECIAL_METHODS = ('__ray_ready__', '__ray_terminate__')

_handle_uids = itertools.count()


class _ActorExit(SystemExit):
    """Raised by exit_actor() to end the actor after the current method."""


def exit_actor():
    """Intentionally exit the current actor. Its pending method calls fail."""
    ctx = api._current_context()
    if ctx is None or ctx.actor_id is None:
        raise TypeError('exit_actor API is called on a non-actor worker.')
    raise _ActorExit(0)


def method(*args, **kwargs):
    """Annotates an actor method, e.g. ``@ray.method(num_returns=2)``."""
    valid = set(_options.ACTOR_METHOD_OPTIONS) | {'_generator_backpressure_num_objects'}
    if len(args) == 1 and not kwargs and callable(args[0]):
        raise ValueError('The @ray.method decorator must be applied using at least one '
                         'of the arguments.')
    if args:
        raise ValueError('The @ray.method decorator must be applied only with keyword '
                         'arguments.')
    for key in kwargs:
        if key not in valid:
            raise ValueError('Unexpected keyword argument to @ray.method: "' + key + '".')
    _options.validate({k: v for k, v in kwargs.items() if k in _options.ACTOR_METHOD_OPTIONS},
                      _options.ACTOR_METHOD_OPTIONS, 'actor methods')

    def annotate(m):
        m.__ray_num_returns__ = kwargs.get('num_returns', 1)
        return m
    return annotate


def _method_meta(cls):
    meta = {}
    for name, member in inspect.getmembers(cls, lambda m: inspect.isfunction(m) or
                                           inspect.ismethod(m)):
        if name.startswith('__') and name.endswith('__'):
            continue
        meta[name] = getattr(member, '__ray_num_returns__', 1)
    for name in _SPECIAL_METHODS:
        meta[name] = 1
    return meta


class _ActorID(object):

    def __init__(self, hex_id):
        self._hex = hex_id

    def hex(self):
        return self._hex

    def binary(self):
        return bytes.fromhex(self._hex)

    def __eq__(self, other):
        return isinstance(other, _ActorID) and other._hex == self._hex

    def __hash__(self):
        return hash(self._hex)

    def __repr__(self):
        return 'ActorID(' + self._hex + ')'


class ActorClass(object):
    """A remote class. ``.remote(...)`` creates an actor and returns its handle."""

    def __init__(self, cls, options):
        if not inspect.isclass(cls):
            raise TypeError('ActorClass expects a class, got ' + repr(cls))
        self._cls = cls
        self._class_name = cls.__name__
        self.__ray_target__ = cls
        self._default_options = _options.validate(options, _options.ACTOR_OPTIONS,
                                                  'actors')
        self._method_meta = _method_meta(cls)
        self._desc = None
        functools.update_wrapper(self, cls, updated=())

    def __call__(self, *args, **kwargs):
        raise TypeError("Actors cannot be instantiated directly. Instead of '" +
                        self._cls.__name__ + "()', use '" + self._cls.__name__ +
                        ".remote()'.")

    def __reduce__(self):
        return (_rebuild_actor_class, (describe(self._cls, self), self._default_options))

    def options(self, **options):
        merged = dict(self._default_options)
        merged.update(_options.validate(options, _options.ACTOR_OPTIONS, 'actors'))
        return _ActorClassWithOptions(self, merged)

    def remote(self, *args, **kwargs):
        return self._remote(args, kwargs, self._default_options)

    def bind(self, *args, **kwargs):
        raise NotImplementedError('Ray DAGs (.bind) are not supported by charm4py.ray')

    def _spec(self):
        """What the host needs to create this actor, without importing it."""
        if self._desc is None:
            self._desc = describe(self._cls, self)
        return (self._desc, self._class_name, self._method_meta)

    @staticmethod
    def _from_spec(spec):
        actor_class = ActorClass.__new__(ActorClass)
        actor_class._cls = None
        actor_class._desc, actor_class._class_name, actor_class._method_meta = spec
        actor_class._default_options = {}
        return actor_class

    def _remote(self, args, kwargs, options):
        api._ensure_initialized()
        if api._client is not None:
            args, kwargs = api._client_args(args, kwargs)
            return api._client.call('_create_actor', self._spec(), args, kwargs, options)
        charm = api._charm()
        if self._desc is None:
            self._desc = describe(self._cls, self)
        name = options.get('name')
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError('Actor name must be a nonempty string, got ' + repr(name))
        namespace_opt = options.get('namespace')
        namespace = namespace_opt if namespace_opt is not None else api._namespace()
        key = (charm.myPe(), next(api._actor_counter))
        actor_id = format(key[0], '08x') + format(key[1], '016x') + api._session_id[:8]
        state = (key, self._class_name, self._method_meta, actor_id)
        meta = {
            'key': key, 'actor_id': actor_id, 'class_name': self._class_name,
            'name': name, 'namespace': namespace, 'handle_state': state,
            'get_if_exists': bool(options.get('get_if_exists')),
            'request': _options.resource_request(options, default_cpus=0),
            'strategy': _options.scheduling_strategy(options),
            'runtime_env': api._merged_runtime_env(options.get('runtime_env')),
            'job': api._current_job(),
        }
        if name is not None:
            result = api._control.create_actor(meta, ret=True).get()
            if result[0] == 'exists':
                return ActorHandle._from_state(result[1])
            if result[0] == 'taken':
                raise ActorAlreadyExistsError(
                    'The name ' + name + ' (namespace=' + str(namespace_opt) + ') is '
                    'already taken. Please use a different name or get the existing '
                    "actor using ray.get_actor('" + name + "', namespace='" +
                    str(namespace_opt) + "')")
        else:
            api._control.create_actor(meta)
        handle = ActorHandle._from_state(state)
        handle._construct(self._desc, args, kwargs)
        return handle


class _ActorClassWithOptions(object):

    def __init__(self, actor_class, options):
        self._actor_class = actor_class
        self._options = options

    def remote(self, *args, **kwargs):
        return self._actor_class._remote(args, kwargs, self._options)

    def options(self, **options):
        merged = dict(self._options)
        merged.update(_options.validate(options, _options.ACTOR_OPTIONS, 'actors'))
        return _ActorClassWithOptions(self._actor_class, merged)

    def bind(self, *args, **kwargs):
        raise NotImplementedError('Ray DAGs (.bind) are not supported by charm4py.ray')


def _rebuild_actor_class(desc, options):
    from ._serialization import resolve
    return ActorClass(resolve(desc), options)


class ActorHandle(object):
    """A handle to an actor. Calls made through one handle run in the order they
    were made; each deserialized copy of a handle is a separate caller."""

    def __init__(self, key, class_name, method_meta, actor_id):
        self._key = key
        self._class_name = class_name
        self._method_meta = method_meta
        self._actor_id = _ActorID(actor_id)
        self._uid = next(_handle_uids)
        # in a worker process the host-side copy of this handle is the caller
        self._caller = None if api._client is not None else (
            api._charm().myPe(), next(api._caller_counter))
        self._seq = 0
        self._killed = None
        self._inflight = collections.deque()
        self._proxy = None

    @staticmethod
    def _from_state(state):
        return ActorHandle(*state)

    def _state(self):
        return (self._key, self._class_name, self._method_meta, self._actor_id.hex())

    def __reduce__(self):
        return (ActorHandle._from_state, (self._state(),))

    def __getattr__(self, name):
        meta = self.__dict__.get('_method_meta')
        if meta is not None and name in meta:
            return ActorMethod(self, name, meta[name])
        raise AttributeError("'ActorHandle' object has no attribute '" + name + "'")

    def __eq__(self, other):
        return isinstance(other, ActorHandle) and other._key == self._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return 'Actor(' + self._class_name + ', ' + self._actor_id.hex() + ')'

    def _element(self):
        if self._proxy is None:
            self._proxy = api._actors[self._key]
        return self._proxy

    def _is_local(self):
        charm = api._charm()
        return self._key in charm.arrays.get(api._actors.aid, ())

    def _construct(self, desc, args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        if self._is_local():
            args, kwargs = api._copy_if_local(api._charm().myPe(), args, kwargs)
        self._element().construct(desc, args, kwargs)

    def _invoke(self, method_name, args, kwargs, num_returns):
        api._ensure_initialized()
        if api._client is not None:
            args, kwargs = api._client_args(args, kwargs)
            return api._client.call('_invoke', self, method_name, args, kwargs, num_returns)
        refs = [api._new_ref() for _ in range(num_returns)]
        if self._killed is not None:
            for ref in refs:
                api._store_error(ref, self._killed)
            return _returns(refs)
        seq = self._seq
        self._seq += 1
        if refs:
            # weakly: a call whose results nobody references need not be failed
            self._inflight.append([weakref.ref(ref) for ref in refs])
            if len(self._inflight) > _INFLIGHT_LIMIT:
                self._inflight.popleft()
        args, kwargs = list(args), dict(kwargs)
        if self._is_local():
            args, kwargs = api._copy_if_local(api._charm().myPe(), args, kwargs)
        self._element().invoke(self._caller, seq, method_name, args, kwargs, refs)
        return _returns(refs)

    def _mark_killed(self):
        """Fails the calls made through this handle that may still be running.
        The object store keeps the first value stored for an object, so calls that
        already finished keep their results."""
        self._killed = ActorDiedError(actor_id=self._actor_id.hex(), error_msg=
                                      killed_message(self._class_name, self._actor_id.hex()))
        for refs in self._inflight:
            for weak in refs:
                ref = weak()
                if ref is not None:
                    api._store_error(ref, self._killed)
        self._inflight.clear()


def killed_message(class_name, actor_id):
    return ('The actor died unexpectedly before finishing this task.\n\tclass_name: ' +
            class_name + '\n\tactor_id: ' + actor_id + '\n\tnamespace: ' +
            str(api._namespace()) + '\n\tip: ' + api.get_node_ip_address() +
            '\nThe actor was killed by `ray.kill`.')


class ActorMethod(object):

    def __init__(self, handle, name, num_returns):
        self._handle = handle
        self._name = name
        self._num_returns = num_returns

    def __call__(self, *args, **kwargs):
        raise TypeError("Actor methods cannot be called directly. Instead of running "
                        "'object." + self._name + "()', try 'object." + self._name +
                        ".remote()'.")

    def remote(self, *args, **kwargs):
        return self._handle._invoke(self._name, args, kwargs, self._num_returns)

    def options(self, **options):
        options = _options.validate(options, _options.ACTOR_METHOD_OPTIONS,
                                    'actor methods')
        return ActorMethod(self._handle, self._name,
                           options.get('num_returns', self._num_returns))

    def bind(self, *args, **kwargs):
        raise NotImplementedError('Ray DAGs (.bind) are not supported by charm4py.ray')
