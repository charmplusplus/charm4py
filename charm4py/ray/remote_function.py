"""``@ray.remote`` functions (Ray tasks)."""

import functools

from . import api, _options
from ._serialization import describe


def _returns(refs):
    if len(refs) == 1:
        return refs[0]
    if not refs:
        return None
    return refs


class RemoteFunction(object):
    """A remote function. ``.remote()`` runs it as a task on some PE and returns
    ObjectRefs for its results; ``.options()`` returns a copy with other options."""

    def __init__(self, function, options):
        self._function = function
        self.__ray_target__ = function
        self._default_options = _options.validate(options, _options.TASK_OPTIONS,
                                                  'remote functions')
        self._desc = None
        functools.update_wrapper(self, function, updated=())

    def __call__(self, *args, **kwargs):
        name = self._function.__module__ + '.' + self._function.__qualname__
        raise TypeError("Remote functions cannot be called directly. Instead of running '" +
                        name + "()', try '" + name + ".remote()'.")

    def __reduce__(self):
        return (_rebuild_remote_function, (describe(self._function, self),
                                           self._default_options))

    def options(self, **options):
        merged = dict(self._default_options)
        merged.update(_options.validate(options, _options.TASK_OPTIONS,
                                        'remote functions'))
        return _FunctionWithOptions(self, merged)

    def remote(self, *args, **kwargs):
        return self._remote(args, kwargs, self._default_options)

    def bind(self, *args, **kwargs):
        raise NotImplementedError('Ray DAGs (.bind) are not supported by charm4py.ray')

    def _remote(self, args, kwargs, options):
        api._ensure_initialized()
        if api._client is not None:
            return api._client.call('_submit_task', self, args, kwargs, options)
        if self._desc is None:
            self._desc = describe(self._function, self)
        num_returns = options.get('num_returns', 1)
        refs = [api._new_ref() for _ in range(num_returns)]
        pe = api._choose_task_pe(options)
        args, kwargs = api._copy_if_local(pe, list(args), dict(kwargs))
        pg_info = None
        strategy = _options.scheduling_strategy(options)
        from .util.scheduling_strategies import PlacementGroupSchedulingStrategy
        if isinstance(strategy, PlacementGroupSchedulingStrategy) and \
                strategy.placement_group_capture_child_tasks:
            pg = strategy.placement_group
            pg_info = (pg.id.hex(), pg.bundle_specs)
        api._workers[pe].execute_task(self._desc, args, kwargs, refs,
                                      options.get('name'), pg_info, api._current_job())
        return _returns(refs)


class _FunctionWithOptions(object):

    def __init__(self, function, options):
        self._function = function
        self._options = options

    def remote(self, *args, **kwargs):
        return self._function._remote(args, kwargs, self._options)

    def options(self, **options):
        merged = dict(self._options)
        merged.update(_options.validate(options, _options.TASK_OPTIONS,
                                        'remote functions'))
        return _FunctionWithOptions(self._function, merged)

    def bind(self, *args, **kwargs):
        raise NotImplementedError('Ray DAGs (.bind) are not supported by charm4py.ray')


def _rebuild_remote_function(desc, options):
    from ._serialization import resolve
    return RemoteFunction(resolve(desc), options)
