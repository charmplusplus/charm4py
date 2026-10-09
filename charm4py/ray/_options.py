"""Validation of ``@ray.remote(...)`` and ``.options(...)`` keyword arguments.

Every option Ray 2.56 accepts is listed here, in one of three groups: options this
implementation honors, options it accepts because they have no observable effect
here, and options it rejects (with NotImplementedError) whenever they are set to
anything but their default. Unknown options raise the same ValueError as Ray."""

import numbers

# option -> default value. Options set to their default are always accepted.
_COMMON = {
    'accelerator_type': None,
    'memory': None,
    'num_cpus': None,
    'num_gpus': None,
    'object_store_memory': None,
    'placement_group': 'default',
    'placement_group_bundle_index': -1,
    'placement_group_capture_child_tasks': None,
    'resources': None,
    'runtime_env': None,
    'scheduling_strategy': None,
    'enable_task_events': None,
    'label_selector': None,
    'fallback_strategy': None,
    '_labels': None,
}

TASK_OPTIONS = dict(_COMMON, **{
    'num_returns': 1,
    'max_calls': None,
    'max_retries': None,
    'retry_exceptions': None,
    'name': None,
    '_generator_backpressure_num_objects': None,
})

ACTOR_OPTIONS = dict(_COMMON, **{
    'name': None,
    'namespace': None,
    'get_if_exists': False,
    'lifetime': None,
    'max_concurrency': None,
    'max_restarts': None,
    'max_task_retries': None,
    'max_pending_calls': None,
    'allow_out_of_order_execution': None,
    'concurrency_groups': None,
    'enable_tensor_transport': None,
})

ACTOR_METHOD_OPTIONS = {
    'num_returns': 1,
    'name': None,
    'concurrency_group': None,
    'max_task_retries': None,
    'retry_exceptions': None,
    'enable_task_events': None,
    'tensor_transport': None,
}

# Accepted at any value. They either only affect process reuse, failure handling
# that cannot occur here, or bookkeeping (memory) that is not enforced yet.
_NO_EFFECT = {'memory', 'object_store_memory', 'enable_task_events', 'max_calls',
              'max_retries', 'max_task_retries', 'allow_out_of_order_execution', 'name',
              'namespace', 'get_if_exists'}

# The accelerator kinds Ray reports from RuntimeContext.get_accelerator_ids().
ACCELERATOR_KEYS = ('HPU', 'TPU', 'neuron_cores', 'RBLN', 'FURIOSA', 'NPU', 'GPU')


def _unsupported(kind, key, value, why=''):
    msg = ('charm4py.ray does not support the ' + kind + ' option ' + key + '=' +
           repr(value) + ' yet')
    if why:
        msg += ' (' + why + ')'
    raise NotImplementedError(msg)


def _check_resource_amount(key, value):
    if not isinstance(value, numbers.Number) or isinstance(value, bool) or value < 0:
        raise ValueError('Resource quantities must be nonnegative numbers. Got ' + key +
                         '=' + repr(value))
    if key in ACCELERATOR_KEYS and value > 0 and value != int(value):
        _unsupported('resource', key, value, 'fractional accelerators')


def validate(options, valid, kind):
    """Checks ``options`` and returns them as a new dict."""
    for key, value in options.items():
        if key not in valid:
            raise ValueError('Invalid option keyword ' + key + ' for ' + kind +
                             '. Valid ones are ' + str(sorted(valid)) + '.')
        default = valid[key]
        if value is None or value == default or key in _NO_EFFECT:
            continue
        if key in ('num_cpus', 'num_gpus'):
            _check_resource_amount('GPU' if key == 'num_gpus' else 'CPU', value)
        elif key == 'resources':
            if not isinstance(value, dict):
                raise TypeError('resources must be a dict, got ' + repr(type(value)))
            for rkey, amount in value.items():
                if rkey in ('CPU', 'GPU'):
                    raise ValueError("Use the 'num_cpus' and 'num_gpus' keyword instead "
                                     "of 'CPU' and 'GPU' in 'resources' keyword")
                _check_resource_amount(rkey, amount)
        elif key == 'num_returns':
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _unsupported(kind, key, value, 'only nonnegative integers')
        elif key == 'runtime_env':
            if not isinstance(value, dict):
                raise TypeError('runtime_env must be a dict')
            unknown = set(value) - {'env_vars', 'nsight'}
            if unknown:
                _unsupported(kind, key, value, 'only env_vars is supported')
            if value.get('nsight'):
                _unsupported(kind, key, value, 'nsight')
            if kind == 'remote functions' and value.get('env_vars'):
                _unsupported(kind, key, value, 'task env_vars would leak into the '
                             'shared worker process')
        elif key == 'scheduling_strategy':
            from .util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy)
            if value not in ('DEFAULT', 'SPREAD') and not isinstance(
                    value, (PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy)):
                _unsupported(kind, key, value)
        elif key in ('placement_group', 'placement_group_bundle_index'):
            pass
        elif key == 'max_concurrency' and value == 1:
            pass
        elif key == 'max_restarts' and value == 0:
            pass
        elif key == 'lifetime' and value == 'non_detached':
            pass
        elif key == 'max_pending_calls' and value == -1:
            pass
        else:
            _unsupported(kind, key, value)
    return dict(options)


def scheduling_strategy(options):
    """Returns the scheduling strategy implied by ``options``, folding in the legacy
    ``placement_group=`` / ``placement_group_bundle_index=`` form."""
    strategy = options.get('scheduling_strategy')
    pg = options.get('placement_group', 'default')
    if strategy is None and pg not in ('default', None):
        from .util.scheduling_strategies import PlacementGroupSchedulingStrategy
        strategy = PlacementGroupSchedulingStrategy(
            pg, options.get('placement_group_bundle_index', -1))
    return strategy


def resource_request(options, default_cpus):
    """The resources a task or actor holds while it runs."""
    req = {}
    num_cpus = options.get('num_cpus')
    req['CPU'] = float(default_cpus if num_cpus is None else num_cpus)
    num_gpus = options.get('num_gpus')
    if num_gpus:
        req['GPU'] = float(num_gpus)
    for key, amount in (options.get('resources') or {}).items():
        if amount:
            req[key] = float(amount)
    return {k: v for k, v in req.items() if v > 0}
