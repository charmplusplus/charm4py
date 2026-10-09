"""``ray.runtime_env``: RuntimeEnv is a dict, usable wherever a runtime_env dict
is (only ``env_vars`` and ``config`` take effect in charm4py.ray; see _options)."""


class RuntimeEnvConfig(dict):

    known_fields = {'setup_timeout_seconds', 'eager_install', 'log_files'}

    def __init__(self, setup_timeout_seconds=600, eager_install=True, log_files=None):
        super().__init__()
        if not isinstance(setup_timeout_seconds, int):
            raise TypeError('setup_timeout_seconds must be of type int, got: ' +
                            str(type(setup_timeout_seconds)))
        if setup_timeout_seconds <= 0 and setup_timeout_seconds != -1:
            raise ValueError('setup_timeout_seconds must be greater than zero or equals '
                             'to -1, got: ' + str(setup_timeout_seconds))
        self['setup_timeout_seconds'] = setup_timeout_seconds
        self['eager_install'] = eager_install
        self['log_files'] = list(log_files or [])


class RuntimeEnv(dict):
    """A runtime environment for a job, task or actor."""

    def __init__(self, *, py_modules=None, py_executable=None, working_dir=None, pip=None,
                 conda=None, container=None, env_vars=None, worker_process_setup_hook=None,
                 nsight=None, rocprof_sys=None, config=None, _validate=True,
                 image_uri=None, uv=None, **kwargs):
        super().__init__()
        runtime_env = dict(kwargs)
        for key, value in (('py_modules', py_modules), ('py_executable', py_executable),
                           ('working_dir', working_dir), ('pip', pip), ('uv', uv),
                           ('conda', conda), ('_nsight', nsight),
                           ('_rocprof_sys', rocprof_sys), ('container', container),
                           ('env_vars', env_vars), ('config', config),
                           ('worker_process_setup_hook', worker_process_setup_hook),
                           ('image_uri', image_uri)):
            if value is not None:
                runtime_env[key] = value
        self.update(runtime_env)
        if _validate and sum(self.get(k) is not None for k in ('conda', 'pip', 'uv')) > 1:
            raise ValueError("The 'pip' field, 'uv' field, and 'conda' field of "
                             'runtime_env cannot be specified at the same time.')
        if env_vars is not None and not (
                isinstance(env_vars, dict) and
                all(isinstance(k, str) and isinstance(v, str) for k, v in env_vars.items())):
            raise TypeError('runtime_env.env_vars must be a dict of str -> str, got ' +
                            repr(env_vars))

    def env_vars(self):
        return dict(self.get('env_vars') or {})

    def has_working_dir(self):
        return self.get('working_dir') is not None

    def to_dict(self):
        return dict(self)


__all__ = ['RuntimeEnvConfig', 'RuntimeEnv']
