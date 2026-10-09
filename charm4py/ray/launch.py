"""Run an unmodified Ray program on Charm4py.

    python -m charmrun.start +p4 -m charm4py.ray.launch script.py [args...]

Every process installs ``charm4py.ray`` as the ``ray`` package (and each
``ray.*`` submodule that charm4py.ray provides), then starts Charm4py. The driver
(PE 0) runs ``script.py`` as ``__main__``. Every other PE first imports the
script as ``__mp_main__`` -- exactly like multiprocessing's spawn mode -- so the
functions and classes it defines can be found by name when tasks and actors
arrive. As with spawn, the script's driver code must be under
``if __name__ == '__main__':``."""

import importlib
import os
import pkgutil
import sys
import traceback
import types


def install_ray_facade():
    """Makes ``import ray`` (and ``import ray.util.placement_group`` etc.) load
    charm4py.ray. Every submodule is registered under both names up front, so a
    module is never loaded twice under different names."""
    import charm4py.ray as pkg
    existing = sys.modules.get('ray')
    if existing is not None and existing is not pkg:
        raise RuntimeError('the real ray package is already imported in this process')
    sys.modules['ray'] = pkg
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        if info.name == pkg.__name__ + '.launch':
            continue
        module = importlib.import_module(info.name)
        sys.modules['ray' + info.name[len(pkg.__name__):]] = module
    return pkg


def _load_script(path, run_name):
    module = types.ModuleType(run_name)
    module.__file__ = path
    module.__builtins__ = __builtins__
    # pickled references name the module '__main__' on the driver and
    # '__mp_main__' on the other PEs; both names must resolve everywhere
    sys.modules['__main__'] = module
    sys.modules['__mp_main__'] = module
    sys.path.insert(0, os.path.dirname(path))
    with open(path) as f:
        code = compile(f.read(), path, 'exec')
    exec(code, module.__dict__)
    return module


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python -m charmrun.start +pN -m charm4py.ray.launch script.py [args]')
        sys.exit(2)
    install_ray_facade()
    from charm4py import charm, Chare, Group

    _script = os.path.abspath(sys.argv[1])

    class _ScriptLoader(Chare):
        """Imports the script on every PE but the driver's."""

        def __init__(self, argv):
            if charm.myPe() != 0:
                sys.argv = list(argv)
                _load_script(_script, '__mp_main__')

    def _driver(args):
        # args is argv with Charm's own options (+pe N, ...) removed
        argv = list(args)
        for i, arg in enumerate(argv):
            if os.path.abspath(arg) == _script:
                argv = argv[i:]
                break
        charm.awaitCreation(Group(_ScriptLoader, args=[argv]))
        sys.argv = argv
        code = 0
        try:
            _load_script(_script, '__main__')
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:
            traceback.print_exc()
            code = 1
        try:
            from charm4py.ray import api
            if api.is_initialized():
                api.shutdown()
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        charm.exit(code)

    charm.start(_driver)
