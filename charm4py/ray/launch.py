"""Run an unmodified Ray program on Charm4py.

    python -m charmrun.start +p4 -m charm4py.ray.launch [options] script.py [args...]
    python -m charmrun.start +p4 -m charm4py.ray.launch [options] -m module [args...]

Options (before the script):
  --in-process          see below
  --num-cpus N          resources of each node (physical host) when the
  --num-gpus N          runtime starts, like `ray start --num-cpus ...`;
  --resources JSON      ray.init(num_cpus=...) of the first job overrides them

Every process installs ``charm4py.ray`` as the ``ray`` package (and each
``ray.*`` submodule that charm4py.ray provides), then starts Charm4py.

By default the program runs like a Ray program: the driver -- ``script.py`` run
as ``__main__`` -- is a process of its own, served by PE 0, and tasks and actors
run in worker processes (see _hosts.py). Every PE and every worker imports the
script as ``__mp_main__`` -- exactly like multiprocessing's spawn mode -- so the
functions and classes it defines can be found by name. As with spawn, the
script's driver code must be guarded by a ``__name__ == '__main__'`` check.

``--in-process`` runs the driver as a coroutine of PE 0 and tasks and actors as
coroutines of the PEs (no worker processes): lighter, but code that blocks
stalls its PE.

Processes started by the job's processes (by the driver, say) inherit a
PYTHONPATH entry that makes their ``import ray`` load charm4py.ray too (see
_shim/ray), and CHARM4PY_RAY_ADDRESS, through which their ``ray.init()`` joins
the job."""

import importlib
import os
import pkgutil
import sys
import traceback
import types


def install_ray_facade(worker=False):
    """Makes ``import ray`` (and ``import ray.util.placement_group`` etc.) load
    charm4py.ray. Every submodule is registered under both names up front, so a
    module is never loaded twice under different names. ``worker``: in a worker
    process, which does not load Charm++ (so not the Charm chares)."""
    import charm4py.ray as pkg
    existing = sys.modules.get('ray')
    if existing is not None and existing is not pkg:
        raise RuntimeError('the real ray package is already imported in this process')
    sys.modules['ray'] = pkg
    skip = {pkg.__name__ + '.launch', pkg.__name__ + '._worker_main'}
    if worker:
        skip.update((pkg.__name__ + '._hosts', pkg.__name__ + '._control'))
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        if info.name in skip:
            continue
        module = importlib.import_module(info.name)
        sys.modules['ray' + info.name[len(pkg.__name__):]] = module
    return pkg


def _load_script(path, run_name):
    module = types.ModuleType(run_name)
    module.__file__ = path
    module.__builtins__ = __builtins__
    # pickled references name the module '__main__' in the driver and
    # '__mp_main__' elsewhere; both names must resolve everywhere
    sys.modules['__main__'] = module
    sys.modules['__mp_main__'] = module
    sys.path.insert(0, os.path.dirname(path))
    with open(path) as f:
        code = compile(f.read(), path, 'exec')
    exec(code, module.__dict__)
    return module


def _parse_options(args):
    """(options, rest of args): the launcher's options come before the script."""
    import json
    options = {'in_process': False, 'num_cpus': None, 'num_gpus': None, 'resources': None}
    while args and args[0].startswith('--'):
        flag = args[0]
        if flag == '--in-process':
            options['in_process'] = True
            args = args[1:]
            continue
        key = flag[2:].replace('-', '_')
        if key not in options or len(args) < 2:
            raise SystemExit('charm4py.ray.launch: unknown option or missing value: ' + flag)
        value = args[1]
        options[key] = json.loads(value) if key == 'resources' else float(value)
        args = args[2:]
    return options, args


def run_main(script, module):
    """Runs the driver's code: the script, or the module (python -m)."""
    if module is not None:
        import runpy
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    else:
        _load_script(script, '__main__')


def _launch():
    options, args = _parse_options(sys.argv[1:])
    in_process = options['in_process']
    module = None
    if args and args[0] == '-m':
        # like python -m: run a module (pytest, say) as the driver
        module, args = (args[1] if len(args) > 1 else None), args[1:]
    if not args:
        print('usage: python -m charmrun.start +pN -m charm4py.ray.launch [--in-process] '
              '[--num-cpus N] [--num-gpus N] [--resources JSON] (script.py | -m module) '
              '[args]')
        sys.exit(2)
    install_ray_facade()
    if not in_process:
        # Ctrl-C is for the driver (a process of its own), which ends the job
        # when it exits; the Charm4py processes keep serving it until then
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    shim = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_shim')
    os.environ['PYTHONPATH'] = os.pathsep.join(
        [shim] + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p])
    from charm4py import charm, Chare, Group
    from charm4py.ray import api

    script = None if module else os.path.abspath(args[0])
    # worker processes import the script too (as __mp_main__)
    api._main_script = script
    api._main_module = module
    api._default_isolation = 'none' if in_process else 'process'
    api._default_resources = {k: options[k] for k in ('num_cpus', 'num_gpus', 'resources')}

    class _ScriptLoader(Chare):
        """Imports the script on a PE, without running its driver code, so the
        functions and classes it defines can be found by name. With
        --in-process, PE 0 runs the driver itself and is skipped."""

        def __init__(self, argv):
            if script is not None and not (in_process and charm.myPe() == 0):
                sys.argv = list(argv)
                _load_script(script, '__mp_main__')

    def run_driver_here(argv):
        sys.argv = argv
        code = 0
        try:
            run_main(script, module)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:
            traceback.print_exc()
            code = 1
        try:
            if api.is_initialized():
                api.shutdown()
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        charm.exit(code)

    def main(charm_args):
        # charm_args is argv with Charm's own options (+pe N, ...) removed
        argv = list(charm_args)
        for i, arg in enumerate(argv):
            if module is not None and arg == '-m' and i + 1 < len(argv) and \
                    argv[i + 1] == module:
                argv = argv[i + 1:]
                break
            if module is None and os.path.abspath(arg) == script:
                argv = argv[i:]
                break
        charm.awaitCreation(Group(_ScriptLoader, args=[argv]))
        from charm4py.ray._hosts import _Gateway
        if in_process:
            # the driver runs as a coroutine of PE 0 (blocking code stalls PE 0)
            Chare(_Gateway, args=[None], onPE=0).address(ret=True).get()
            run_driver_here(argv)
        else:
            # the driver runs in a process of its own, like a Ray driver; the job
            # ends when it does
            Chare(_Gateway, args=[argv], onPE=0)

    # charm.start registers the Chare classes of __main__ (this module)
    globals()['_ScriptLoader'] = _ScriptLoader
    charm.start(main)


if __name__ == '__main__':
    _launch()
