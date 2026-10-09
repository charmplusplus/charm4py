"""Ray Core contract tests, run unchanged against real Ray (the oracle) and
against charm4py.ray. See tests/ray_compat/README.md.

    # oracle (real Ray 2.56.x)
    python tests/ray_compat/contract.py
    # Charm4py
    python -m charmrun.start +p4 -m charm4py.ray.launch tests/ray_compat/contract.py

Arguments select tests by substring. The last output line is a JSON summary.
Resources assumed: one node with 4 CPUs and 2 (logical) GPUs, set by ray.init."""

import json
import os
import socket
import sys
import time
import traceback

import numpy as np
import ray
from ray.util.placement_group import (placement_group, placement_group_table,
                                      remove_placement_group, get_current_placement_group)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

BACKEND = 'charm4py' if getattr(ray, '__charm4py__', False) else 'ray-' + ray.__version__
NUM_CPUS = 4
NUM_GPUS = 2


# ------------------------------------------------------------------ helpers

class Fail(AssertionError):
    pass


def check(cond, msg=''):
    if not cond:
        raise Fail(msg)


def raises(exc_types, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_types as e:
        return e
    except Exception as e:
        raise Fail('expected ' + str(exc_types) + ', got ' + type(e).__name__ + ': ' +
                   str(e)[:300])
    raise Fail('expected ' + str(exc_types) + ', nothing was raised')


def eventually(fn, timeout=10.0, interval=0.1):
    deadline = time.time() + timeout
    while True:
        value = fn()
        if value or time.time() > deadline:
            return value
        time.sleep(interval)


# ------------------------------------------------------------------ remote code

@ray.remote
def add(x, y=10):
    return x + y


@ray.remote
def ident(x):
    return x


@ray.remote
def noop():
    pass


@ray.remote(num_returns=2)
def two():
    return 1, 2


@ray.remote
def three():
    return 1, 2, 3


@ray.remote
def boom():
    raise ValueError('bad')


@ray.remote
def returns_exception():
    return ValueError('just a value')


@ray.remote
def slow(t, value=None):
    time.sleep(t)
    return t if value is None else value


@ray.remote
def fib(n):
    if n < 2:
        return n
    return sum(ray.get([fib.remote(n - 1), fib.remote(n - 2)]))


@ray.remote
def array_sum(a):
    return float(np.sum(a))


@ray.remote
def make_array(n):
    return np.arange(n, dtype=np.float64)


@ray.remote
def sum_of_arrays(count, n):
    # some of these large results are produced on this task's own PE, which is
    # also their owner (home) PE
    return sum(float(a.sum()) for a in ray.get([make_array.remote(n) for _ in range(count)]))


@ray.remote
def task_context():
    ctx = ray.get_runtime_context()
    return ctx.get_task_id(), ctx.get_actor_id(), ctx.get_node_id()


@ray.remote
def call_actor(handle, n):
    return ray.get([handle.incr.remote() for _ in range(n)])[-1]


@ray.remote
def make_counter(start):
    return [Counter.remote(start=start)]


@ray.remote
class Counter:
    def __init__(self, start=0, step=1):
        self.value = start
        self.step = step
        self.log = []

    def incr(self, by=None):
        self.value += self.step if by is None else by
        return self.value

    def get(self):
        return self.value

    def append(self, i):
        self.log.append(i)

    def get_log(self):
        return self.log

    def fail(self):
        raise ValueError('actor method failed')

    def sleep(self, t):
        time.sleep(t)
        return t

    def leave(self):
        ray.actor.exit_actor()

    def add_ref(self, x):
        self.value += x
        return self.value

    @ray.method(num_returns=2)
    def pair(self):
        return self.value, -self.value

    def context(self):
        ctx = ray.get_runtime_context()
        pg = get_current_placement_group()
        return {'node_id': ctx.get_node_id(), 'actor_id': ctx.get_actor_id(),
                'task_id': ctx.get_task_id(), 'gpus': ctx.get_accelerator_ids()['GPU'],
                'pg_id': ctx.get_placement_group_id(),
                'current_pg': pg.id.hex() if pg is not None else None}


@ray.remote
class BadInit:
    def __init__(self):
        raise KeyError('constructor failed')

    def m(self):
        return 1


@ray.remote
class FakeVllmWorker:
    """The calls RayExecutorV2 makes on RayWorkerProc, without a model."""

    def __init__(self, vllm_config, rank, input_shm_handle, is_driver_worker,
                 is_driver_node=False):
        self.rank = rank
        self.config = vllm_config
        self.is_driver_worker = is_driver_worker
        self.is_driver_node = is_driver_node
        self.local_rank = None

    def get_node_and_physical_gpu_ids(self):
        ctx = ray.get_runtime_context()
        return ctx.get_node_id(), [int(x) for x in ctx.get_accelerator_ids()['GPU']]

    def create_dist_init_method(self):
        s = socket.socket()
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return 'tcp://' + ray.util.get_node_ip_address() + ':' + str(port)

    def initialize_worker(self, local_rank, env_vars, distributed_init_method,
                          driver_env_vars=None, assigned_physical_gpu_ids=None):
        for k, v in (driver_env_vars or {}).items():
            os.environ.setdefault(k, v)
        os.environ.update(env_vars)
        self.local_rank = local_rank
        self.init_method = distributed_init_method
        self.gpu_ids = assigned_physical_gpu_ids
        return None

    def wait_for_init(self):
        return {'status': 'READY', 'handle': ('mq', self.rank, self.local_rank)}

    def run(self, seconds):
        time.sleep(seconds)
        return 'finished'


# ------------------------------------------------------------------ tests

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_is_initialized():
    check(ray.is_initialized())


@test
def test_init_twice():
    raises(RuntimeError, ray.init)
    ray.init(ignore_reinit_error=True)
    check(ray.is_initialized())


@test
def test_task_basic():
    check(ray.get(add.remote(1, 2)) == 3)


@test
def test_task_kwargs_and_defaults():
    check(ray.get(add.remote(1, y=5)) == 6)
    check(ray.get(add.remote(x=2)) == 12)
    check(ray.get(add.remote(y=1, x=2)) == 3)


@test
def test_task_num_returns():
    a, b = two.remote()
    check(ray.get([a, b]) == [1, 2])
    refs = three.options(num_returns=3).remote()
    check(isinstance(refs, list) and len(refs) == 3)
    check(ray.get(refs) == [1, 2, 3])
    # options() returns a copy; the original keeps num_returns=1
    check(ray.get(three.remote()) == (1, 2, 3))
    check(add.options(num_returns=0).remote(1) is None)


@test
def test_task_num_returns_mismatch():
    refs = three.options(num_returns=2).remote()
    raises(Exception, ray.get, refs[0])


@test
def test_direct_calls_raise():
    raises(TypeError, add, 1)
    raises(TypeError, Counter)
    c = Counter.remote()
    raises(TypeError, c.get)


@test
def test_invalid_options():
    raises(ValueError, add.options, bogus=1)
    raises(ValueError, Counter.options, bogus=1)


@test
def test_returns_none():
    check(ray.get(noop.remote()) is None)
    check(ray.get(ray.put(None)) is None)


@test
def test_put_get():
    value = {'a': [1, 2, 3], 'b': 'text'}
    check(ray.get(ray.put(value)) == value)
    refs = [ray.put(i) for i in range(10)]
    check(ray.get(refs) == list(range(10)))
    check(ray.get([]) == [])
    check(ray.get(ray.put(7), timeout=0) == 7)


@test
def test_put_objectref_raises():
    raises(TypeError, ray.put, ray.put(1))


@test
def test_get_invalid_type():
    raises(ValueError, ray.get, 5)
    raises(ValueError, ray.get, (ray.put(1),))


@test
def test_large_numpy():
    a = np.arange(1 << 20, dtype=np.float64)
    ref = ray.put(a)
    check(np.array_equal(ray.get(ref), a))
    check(ray.get(array_sum.remote(ref)) == float(a.sum()))
    check(ray.get(array_sum.remote(make_array.remote(1 << 18))) ==
          float(np.arange(1 << 18).sum()))


@test
def test_large_results_owned_by_tasks():
    n = 1 << 12
    expected = 8 * float(np.arange(n).sum())
    check(ray.get(sum_of_arrays.remote(8, n), timeout=30) == expected)
    check(ray.get([sum_of_arrays.remote(8, n) for _ in range(4)], timeout=30) ==
          [expected] * 4)


@test
def test_ref_args():
    check(ray.get(ident.remote(ray.put(5))) == 5)
    inner = ray.put(9)
    out = ray.get(ident.remote([inner]))
    check(isinstance(out[0], ray.ObjectRef), 'refs nested in a list are not resolved')
    check(out[0] == inner and hash(out[0]) == hash(inner))
    check(ray.get(out[0]) == 9)


@test
def test_chained_and_nested_tasks():
    check(ray.get(add.remote(add.remote(1, 2), add.remote(3, 4))) == 10)
    check(ray.get(fib.remote(10)) == 55)


@test
def test_task_exception():
    e = raises(ray.exceptions.RayTaskError, ray.get, boom.remote())
    check(isinstance(e, ValueError), 'RayTaskError is an instance of the cause class')
    check('bad' in str(e))
    check(isinstance(e.cause, ValueError))


@test
def test_dependency_failure():
    e = raises(ray.exceptions.RayTaskError, ray.get, ident.remote(boom.remote()))
    check(isinstance(e, ValueError))


@test
def test_exception_as_value():
    value = ray.get(returns_exception.remote())
    check(isinstance(value, ValueError) and str(value) == 'just a value')


@test
def test_get_timeout():
    ref = slow.remote(1.5)
    t0 = time.time()
    e = raises(ray.exceptions.GetTimeoutError, ray.get, ref, timeout=0.2)
    check(isinstance(e, TimeoutError))
    check(time.time() - t0 < 1.2, 'timeout took too long')
    check(ray.get(ref) == 1.5)


@test
def test_wait_order_and_num_returns():
    refs = [slow.remote(2.0, 'slow'), ident.remote(1), ident.remote(2)]
    ready, not_ready = ray.wait(refs, num_returns=2, timeout=10)
    check(ready == [refs[1], refs[2]], 'ready refs keep input order')
    check(not_ready == [refs[0]])
    ray.get(refs)
    ready, not_ready = ray.wait(refs, num_returns=1)
    check(len(ready) == 1 and ready[0] in refs and len(not_ready) == 2)
    check(sorted(map(refs.index, ready + not_ready)) == [0, 1, 2])
    check([r for r in refs if r in not_ready] == not_ready, 'not_ready keeps input order')


@test
def test_wait_timeout():
    ref = slow.remote(1.5)
    t0 = time.time()
    ready, not_ready = ray.wait([ref], timeout=0.2)
    check(ready == [] and not_ready == [ref])
    check(time.time() - t0 < 1.2)
    ready, not_ready = ray.wait([ref], timeout=0)
    check(ready == [] and not_ready == [ref])
    put_ref = ray.put(3)
    ready, _ = ray.wait([put_ref], timeout=0)
    check(ready == [put_ref], 'a put object is ready at once')
    ray.get(ref)


@test
def test_wait_errors():
    r = ray.put(1)
    raises(ValueError, ray.wait, [r, r])
    raises(ValueError, ray.wait, [r], num_returns=2)
    raises(ValueError, ray.wait, [r], num_returns=0)
    raises(TypeError, ray.wait, r)
    raises(TypeError, ray.wait, (r,))
    check(ray.wait([]) == ([], []))


@test
def test_objectref_type():
    from ray.types import ObjectRef
    ref = add.remote(1, 1)
    check(isinstance(ref, ray.ObjectRef))
    # ray.types.ObjectRef is only a generic type for annotations
    check(not isinstance(ref, ObjectRef) and ObjectRef[int] is not None)
    check(isinstance(ref.hex(), str) and len(ref.hex()) == 56)
    check(ref == ref and len({ref, ref}) == 1)


@test
def test_actor_state():
    c = Counter.remote(start=5, step=2)
    check(isinstance(c, ray.actor.ActorHandle))
    for _ in range(3):
        c.incr.remote()
    check(ray.get(c.get.remote()) == 11)
    check(ray.get(c.incr.remote(by=100)) == 111)


@test
def test_actor_call_order():
    c = Counter.remote()
    for i in range(200):
        c.append.remote(i)
    check(ray.get(c.get_log.remote()) == list(range(200)))


@test
def test_actor_method_num_returns():
    c = Counter.remote(start=4)
    a, b = c.pair.remote()
    check(ray.get([a, b]) == [4, -4])


@test
def test_actor_ref_args():
    c = Counter.remote(start=ray.put(10))
    check(ray.get(c.add_ref.remote(ray.put(5))) == 15)
    check(ray.get(c.add_ref.remote(add.remote(1, 1))) == 17)


@test
def test_actor_handle_in_task():
    c = Counter.remote()
    check(ray.get(call_actor.remote(c, 5)) == 5)
    check(ray.get(c.get.remote()) == 5)


@test
def test_actor_created_in_task():
    [c] = ray.get(make_counter.remote(40))
    check(ray.get(c.incr.remote()) == 41)


@test
def test_actor_ready():
    c = Counter.remote()
    check(ray.get(c.__ray_ready__.remote()) is True)


@test
def test_named_actors():
    c = Counter.options(name='contract_named').remote(start=1)
    c.incr.remote()
    h = ray.get_actor('contract_named')
    check(isinstance(h, ray.actor.ActorHandle))
    check(ray.get(h.get.remote()) == 2)
    e = raises(ValueError, Counter.options(name='contract_named').remote)
    check('already taken' in str(e))
    same = Counter.options(name='contract_named', get_if_exists=True).remote()
    check(ray.get(same.get.remote()) == 2)
    raises(ValueError, ray.get_actor, 'contract_missing')
    ray.kill(c)
    eventually(lambda: raises(ValueError, ray.get_actor, 'contract_named'), 10)


@test
def test_actor_method_exception():
    c = Counter.remote(start=3)
    e = raises(ray.exceptions.RayTaskError, ray.get, c.fail.remote())
    check(isinstance(e, ValueError))
    check(ray.get(c.get.remote()) == 3, 'actor survives a method exception')


@test
def test_actor_constructor_failure():
    a = BadInit.remote()
    e = raises(ray.exceptions.RayActorError, ray.get, a.m.remote(), timeout=30)
    check(isinstance(e, ray.exceptions.ActorDiedError))


@test
def test_kill():
    c = Counter.remote()
    check(ray.get(c.incr.remote()) == 1)
    ray.kill(c)

    def dead():
        # ray.kill is asynchronous in Ray: a call made right after it may still run
        try:
            ray.get(c.get.remote(), timeout=10)
            return False
        except ray.exceptions.RayActorError:
            return True
    check(eventually(dead, 20), 'calls fail after kill')
    raises(ValueError, ray.kill, ray.put(1))


@test
def test_kill_fails_pending_call():
    c = Counter.remote()
    ray.get(c.__ray_ready__.remote())
    pending = c.sleep.remote(2)
    time.sleep(0.3)
    ready, _ = ray.wait([pending], timeout=0)
    check(ready == [], 'still running')
    t0 = time.time()
    ray.kill(c)
    raises(ray.exceptions.RayActorError, ray.get, pending, timeout=10)
    check(time.time() - t0 < 1.5, 'the pending call fails promptly')


@test
def test_exit_actor():
    c = Counter.remote()
    raises(ray.exceptions.RayActorError, ray.get, c.leave.remote(), timeout=30)
    raises(ray.exceptions.RayActorError, ray.get, c.get.remote(), timeout=30)


@test
def test_runtime_context_driver():
    ctx = ray.get_runtime_context()
    node_ids = [n['NodeID'] for n in ray.nodes()]
    check(ctx.get_node_id() in node_ids)
    check(len(ctx.get_node_id()) == 56)
    check(ctx.get_actor_id() is None and ctx.get_task_id() is None)
    check(isinstance(ctx.get_job_id(), str))
    check(isinstance(ctx.namespace, str))
    check(ctx.get_accelerator_ids()['GPU'] == [])
    check(get_current_placement_group() is None)


@test
def test_runtime_context_task_and_actor():
    task_id, actor_id, node_id = ray.get(task_context.remote())
    check(task_id is not None and actor_id is None)
    check(node_id in [n['NodeID'] for n in ray.nodes()])
    c = Counter.remote()
    info = ray.get(c.context.remote())
    check(info['actor_id'] == c._actor_id.hex())
    check(info['task_id'] is not None)
    check(info['gpus'] == [] and info['pg_id'] is None and info['current_pg'] is None)


@test
def test_nodes_and_resources():
    nodes = ray.nodes()
    check(len(nodes) >= 1)
    for n in nodes:
        for key in ('NodeID', 'Alive', 'NodeManagerAddress', 'Resources'):
            check(key in n, key)
        check(n['Alive'] is True)
    total = ray.cluster_resources()
    check(total.get('CPU') == float(NUM_CPUS) and total.get('GPU') == float(NUM_GPUS),
          str(total))
    check(ray.util.get_node_ip_address() == nodes[0]['NodeManagerAddress'])
    from ray._private.state import available_resources_per_node
    per_node = available_resources_per_node()
    check(set(per_node) == set(n['NodeID'] for n in nodes))


@test
def test_actor_gpu_accounting():
    def free_gpus():
        return ray.available_resources().get('GPU', 0)
    check(eventually(lambda: free_gpus() == NUM_GPUS), 'all GPUs free at start')
    c = Counter.options(num_gpus=1).remote()
    info = ray.get(c.context.remote())
    check(len(info['gpus']) == 1)
    check(eventually(lambda: free_gpus() == NUM_GPUS - 1), 'one GPU in use')
    ray.kill(c)
    check(eventually(lambda: free_gpus() == NUM_GPUS), 'GPU released by kill')


@test
def test_placement_group_basic():
    pg = placement_group([{'CPU': 1}, {'CPU': 1}], strategy='PACK')
    check(ray.get(pg.ready(), timeout=30) == pg)
    check(pg.bundle_specs == [{'CPU': 1.0}, {'CPU': 1.0}])
    table = placement_group_table(pg)
    check(table['state'] == 'CREATED' and table['strategy'] == 'PACK')
    check(sorted(table['bundles_to_node_id']) == [0, 1])
    node_ids = [n['NodeID'] for n in ray.nodes()]
    check(all(n in node_ids for n in table['bundles_to_node_id'].values()))
    remove_placement_group(pg)


@test
def test_placement_group_validation():
    raises(ValueError, placement_group, [{'CPU': 1}], strategy='NOPE')
    raises(ValueError, placement_group, [])
    raises(ValueError, placement_group, [{}])


@test
def test_placement_group_infeasible_and_pending():
    big = placement_group([{'GPU': 100}])
    ready, _ = ray.wait([big.ready()], timeout=0.5)
    check(ready == [])
    raises(ray.exceptions.GetTimeoutError, ray.get, big.ready(), timeout=0)
    remove_placement_group(big)
    first = placement_group([{'GPU': NUM_GPUS}])
    ray.get(first.ready(), timeout=30)
    second = placement_group([{'GPU': 1}])
    ready, _ = ray.wait([second.ready()], timeout=0.5)
    check(ready == [], 'no GPU is free')
    remove_placement_group(first)
    check(ray.get(second.ready(), timeout=30) == second, 'ready once GPUs are released')
    remove_placement_group(second)


@test
def test_placement_group_actors():
    pg = placement_group([{'GPU': 1}] * NUM_GPUS, strategy='PACK')
    ray.get(pg.ready(), timeout=30)
    actors = [Counter.options(num_cpus=0, num_gpus=1, scheduling_strategy=
                              PlacementGroupSchedulingStrategy(pg, i)).remote()
              for i in range(NUM_GPUS)]
    infos = ray.get([a.context.remote() for a in actors])
    gpus = sorted(g for info in infos for g in info['gpus'])
    check(gpus == [str(i) for i in range(NUM_GPUS)], 'distinct GPUs: ' + str(gpus))
    check(all(info['pg_id'] == pg.id.hex() for info in infos))
    check(all(info['current_pg'] == pg.id.hex() for info in infos))
    remove_placement_group(pg)
    for a in actors:
        raises(ray.exceptions.RayActorError, ray.get, a.get.remote(), timeout=30)
    check(eventually(lambda: ray.available_resources().get('GPU', 0) == NUM_GPUS),
          'GPUs released')


@test
def test_fake_vllm_v2_lifecycle():
    """RayExecutorV2's startup, monitoring and shutdown sequence."""
    world = NUM_GPUS
    pg = placement_group([{'GPU': 1.0} for _ in range(world)], strategy='PACK')
    ray.get(pg.ready(), timeout=30)
    table = placement_group_table(pg)
    node_ip = {n['NodeID']: n['NodeManagerAddress'] for n in ray.nodes() if n['Alive']}
    driver_node = ray.get_runtime_context().get_node_id()
    bundles = [(i, table['bundles_to_node_id'][i]) for i, b in enumerate(pg.bundle_specs)
               if b.get('GPU')]
    bundles.sort(key=lambda b: (b[1] != driver_node, b[1]))
    check(all(node in node_ip for _, node in bundles))
    workers = []
    for rank, (bundle_index, node) in enumerate(bundles):
        workers.append(FakeVllmWorker.options(
            name='contract_vllm_worker_' + str(rank), num_cpus=0, num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=bundle_index),
            runtime_env={'env_vars': {'CONTRACT_FAKE_VLLM': '1'}},
        ).remote(vllm_config={'model': 'fake'}, rank=rank, input_shm_handle=('shm', 0),
                 is_driver_worker=(rank == 0), is_driver_node=(node == driver_node)))
    ids = ray.get([w.get_node_and_physical_gpu_ids.remote() for w in workers])
    check(sorted(g for _, gs in ids for g in gs) == list(range(world)))
    init_method = ray.get(workers[0].create_dist_init_method.remote())
    check(init_method.startswith('tcp://'))
    node_workers = {}
    for i, (node, _) in enumerate(ids):
        node_workers.setdefault(node, []).append(i)
    ray.get([w.initialize_worker.remote(node_workers[ids[i][0]].index(i), {'X': '1'},
                                        init_method, {'Y': '2'},
                                        assigned_physical_gpu_ids=sorted(ids[i][1]))
             for i, w in enumerate(workers)])
    results = ray.get([w.wait_for_init.remote() for w in workers])
    check(all(r['status'] == 'READY' for r in results))
    run_refs = [w.run.remote(3.0) for w in workers]
    done, _ = ray.wait(run_refs, num_returns=1, timeout=0.5)
    check(done == [], 'workers are running')
    ray.kill(workers[1], no_restart=True)
    t0 = time.time()
    done, _ = ray.wait(run_refs, num_returns=1, timeout=5)
    check(done == [run_refs[1]], 'the monitor sees the killed worker: ' + str(done))
    check(time.time() - t0 < 2.5)
    raises(ray.exceptions.RayActorError, ray.get, run_refs[1], timeout=5)
    for w in workers:
        ray.kill(w)
    for w in workers:
        raises(ray.exceptions.RayActorError, ray.get, w.wait_for_init.remote(), timeout=5)
    remove_placement_group(pg)
    check(eventually(lambda: ray.available_resources().get('GPU', 0) == NUM_GPUS),
          'no leaked reservation')


@test
def test_shutdown_and_reinit():
    ray.shutdown()
    check(not ray.is_initialized())
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, include_dashboard=False,
             log_to_driver=False)
    check(ray.is_initialized())
    check(ray.get(add.remote(2, 3)) == 5)


# ------------------------------------------------------------------ runner

def main(argv):
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, include_dashboard=False,
             log_to_driver=False)
    selected = [t for t in TESTS if not argv or any(a in t.__name__ for a in argv)]
    results = {}
    for t in selected:
        t0 = time.time()
        try:
            t()
            status, detail = 'PASS', ''
        except Exception as e:
            status = 'FAIL'
            detail = type(e).__name__ + ': ' + str(e)[:500]
            if not isinstance(e, Fail):
                detail += '\n' + traceback.format_exc()
        elapsed = time.time() - t0
        results[t.__name__] = status
        print('%-4s %-45s %6.2fs %s' % (status, t.__name__, elapsed, detail), flush=True)
    passed = sum(1 for s in results.values() if s == 'PASS')
    print('%s: %d/%d passed' % (BACKEND, passed, len(results)), flush=True)
    print('RESULTS ' + json.dumps({'backend': BACKEND, 'results': results}), flush=True)
    ray.shutdown()
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
