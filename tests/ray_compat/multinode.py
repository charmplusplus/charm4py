"""Multi-node behavior of the control plane (see charm4py/ray/_control.py):
placement across nodes, placement group strategies, two-phase reservation
under contention, locality, node affinity.

On one machine, split the PEs into logical nodes:

    CHARM4PY_RAY_FAKE_NODES=3 python -m charmrun.start +p6 -m charm4py.ray.launch \\
        tests/ray_compat/multinode.py

It also runs on Ray: on a cluster of 3 nodes with 2 CPUs and 1 (logical) GPU
each, reached through RAY_ADDRESS (see the bottom of this file for one made of
raylets on one machine). The last line is a JSON summary."""

import json
import sys
import time
import traceback

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import (PlacementGroupSchedulingStrategy,
                                            NodeAffinitySchedulingStrategy)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def check(cond, msg=''):
    if not cond:
        raise AssertionError(msg)


def eventually(fn, timeout=20.0):
    deadline = time.time() + timeout
    while True:
        value = fn()
        if value or time.time() > deadline:
            return value
        time.sleep(0.05)


@ray.remote
class Where:
    def node(self):
        return ray.get_runtime_context().get_node_id()

    def gpus(self):
        return ray.get_runtime_context().get_accelerator_ids()['GPU']

    def spawn(self, **options):
        child = Where.options(**options).remote()
        return ray.get(child.node.remote()), child


def nodes():
    return [n['NodeID'] for n in ray.nodes() if n['Alive']]


@test
def test_node_table():
    ids = nodes()
    check(len(ids) >= 3 and len(set(ids)) == len(ids), str(ids))
    per_node = ray._private.state.available_resources_per_node()
    check(sorted(per_node) == sorted(ids))
    check(all(r.get('GPU') == 1 for r in per_node.values()), str(per_node))


@test
def test_actors_spill_to_other_nodes_and_wait():
    n = len(nodes())
    actors = [Where.options(num_gpus=1, num_cpus=0).remote() for _ in range(n)]
    where = ray.get([a.node.remote() for a in actors], timeout=30)
    check(len(set(where)) == n, 'one GPU actor per node: ' + str(where))
    check(all(ray.get(a.gpus.remote()) == ['0'] for a in actors))
    extra = Where.options(num_gpus=1, num_cpus=0).remote()
    ready, _ = ray.wait([extra.node.remote()], timeout=1.0)
    check(not ready, 'no GPU is free: the extra actor waits')
    freed = where[1]
    ray.kill(actors[1])
    check(ray.get(extra.node.remote(), timeout=30) == freed,
          'placed where a GPU was freed')
    for a in actors + [extra]:
        ray.kill(a)
    check(eventually(lambda: ray.available_resources().get('GPU') == n))


@test
def test_spread_actors_use_every_node():
    # vLLM's data-parallel engines (and bench managers) are SPREAD actors that
    # need no resources; they must not all land on the creator's node
    n = len(nodes())
    actors = [Where.options(num_cpus=0, scheduling_strategy='SPREAD').remote()
              for _ in range(2 * n)]
    where = ray.get([a.node.remote() for a in actors], timeout=30)
    check(len(set(where)) == n, 'SPREAD actors left nodes unused: ' + str(where))
    for a in actors:
        ray.kill(a)


@test
def test_node_affinity():
    ids = nodes()
    target = ids[-1]
    a = Where.options(num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(
        target, soft=False)).remote()
    check(ray.get(a.node.remote(), timeout=30) == target)
    ray.kill(a)


@test
def test_actor_created_in_actor_prefers_its_node():
    ids = nodes()
    parent = Where.options(num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(
        ids[1], soft=False)).remote()
    child_node, child = ray.get(parent.spawn.remote(num_cpus=0), timeout=30)
    check(child_node == ids[1], 'local first: ' + str((child_node, ids[1])))
    ray.kill(child)
    ray.kill(parent)


def bundle_nodes(pg):
    table = ray.util.placement_group_table(pg)
    return [table['bundles_to_node_id'][i] for i in range(len(pg.bundle_specs))]


@test
def test_pg_strategies():
    n = len(nodes())
    spread = placement_group([{'GPU': 1}] * n, strategy='STRICT_SPREAD')
    ray.get(spread.ready(), timeout=30)
    check(len(set(bundle_nodes(spread))) == n, 'one bundle per node')
    # no GPU left: a packed group must wait, then fit once the first is removed
    pack = placement_group([{'GPU': 1}], strategy='STRICT_PACK')
    check(not pack.wait(1), 'waits for a GPU')
    remove_placement_group(spread)
    check(pack.wait(30), 'created once GPUs are free')
    infeasible = placement_group([{'GPU': 1}] * 2, strategy='STRICT_PACK')
    check(not infeasible.wait(1), 'two GPUs on one node never fit')
    remove_placement_group(infeasible)
    remove_placement_group(pack)
    check(eventually(lambda: ray.available_resources().get('GPU') == n))


@test
def test_pg_actors_on_bundle_nodes():
    n = len(nodes())
    pg = placement_group([{'GPU': 1, 'CPU': 1}] * n, strategy='SPREAD')
    ray.get(pg.ready(), timeout=30)
    expected = bundle_nodes(pg)
    actors = [Where.options(num_gpus=1, num_cpus=1,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i))
              .remote() for i in range(n)]
    check(ray.get([a.node.remote() for a in actors], timeout=30) == expected)
    remove_placement_group(pg)
    for a in actors:
        try:
            ray.get(a.node.remote(), timeout=30)
            check(False, 'actors die with their group')
        except ray.exceptions.RayActorError:
            pass


@test
def test_pg_reservation_under_contention():
    """Many groups at once, more than fit: never oversubscribed, and every
    group is eventually created as others are removed."""
    n = len(nodes())
    groups = [placement_group([{'GPU': 1}] * 2, strategy='SPREAD') for _ in range(2 * n)]
    deadline = time.time() + 60
    created = set()
    while len(created) < len(groups) and time.time() < deadline:
        ready, _ = ray.wait([g.ready() for g in groups if g.id.hex() not in created],
                            num_returns=1, timeout=5)
        reserved = sum(len(g.bundle_specs) for g in groups
                       if ray.util.placement_group_table(g)['state'] == 'CREATED')
        check(reserved <= n, 'oversubscribed: ' + str(reserved) + ' GPUs reserved')
        for g in groups:
            if g.id.hex() not in created and \
                    ray.util.placement_group_table(g)['state'] == 'CREATED':
                created.add(g.id.hex())
                remove_placement_group(g)
    check(len(created) == len(groups), str(len(created)) + ' of ' + str(len(groups)))
    check(eventually(lambda: ray.available_resources().get('GPU') == n))


@test
def test_pg_concurrent_reservations_fill_the_cluster():
    """Groups created back to back that exactly fill the cluster are all created:
    a group whose chosen node filled up in the meantime must look elsewhere,
    not wait for that node to free resources."""
    n = len(nodes())
    for _ in range(3):
        groups = [placement_group([{'GPU': 1}]) for _ in range(n)]
        ready, _ = ray.wait([g.ready() for g in groups], num_returns=n, timeout=20)
        check(len(ready) == n, str(len(ready)) + ' of ' + str(n) + ' groups created')
        for g in groups:
            remove_placement_group(g)
        check(eventually(lambda: ray.available_resources().get('GPU') == n))


@test
def test_named_actors_across_nodes():
    ids = nodes()
    actors = [Where.options(name='mn_' + str(i), namespace='mn', num_cpus=0,
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                ids[i % len(ids)], soft=False)).remote()
              for i in range(6)]
    for i in range(6):
        found = ray.get_actor('mn_' + str(i), namespace='mn')
        check(ray.get(found.node.remote(), timeout=30) == ids[i % len(ids)])
    for a in actors:
        ray.kill(a)
    check(eventually(lambda: _lookup_fails('mn_0', 'mn')))


def _lookup_fails(name, namespace):
    try:
        ray.get_actor(name, namespace=namespace)
        return False
    except ValueError:
        return True


def main(argv):
    import os
    if os.environ.get('RAY_ADDRESS'):
        ray.init()   # an existing cluster (ray.cluster_utils.Cluster, say)
    else:
        ray.init(num_cpus=2, num_gpus=1)
    results = {}
    for t in TESTS:
        if argv and not any(a in t.__name__ for a in argv):
            continue
        t0 = time.time()
        try:
            t()
            status, detail = 'PASS', ''
        except Exception as e:
            status, detail = 'FAIL', type(e).__name__ + ': ' + str(e)[:300] + '\n' + \
                traceback.format_exc()
        results[t.__name__] = status
        print('%-4s %-45s %6.2fs %s' % (status, t.__name__, time.time() - t0, detail),
              flush=True)
    passed = sum(1 for s in results.values() if s == 'PASS')
    print('multinode: %d/%d passed' % (passed, len(results)), flush=True)
    print('RESULTS ' + json.dumps(results), flush=True)
    ray.shutdown()
    return 0 if passed == len(results) else 1


def ray_oracle_cluster(n=3):
    """Real Ray: n raylets on this machine, like the fake nodes above.
        python -c 'import multinode as m; m.ray_oracle_cluster()' """
    import os
    import subprocess
    from ray.cluster_utils import Cluster
    cluster = Cluster(initialize_head=True, head_node_args={'num_cpus': 2, 'num_gpus': 1})
    for _ in range(n - 1):
        cluster.add_node(num_cpus=2, num_gpus=1)
    cluster.wait_for_nodes()
    env = dict(os.environ, RAY_ADDRESS=cluster.address)
    code = subprocess.call([sys.executable, __file__] + sys.argv[1:], env=env)
    cluster.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
