"""Control-plane load of many vLLM replicas, without models (step 5, H1).

Each replica is driven by a manager actor, as vLLM's data-parallel engines
each drive their own RayExecutorV2: the manager reserves a placement group of
`--world` GPU bundles, creates one worker actor per bundle with
`num_cpus=0, num_gpus=1` and a PlacementGroupSchedulingStrategy, makes the
calls RayExecutorV2 makes during startup (node and GPU IDs, the distributed
init method, initialize_worker, wait_for_init), and starts the long-running
`run` call whose pending reference vLLM keeps as a liveness sentinel. Stopping
kills the workers, waits until every run reference fails, and removes the
placement group.

The same script runs on charm4py.ray and on Ray:

    python -m charmrun.start +p8 -m charm4py.ray.launch --num-gpus 8 \\
        tests/ray_compat/bench/replicas.py --replicas 8 --world 1
    RAY_ADDRESS=auto python tests/ray_compat/bench/replicas.py --replicas 8 --world 1

GPUs can be logical (`--num-gpus`, `ray start --num-gpus`): workers never
touch a device. Each round starts all replicas at once, waits until all are
running, then stops them all. Prints one JSON line per replica and round
(REPLICA ...) and one per round (ROUND ...). Phase times come from the
manager's clock only, so clock skew between nodes does not enter them; round
times come from the driver's clock.
"""

import argparse
import json
import os
import socket
import sys
import time

import ray
from ray.util.placement_group import (placement_group, placement_group_table,
                                      remove_placement_group)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class Worker:
    """The calls RayExecutorV2 makes on RayWorkerProc, without a model."""

    def __init__(self, rank, init_delay):
        self.rank = rank
        self.init_delay = init_delay

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
                          assigned_physical_gpu_ids=None):
        os.environ.update(env_vars)
        self.local_rank = local_rank
        if self.init_delay:
            time.sleep(self.init_delay)   # stands in for loading weights

    def wait_for_init(self):
        return {'status': 'READY', 'rank': self.rank}

    def run(self):
        while True:
            time.sleep(3600)


@ray.remote(num_cpus=0)
class ReplicaManager:

    def __init__(self, replica_id):
        self.id = replica_id
        self.pg = None
        self.workers = []
        self.run_refs = []

    def node(self):
        return ray.get_runtime_context().get_node_id()

    def start(self, world, strategy, init_delay):
        t = {'start': time.time()}
        self.pg = placement_group([{'GPU': 1.0}] * world, strategy=strategy)
        ray.get(self.pg.ready())
        t['pg_ready'] = time.time()
        table = placement_group_table(self.pg)
        here = ray.get_runtime_context().get_node_id()
        bundles = sorted(range(world), key=lambda i: (table['bundles_to_node_id'][i] != here,
                                                      table['bundles_to_node_id'][i], i))
        self.workers = [Worker.options(
            num_cpus=0, num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=b),
        ).remote(rank, init_delay) for rank, b in enumerate(bundles)]
        t['actors_submitted'] = time.time()
        ids = ray.get([w.get_node_and_physical_gpu_ids.remote() for w in self.workers])
        t['first_call'] = time.time()
        init_method = ray.get(self.workers[0].create_dist_init_method.remote())
        by_node = {}
        for i, (node, _) in enumerate(ids):
            by_node.setdefault(node, []).append(i)
        ray.get([w.initialize_worker.remote(by_node[ids[i][0]].index(i), {'VLLM_FAKE': '1'},
                                            init_method, assigned_physical_gpu_ids=ids[i][1])
                 for i, w in enumerate(self.workers)])
        t['initialized'] = time.time()
        assert all(r['status'] == 'READY'
                   for r in ray.get([w.wait_for_init.remote() for w in self.workers]))
        t['ready'] = time.time()
        self.run_refs = [w.run.remote() for w in self.workers]
        done, _ = ray.wait(self.run_refs, timeout=0)
        assert not done
        t['running'] = time.time()
        t['nodes'] = sorted(set(node for node, _ in ids))
        return t

    def stop(self):
        t = {'stop': time.time()}
        for w in self.workers:
            ray.kill(w, no_restart=True)
        t['killed'] = time.time()
        pending = list(self.run_refs)
        while pending:
            _, pending = ray.wait(pending, num_returns=len(pending), timeout=30)
            if pending and time.time() - t['killed'] > 120:
                raise RuntimeError('run references still pending 120 s after kill')
        t['detected'] = time.time()
        remove_placement_group(self.pg)
        t['pg_removed'] = time.time()
        self.workers, self.run_refs, self.pg = [], [], None
        return t


def percentiles(xs, ps=(50, 95, 99)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {'p%d' % p: round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 4)
            for p in ps} | {'max': round(xs[-1], 4)}


def gpus_free():
    return ray.available_resources().get('GPU', 0.0)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--replicas', type=int, required=True)
    ap.add_argument('--world', type=int, default=1, help='GPU workers per replica (TP*PP)')
    ap.add_argument('--strategy', default='PACK')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--init-delay', type=float, default=0.0,
                    help='seconds each worker sleeps in initialize_worker')
    ap.add_argument('--label', default='')
    args = ap.parse_args(argv)

    ray.init()
    backend = 'charm4py' if getattr(ray, '__charm4py__', False) else 'ray-' + ray.__version__
    nodes = [n for n in ray.nodes() if n['Alive']]
    total = ray.cluster_resources().get('GPU', 0.0)
    need = args.replicas * args.world
    meta = {'backend': backend, 'label': args.label, 'nodes': len(nodes), 'gpus': total,
            'replicas': args.replicas, 'world': args.world, 'strategy': args.strategy,
            'init_delay': args.init_delay}
    print('BENCH_META ' + json.dumps(meta), flush=True)
    if need > total:
        print('not enough GPUs: need %d, have %g' % (need, total), file=sys.stderr)
        return 2

    t0 = time.time()
    managers = [ReplicaManager.options(scheduling_strategy='SPREAD').remote(i)
                for i in range(args.replicas)]
    manager_nodes = ray.get([m.node.remote() for m in managers])
    print('MANAGERS ' + json.dumps({'create_s': round(time.time() - t0, 4),
                                    'nodes_used': len(set(manager_nodes))}), flush=True)

    for rnd in range(args.rounds):
        r0 = time.time()
        refs = {m.start.remote(args.world, args.strategy, args.init_delay): i
                for i, m in enumerate(managers)}
        ready_at, starts = {}, {}
        pending = list(refs)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                starts[refs[ref]] = ray.get(ref)
                ready_at[refs[ref]] = time.time() - r0
        up = time.time() - r0
        s0 = time.time()
        stops = ray.get([m.stop.remote() for m in managers])
        down = time.time() - s0
        while gpus_free() < total and time.time() - s0 < 120:
            time.sleep(0.05)
        released = time.time() - s0
        phases = {'pg_ready': [], 'actors_submitted': [], 'first_call': [], 'initialized': [],
                  'ready': [], 'running': [], 'detect': [], 'pg_remove': []}
        for i in range(args.replicas):
            s, e = starts[i], stops[i]
            rec = {'round': rnd, 'replica': i, 'ready_from_round_s': round(ready_at[i], 4),
                   'nodes': len(s['nodes']),
                   'pg_ready': s['pg_ready'] - s['start'],
                   'actors_submitted': s['actors_submitted'] - s['pg_ready'],
                   'first_call': s['first_call'] - s['actors_submitted'],
                   'initialized': s['initialized'] - s['first_call'],
                   'ready': s['ready'] - s['initialized'],
                   'running': s['running'] - s['ready'],
                   'detect': e['detected'] - e['killed'],
                   'pg_remove': e['pg_removed'] - e['detected']}
            for k in phases:
                phases[k].append(rec[k])
            print('REPLICA ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in rec.items()}), flush=True)
        summary = dict(meta, round=rnd, startup_makespan_s=round(up, 4),
                       teardown_s=round(down, 4), resources_released_s=round(released, 4),
                       leaked_gpus=round(total - gpus_free(), 4),
                       ready_from_round=percentiles(list(ready_at.values())),
                       phases={k: percentiles(v) for k, v in phases.items()})
        print('ROUND ' + json.dumps(summary), flush=True)

    for m in managers:
        ray.kill(m)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
