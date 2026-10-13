"""Control-plane benchmark (hypothesis H1 in vllm_charm4py_research_plan.md):
the startup, failure-detection and teardown sequence of RayExecutorV2, for many
vLLM replicas at once, without models or GPUs.

Each replica is one RayExecutorV2: a PACK placement group of `--tp` one-GPU
bundles, one named worker actor per bundle, the four rounds of startup calls,
a pending `run` ref per worker as the liveness sentinel, then (with --kill)
one worker killed and detected through `ray.wait`, then teardown. Replicas run
concurrently, one driver thread each, like DP engines starting together.
GPUs are logical: give each node N of them.

    # charm4py.ray (one node; across nodes start the launcher with srun, see
    # GPU_RUNBOOK.md section 4)
    python -m charmrun.start +p8 -m charm4py.ray.launch --num-gpus 8 \\
        tests/ray_compat/replica_bench.py --tp 2 --rounds 3 --kill
    # Ray: a running cluster whose nodes were started with --num-gpus 8
    RAY_ADDRESS=auto python tests/ray_compat/replica_bench.py --tp 2 --rounds 3 --kill

Per-replica records go to --out as JSON lines; the last output line is a JSON
summary. Phase times are seconds from the replica's start:
    pg      placement group created and ready
    actors  all workers created and answered their first call
    init    dist init method, initialize_worker, wait_for_init done
    run     run refs issued (the replica is serving)
and, measured from shared barriers rather than per replica:
    detect  ray.kill of one worker to ray.wait returning its run ref
    release teardown start to all GPUs available again (one value per round)"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback

import ray
from ray.util.placement_group import (placement_group, placement_group_table,
                                      remove_placement_group)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

BACKEND = 'charm4py' if getattr(ray, '__charm4py__', False) else 'ray-' + ray.__version__
PHASES = ('pg', 'actors', 'init', 'run')
TIMEOUT = 600.0   # any single wait longer than this is reported as a failure


@ray.remote
class FakeVllmWorker:
    """The calls RayExecutorV2 makes on RayWorkerProc, without a model (as in
    contract.py)."""

    def __init__(self, vllm_config, rank, input_shm_handle, is_driver_worker,
                 is_driver_node=False):
        self.rank = rank
        self.config = vllm_config
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
        return None

    def wait_for_init(self):
        return {'status': 'READY', 'handle': ('mq', self.rank, self.local_rank)}

    def run(self, seconds):
        time.sleep(seconds)
        return 'finished'


class Replica:
    def __init__(self, rid, rnd, tp, driver_node):
        self.rid, self.rnd, self.tp, self.driver_node = rid, rnd, tp, driver_node
        self.rec = {'replica': rid, 'round': rnd, 'error': None}
        self.pg = None
        self.workers = []
        self.run_refs = []

    def start(self):
        """Steps 3-9 of RayExecutorV2 startup (the plan's list); returns when
        the replica is serving."""
        t0 = time.perf_counter()
        self.rec['t_start'] = time.time()
        mark = lambda phase: self.rec.__setitem__(phase, time.perf_counter() - t0)

        self.pg = placement_group([{'GPU': 1.0} for _ in range(self.tp)], strategy='PACK')
        ray.get(self.pg.ready(), timeout=TIMEOUT)
        mark('pg')

        bundle_nodes = placement_group_table(self.pg)['bundles_to_node_id']
        bundles = sorted(range(self.tp), key=lambda i: (bundle_nodes[i] != self.driver_node,
                                                        bundle_nodes[i], i))
        for rank, b in enumerate(bundles):
            self.workers.append(FakeVllmWorker.options(
                name='bench_r%d_%d_w%d' % (self.rnd, self.rid, rank), num_cpus=0, num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg, placement_group_bundle_index=b),
                runtime_env={'env_vars': {'BENCH_FAKE_VLLM': '1'}},
            ).remote(vllm_config={'model': 'fake'}, rank=rank, input_shm_handle=('shm', 0),
                     is_driver_worker=(rank == 0),
                     is_driver_node=(bundle_nodes[b] == self.driver_node)))
        ids = ray.get([w.get_node_and_physical_gpu_ids.remote() for w in self.workers],
                      timeout=TIMEOUT)
        mark('actors')
        self.rec['nodes'] = len({node for node, _ in ids})

        init_method = ray.get(self.workers[0].create_dist_init_method.remote(),
                              timeout=TIMEOUT)
        node_workers = {}
        for i, (node, _) in enumerate(ids):
            node_workers.setdefault(node, []).append(i)
        ray.get([w.initialize_worker.remote(node_workers[ids[i][0]].index(i), {'X': '1'},
                                            init_method, {'Y': '2'},
                                            assigned_physical_gpu_ids=sorted(ids[i][1]))
                 for i, w in enumerate(self.workers)], timeout=TIMEOUT)
        results = ray.get([w.wait_for_init.remote() for w in self.workers], timeout=TIMEOUT)
        if not all(r['status'] == 'READY' for r in results):
            raise RuntimeError('a worker is not READY: %s' % results)
        mark('init')

        self.run_refs = [w.run.remote(3600.0) for w in self.workers]
        done, _ = ray.wait(self.run_refs, num_returns=1, timeout=0)
        if done:
            raise RuntimeError('a run ref finished early')
        mark('run')

    def kill_one(self):
        """ray.kill one worker (not rank 0) and time how long the monitor's
        ray.wait takes to see it."""
        victim = self.tp - 1
        t0 = time.perf_counter()
        ray.kill(self.workers[victim], no_restart=True)
        done, _ = ray.wait(self.run_refs, num_returns=1, timeout=TIMEOUT)
        self.rec['detect'] = time.perf_counter() - t0
        if done != [self.run_refs[victim]]:
            raise RuntimeError('ray.wait returned %r, expected the killed worker' % (done,))

    def teardown(self):
        for w in self.workers:
            ray.kill(w)
        if self.pg is not None:
            remove_placement_group(self.pg)


def in_threads(fn, items):
    """Runs fn(item) in one thread per item, all released together."""
    barrier = threading.Barrier(len(items))
    errors = []

    def body(item):
        barrier.wait()
        try:
            fn(item)
        except Exception as e:
            item.rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
            errors.append(item)
            traceback.print_exc()

    threads = [threading.Thread(target=body, args=(item,), daemon=True) for item in items]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, errors


def wait_released(total_gpus, timeout=TIMEOUT):
    t0 = time.perf_counter()
    while ray.available_resources().get('GPU', 0) < total_gpus:
        if time.perf_counter() - t0 > timeout:
            return None
        time.sleep(0.02)
    return time.perf_counter() - t0


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
    return {'n': len(values), 'p50': pick(0.5), 'p95': pick(0.95), 'p99': pick(0.99),
            'max': values[-1], 'mean': sum(values) / len(values)}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--tp', type=int, default=2, help='workers (GPUs) per replica')
    p.add_argument('--replicas', type=int, default=0,
                   help='replicas per round (default: all GPUs in the cluster / tp)')
    p.add_argument('--rounds', type=int, default=3,
                   help='rounds; round 0 is warm-up and left out of the summary')
    p.add_argument('--kill', action='store_true', help='measure failure detection')
    p.add_argument('--out', default=None, help='per-replica JSON lines')
    p.add_argument('--tag', default='', help='free-form label copied into the summary')
    args = p.parse_args(argv)

    t_init = time.perf_counter()
    ray.init(log_to_driver=False)
    t_init = time.perf_counter() - t_init
    nodes = [n for n in ray.nodes() if n['Alive']]
    total_gpus = int(ray.cluster_resources().get('GPU', 0))
    replicas = args.replicas or total_gpus // args.tp
    if replicas < 1 or replicas * args.tp > total_gpus:
        print('need %d GPUs for %d replicas of tp=%d, cluster has %d' %
              (replicas * args.tp, replicas, args.tp, total_gpus), flush=True)
        return 2
    driver_node = ray.get_runtime_context().get_node_id()
    print('%s: %d nodes, %d GPUs, %d replicas x tp=%d, %d rounds, ray.init %.2fs' %
          (BACKEND, len(nodes), total_gpus, replicas, args.tp, args.rounds, t_init),
          flush=True)

    out = open(args.out, 'a') if args.out else None
    records, rounds = [], []
    for rnd in range(args.rounds):
        reps = [Replica(i, rnd, args.tp, driver_node) for i in range(replicas)]
        makespan, errors = in_threads(Replica.start, reps)
        info = {'round': rnd, 'makespan': makespan, 'start_errors': len(errors)}
        live = [r for r in reps if r.rec['error'] is None]
        if args.kill and live:
            info['kill_wall'], kill_errors = in_threads(Replica.kill_one, live)
            info['kill_errors'] = len(kill_errors)
        t0 = time.perf_counter()
        for r in reps:
            try:
                r.teardown()
            except Exception as e:
                r.rec['error'] = r.rec['error'] or 'teardown %s: %s' % (type(e).__name__, e)
        released = wait_released(total_gpus)
        info['release'] = None if released is None else time.perf_counter() - t0
        rounds.append(info)
        for r in reps:
            r.rec.update(backend=BACKEND, tp=args.tp, replicas=replicas, n_nodes=len(nodes),
                         tag=args.tag)
            records.append(r.rec)
            if out:
                out.write(json.dumps(r.rec) + '\n')
        if out:
            out.flush()
        print('round %d: makespan %.3fs, release %s, errors %d, %s' % (
            rnd, makespan, '%.3fs' % info['release'] if info['release'] is not None
            else 'TIMEOUT', sum(1 for r in reps if r.rec['error']),
            ' '.join('%s p50 %.3f' % (ph, percentiles([r.rec.get(ph) for r in reps])['p50'])
                     for ph in PHASES + (('detect',) if args.kill else ())
                     if percentiles([r.rec.get(ph) for r in reps]))), flush=True)
        if released is None:
            print('resources were not released; stopping', flush=True)
            break

    measured = [r for r in records if r['round'] > 0] or records
    summary = {
        'backend': BACKEND, 'tag': args.tag, 'nodes': len(nodes), 'gpus': total_gpus,
        'replicas': replicas, 'tp': args.tp, 'rounds': len(rounds), 'ray_init': t_init,
        'errors': sum(1 for r in records if r['error']),
        'phases': {ph: percentiles([r.get(ph) for r in measured])
                   for ph in PHASES + ('detect',)},
        'makespan': percentiles([i['makespan'] for i in rounds[1:]] or
                                [i['makespan'] for i in rounds]),
        'release': percentiles([i['release'] for i in rounds[1:]] or
                               [i['release'] for i in rounds]),
        'round_info': rounds,
    }
    print('SUMMARY ' + json.dumps(summary), flush=True)
    ray.shutdown()
    return 0 if summary['errors'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
