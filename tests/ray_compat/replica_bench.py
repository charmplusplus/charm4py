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
    # H3: crash a worker process of 2 replicas per round and restart them
    ... replica_bench.py --tp 2 --kill --kill-mode sigkill --kill-count 2 --recover

Per-replica records go to --out as JSON lines; the last output line is a JSON
summary. Phase times are seconds from the replica's start:
    pg      placement group created and ready
    actors  all workers created and answered their first call
    init    dist init method, initialize_worker, wait_for_init done
    run     run refs issued (the replica is serving)
and, measured from shared barriers rather than per replica:
    detect  ray.kill of one worker to ray.wait returning its run ref
    release teardown start to all GPUs available again (one value per round)
With --kill-mode sigkill the worker gets SIGKILL from a helper actor on its
node, as a crash would, instead of ray.kill. With --recover the replica then
tears itself down and starts again, as a restarted vLLM engine would:
    recover          kill to serving again
    restart_<phase>  the phases above, from the start of the restart

With --engine-actors each replica is laid out as vLLM lays out a data-parallel
rank with the Ray backend (CoreEngineActorManager): the driver reserves a
STRICT_PACK group pinned to one node (`--tp` GPU bundles and a CPU bundle for
the engine) and creates an engine actor in the CPU bundle, passing it the
group; the engine runs the executor -- workers and their startup calls, then
the kill, its detection and the teardown of its workers -- so those Ray calls
come from the engine's process, not the driver. Driver threads only reserve,
create engines and wait for them. Phases keep their meaning (seconds from the
replica's start, now including the engine's start) and add
    engine  the engine actor began its executor
A restart is a new engine in a new group, as a restarted DP rank would be.
Under charm4py.ray start the launcher with --num-cpus (one CPU per replica's
engine), as Ray nodes are."""

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
from ray.util.scheduling_strategies import (NodeAffinitySchedulingStrategy,
                                            PlacementGroupSchedulingStrategy)

BACKEND = 'charm4py' if getattr(ray, '__charm4py__', False) else 'ray-' + ray.__version__
PHASES = ('pg', 'engine', 'actors', 'init', 'run')
RESTART = tuple('restart_' + ph for ph in PHASES)
TIMEOUT = 120.0   # any single wait longer than this is reported as a failure


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

    def pid(self):
        return os.getpid()


@ray.remote(num_cpus=0)
class Killer:
    """Sends SIGKILL to a process on its node."""

    def ping(self):
        return True

    def kill(self, pid):
        os.kill(pid, 9)
        return True


class Replica:
    def __init__(self, rid, rnd, tp, driver_node, want_pids=False, engine_ip=None):
        self.rid, self.rnd, self.tp, self.driver_node = rid, rnd, tp, driver_node
        self.want_pids = want_pids   # for SIGKILL; asked before `run` occupies the workers
        self.engine_ip = engine_ip   # --engine-actors: the node this replica's group is pinned to
        self.engine = None
        self.pids = []
        self.rec = {'replica': rid, 'round': rnd, 'error': None}
        self.pg = None
        self.workers = []
        self.ids = []
        self.run_refs = []
        self.gen = 0   # starts so far; a restart's workers get new names

    def start(self, prefix=''):
        """Steps 3-9 of RayExecutorV2 startup (the plan's list); returns when
        the replica is serving."""
        if self.engine_ip is not None:
            return self._start_engine(prefix)
        clock = [time.perf_counter()]   # phase times are from clock[0]
        self.rec[prefix + 't_start'] = time.time()
        self.gen += 1

        self.pg = placement_group([{'GPU': 1.0} for _ in range(self.tp)], strategy='PACK')
        ray.get(self.pg.ready(), timeout=TIMEOUT)
        self.rec[prefix + 'pg'] = time.perf_counter() - clock[0]
        self._start_workers(clock, prefix)

    def _start_engine(self, prefix):
        """--engine-actors: the group and the engine, from the driver; the
        engine starts the workers."""
        t0 = time.perf_counter()
        self.rec[prefix + 't_start'] = time.time()
        self.gen += 1
        pin = {'node:' + self.engine_ip: 0.001}
        self.pg = placement_group([dict(pin, GPU=1.0) for _ in range(self.tp)] +
                                  [dict(pin, CPU=1.0)], strategy='STRICT_PACK')
        ray.get(self.pg.ready(), timeout=TIMEOUT)
        self.rec[prefix + 'pg'] = time.perf_counter() - t0
        self.engine = FakeEngineCore.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=self.tp),
            runtime_env={'env_vars': {'BENCH_FAKE_VLLM': '1'}},
        ).remote(self.rid, self.rnd, self.tp, self.gen, self.want_pids, self.pg)
        done = ray.get(self.engine.wait_for_init.remote(), timeout=TIMEOUT)
        if done['error']:
            raise RuntimeError('engine: ' + done['error'])
        # the engine's phases are from its own start; it began `took` seconds
        # before the reply reached this thread
        began = time.perf_counter() - t0 - done['took']
        self.rec[prefix + 'engine'] = began
        for ph in ('actors', 'init', 'run'):
            self.rec[prefix + ph] = began + done['rec'][ph]
        self.rec[prefix + 'nodes'] = done['rec']['nodes']

    def _start_workers(self, clock, prefix=''):
        """The executor's part: workers in self.pg's GPU bundles, their four
        rounds of startup calls, a pending `run` ref per worker. Phase times
        are from clock[0]."""
        mark = lambda phase: self.rec.__setitem__(prefix + phase, time.perf_counter() - clock[0])
        self.workers, self.run_refs = [], []
        bundle_nodes = placement_group_table(self.pg)['bundles_to_node_id']
        bundles = sorted(range(self.tp), key=lambda i: (bundle_nodes[i] != self.driver_node,
                                                        bundle_nodes[i], i))
        for rank, b in enumerate(bundles):
            self.workers.append(FakeVllmWorker.options(
                name='bench_r%d_%d_g%d_w%d' % (self.rnd, self.rid, self.gen, rank),
                num_cpus=0, num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg, placement_group_bundle_index=b),
                runtime_env={'env_vars': {'BENCH_FAKE_VLLM': '1'}},
            ).remote(vllm_config={'model': 'fake'}, rank=rank, input_shm_handle=('shm', 0),
                     is_driver_worker=(rank == 0),
                     is_driver_node=(bundle_nodes[b] == self.driver_node)))
        ids = ray.get([w.get_node_and_physical_gpu_ids.remote() for w in self.workers],
                      timeout=TIMEOUT)
        mark('actors')
        self.ids = ids
        self.rec[prefix + 'nodes'] = len({node for node, _ in ids})

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

        if self.want_pids:   # not part of RayExecutorV2: left out of the phase times
            p0 = time.perf_counter()
            self.pids = ray.get([w.pid.remote() for w in self.workers], timeout=TIMEOUT)
            clock[0] += time.perf_counter() - p0
        self.run_refs = [w.run.remote(3600.0) for w in self.workers]
        done, _ = ray.wait(self.run_refs, num_returns=1, timeout=0)
        if done:
            raise RuntimeError('a run ref finished early')
        mark('run')

    def kill_one(self, killers=None, recover=False):
        """Kill one worker (not rank 0; ray.kill, or SIGKILL through the
        Killer on its node when `killers` is given) and time how long the
        monitor's ray.wait takes to see it; then, with `recover`, restart."""
        t0 = time.perf_counter()
        if self.engine is not None:
            # the engine kills its worker and detects it
            done = ray.get(self.engine.kill_one.remote(killers), timeout=TIMEOUT)
            if done['error']:
                raise RuntimeError('engine: ' + done['error'])
            self.rec['detect'] = done['detect']
            if recover:
                self.teardown()
                self.start(prefix='restart_')
                self.rec['recover'] = time.perf_counter() - t0
            return
        victim = self.tp - 1
        if killers is not None:
            ray.get(killers[self.ids[victim][0]].kill.remote(self.pids[victim]),
                    timeout=TIMEOUT)
        else:
            ray.kill(self.workers[victim], no_restart=True)
        done, _ = ray.wait(self.run_refs, num_returns=1, timeout=TIMEOUT)
        self.rec['detect'] = time.perf_counter() - t0
        if done != [self.run_refs[victim]]:
            raise RuntimeError('ray.wait returned %r, expected the killed worker' % (done,))
        if recover:
            self.teardown()
            self.start(prefix='restart_')
            self.rec['recover'] = time.perf_counter() - t0

    def teardown(self):
        if self.engine is not None:
            # removing the group kills the workers in it too
            ray.kill(self.engine)
            self.engine = None
        for w in self.workers:
            ray.kill(w)
        if self.pg is not None:
            remove_placement_group(self.pg)


@ray.remote
class FakeEngineCore:
    """vLLM's DPEngineCoreActor without a model, for --engine-actors: its
    __init__ builds the executor (Replica._start_workers) in the placement
    group the driver passes it, as vLLM passes the group in the config."""

    def __init__(self, rid, rnd, tp, gen, want_pids, pg):
        self.t0 = time.perf_counter()
        node = ray.get_runtime_context().get_node_id()
        self.rep = Replica(rid, rnd, tp, node, want_pids=want_pids)
        self.rep.gen, self.rep.pg = gen, pg
        self.error = None
        try:
            self.rep._start_workers([self.t0])
        except Exception as e:
            self.error = '%s: %s' % (type(e).__name__, str(e)[:300])

    def wait_for_init(self):
        """The executor's phase times (from this engine's start) and how long
        ago it started."""
        rec = {k: self.rep.rec.get(k) for k in ('actors', 'init', 'run', 'nodes')}
        return {'error': self.error, 'rec': rec, 'took': time.perf_counter() - self.t0}

    def kill_one(self, killers):
        try:
            self.rep.kill_one(killers)
        except Exception as e:
            return {'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
        return {'error': None, 'detect': self.rep.rec['detect']}


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
    p.add_argument('--kill-mode', choices=('api', 'sigkill'), default='api',
                   help='ray.kill, or SIGKILL of the worker process (a crash)')
    p.add_argument('--kill-count', type=int, default=0,
                   help='replicas that lose a worker per round (default: all)')
    p.add_argument('--recover', action='store_true',
                   help='with --kill, restart the replicas that lost a worker')
    p.add_argument('--engine-actors', action='store_true',
                   help='lay replicas out as vLLM data-parallel ranks: an engine actor '
                        'per replica runs its executor (see the module docstring)')
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
    engine_ips = [None] * replicas
    if args.engine_actors:
        # as CoreEngineActorManager.create_dp_placement_groups: whole ranks per
        # node, the driver's node first
        engine_ips = [n['NodeManagerAddress']
                      for n in sorted(nodes, key=lambda n: n['NodeID'] != driver_node)
                      for _ in range(int(n['Resources'].get('GPU', 0)) // args.tp)]
        if len(engine_ips) < replicas:
            print('only %d replicas of tp=%d fit whole on a node' % (len(engine_ips), args.tp),
                  flush=True)
            return 2
    print('%s: %d nodes, %d GPUs, %d replicas x tp=%d, %d rounds, ray.init %.2fs%s' %
          (BACKEND, len(nodes), total_gpus, replicas, args.tp, args.rounds, t_init,
           ', engine actors' if args.engine_actors else ''), flush=True)

    killers = None
    if args.kill and args.kill_mode == 'sigkill':
        killers = {n['NodeID']: Killer.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            n['NodeID'], soft=False)).remote() for n in nodes}
        ray.get([k.ping.remote() for k in killers.values()], timeout=TIMEOUT)
    measured_phases = PHASES + (('detect',) if args.kill else ()) + \
        (('recover',) + RESTART if args.kill and args.recover else ())

    out = open(args.out, 'a') if args.out else None
    records, rounds = [], []
    for rnd in range(args.rounds):
        reps = [Replica(i, rnd, args.tp, driver_node, want_pids=killers is not None,
                        engine_ip=engine_ips[i])
                for i in range(replicas)]
        makespan, errors = in_threads(Replica.start, reps)
        info = {'round': rnd, 'makespan': makespan, 'start_errors': len(errors)}
        live = [r for r in reps if r.rec['error'] is None]
        if args.kill and live:
            victims = live[:args.kill_count] if args.kill_count else live
            info['kill_wall'], kill_errors = in_threads(
                lambda r: r.kill_one(killers, args.recover), victims)
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
                     for ph in measured_phases
                     if percentiles([r.rec.get(ph) for r in reps]))), flush=True)
        if released is None:
            print('resources were not released; stopping', flush=True)
            break

    measured = [r for r in records if r['round'] > 0] or records
    summary = {
        'backend': BACKEND, 'tag': args.tag, 'nodes': len(nodes), 'gpus': total_gpus,
        'engine_actors': args.engine_actors,
        'replicas': replicas, 'tp': args.tp, 'rounds': len(rounds), 'ray_init': t_init,
        'errors': sum(1 for r in records if r['error']),
        'phases': {ph: percentiles([r.get(ph) for r in measured])
                   for ph in PHASES + ('detect', 'recover') + RESTART},
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
