"""What the driver sees when a worker process or a whole PE dies (step 5, H3).

Starts `--actors` GPU actors with a long-running `run` call and monitors the
pending references the way RayExecutorV2 does (ray.wait, 5 s timeout). Then
kills, with SIGKILL, either one actor process (`--kill worker`) or one runtime
process that is not the driver's (`--kill pe`: a Charm PE under charm4py.ray,
a raylet under Ray), and reports when and how the loss shows up: the failed
reference and its error, a hang, or the job ending.

    CHARM4PY_RAY_FAKE_NODES=2 python -m charmrun.start +p4 -m charm4py.ray.launch \\
        --num-gpus 2 tests/ray_compat/bench/failure_probe.py --kill worker
"""

import argparse
import json
import os
import signal
import sys
import time

import psutil
import ray


@ray.remote(num_cpus=0, num_gpus=1)
class Victim:
    def pid(self):
        return os.getpid()

    def node(self):
        return ray.get_runtime_context().get_node_id()

    def run(self):
        while True:
            time.sleep(3600)


def runtime_processes(exclude):
    """This user's runtime processes on this machine, other than `exclude`."""
    keys = ('charm4py.ray.launch', 'raylet')
    out = []
    for p in psutil.process_iter(['pid', 'cmdline', 'uids']):
        try:
            text = ' '.join(p.info['cmdline'] or [])
            if p.info['uids'].real == os.getuid() and p.info['pid'] not in exclude \
                    and any(k in text for k in keys) and 'lcrun' not in text:
                out.append(p.info['pid'])
        except (psutil.Error, TypeError):
            pass
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--actors', type=int, default=4)
    ap.add_argument('--kill', choices=('worker', 'pe'), required=True)
    ap.add_argument('--watch', type=float, default=60.0, help='seconds to watch after the kill')
    ap.add_argument('--ray-local-cluster', action='store_true',
                    help='real Ray only: a head without GPUs plus a second local node that '
                         'holds them, so killing that node\'s raylet loses a non-head node')
    args = ap.parse_args(argv)
    cluster = None
    if args.ray_local_cluster:
        from ray.cluster_utils import Cluster
        cluster = Cluster(initialize_head=True, head_node_args={'num_cpus': 2, 'num_gpus': 0})
        cluster.add_node(num_cpus=2, num_gpus=args.actors)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
    else:
        ray.init()
    backend = 'charm4py' if getattr(ray, '__charm4py__', False) else 'ray-' + ray.__version__
    actors = [Victim.remote() for _ in range(args.actors)]
    pids = ray.get([a.pid.remote() for a in actors])
    refs = [a.run.remote() for a in actors]
    done, _ = ray.wait(refs, timeout=1)
    assert not done
    me = os.getpid()
    parent = psutil.Process(me).ppid()
    if args.kill == 'worker':
        victim = pids[-1]
    else:
        candidates = runtime_processes({me, parent})
        # prefer one that hosts an actor: an ancestor of an actor process
        hosting = set()
        for pid in pids:
            try:
                hosting.update(a.pid for a in psutil.Process(pid).parents())
            except psutil.Error:
                pass
        candidates = [c for c in candidates if c in hosting] or candidates
        if not candidates:
            print('PROBE ' + json.dumps({'backend': backend, 'kill': args.kill,
                                         'error': 'no runtime process to kill'}), flush=True)
            return 2
        victim = candidates[-1]
    print('PROBE_KILL ' + json.dumps({'backend': backend, 'kill': args.kill, 'pid': victim,
                                      'cmd': ' '.join(psutil.Process(victim).cmdline())[:120]}),
          flush=True)
    t0 = time.time()
    os.kill(victim, signal.SIGKILL)
    events = []
    pending = list(refs)
    while pending and time.time() - t0 < args.watch:
        done, pending = ray.wait(pending, num_returns=1, timeout=5)   # vLLM's monitor loop
        for ref in done:
            try:
                ray.get(ref, timeout=5)
                err = None
            except Exception as e:  # noqa: BLE001 - we report whatever arrives
                err = type(e).__name__ + ': ' + str(e).splitlines()[0][:160]
            events.append({'after_s': round(time.time() - t0, 3), 'actor': refs.index(ref),
                           'error': err})
            print('PROBE_EVENT ' + json.dumps(events[-1]), flush=True)
    alive = []
    for a in actors:
        try:
            ray.get(a.pid.remote(), timeout=10)
            alive.append(True)
        except Exception:  # noqa: BLE001
            alive.append(False)
    print('PROBE ' + json.dumps({'backend': backend, 'kill': args.kill,
                                 'first_detection_s': events[0]['after_s'] if events else None,
                                 'failed_refs': len(events), 'actors': len(actors),
                                 'actors_still_answering': sum(alive),
                                 'watched_s': round(time.time() - t0, 1)}), flush=True)
    for a in actors:
        try:
            ray.kill(a)
        except Exception:  # noqa: BLE001
            pass
    if cluster is not None:
        ray.shutdown()
        cluster.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
