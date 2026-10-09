"""vLLM's process layout: the driver never calls ray.init(); it starts an engine
process (forked by default, spawned when CUDA is initialized), and that process
initializes Ray, reserves a placement group and creates the worker actors.

    python tests/ray_compat/engine_process.py [fork|spawn]          # real Ray
    python -m charmrun.start +p4 -m charm4py.ray.launch \\
        tests/ray_compat/engine_process.py [fork|spawn]               # charm4py

Prints PASS or FAIL and exits accordingly."""

import multiprocessing
import os
import sys

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

WORLD = 2


@ray.remote
class Worker:
    def __init__(self, rank):
        self.rank = rank

    def info(self):
        ctx = ray.get_runtime_context()
        return (self.rank, ctx.get_node_id(), ctx.get_accelerator_ids()['GPU'],
                os.environ.get('CUDA_VISIBLE_DEVICES'))


def engine(results):
    try:
        if ray.is_initialized():
            raise AssertionError('Ray is initialized in a fresh engine process')
        ray.init(num_gpus=WORLD)   # vLLM: ray.init(address=ray_address, ...)
        pg = placement_group([{'GPU': 1}] * WORLD, strategy='PACK')
        ray.get(pg.ready(), timeout=60)
        workers = [Worker.options(
            num_cpus=0, num_gpus=1, scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i)).remote(i)
            for i in range(WORLD)]
        infos = ray.get([w.info.remote() for w in workers], timeout=60)
        for w in workers:
            ray.kill(w)
        remove_placement_group(pg)
        ray.shutdown()
        results.put(('ok', infos))
    except BaseException as e:
        results.put(('error', repr(e)))


def main(method):
    mp = multiprocessing.get_context(method)
    results = mp.Queue()
    proc = mp.Process(target=engine, args=(results,))
    proc.start()
    reply = results.get(timeout=180)
    proc.join(60)
    ok = (reply[0] == 'ok' and proc.exitcode == 0 and
          sorted(g for _, _, gs, _ in reply[1] for g in gs) == [str(i) for i in range(WORLD)]
          and all(env == ','.join(gs) for _, _, gs, env in reply[1]))
    print(('PASS' if ok else 'FAIL') + ' engine process (' + method + '): ' + str(reply),
          flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'fork'))
