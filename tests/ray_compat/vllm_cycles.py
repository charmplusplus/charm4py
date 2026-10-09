"""Repeated vLLM create/generate/shutdown cycles on the Ray executor: every
cycle must produce the same text, and afterwards nothing may be left behind
(resources, objects, worker processes). Per-cycle startup times show drift.

    VLLM_CPU_RAY=1 VLLM_CPU_KVCACHE_SPACE=1 python -m charmrun.start +p4 \\
        -m charm4py.ray.launch --num-gpus 2 tests/ray_compat/vllm_cycles.py 20 [TP]"""

import gc
import json
import sys
import time


def worker_processes():
    """charm4py.ray worker processes, and vLLM's (which rename themselves)."""
    import psutil
    found = []
    for p in psutil.process_iter(['cmdline', 'name']):
        text = ' '.join(p.info['cmdline'] or []) + ' ' + (p.info['name'] or '')
        if 'charm4py.ray._worker_main' in text or 'VLLM::' in text:
            found.append(p)
    return found


def _pe_memory():
    """RSS (MB) of this machine's Charm4py processes, to spot growth."""
    import psutil
    return sorted(int(p.memory_info().rss / 2**20) for p in psutil.process_iter(['cmdline'])
                  if 'charm4py.ray.launch' in ' '.join(p.info['cmdline'] or []))


def main(argv):
    from vllm import LLM, SamplingParams
    cycles = int(argv[0]) if argv else 20
    tp = int(argv[1]) if len(argv) > 1 else 2
    params = SamplingParams(temperature=0.0, max_tokens=16)
    baseline_procs = len(worker_processes())
    first, times = None, []
    for i in range(cycles):
        t0 = time.time()
        llm = LLM(model='facebook/opt-125m', tensor_parallel_size=tp,
                  distributed_executor_backend='ray', enforce_eager=True,
                  max_model_len=256, seed=0)
        t1 = time.time()
        text = [o.outputs[0].text for o in llm.generate(['The capital of France is'], params)]
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        times.append(round(t1 - t0, 2))
        first = first or text
        if text != first:
            print('FAIL cycle', i, 'text differs:', text, first, flush=True)
            return 1
        print('cycle', i, 'startup', times[-1], 's', _pe_memory(), flush=True)
    import ray
    ok = True
    if getattr(ray, '__charm4py__', False):
        ray.init()
        deadline = time.time() + 30
        while time.time() < deadline:
            gpus = ray.available_resources().get('GPU', 0)
            objects = sum(o for o, _ in ray._object_counts())
            procs = len(worker_processes())
            if gpus == ray.cluster_resources().get('GPU', 0) and objects == 0 and \
                    procs <= baseline_procs:
                break
            time.sleep(0.5)
        print('left: GPUs free', gpus, 'of', ray.cluster_resources().get('GPU'),
              '| objects', objects, '| worker processes', procs, 'vs', baseline_procs,
              flush=True)
        ok = gpus == ray.cluster_resources().get('GPU', 0) and objects == 0 and \
            procs <= baseline_procs
        ray.shutdown()
    print('VLLM_CYCLES ' + json.dumps({'cycles': cycles, 'startup_s': times}), flush=True)
    print(('PASS' if ok else 'FAIL') + ' vllm cycles: ' + str(cycles), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
