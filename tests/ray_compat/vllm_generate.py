"""Greedy generation with vLLM's Ray executor (RayExecutorV2), to compare backends.

    VLLM_CPU_RAY=1 python -m charmrun.start +p4 -m charm4py.ray.launch --num-gpus 2 \\
        tests/ray_compat/vllm_generate.py [TP] [PP]
    # real Ray:  ray start --head --num-gpus 2;  VLLM_CPU_RAY=1 RAY_ADDRESS=auto \\
    #            python tests/ray_compat/vllm_generate.py [TP] [PP]
    # reference: python tests/ray_compat/vllm_generate.py [TP] [PP] mp

On a machine without GPUs the vllm_plugin (pip install -e vllm_plugin) supplies
a CPU platform the Ray executor accepts. Prints one JSON line with the outputs."""

import json
import sys
import time

PROMPTS = ['Hello, my name is', 'The capital of France is',
           'The future of AI is', 'Charm++ is a parallel programming system that']


def main(argv):
    from vllm import LLM, SamplingParams
    tp = int(argv[0]) if len(argv) > 0 else 1
    pp = int(argv[1]) if len(argv) > 1 else 1
    backend = argv[2] if len(argv) > 2 else 'ray'
    t0 = time.time()
    llm = LLM(model='facebook/opt-125m', tensor_parallel_size=tp,
              pipeline_parallel_size=pp, distributed_executor_backend=backend,
              enforce_eager=True, max_model_len=256, seed=0)
    t1 = time.time()
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=24))
    t2 = time.time()
    texts = [o.outputs[0].text for o in outputs]
    print('VLLM_RESULT ' + json.dumps({'backend': backend, 'tp': tp, 'pp': pp,
                                       'startup_s': round(t1 - t0, 2),
                                       'generate_s': round(t2 - t1, 2), 'texts': texts}),
          flush=True)
    del llm
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
