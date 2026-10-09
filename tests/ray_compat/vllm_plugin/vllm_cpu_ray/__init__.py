"""Test-only vLLM platform plugin, active when VLLM_CPU_RAY=1.

vLLM's Ray executor needs a platform with a Ray accelerator key, so on a machine
without GPUs (a Mac, CI) it cannot run at all. This plugin is vLLM's CPU platform
with each worker's slot exposed to Ray as a logical GPU: start Ray with logical
GPUs (`ray start --num-gpus N`, or charm4py.ray.launch --num-gpus N) and vLLM's
unmodified RayExecutorV2 runs end to end on CPUs, with gloo collectives."""

import os


def register():
    if os.environ.get('VLLM_CPU_RAY') == '1':
        return 'vllm_cpu_ray.platform.CpuRayPlatform'
    return None
