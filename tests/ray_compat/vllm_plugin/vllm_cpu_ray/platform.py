from vllm.platforms.cpu import CpuPlatform


class CpuRayPlatform(CpuPlatform):
    # vLLM asks Ray for one of these per worker; they are logical GPUs here
    ray_device_key = 'GPU'
