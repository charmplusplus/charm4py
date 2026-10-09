"""``import ray`` in a process started by a process of a charm4py.ray job (the
driver starting a process with multiprocessing, say): charm4py.ray.launch puts
this directory on PYTHONPATH, and this module replaces itself with charm4py.ray.
Such a process does not run Charm4py; its ray.init() connects to the job."""

import sys

del sys.modules['ray']
from charm4py.ray.launch import install_ray_facade  # noqa: E402

install_ray_facade(worker=True)
