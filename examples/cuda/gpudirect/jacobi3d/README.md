# Jacobi3D on GPUs with device-to-device ghost exchange

A 3-D Jacobi iteration (7-point stencil) on a `-x` by `-y` by `-z` grid, split into
one block per chare. The blocks, their halos and the ghost buffers live on the GPU;
the stencil, packing and unpacking are numba.cuda kernels (`kernels.py`). Ported from
charm4py PR #208 (Charm4py 1.x) to the Charm4py 2 API.

| file | what |
|---|---|
| `jacobi3d.py` | options, decomposition, creation of the blocks, validation |
| `block.py` | the `Block` chare: allocation, ghost exchange in three modes, timing |
| `kernels.py` | the numba.cuda kernels (three packing kernels fixed, see its docstring) |
| `scripts/charm4py.sh` | timing runs of every mode, with `srun` on Delta or `charmrun.start` locally |
| `scripts/process_times.py` | mean and standard deviation of the times per mode, as CSV |

## Exchange modes (`-m`)

- `staged`: pack on the GPU, copy the ghost to pinned host memory, send it as a
  numpy array, copy it to the device on the receiver. Needs numba and a GPU, not a
  CUDA build of Charm++ (PR #208 without `-d`).
- `device` (default; `-d` is accepted as in PR #208): the device ghost array is an
  argument of the message (a DEVICE slot; Charm++ moves it with a device copy,
  CUDA IPC or GPUDirect RDMA). Each block posts its receive ghosts with
  `Chare.post_buffer`, so a ghost lands in the same device array every iteration.
  One entry method per direction (`recv_left` ... `recv_back`): a post covers one
  method argument.
- `persistent`: each block registers its receive ghosts once with
  `charm.DevicePersistent` and sends the handles to its neighbors. A block puts its
  packed face into the neighbor's buffer (`put(handle)`, a future that is set when
  the bytes have arrived) and then sends the neighbor a message with the iteration
  number. This is the form of the Charm++ Jacobi3D's persistent GPU-direct buffers.

PR #208 exchanged ghosts over channels (`Channel.send(gpu_src_ptrs=...)`,
`Channel.recv(post_buf_addresses=...)`). Charm4py 2 channels carry device arrays
too (they arrive as `DeviceBuffer`), but have no posted receives, so the `device`
mode uses entry methods with posted receives instead.

`block.py` explains the ordering: a neighbor can be one iteration ahead, and the
buffers that a transfer may still use are doubled by iteration parity.

## Options

    -x, -y, -z      grid size (default 512 each); must divide by the chare grid
    -c              number of chares (default: number of PEs); the chare grid is the
                    decomposition with the smallest surface
    -i, -w          timed and warm-up iterations (default 100 and 10)
    -m              staged | device | persistent
    --validate      compare the sum of the final grid with numpy on PE 0 (small grids)
    --dry-run       print the configuration and exit (no GPU, no numba needed)

The output gives the elapsed time, the average time per iteration, the
communication time per iteration (packing, sending, and waiting for every ghost of
the iteration) and the sum of the interior of the final grid, which is the same in
every mode.

## Running on one node

Requirements: numba with numba-cuda, a CUDA build of Charm++ under Charm4py for the
`device` and `persistent` modes (see `docs/gpus.rst`). One process per GPU:

    python -m charmrun.start +p2 jacobi3d.py -x 256 -y 256 -z 256 -m device
    python -m charmrun.start +p2 jacobi3d.py -x 64 -y 64 -z 64 -c 8 -m persistent --validate

Without a GPU, `--dry-run` checks the options and the decomposition, and the numba
CUDA simulator runs the `staged` mode on the CPU (slowly; small grids only):

    NUMBA_ENABLE_CUDASIM=1 python -m charmrun.start +p2 jacobi3d.py -x 8 -y 8 -z 8 -c 8 -m staged -i 5 --validate

## Running on NCSA Delta

With the workspace of `tools/delta/README.md` (`env-cuda.sh` loads the modules and
the venv). Every GPU of the node should be visible to every process: Charm++ gives
process k of a node device k, and CUDA IPC between processes of a node needs the
other process's GPU visible. numba-cuda 0.30.4 calls `numpy.row_stack`, removed in
numpy 2.5, so the kernels need `$WS/numpy-compat` (numpy < 2.5) on `PYTHONPATH`:

    source $WS/env-cuda.sh
    cd $WS/charm4py/examples/cuda/gpudirect/jacobi3d
    srun --mpi=pmix -n 2 --gpus=2 --gpu-bind=none --cpu-bind=none \
        env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$WS/numpy-compat \
        python jacobi3d.py -x 512 -y 512 -z 512 -m persistent +pe 2

inside an allocation such as `salloc -A mzu-delta-gpu -p gpuA100x4-interactive -N 1
--ntasks=2 --cpus-per-task=4 --gpus-per-node=2 --mem=32g -t 30`. For timing runs of
all modes on several nodes, `scripts/charm4py.sh` is a batch script:

    sbatch -N 2 scripts/charm4py.sh                  # 8 processes, one per GPU
    GRID=1024 MODES="device persistent" sbatch -N 4 scripts/charm4py.sh
    scripts/process_times.py jacobi3d-charm4py 1 4   # after renaming outputs to <job>-n<nodes>-*.out
