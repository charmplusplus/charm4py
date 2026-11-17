# Thin charm4py on NCSA Delta (CUDA, reconverse)

Scripts for building the thin charm4py stack against a CUDA reconverse build of Charm++
on Delta and running its GPU gate. Recorded results: `docs/design/SPEC-3.md`, section 7,
"Delta". Machine notes (Duo, partitions, `srun --mpi=pmix`): charm-notes
`machines/delta.md`.

| file | what |
|---|---|
| `env-cuda.sh` | modules, `FI_PROVIDER=cxi`, build paths, venv; sourced by everything else |
| `build.sh` | `./build charm4py reconverse-linux-x86_64 cuda ...` into `$CBUILD` (for the Python package) |
| `build-cxx.sh` | `./build charm++ ...` of the same source into `$CBUILD_CXX` (for ckdyntest and hapitest; a charm4py-target build aborts charmxi programs) |
| `gpu-gate.sbatch` | the gate: one node, 2 GPUs, 4 tasks; C++ tests, Python GPU tests, the full thin suite |
| `run_thin.py` | the suite runner used by the gate (test_config.json plus tests/thin, one `srun` per test) |
| `gpu-2node.sbatch` | two nodes, one GPU each: ckdyntest across the fabric and the one-way measurement |
| `bench_zc_delta.py` | one-way time of a float64 array as BUFFER, NCPY or DEVICE (`MODE BYTES +pe 2`) |
| `gpu-devpersist.sbatch` | SPEC-7-device: ckdyntest (`devpersist` phases), `device_persistent.py`, `posted_receive.py`, `device.py` per shape (one node: `-n 1`, `-n 2`, `-n 2 +gpushm`, and both again with both GPUs visible to both tasks; two nodes with `-N 2`), then the measurements |
| `bench_devpersist.py` | time of a persistent `get` or `put` between two PEs (`OP BYTES +pe 2`), with its transfer mode |
| `build-incr-cpu.sbatch` | incremental `make` of both builds and the charm4py reinstall as a cpu-interactive job (no GPU needed; `CUDA_CAP` picks the target, default 86); the login nodes should not run parallel builds |

## Environment

The scripts hard-code no allocation or path. Set these in the login shell (for example in
`~/.bashrc`); there are no defaults, and a script stops with a one-line message when the
variable it needs is unset:

| variable | what |
|---|---|
| `CHARM4PY_DELTA_WS` | the workspace directory: holds `charm/`, `charm4py/`, `venv/`, these scripts and the job output |
| `CHARM4PY_DELTA_ACCOUNT_GPU` | the Slurm account of GPU jobs (an allocation such as `abcd-delta-gpu`) |
| `CHARM4PY_DELTA_ACCOUNT_CPU` | the Slurm account of CPU jobs (`build-incr-cpu.sbatch`) |

An `#SBATCH --account=` line cannot read an environment variable, so the batch scripts
carry no account line: pass it on the command line, as in every `sbatch` example below.
`${VAR:?}` makes the shell refuse the command when the variable is unset:

    sbatch --account=${CHARM4PY_DELTA_ACCOUNT_GPU:?} -p gpuA40x4-interactive gpu-gate.sbatch

`sbatch` passes the submitting shell's environment to the job, so `CHARM4PY_DELTA_WS`
reaches the batch script.

## Setup (once, on a login node)

    WS=$CHARM4PY_DELTA_WS                                # copy these scripts there
    cd $WS
    git clone -b main https://github.com/charmplusplus/charm4py.git charm4py
    git clone https://github.com/charmplusplus/charm.git charm
    git -C charm checkout $(awk '/^commit/ {print $2}' charm4py/charm_version.txt)
    git -C charm submodule update --init contrib/reconverse
    module load python/3.13.5-gcc13.3.1 && python3 -m venv venv
    source env-cuda.sh
    pip install numpy greenlet cython scikit-build-core nanobind ninja patchelf \
        cupy-cuda13x numba numba-cuda matplotlib
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ./build.sh && ./build-cxx.sh                         # about 3 minutes each
    (cd charm4py && pip install --no-build-isolation -e .)

`examples/cuda/hapi/multi_gpu_callback.py` needs a CUDA torch; the venv holds the CPU one
(for the Ray examples). The gate puts `$WS/torch-cuda` on `PYTHONPATH` for that one step
when it exists:

    pip install --target $WS/torch-cuda torch --index-url https://download.pytorch.org/whl/cu130

`examples/cuda/hapi/hapi-cuda-callback.py` jit-compiles a kernel with numba-cuda, and
numba-cuda 0.30.4 (the newest on 2026-11-08) calls `numpy.row_stack`, which numpy 2.5
removed. The gate puts `$WS/numpy-compat` on `PYTHONPATH` for those two steps:

    pip install --target $WS/numpy-compat "numpy<2.5"

Nothing from a CUDA build runs on a login node (LCI calls `cuInit` at startup), so every
run, including `+pe 1`, goes through `sbatch`.

## After pulling new commits

    source env-cuda.sh
    git -C charm pull && git -C charm submodule update --init contrib/reconverse
    make -C $CBUILD -j16 && make -C $CBUILD_CXX -j16
    (cd charm4py && git pull && pip install --no-build-isolation -e .)

The `pip install` step is needed after every rebuild of `$CBUILD`: the package bundles
copies of libcharm and the reconverse libraries in `charm4py/.libs`.

## Running the gate by hand

    cd $WS
    A=--account=${CHARM4PY_DELTA_ACCOUNT_GPU:?}
    sbatch $A -p gpuA40x4-interactive gpu-gate.sbatch       # or gpuA100x4-interactive
    sbatch $A --export=ALL,SKIP_SUITE=1 -p gpuA40x4-interactive gpu-gate.sbatch   # no full suite
    sbatch $A -p gpuA40x4-interactive gpu-2node.sbatch      # two nodes

Switches (environment): `SKIP_CXX`, `SKIP_PY`, `SKIP_SUITE`, `FILTER` (regex on suite
labels), `MODE` (`srun` or `charmrun`), `GATE_HISTORY` (file to append a summary line to).
The interactive partitions allow 1 running and 2 submitted jobs per user; submit one at a
time.

Output: `gate-<jobid>.out` with one `RESULT <step> exit=<rc> <s>s banner=[...]` line per
step, and `gate-<jobid>/<step>.log` per step (suite logs under `gate-<jobid>/thin/`).
Expected non-zero exits: `ckdyn_n1_stalesend` and `ckdyn_n1_zcbcast` (abort tests; the
`..._message found=1` line is the check), `hapitest_small_n1` (OOM, upstream defect, see
SPEC-3 section 7), and the suite's n=1 runs of `charm.pool` examples (square, ray/simple,
ray/fib, ray/mandelbrot), which need more than one PE. A multi-process result counts only
when its log shows `Starting Reconverse with N processes`.

## Scheduling the gate

To run the gate on a schedule, add an entry with `scrontab -e` on a login node
(`scrontab -l` lists it). scrontab does not read `~/.bashrc`, so the entry names the account
and the workspace itself. For example, Sundays at 03:00, with a summary line per run
appended to `gate-history.txt`:

    #SCRON -A <gpu-account> -p gpuA40x4-interactive -t 00:30:00 -N 1 --ntasks=4 --cpus-per-task=4 --gpus-per-node=2 --mem=32g -J c4p-gate -o <workspace>/gate-%j.out
    0 3 * * 0 cd <workspace> && CHARM4PY_DELTA_WS=<workspace> GATE_HISTORY=<workspace>/gate-history.txt bash gpu-gate.sbatch

The entry runs whatever `$WS/charm` and `$WS/charm4py` hold; it does not rebuild them.
