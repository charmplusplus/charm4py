#!/bin/bash
# GPU gate for the thin charm4py stack on Delta (one node, two GPUs, four tasks).
# Submit from the workspace:  sbatch -p gpuA40x4-interactive  gpu-gate.sbatch
#                         or  sbatch -p gpuA100x4-interactive gpu-gate.sbatch
# Environment switches: SKIP_CXX, SKIP_PY (skip the C++ or the Python part),
# SKIP_SUITE (skip the full test_config.json suite), FILTER (regex on suite labels),
# MODE (srun or charmrun for the suite).
# Every step writes $OUT/<name>.log and one "RESULT <name> exit=<rc>" line.
#SBATCH -A mzu-delta-gpu
#SBATCH -J c4thin-gate
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=32g
#SBATCH -t 00:45:00
#SBATCH -o gate-%j.out
ulimit -c 0
WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
export CUDA_ARCH=sm_$cap
source $WS/env-cuda.sh
unset SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none SRUN_CPU_BIND=none
OUT=$WS/gate-$SLURM_JOB_ID; mkdir -p $OUT
echo "JOB $SLURM_JOB_ID node $(hostname) partition $SLURM_JOB_PARTITION CUDA_ARCH=$CUDA_ARCH start $(date '+%F %T %Z')"
nvidia-smi -L
echo "charm $(git -C $CHARM_SRC log --oneline -1) reconverse $(git -C $CHARM_SRC/contrib/reconverse log --oneline -1)"
echo "charm4py $(git -C $CHARM4PY_SRC log --oneline -1)"
step() {  # step <name> <seconds> <cmd...>; log in $OUT/<name>.log
  local name=$1 secs=$2; shift 2
  local t0=$(date +%s)
  timeout $secs stdbuf -o0 -e0 "$@" > $OUT/$name.log 2>&1 < /dev/null
  local rc=$?
  echo "RESULT $name exit=$rc $(( $(date +%s)-t0 ))s banner=[$(grep -o 'Starting Reconverse with [0-9]* process' $OUT/$name.log | head -1)]"
  return $rc
}
SR1="srun --mpi=pmix -n 1 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"
SR2="srun --mpi=pmix -n 2 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"

if [ -z "$SKIP_CXX" ]; then
# --- C++ tests run from the charm++-target build (CBUILD_CXX, build-cxx.sh): the
# charm4py-target build aborts charmxi programs ("No callback for CkRegisterMainModule").
# Its lib/ goes first on LD_LIBRARY_PATH, which outranks the binaries' RUNPATH.
export LD_LIBRARY_PATH=$CBUILD_CXX/lib:$LD_LIBRARY_PATH
make -C $CBUILD_CXX/tests/charm++/ckdyn > $OUT/ckdyn-build.log 2>&1; echo "RESULT ckdyn_build exit=$?"
# --- C++: ckdyntest (Makefile test target, +p4 -> -n 2 +pe 2, +p1 -> -n 1 +pe 1)
cd $CBUILD_CXX/tests/charm++/ckdyn && rm -rf ckdyntest.ckpt
ldd ./ckdyntest | grep -E "libcharm|reconverse"
step ckdyn_n2_rand 300 $SR2 ./ckdyntest +pe 2 +balancer RotateLB +randomized_msgq -skip priority -requiremoves
step ckdyn_n2_rotate 300 $SR2 ./ckdyntest +pe 2 +balancer RotateLB -requiremoves
step ckdyn_n2_restart 300 $SR2 ./ckdyntest +pe 2 +restart ckdyntest.ckpt
step ckdyn_n2_pe4 300 $SR2 ./ckdyntest +pe 4
step ckdyn_n1 300 $SR1 ./ckdyntest +pe 1
step ckdyn_n1_restart 300 $SR1 ./ckdyntest +pe 1 +restart ckdyntest.ckpt
step ckdyn_n1_stalesend 120 $SR1 ./ckdyntest +pe 1 -stalesend -skip pingpong -skip pingpong_inline
echo "RESULT ckdyn_n1_stalesend_message found=$(grep -c 'send to a deleted section .*stale generation' $OUT/ckdyn_n1_stalesend.log) (expect exit!=0 and found>=1)"
step ckdyn_n1_zcbcast 120 $SR1 ./ckdyntest +pe 1 -zcbcast -skip pingpong -skip pingpong_inline
echo "RESULT ckdyn_n1_zcbcast_message found=$(grep -c 'slot 2 is an NCPY slot; a message with zero-copy slots is sent point to point only, not as an array broadcast' $OUT/ckdyn_n1_zcbcast.log) (expect exit!=0 and found>=1)"
for f in $OUT/ckdyn_n*.log; do
  echo "PHASES $(basename $f .log): $(grep -E '^(PASS|SKIP) (ncpy|device|hapi)' $f | sed 's/ (.*//' | tr '\n' ',')"
done
grep -h -E "^(SKIP|FAIL)" $OUT/ckdyn_*.log | sort | uniq -c

# --- C++: hapitest, built here for this node's GPU. Phase 1 saw it OOM-killed at 33 GB;
# one bounded run: small sizes, an 8 GB step limit, 90 s.
cd $CBUILD_CXX/tests/charm++/cuda/hapitest && make clean > /dev/null && make CUDA_ARCH=$CUDA_ARCH > $OUT/hapitest-build.log 2>&1
echo "RESULT hapitest_build exit=$?"
step hapitest_small_n1 90 srun --mpi=pmix -n 1 -c 4 --gpus-per-task=1 --mem=8g --cpu-bind=none ./hapitest -v -c 2 -k 1 -n 64 +pe 1
sacct -j $SLURM_JOB_ID --format=JobID%20,State,ExitCode,MaxRSS,Elapsed | tail -3
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH#$CBUILD_CXX/lib:}
fi
[ -n "$SKIP_PY" ] && { echo "END $(date '+%F %T %Z')"; exit 0; }

# --- Python, the step 3 tests and the GPU examples
cd $CHARM4PY_SRC
step py_zerocopy_n1 150 $SR1 python tests/thin/zerocopy.py +pe 1
step py_zerocopy_n2 150 $SR2 python tests/thin/zerocopy.py +pe 2
step py_device_n1 150 $SR1 python tests/thin/device.py +pe 1
step py_device_n2 150 $SR2 python tests/thin/device.py +pe 2
# numba-cuda 0.30.4 calls numpy.row_stack, which numpy 2.5 removed; $WS/numpy-compat
# (pip install --target $WS/numpy-compat "numpy<2.5") supplies an older numpy for these two
NPP=; [ -d $WS/numpy-compat ] && NPP=$WS/numpy-compat
step py_hapi_callback_n1 150 env PYTHONPATH=$NPP $SR1 python examples/cuda/hapi/hapi-cuda-callback.py +pe 1
step py_hapi_callback_n2 150 env PYTHONPATH=$NPP $SR2 python examples/cuda/hapi/hapi-cuda-callback.py +pe 2
# one process with both GPUs; needs a CUDA torch (the venv has the CPU one), which
# $WS/torch-cuda provides when present
TPP=; [ -d $WS/torch-cuda ] && TPP=$WS/torch-cuda
step py_multi_gpu_n1 200 env PYTHONPATH=$TPP srun --mpi=pmix -n 1 -c 4 --gpus-per-task=2 --cpu-bind=none \
    --kill-on-bad-exit=1 python examples/cuda/hapi/multi_gpu_callback.py +pe 1
step py_param_server_sync_n2 300 $SR2 python examples/ray/parameter_server.py sync +pe 2
step py_param_server_async_n2 300 $SR2 python examples/ray/parameter_server.py async +pe 2
for f in $OUT/py_*.log; do
  echo "TAIL $(basename $f .log): $(grep -v -E 'PMIX ERROR|^$' $f | tail -2 | tr '\n' '|' | cut -c1-300)"
done

# --- thin test suite (test_config.json and tests/thin), one srun per test
if [ -z "$SKIP_SUITE" ]; then
python $WS/run_thin.py $OUT/thin --mode ${MODE:-srun} --counts 1,2 ${FILTER:+--filter "$FILTER"} 2>&1 | tee $OUT/suite.txt
fi
echo "END $(date '+%F %T %Z')"
# one summary line per run, appended to $GATE_HISTORY when set (the weekly scrontab entry sets it).
# The two abort steps (stalesend, zcbcast) are expected to exit nonzero; they count as
# passing when their message was found.
if [ -n "$GATE_HISTORY" ]; then
  log=$WS/gate-$SLURM_JOB_ID.out; [ -f "$log" ] || log=${SLURM_STDOUT:-/dev/null}
  nres=$(ls $OUT/*.log 2>/dev/null | wc -l)
  bad=$(cd $OUT && for f in *.log; do n=${f%.log}; case $n in *build*|*stalesend|*zcbcast) continue;; esac;
        grep -q "exit=0" <<< "$(grep -h "^RESULT $n " $log 2>/dev/null)" || echo -n "$n "; done)
  aborts=$(grep -h -E "RESULT ckdyn_n1_(stalesend|zcbcast)_message found=[1-9]" $log 2>/dev/null | wc -l)
  echo "$(date '+%F %T %Z') job=$SLURM_JOB_ID node=$(hostname -s) charm=$(git -C $CHARM_SRC rev-parse --short HEAD) charm4py=$(git -C $CHARM4PY_SRC rev-parse --short HEAD) steps-not-exit0=[${bad% }] expected-aborts=$aborts/2 suite=[$(grep -h SUMMARY $OUT/suite.txt 2>/dev/null)]" >> $GATE_HISTORY
fi
