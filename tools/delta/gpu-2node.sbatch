#!/bin/bash
# Two nodes, one process and one GPU per node: ckdyntest's ncpy and device phases and
# the BUFFER / NCPY / DEVICE one-way measurement (SPEC-3.md section 6) across Slingshot.
# Submit from the workspace:  sbatch -p gpuA40x4-interactive gpu-2node.sbatch
#SBATCH -A mzu-delta-gpu
#SBATCH -J c4thin-2node
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 00:30:00
#SBATCH -o gate2n-%j.out
ulimit -c 0
WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
export CUDA_ARCH=sm_$cap
source $WS/env-cuda.sh
unset SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none SRUN_CPU_BIND=none
OUT=$WS/gate2n-$SLURM_JOB_ID; mkdir -p $OUT
echo "JOB $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST partition $SLURM_JOB_PARTITION CUDA_ARCH=$CUDA_ARCH start $(date '+%F %T %Z')"
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
SRN="srun --mpi=pmix -N 2 --ntasks-per-node=1 -c 8 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"

# --- C++ ckdyntest across the two nodes (charm++-target build)
export LD_LIBRARY_PATH=$CBUILD_CXX/lib:$LD_LIBRARY_PATH
cd $CBUILD_CXX/tests/charm++/ckdyn && rm -rf ckdyntest.ckpt
step ckdyn_2node 300 $SRN ./ckdyntest +pe 2
echo "PHASES ckdyn_2node: $(grep -E '^(PASS|SKIP) (ncpy|device|hapi)' $OUT/ckdyn_2node.log | tr '\n' ',')"
step ckdyn_2node_only_zc 300 env LCI_LOG_LEVEL=info $SRN ./ckdyntest +pe 2 -skip pingpong -skip pingpong_inline
grep -i -E "hmem|gdr|gpudirect|ipc|staging|device|rdma|cuda" $OUT/ckdyn_2node*.log | grep -v -E "^.*(PASS|SKIP) " | sort | uniq -c | head -40
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH#$CBUILD_CXX/lib:}

# --- one-way time, charm4py
cd $CHARM4PY_SRC
for size in 65536 262144 1048576 4194304 16777216 67108864; do
  for mode in buffer ncpy; do
    step bench_${mode}_$size 180 $SRN python $WS/bench_zc_delta.py $mode $size +pe 2
    grep -h BENCH $OUT/bench_${mode}_$size.log
  done
done
for rep in 1 2 3; do
  for mode in buffer ncpy device; do
    step bench16_${mode}_r$rep 180 $SRN python $WS/bench_zc_delta.py $mode 16777216 +pe 2
    grep -h BENCH $OUT/bench16_${mode}_r$rep.log
  done
done
for size in 1048576 67108864; do
  step bench_device_$size 180 $SRN python $WS/bench_zc_delta.py device $size +pe 2
  grep -h BENCH $OUT/bench_device_$size.log
done
echo "END $(date '+%F %T %Z')"
