#!/bin/bash
# Persistent device buffers, posted receives and the +gpushm IPC path (SPEC-7-device.md).
# One node, two GPUs (the default) or, with -N 2, two nodes with one GPU each:
#   sbatch -p gpuA40x4-interactive gpu-devpersist.sbatch
#   sbatch -p gpuA40x4-interactive -N 2 --ntasks-per-node=1 --gpus-per-node=1 gpu-devpersist.sbatch
# Switches: SKIP_CXX, SKIP_PY, SKIP_BENCH. Logs: $WS/devp-<jobid>/<step>.log, one RESULT line each.
#SBATCH -A mzu-delta-gpu
#SBATCH -J c4thin-devp
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=32g
#SBATCH -t 00:45:00
#SBATCH -o devp-%j.out
ulimit -c 0
WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
export CUDA_ARCH=sm_$cap
source $WS/env-cuda.sh
unset SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none SRUN_CPU_BIND=none
OUT=$WS/devp-$SLURM_JOB_ID; mkdir -p $OUT
NN=$SLURM_JOB_NUM_NODES
echo "JOB $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST ($NN) partition $SLURM_JOB_PARTITION CUDA_ARCH=$CUDA_ARCH start $(date '+%F %T %Z')"
nvidia-smi -L
echo "charm $(git -C $CHARM_SRC log --oneline -1) reconverse $(git -C $CHARM_SRC/contrib/reconverse log --oneline -1)"
echo "charm4py $(git -C $CHARM4PY_SRC log --oneline -1)"
step() {  # step <name> <seconds> <cmd...>
  local name=$1 secs=$2; shift 2
  local t0=$(date +%s)
  timeout $secs stdbuf -o0 -e0 "$@" > $OUT/$name.log 2>&1 < /dev/null
  local rc=$?
  echo "RESULT $name exit=$rc $(( $(date +%s)-t0 ))s banner=[$(grep -o 'Starting Reconverse with [0-9]* process' $OUT/$name.log | head -1)] modes=[$(grep -h -o 'CkDyn> [^:]*: [A-Za-z/ ]*' $OUT/$name.log | sort -u | tr '\n' ';')]"
  return $rc
}
if [ "$NN" = 1 ]; then
  # one GPU per task, and both GPUs visible to both tasks (hapi's block mapping
  # gives process k device k); the second form is what +gpushm IPC needs if the first fails
  SR1="srun --mpi=pmix -n 1 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"
  SR2="srun --mpi=pmix -n 2 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"
  SR2V="srun --mpi=pmix -n 2 -c 4 --gpus=2 --gpu-bind=none --cpu-bind=none --kill-on-bad-exit=1"
  SHAPES="n1 n2 n2shm n2vis n2visshm"
else
  SR2="srun --mpi=pmix -N 2 --ntasks-per-node=1 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"
  SHAPES="2node"
fi
cmd_of() {   # the launcher and extra arguments of a shape
  case $1 in
    n1) echo "$SR1|+pe 1";; n2) echo "$SR2|+pe 2";; n2shm) echo "$SR2|+pe 2 +gpushm";;
    n2vis) echo "$SR2V|+pe 2";; n2visshm) echo "$SR2V|+pe 2 +gpushm";; 2node) echo "$SR2|+pe 2";;
  esac
}

if [ -z "$SKIP_CXX" ]; then
export LD_LIBRARY_PATH=$CBUILD_CXX/lib:$LD_LIBRARY_PATH
make -C $CBUILD_CXX/tests/charm++/ckdyn > $OUT/ckdyn-build.log 2>&1; echo "RESULT ckdyn_build exit=$?"
cd $CBUILD_CXX/tests/charm++/ckdyn && rm -rf ckdyntest.ckpt
for sh in $SHAPES; do
  IFS='|' read L A <<< "$(cmd_of $sh)"
  step ckdyn_$sh 300 $L ./ckdyntest $A -skip sections -skip checkpoint -skip migration -skip late
  echo "PHASES ckdyn_$sh: $(grep -E '^(PASS|SKIP|FAIL) (ncpy|device|hapi|devpersist)' $OUT/ckdyn_$sh.log | sed 's/ (.*//' | tr '\n' ',')"
  grep -h "devpersist/" $OUT/ckdyn_$sh.log | grep mode
done
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH#$CBUILD_CXX/lib:}
fi

if [ -z "$SKIP_PY" ]; then
cd $CHARM4PY_SRC
for sh in $SHAPES; do
  IFS='|' read L A <<< "$(cmd_of $sh)"
  for t in device_persistent posted_receive device; do
    step py_${t}_$sh 200 $L python tests/thin/$t.py $A
    echo "TAIL py_${t}_$sh: $(grep -v -E 'PMIX ERROR|^$|^Charm|^Reconverse|^HAPI|^CharmLB|^Isomalloc' $OUT/py_${t}_$sh.log | tail -3 | tr '\n' '|' | cut -c1-400)"
  done
done
[ "$NN" = 1 ] && { step py_posted_mismatch 120 $SR1 python tests/thin/posted_receive.py +pe 1 -mismatch;
  echo "RESULT py_posted_mismatch_message found=$(grep -c 'the posted buffer has 1048576 bytes, the message.s slot 2097152' $OUT/py_posted_mismatch.log)"; }
fi

if [ -z "$SKIP_BENCH" ]; then
cd $CHARM4PY_SRC
B=$CHARM4PY_SRC/tools/delta
if [ "$NN" = 1 ]; then BSHAPES="n2 n2shm n2vis n2visshm"; SIZES="16777216"; else BSHAPES="2node"; SIZES="1048576 4194304 16777216 67108864"; fi
for sh in $BSHAPES; do
  IFS='|' read L A <<< "$(cmd_of $sh)"
  step bench_device_$sh 180 $L python $B/bench_zc_delta.py device 16777216 $A
  grep -h BENCH $OUT/bench_device_$sh.log
  for size in $SIZES; do
    for op in get put; do
      step bench_p${op}_${size}_$sh 180 $L python $B/bench_devpersist.py $op $size $A
      grep -h BENCH $OUT/bench_p${op}_${size}_$sh.log
    done
  done
done
fi
echo "END $(date '+%F %T %Z')"
