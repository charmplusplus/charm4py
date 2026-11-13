#!/bin/bash
# C++ device ping-pong latency, no Python: benchmarks/charm++/cuda/gpudirect/latency
# (host-staged message, then nocopydevice zero-copy parameter) and latency-persistent
# (CkDevicePersistent put, done callback on the receiver), from the Charm++ tree (both
# exist on recharm). Both print "Latency for N bytes: mean += stdev us", one-way = round trip / 2.
# Layouts (one node: three; two nodes: one):
#   n2       srun -n 2 --gpus-per-task=1: each process sees one GPU, so the CUDA IPC open of
#            the peer's buffer fails and both paths use RDMA within the host.
#   n2vis    srun -n 2 --gpus=2 --gpu-bind=none, CUDA_VISIBLE_DEVICES=0,1: both GPUs visible;
#            persistent opens a CUDA IPC handle, the zero-copy parameter stays on RDMA;
#   n2visshm the same with +gpushm: the zero-copy parameter goes through HAPI's IPC buffer;
#            +gpucommbuffer 256 (MB) because the default 64 MB buffer aborts a 64 MB message.
#   2node    srun -N 2 --ntasks-per-node=1 --gpus-per-task=1: RDMA over Slingshot.
# Submit:  sbatch -N 1 cxx-latency.sbatch
#          sbatch -N 2 --ntasks-per-node=1 --gpus-per-node=1 cxx-latency.sbatch
# Sources $WS/charm4py/tools/delta/env-cuda.sh (modules, CBUILD_CXX = the charm++-target
# CUDA reconverse build). Change WS (or CBUILD_CXX after sourcing) for another workspace.
#SBATCH -A mzu-delta-gpu
#SBATCH -p gpuA40x4-interactive
#SBATCH -J cxxlat
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=32g
#SBATCH -t 00:40:00
#SBATCH -o cxxlat-%j.out
ulimit -c 0
WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
export CUDA_ARCH=sm_$cap
source $WS/charm4py/tools/delta/env-cuda.sh
export LD_LIBRARY_PATH=$CBUILD_CXX/lib:$LD_LIBRARY_PATH
unset SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none SRUN_CPU_BIND=none
NN=$SLURM_JOB_NUM_NODES
OPTS=${OPTS:-"-s 1024 -x 67108864 -i 1000 -l 100 -w 10"}
echo "JOB $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST ($NN) partition $SLURM_JOB_PARTITION start $(date '+%F %T %Z')"
echo "charm $(git -C $CHARM_SRC log --oneline -1) build $CBUILD_CXX options [$OPTS]"
srun -n $NN --ntasks-per-node=1 bash -c 'echo "GPUS $(hostname): $(nvidia-smi -L | tr "\n" ";")"'
B=$CBUILD_CXX/benchmarks/charm++/cuda/gpudirect
for d in latency latency-persistent; do
  make -C $B/$d clean > /dev/null 2>&1
  make -C $B/$d > /dev/null 2>&1; echo "BUILD $d exit=$?"
done
if [ "$NN" = 1 ]; then
  SR2="srun --mpi=pmix -n 2 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"
  SR2V="srun --mpi=pmix -n 2 -c 4 --gpus=2 --gpu-bind=none --cpu-bind=none --kill-on-bad-exit=1 env CUDA_VISIBLE_DEVICES=0,1"
  RUNS=("n2|$SR2|latency|" "n2|$SR2|latency-persistent|+gpupersistverbose"
        "n2vis|$SR2V|latency|" "n2visshm|$SR2V|latency|+gpushm +gpucommbuffer 256"
        "n2vis|$SR2V|latency-persistent|+gpupersistverbose")
else
  SR2="srun --mpi=pmix -N 2 --ntasks-per-node=1 -c 4 --gpus-per-task=1 --cpu-bind=none --kill-on-bad-exit=1"
  RUNS=("2node|$SR2|latency|" "2node|$SR2|latency-persistent|+gpupersistverbose")
fi
for r in "${RUNS[@]}"; do
  IFS='|' read shape L prog extra <<< "$r"
  echo "RUN $shape $prog [$extra] $(date '+%T')"
  cd $B/$prog
  t0=$(date +%s)
  timeout 300 stdbuf -o0 -e0 $L ./latency $OPTS +pe 2 $extra < /dev/null 2>&1 \
    | grep -v 'PMIX ERROR' | sed "s/^/[$shape $prog] /"
  echo "DONE $shape $prog exit=${PIPESTATUS[0]} $(( $(date +%s)-t0 ))s"
done
echo "END $(date '+%F %T %Z')"
