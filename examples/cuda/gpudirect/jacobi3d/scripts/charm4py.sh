#!/bin/bash
# Jacobi3D timing runs: three runs of each ghost-exchange mode (staged, device,
# persistent). PR #208 ran this on Summit with jsrun (6 GPUs per node, one process
# per GPU); this version runs on NCSA Delta with srun, or on one machine with
# python -m charmrun.start when it is not inside a Slurm job.
#
# Delta (one process per GPU; 4 GPUs per node on gpuA100x4 and gpuA40x4):
#   sbatch -N 2 scripts/charm4py.sh          # from examples/cuda/gpudirect/jacobi3d
# One machine with NGPUS GPUs:
#   NGPUS=2 bash scripts/charm4py.sh
# Variables: GRID (cube edge, default 768), ITERS (100), WARMUP (10),
#            MODES ("staged device persistent"), RUNS (3), WS (Delta workspace).
# Each run prints "Average time per iteration" and "Communication time per
# iteration"; scripts/process_times.py averages them per mode.
#SBATCH -A mzu-delta-gpu
#SBATCH -J jacobi3d-charm4py
#SBATCH -p gpuA100x4
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=64g
#SBATCH -t 00:30:00
#SBATCH -o jacobi3d-charm4py-n%N-%j.out

GRID=${GRID:-768}
ITERS=${ITERS:-100}
WARMUP=${WARMUP:-10}
MODES=${MODES:-"staged device persistent"}
RUNS=${RUNS:-3}

exe() { echo "\$ $*" ; "$@" ; }

if [ -n "$SLURM_JOB_ID" ]; then
  WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
  source $WS/env-cuda.sh       # modules, FI_PROVIDER=cxi, the venv (tools/delta/env-cuda.sh)
  # numba-cuda 0.30.4 calls numpy.row_stack, removed in numpy 2.5 (tools/delta/README.md)
  [ -d $WS/numpy-compat ] && export PYTHONPATH=$WS/numpy-compat${PYTHONPATH:+:$PYTHONPATH}
  cd ${SLURM_SUBMIT_DIR:-$WS/charm4py/examples/cuda/gpudirect/jacobi3d}
  NP=$SLURM_NTASKS
  GPN=${SLURM_GPUS_PER_NODE:-4}; GPN=${GPN##*:}
  # every GPU of a node visible to every process (HAPI gives process k of the node
  # device k, and CUDA IPC between processes of a node needs both GPUs visible)
  LAUNCH="srun --mpi=pmix -n $NP --gpus-per-node=$GPN --gpu-bind=none --cpu-bind=none
          env CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPN-1)))"
  PES="+pe $NP"
else
  NP=${NGPUS:-2}
  LAUNCH="python -m charmrun.start +p$NP"
  PES=""
fi

echo "# Charm4py Jacobi3D, $NP processes, grid ${GRID}^3, $ITERS iterations, $WARMUP warm-up"
for mode in $MODES; do
  echo "# Mode $mode"
  for run in $(seq 1 $RUNS); do
    date
    echo "# Run $run"
    if [ -n "$SLURM_JOB_ID" ]; then
      exe $LAUNCH python jacobi3d.py -x $GRID -y $GRID -z $GRID -w $WARMUP -i $ITERS -m $mode $PES
    else
      exe $LAUNCH jacobi3d.py -x $GRID -y $GRID -z $GRID -w $WARMUP -i $ITERS -m $mode
    fi
  done
done
