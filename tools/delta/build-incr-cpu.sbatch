#!/bin/bash
# Incremental rebuild of both CUDA builds and the charm4py reinstall, inside an
# allocation (keeps parallel compiles off the login nodes). Submit from the workspace:
#   sbatch -p cpu-interactive build-incr-cpu.sbatch   (no GPU needed; CUDA_ARCH is set below)
#SBATCH -A mzu-delta-cpu
#SBATCH -J c4thin-build
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH -t 00:30:00
#SBATCH -o build-%j.out
WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
cap=${CUDA_CAP:-86}   # compute capability of the GPUs the build targets (86 A40, 80 A100); no GPU on the build node
export CUDA_ARCH=sm_$cap
source $WS/env-cuda.sh
echo "JOB $SLURM_JOB_ID node $(hostname) start $(date '+%F %T %Z')"
echo "charm $(git -C $CHARM_SRC log --oneline -1) charm4py $(git -C $CHARM4PY_SRC log --oneline -1)"
make -C $CBUILD -j16 > $WS/build-$SLURM_JOB_ID-c4p.log 2>&1; echo "RESULT make_c4p exit=$?"
make -C $CBUILD_CXX -j16 > $WS/build-$SLURM_JOB_ID-cxx.log 2>&1; echo "RESULT make_cxx exit=$?"
grep -h -E "error|Error " $WS/build-$SLURM_JOB_ID-c4p.log $WS/build-$SLURM_JOB_ID-cxx.log | head -40
(cd $CHARM4PY_SRC && pip install --no-build-isolation -e . > $WS/build-$SLURM_JOB_ID-pip.log 2>&1); echo "RESULT pip exit=$?"
grep -h -E "error" $WS/build-$SLURM_JOB_ID-pip.log | head -20
echo "END $(date '+%F %T %Z')"
