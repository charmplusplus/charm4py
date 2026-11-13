# Delta environment for the thin charm4py stack on a CUDA reconverse build.
# Source from the login shell and from every batch script:  source $WS/env-cuda.sh
# Override CUDA_ARCH (sm_80 A100, sm_86 A40) before sourcing, or let the batch
# script set it from nvidia-smi.
if [ -z "$LMOD_CMD" ]; then source /etc/profile.d/zz-lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash; fi
# Login defaults already give PrgEnv-gnu/8.7.0 (gcc 14.2.1), cudatoolkit/26.5_13.2,
# libfabric/2.3.1, craype-accel-nvidia80; hwloc is the system /usr/lib64 one.
module load cmake/3.31.8 2>/dev/null
module load python/3.13.5-gcc13.3.1 2>/dev/null
export WS=${WS:-/projects/mzu/lkale/software/charm4py-thin}
export CHARM_SRC=$WS/charm
export CHARM4PY_SRC=$WS/charm4py
export CBUILD=$CHARM_SRC/reconverse-linux-x86_64-cuda-charm4py
# charm++-target build of the same source, for the C++ tests (build-cxx.sh)
export CBUILD_CXX=$CHARM_SRC/reconverse-linux-x86_64-cuda-cxx
export CUDA_HOME=${CUDA_HOME:-$CUDATOOLKIT_HOME}
export CUDATOOLKIT_HOME=${CUDATOOLKIT_HOME:-$CUDA_HOME}
export CUDA_ARCH=${CUDA_ARCH:-sm_80}
export LD_LIBRARY_PATH=$CBUILD/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FI_PROVIDER=cxi
export LCI_NETWORK_BACKENDS=ofi
# for pip install -e of charm4py against this build
export CHARM4PY_CHARM_DIR=$CHARM_SRC
export CHARM4PY_BUILD_TRIPLET=reconverse-linux-x86_64
export CHARM4PY_CHARM_BUILD_DIR=$CBUILD
if [ -f $WS/venv/bin/activate ]; then . $WS/venv/bin/activate; fi
true
