#!/bin/bash
# Build the charm++ target (same source, CUDA, reconverse) for the C++ tests
# (ckdyntest, hapitest). The charm4py-target build sets CMK_CHARM4PY, under which
# a charmxi program aborts with "No callback for CkRegisterMainModule".
# usage: build-cxx.sh [--force]    (login node)
set -o pipefail
source "$(dirname "$(readlink -f "$0")")/env-cuda.sh"
cd $CHARM_SRC || exit 1
FORCE=; [ "$1" = --force ] && FORCE=--force
t0=$(date +%s)
./build charm++ reconverse-linux-x86_64 cuda -j16 --with-production --disable-fortran \
    "--with-refnum-type=unsigned int" \
    --with-cmake-args=-DLCI_OPTIMIZE_FOR_NATIVE=OFF \
    --destination=$CBUILD_CXX $FORCE 2>&1 | tee $WS/build-cxx.log
rc=$?
echo "BUILD exit=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a $WS/build-cxx.log
exit $rc
