#!/bin/bash
# Build Charm++ (charm4py target, reconverse, CUDA) for the thin charm4py stack on Delta.
# usage: build.sh [--force]    (login node; about 6-10 minutes)
set -o pipefail
source "$(dirname "$(readlink -f "$0")")/env-cuda.sh"
cd $CHARM_SRC || exit 1
git submodule update --init contrib/reconverse
FORCE=; [ "$1" = --force ] && FORCE=--force
t0=$(date +%s)
./build charm4py reconverse-linux-x86_64 cuda -j16 --with-production --disable-fortran \
    "--with-refnum-type=unsigned int" \
    --with-cmake-args=-DLCI_OPTIMIZE_FOR_NATIVE=OFF \
    --destination=$CBUILD $FORCE 2>&1 | tee $WS/build-charm.log
rc=$?
echo "BUILD exit=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a $WS/build-charm.log
exit $rc
