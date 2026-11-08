#!/bin/sh
# Builds a small libfabric (the tcp, sockets, udp and rxm providers, plus shm on
# Linux) from a release tarball and installs it into PREFIX. The wheel builds
# (cibuildwheel, pyproject.toml [tool.cibuildwheel]) link LCI's ofi backend
# against this library, and auditwheel or delocate copies it into the wheel.
# A library built this way does not depend on the host's RDMA libraries
# (libibverbs, librdmacm, libpsm2), which a wheel cannot assume.
#
# usage: tools/build_libfabric.sh PREFIX
# environment: LIBFABRIC_VERSION (default below), MACOSX_DEPLOYMENT_TARGET on macOS
set -eu
prefix=${1:?usage: build_libfabric.sh PREFIX}
version=${LIBFABRIC_VERSION:-2.2.0}
if [ -f "$prefix/lib/pkgconfig/libfabric.pc" ] && \
   grep -q "^Version: $version" "$prefix/lib/pkgconfig/libfabric.pc"; then
  echo "build_libfabric: libfabric $version is already in $prefix"
  exit 0
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
url=https://github.com/ofiwg/libfabric/releases/download/v$version/libfabric-$version.tar.bz2
echo "build_libfabric: $url"
curl -fsSL -o libfabric.tar.bz2 "$url"
tar xjf libfabric.tar.bz2
cd "libfabric-$version"
providers="--enable-tcp=yes --enable-sockets=yes --enable-udp=yes --enable-rxm=yes"
if [ "$(uname -s)" = Linux ]; then
  providers="$providers --enable-shm=yes"
fi
# shellcheck disable=SC2086
./configure --prefix="$prefix" --enable-only $providers \
  --disable-static --disable-dependency-tracking
jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
make -j"$jobs"
make install
echo "build_libfabric: installed libfabric $version into $prefix"
