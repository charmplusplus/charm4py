#!/usr/bin/env python3
"""Find or build the Charm++ libcharm that charm4py's extension links against.
CMakeLists.txt runs this at configure time when CHARM4PY_CHARM_BUILD_DIR is
not set, and writes the build directory it reports into its cache.

Where the Charm++ source comes from:
  CHARM4PY_CHARM_DIR set   that checkout (a developer's tree); nothing is cloned
  otherwise                a clone of the repository and commit in
                           charm_version.txt, made under the cache directory
                           (CHARM4PY_CHARM_CACHE_DIR, default <source>/build/charm)

The build directory is <charm source>/<triplet>-charm4py. When it already has
lib/libcharm.* and include/ckdyn.h it is used as it is. Otherwise this script
runs, in the Charm++ source,

  ./build charm4py <triplet> -j<N> --with-production --disable-fortran
          --with-refnum-type="unsigned int"
          --with-cmake-args=-DLCI_OPTIMIZE_FOR_NATIVE=OFF
          --destination=<build dir> $CHARM_EXTRA_BUILD_OPTS

<triplet> is CHARM4PY_BUILD_TRIPLET or the Reconverse triplet of this host;
<N> is CHARM_BUILD_PROCESSES or the number of cores. CHARM4PY_WHEEL_BUILD=1
adds the options of a wheel build: LCI's libfabric backend only, and no CPU
affinity (no hwloc). A build directory that
this script created (it holds charm4py-build.txt) and that is incomplete or
was configured with other options is rebuilt with --force; any other
incomplete directory is an error, so that a developer's build is never
deleted.

usage: build_charm.py --source-dir DIR --result FILE
"""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys

STAMP = 'charm4py-build.txt'


def log(msg):
    print('build_charm: ' + msg, file=sys.stderr, flush=True)


def fail(msg):
    log('error: ' + msg)
    sys.exit(1)


def read_pin(source_dir):
    pin = {}
    path = os.path.join(source_dir, 'charm_version.txt')
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                key, _, value = line.partition(' ')
                pin[key] = value.strip()
    for key in ('repository', 'commit'):
        if not pin.get(key):
            fail(path + ' has no "' + key + '" line')
    return pin


def host_triplet():
    system = platform.system()
    machine = platform.machine().lower()
    if system == 'Darwin' and machine in ('arm64', 'aarch64'):
        return 'reconverse-darwin-arm8'
    if system == 'Linux' and machine in ('x86_64', 'amd64'):
        return 'reconverse-linux-x86_64'
    if system == 'Linux' and machine in ('aarch64', 'arm64'):
        return 'reconverse-linux-arm8'
    fail('no Reconverse triplet for %s %s; set CHARM4PY_BUILD_TRIPLET, or build Charm++ '
         'yourself and set CHARM4PY_CHARM_BUILD_DIR' % (system, machine))


def run(cmd, cwd=None):
    log('$ ' + shlex.join(cmd) + ('  (in ' + cwd + ')' if cwd else ''))
    subprocess.check_call(cmd, cwd=cwd)


def clone(pin, charm_dir):
    """Clones the pinned commit and its contrib/reconverse submodule into
    charm_dir, through a temporary directory so that an interrupted clone is
    not mistaken for a complete one."""
    if shutil.which('git') is None:
        fail('git is required to fetch Charm++ (or set CHARM4PY_CHARM_DIR to a Charm++ checkout)')
    tmp = charm_dir + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    run(['git', 'init', '-q'], cwd=tmp)
    run(['git', 'remote', 'add', 'origin', pin['repository']], cwd=tmp)
    try:
        run(['git', 'fetch', '-q', '--depth', '1', 'origin', pin['commit']], cwd=tmp)
    except subprocess.CalledProcessError:
        # servers that do not serve a commit by its id: fetch the branch
        log('fetching the commit by id failed; fetching branch ' + pin.get('branch', '(none)'))
        if not pin.get('branch'):
            raise
        run(['git', 'fetch', '-q', 'origin', pin['branch']], cwd=tmp)
    run(['git', '-c', 'advice.detachedHead=false', 'checkout', '-q', pin['commit']], cwd=tmp)
    try:
        run(['git', 'submodule', 'update', '--init', '--depth', '1', 'contrib/reconverse'], cwd=tmp)
    except subprocess.CalledProcessError:
        run(['git', 'submodule', 'update', '--init', 'contrib/reconverse'], cwd=tmp)
    os.replace(tmp, charm_dir)


def is_complete(build_dir):
    lib = 'libcharm.dylib' if platform.system() == 'Darwin' else 'libcharm.so'
    return (os.path.isfile(os.path.join(build_dir, 'lib', lib)) and
            os.path.isfile(os.path.join(build_dir, 'include', 'ckdyn.h')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', required=True, help='the charm4py source tree')
    ap.add_argument('--result', required=True, help='file to write the build directory into')
    a = ap.parse_args()
    source_dir = os.path.abspath(a.source_dir)

    triplet = os.environ.get('CHARM4PY_BUILD_TRIPLET') or host_triplet()
    if not triplet.startswith('reconverse-'):
        fail('charm4py 2 needs a Reconverse build of Charm++; ' + triplet + ' is not one')

    if os.environ.get('CHARM4PY_CHARM_DIR'):
        charm_dir = os.path.abspath(os.path.expanduser(os.environ['CHARM4PY_CHARM_DIR']))
        if not os.path.isfile(os.path.join(charm_dir, 'build')):
            fail('CHARM4PY_CHARM_DIR=' + charm_dir + ' is not a Charm++ source tree')
        if not os.listdir(os.path.join(charm_dir, 'contrib', 'reconverse')):
            fail('contrib/reconverse is empty in ' + charm_dir + '; run\n'
                 '  git -C ' + charm_dir + ' submodule update --init contrib/reconverse')
        pin = None
    else:
        pin = read_pin(source_dir)
        cache = os.environ.get('CHARM4PY_CHARM_CACHE_DIR') or os.path.join(source_dir, 'build', 'charm')
        cache = os.path.abspath(os.path.expanduser(cache))
        os.makedirs(cache, exist_ok=True)
        charm_dir = os.path.join(cache, 'charm-' + pin['commit'][:12])
        if not os.path.isdir(os.path.join(charm_dir, '.git')):
            log('cloning Charm++ ' + pin['commit'] + ' from ' + pin['repository'])
            clone(pin, charm_dir)

    build_dir = os.path.join(charm_dir, triplet + '-charm4py')
    jobs = os.environ.get('CHARM_BUILD_PROCESSES') or str(os.cpu_count() or 4)
    # no --enable-tracing: libcharm then links the projections and summary
    # trace modules, and every run writes trace files (SPEC-6.md section 2)
    options = ['--with-production', '--disable-fortran', '--with-refnum-type=unsigned int',
               '--with-cmake-args=-DLCI_OPTIMIZE_FOR_NATIVE=OFF']
    if os.environ.get('CHARM4PY_WHEEL_BUILD', '0') not in ('', '0'):
        # a wheel: LCI's libfabric backend only (the wheel carries a libfabric;
        # it cannot carry libibverbs) and no hwloc, which Reconverse would
        # otherwise link from the build host for CPU affinity
        options += ['--with-cmake-args=-DLCI_NETWORK_BACKENDS=ofi',
                    '--with-cmake-args=-DRECONVERSE_ENABLE_CPU_AFFINITY=OFF']
    options += shlex.split(os.environ.get('CHARM_EXTRA_BUILD_OPTS', ''))
    description = 'triplet %s\noptions %s\n' % (triplet, shlex.join(options))
    if pin:
        description += 'commit %s\n' % pin['commit']
    stamp = os.path.join(build_dir, STAMP)
    ours = os.path.isfile(stamp)
    same = ours and open(stamp).read() == description

    if is_complete(build_dir) and (same or not ours):
        log('using the Charm++ build in ' + build_dir)
    else:
        if os.path.exists(build_dir) and not ours:
            fail(build_dir + ' exists but has no libcharm; build it (make in that directory) '
                 'or remove it')
        cmd = ['./build', 'charm4py', triplet, '-j' + jobs] + options + \
              ['--destination=' + build_dir]
        if os.path.exists(build_dir):
            cmd.append('--force')
        log('building Charm++ (several minutes)')
        run(cmd, cwd=charm_dir)
        if not is_complete(build_dir):
            fail('the Charm++ build finished without lib/libcharm and include/ckdyn.h in ' + build_dir)
        with open(stamp, 'w') as f:
            f.write(description)

    with open(a.result, 'w') as f:
        f.write(build_dir)


if __name__ == '__main__':
    main()
