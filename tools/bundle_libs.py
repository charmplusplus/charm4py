#!/usr/bin/env python3
"""Copy libcharm and the Reconverse runtime libraries of a Charm++ build into
a directory that ships as charm4py/.libs, renamed and relinked so that an
installed charm4py does not need the build tree (port of setup.py's
prepare_darwin_libraries / prepare_linux_libraries and the copy step of
build_libcharm).

Unlike setup.py, the build directory is not modified: the libraries are
copied first and the copies are relinked.

usage: bundle_libs.py --charm-build-dir DIR --out DIR [--also DIR ...]

On macOS:   libcharm.dylib -> libcharm4py.dylib, install
            name @rpath/libcharm4py.dylib; libreconverse/liblci/liblct ->
            libcharm4py_*.dylib with @loader_path install names, and every
            reference between them rewritten to @loader_path/<bundled name>.
On Linux:   libcharm.so plus every build library it needs (directly or not),
            renamed libcharm4py_*.so, sonames and DT_NEEDED rewritten, RUNPATH
            $ORIGIN (patchelf required).
Both:       a 'reconverse' marker file when the build is a Reconverse build
            (charmrun/start.py reads it); stale libcharm4py_* files removed.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

DARWIN_RUNTIME = {
    'libreconverse.dylib': 'libcharm4py_reconverse.dylib',
    'liblci.dylib': 'libcharm4py_lci.dylib',
    'liblct.dylib': 'libcharm4py_lct.dylib',
}


def log(msg):
    print('bundle_libs: ' + msg, file=sys.stderr)


def copy_if_changed(src, dst):
    if os.path.exists(dst):
        s, d = os.stat(src), os.stat(dst)
        if s.st_size == d.st_size and int(s.st_mtime) <= int(d.st_mtime):
            return False
    tmp = dst + '.tmp'
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def darwin(library_dir, out):
    libcharm = os.path.join(library_dir, 'libcharm.dylib')
    present = [n for n in DARWIN_RUNTIME if os.path.isfile(os.path.join(library_dir, n))]
    if present and len(present) != len(DARWIN_RUNTIME):
        missing = [n for n in DARWIN_RUNTIME if n not in present]
        sys.exit('bundle_libs: incomplete Reconverse runtime in ' + library_dir +
                 ': missing ' + ', '.join(missing))

    produced = []
    staged = []   # (path, install id)
    tmp_dir = os.path.join(out, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    for name in present:
        dst = os.path.join(tmp_dir, DARWIN_RUNTIME[name])
        shutil.copy2(os.path.join(library_dir, name), dst)
        staged.append((dst, '@loader_path/' + DARWIN_RUNTIME[name]))
    dst = os.path.join(tmp_dir, 'libcharm4py.dylib')
    shutil.copy2(libcharm, dst)
    staged.append((dst, '@rpath/libcharm4py.dylib'))

    for path, install_id in staged:
        os.chmod(path, 0o755)
        subprocess.check_call(['install_name_tool', '-id', install_id, path])
        deps = subprocess.check_output(['otool', '-L', path], text=True).splitlines()[1:]
        for line in deps:
            dep = line.strip().split(' (compatibility')[0]
            base = os.path.basename(dep)
            original = next((n for n, b in DARWIN_RUNTIME.items() if base in (n, b)), None)
            if original is None:
                continue
            replacement = '@loader_path/' + DARWIN_RUNTIME[original]
            if dep != replacement:
                subprocess.check_call(['install_name_tool', '-change', dep, replacement, path])
        # install_name_tool invalidates the ad-hoc signature on arm64
        subprocess.call(['codesign', '--force', '-s', '-', path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for path, _ in staged:
        final = os.path.join(out, os.path.basename(path))
        os.replace(path, final)
        produced.append(final)
    # (the 1.x package also carried a copy named libcharm.dylib; nothing loads it now)
    stale = os.path.join(out, 'libcharm.dylib')
    if os.path.isfile(stale):
        os.unlink(stale)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return produced, bool(present)


def linux(library_dir, out):
    patchelf = shutil.which('patchelf')
    if patchelf is None:
        sys.exit('bundle_libs: patchelf is required to package a Reconverse build (pip install patchelf)')

    def pe(*args):
        return subprocess.check_output([patchelf] + list(args), text=True)

    libcharm_name = 'libcharm.so'
    libcharm = os.path.join(library_dir, libcharm_name)
    prefix = 'libcharm4py_'

    def original_name(name):
        return 'lib' + name[len(prefix):] if name.startswith(prefix) else name

    def bundled_name(name):
        return prefix + name[len('lib'):]

    def is_runtime_library(name):
        return name != libcharm_name and os.path.isfile(os.path.join(library_dir, name))

    runtime_names = []
    to_visit = [libcharm]
    while to_visit:
        for name in pe('--print-needed', to_visit.pop()).split():
            name = original_name(name)
            if is_runtime_library(name) and name not in runtime_names:
                runtime_names.append(name)
                to_visit.append(os.path.join(library_dir, name))

    real_library_dir = os.path.realpath(library_dir)

    def relink(source, target, soname=None):
        tmp = target + '.tmp'
        shutil.copy2(source, tmp)
        if soname is not None:
            subprocess.check_call([patchelf, '--set-soname', soname, tmp])
        for name in pe('--print-needed', tmp).split():
            if original_name(name) in runtime_names and name != bundled_name(original_name(name)):
                subprocess.check_call([patchelf, '--replace-needed', name,
                                       bundled_name(original_name(name)), tmp])
        old_rpath = pe('--print-rpath', tmp).strip()
        kept = [e for e in old_rpath.split(':')
                if e and e != '$ORIGIN' and os.path.realpath(e) != real_library_dir]
        subprocess.check_call([patchelf, '--set-rpath', ':'.join(['$ORIGIN'] + kept), tmp])
        os.replace(tmp, target)

    produced = []
    for name in runtime_names:
        target = os.path.join(out, bundled_name(name))
        relink(os.path.join(library_dir, name), target, soname=bundled_name(name))
        produced.append(target)
    target = os.path.join(out, libcharm_name)
    relink(libcharm, target)
    produced.append(target)
    is_reconverse = any(original_name(n).startswith('libreconverse') for n in runtime_names)
    return produced, is_reconverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charm-build-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--also', action='append', default=[],
                    help='another directory to mirror the result into (the source tree, for editable installs)')
    ap.add_argument('--reconverse', choices=['auto', 'yes', 'no'], default='auto')
    a = ap.parse_args()

    library_dir = os.path.join(os.path.abspath(a.charm_build_dir), 'lib')
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    system = platform.system().lower()
    if system == 'darwin':
        if not os.path.isfile(os.path.join(library_dir, 'libcharm.dylib')):
            sys.exit('bundle_libs: no libcharm.dylib in ' + library_dir)
        produced, is_reconverse = darwin(library_dir, out)
    elif system == 'linux':
        if not os.path.isfile(os.path.join(library_dir, 'libcharm.so')):
            sys.exit('bundle_libs: no libcharm.so in ' + library_dir)
        produced, is_reconverse = linux(library_dir, out)
    else:
        sys.exit('bundle_libs: unsupported platform ' + system)
    if a.reconverse != 'auto':
        is_reconverse = a.reconverse == 'yes'

    names = {os.path.basename(p) for p in produced}
    for d in [out] + [os.path.abspath(x) for x in a.also]:
        os.makedirs(d, exist_ok=True)
        if d != out:
            for p in produced:
                copy_if_changed(p, os.path.join(d, os.path.basename(p)))
        for f in os.listdir(d):
            if f.startswith('libcharm4py_') and f not in names:
                os.unlink(os.path.join(d, f))
        marker = os.path.join(d, 'reconverse')
        if is_reconverse:
            open(marker, 'w').close()
        elif os.path.isfile(marker):
            os.unlink(marker)
    log('bundled ' + ', '.join(sorted(names)) + ' into ' + out +
        (' (reconverse)' if is_reconverse else ''))


if __name__ == '__main__':
    main()
