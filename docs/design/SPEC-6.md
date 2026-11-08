# Step 6 specification: packaging

Written 2026-11-08 with the implementation (branch `thin`), against `ckdyn-dev`
454f5eb10. DESIGN.md 6 (6.1-6.3) and 7.1 step 6 are the plan and the gate; the logs
are in `logs/step-6/` of the project directory.

## 1. The three build paths

`pip install .` runs scikit-build-core, which configures `CMakeLists.txt`. The first
thing the CMake file needs is a Charm++ build directory (`lib/libcharm.*`,
`include/ckdyn.h`):

| set by the user | what happens |
|---|---|
| `CHARM4PY_CHARM_BUILD_DIR` | used as it is (developers; unchanged from step 4a) |
| `CHARM4PY_CHARM_DIR` only | `tools/build_charm.py` uses `$CHARM4PY_CHARM_DIR/<triplet>-charm4py`; builds it there when it has no libcharm |
| nothing | `tools/build_charm.py` clones the pin of `charm_version.txt` into `build/charm/charm-<sha12>/` (or `$CHARM4PY_CHARM_CACHE_DIR`) and builds `<clone>/<triplet>-charm4py` |

`build_charm.py` runs at configure time from `execute_process` and writes the build
directory into a file that CMake reads. The rest of the CMake file is step 4a's:
`tools/bundle_libs.py` copies libcharm, libreconverse, liblci and liblct into
`charm4py/.libs` under `libcharm4py*` names with `@loader_path`/`$ORIGIN` links, and
`_ckdyn` is linked against the copy with an rpath of `.libs`.

The Charm++ command:

    ./build charm4py <triplet> -j<N> --with-production --disable-fortran
            --with-refnum-type="unsigned int"
            --with-cmake-args=-DLCI_OPTIMIZE_FOR_NATIVE=OFF
            [CHARM4PY_WHEEL_BUILD=1: --with-cmake-args=-DLCI_NETWORK_BACKENDS=ofi
                                     --with-cmake-args=-DRECONVERSE_ENABLE_CPU_AFFINITY=OFF]
            --destination=<build dir> $CHARM_EXTRA_BUILD_OPTS

- `<triplet>`: `CHARM4PY_BUILD_TRIPLET`, else from `platform`: `reconverse-darwin-arm8`,
  `reconverse-linux-x86_64`, `reconverse-linux-arm8`. A non-Reconverse triplet is an
  error. There is no Reconverse triplet for macOS x86_64.
- `<N>`: `CHARM_BUILD_PROCESSES`, else `os.cpu_count()`.
- `LCI_OPTIMIZE_FOR_NATIVE=OFF`: LCI otherwise adds `-march=native` to every LCI
  source, which makes a wheel fail on an older CPU, and a source build made on a
  cluster's login node fail on a compute node of another generation.
- The wheel options: LCI's CMake looks for the `ibv` and `ofi` backends and links
  every one it finds; a wheel can carry libfabric but not libibverbs (its provider
  drivers are loaded from the host). Reconverse links the host's hwloc for CPU
  affinity when it finds one; on this Mac that is MacPorts' hwloc, which brings
  libxml2, ICU (33 MB) and libiconv into the wheel. In a wheel, `+pemap` and
  `+setcpuaffinity` are accepted and do nothing.
- Reuse: a build directory with libcharm and `ckdyn.h` is used without running
  `./build`. `build_charm.py` writes `charm4py-build.txt` (triplet, options, commit)
  into a directory it built; when that file is present and the options differ, or the
  build is incomplete, it rebuilds with `--force`. An incomplete directory without the
  file is an error, so a developer's build directory is never deleted.
- The clone: `git init` + `git fetch --depth 1 origin <sha>` (GitHub serves a commit by
  id) with a fallback to fetching the branch, then
  `git submodule update --init --depth 1 contrib/reconverse`, in `<dir>.tmp` renamed
  when complete.

## 2. Deviations from DESIGN.md 6

1. **No `--enable-tracing` in the default build or the wheels** (DESIGN 6.1 item 6
   says wheels have it). In a `charm4py` build with tracing, `libcharm` links
   `trace-summary` and `trace-projections` (Charm++ `CMakeLists.txt`, the
   `target_link_libraries(charm trace-summary trace-projections ...)` line), so every
   run prints "Tracemode Projections enabled" and writes `<app>.sts`, `.0.log.gz`,
   `.sum` and `.projrc` next to the program (seen with the first local wheel). Tracing
   in wheels needs the trace modules to be selected at run time (`+tracemode`), a
   Charm++ change. Until then: `CHARM_EXTRA_BUILD_OPTS=--enable-tracing` in a source
   build.
2. **`tools/bundle_libs.py` is kept for wheels too** (DESIGN 6.2 item 3 expected
   auditwheel/delocate to replace it). The repair tools handle libraries already in the
   wheel with relative links, and the source-build path needs `bundle_libs.py` anyway,
   so both paths install the same layout. auditwheel and delocate only add the
   external libraries (libfabric) under `charm4py.libs/` or `charm4py/.dylibs/`.
3. libfabric is built from its release tarball (`tools/build_libfabric.sh`, 2.2.0,
   providers `tcp sockets udp rxm`, plus `shm` on Linux, `--enable-only`) instead of
   the distribution's package: the AlmaLinux 8 package of a manylinux image links
   libibverbs, librdmacm and libpsm2, and Homebrew's targets the runner's macOS
   version, which delocate would then have to put in the wheel tag.
4. The duplicate `libcharm.dylib` (a second copy of `libcharm4py.dylib`, 3.9 MB, kept
   from the 1.x package) is no longer bundled; nothing loads it.

## 3. The pin file

`charm_version.txt`, `key value` lines:

    repository https://github.com/charmplusplus/charm.git
    branch ckdyn-dev
    commit 454f5eb10ffdcf857227fa9685f226bd151c4616

The commit decides the Reconverse version too (its `contrib/reconverse` gitlink:
ba836e6, reconverse main after #324). When `ckdyn-dev` merges into `recharm`, the pin
moves to that `recharm` commit and `branch` to `recharm`. Updating the pin changes the
clone directory name (`charm-<sha12>`), so a stale build is never reused.

## 4. Wheel contents (macOS arm64, cp313, local build)

    charm4py/_ckdyn.cpython-313-darwin.so          308 KB
    charm4py/.libs/libcharm4py.dylib               3.9 MB   (libcharm)
    charm4py/.libs/libcharm4py_reconverse.dylib    372 KB
    charm4py/.libs/libcharm4py_lci.dylib           269 KB
    charm4py/.libs/libcharm4py_lct.dylib           288 KB
    charm4py/.libs/reconverse                      marker read by charmrun/start.py
    charm4py/.dylibs/libfabric.1.dylib             719 KB   (added by delocate)
    charm4py/*.py, charm4py/ray/*, charmrun/*.py

Tag `macosx_14_0_arm64` (`MACOSX_DEPLOYMENT_TARGET=14.0` for libfabric, Charm++ and
the module). External dependencies left: libSystem, libc++, libz, and the
CoreFoundation, Foundation and IOKit frameworks. Linux wheels have the same layout
with `.so` names, plus `charm4py.libs/libfabric-<hash>.so.1` from auditwheel.

Multi-process start from an installed package: `python -m charmrun.start +p2 app.py`
needs no lcrun. `charmrun/start.py` sees the `reconverse` marker, translates `+pN`
to `N` processes with `+pe N`, and starts them itself with LCI's file PMI
(`LCT_PMI_BACKEND=file,local`, a private rendezvous directory), as `lcrun` does.

## 5. CI (`.github/workflows/thin.yaml`, branch `thin` only)

| job | runners | what |
|---|---|---|
| `sdist` | ubuntu-latest, macos-15 | `python -m build --sdist`; `pip install dist/*.tar.gz` (clones and builds Charm++ against the system libfabric: `libfabric-dev` / Homebrew); hello_world at +p1 and +p2; `auto_test.py` with `CHARM4PY_TEST_NUM_PROCESSES=2`, from a copy of the tests outside the source tree |
| `wheels` | ubuntu-latest (x86_64), ubuntu-24.04-arm (aarch64), macos-15 (arm64) | cibuildwheel 4.3.5 with `[tool.cibuildwheel]`: cp311-cp313; test command hello_world +p1 and +p2; wheels uploaded as artifacts |

`FI_PROVIDER=tcp` in both, as in the 1.x reconverse job. The 1.x workflows
(`charm4py.yml`, `build_wheels.yml`) are unchanged and do not run on pushes to `thin`.

## 6. Known issues

- **CCS crash, resolved 2026-11-08 14:08 PST.** The reconverse fix `ee47fb6` (branch
  `ccs-fn-dispatch`) is pushed to charmplusplus/reconverse; ckdyn-dev `44b27e6c8` records
  it as the submodule commit and `charm_version.txt` pins that commit, so wheels and
  default source builds now carry the fix. The branch becomes a reconverse pull request
  when speculative mode ends.
- No tracing in wheels (section 2, item 1).
- No CUDA/HIP wheels; the source recipe is step 3's (placeholder in `install.rst`).
- macOS x86_64: no Reconverse triplet, no wheel.

## 7. Results (2026-11-08)

Local, this Mac (macOS 15, arm64): `pip wheel .` in a scratch venv (isolated build:
scikit-build-core 1.x, nanobind, numpy, greenlet from PyPI) with
`CHARM4PY_WHEEL_BUILD=1`, `MACOSX_DEPLOYMENT_TARGET=14.0` and `OFI_ROOT` at a
`tools/build_libfabric.sh` prefix; the clone of the pin and the Charm++ build took
about 90 s with `-j6`. `delocate-wheel` (cibuildwheel's macOS repair command) gave
`charm4py-2.0.0.dev0-cp313-cp313-macosx_14_0_arm64.whl`. cibuildwheel itself was not
run here: on macOS it installs python.org interpreters with `sudo installer`.
Installed into a new venv (`venv-wheel`, MacPorts Python 3.13): hello_world passes at
`+p1` and `+p2` (`python -m charmrun.start`, no lcrun), and `auto_test.py` with
`CHARM4PY_TEST_NUM_PROCESSES=2`, run from a copy of the tests outside the source tree,
passes all 83 tests (the 3 numba tests are skipped by condition) in 148 s, once
torch, torchvision and filelock are installed for `examples/ray/parameter_server.py`.

The developer paths: `CHARM4PY_CHARM_BUILD_DIR` (with `CHARM4PY_CHARM_DIR` and
`CHARM4PY_BUILD_TRIPLET`, the project's recipe) and `CHARM4PY_CHARM_DIR` alone both
install editable and reuse `charm/reconverse-darwin-arm8-charm4py`, the build with the
CCS fix.

CI: see DESIGN.md 7.1 step 6 for the runs.
