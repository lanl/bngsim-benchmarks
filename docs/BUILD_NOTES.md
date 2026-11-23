# Build notes

How to get each engine onto a platform the suite does not have a prebuilt
artifact for. The default environment (`pixi.toml` + `pixi.lock`) covers
**linux-64** and **osx-arm64** out of the box: every dependency resolves to a
conda package or a PyPI wheel, and `pixi install` is all you need. These notes
are for the cases that fall outside that — chiefly **linux-aarch64 (ARM)**,
which the cross-platform timing gate of `docs/DESIGN.md` section 9.1 needs and
which PyPI/conda do not fully cover.

Nothing here changes the x86-64 path. The wheels the suite runs on today are
untouched; this is additive.

> **The one rule that governs every native build here — ABI consistency.** Build
> each engine's compiled core with the **same conda-forge toolchain that provides
> the `libstdc++` it will run against** — never the cluster's system `gcc`/`g++`.
> A C++ object built against one `libstdc++` and run against another is undefined
> behaviour, and the usual symptom is a **segfault at runtime, not a link error at
> build time** — so it passes every build step and then crashes in use. The sharp
> edge is `std::filesystem::path`, whose internal layout changed at the **GCC 9
> ABI break**: a core built with a system GCC 8 but run against a conda-forge
> GCC ≥ 9 `libstdc++` corrupts the heap the first time it handles a path (e.g. a
> `.net` filename) and crashes with no traceback. The suite's run environments are
> conda-forge (GCC 13-era `libstdc++`), so **every engine core here is built with
> the conda-forge compiler**, matching the x86 wheels' own toolchain. `bench
> doctor` now actively guards this: it loads a trivial model in a child process
> and reports the ABI mismatch by name if the core crashes, rather than letting a
> run reach a compute node and die silently.

> **Cluster specifics.** The commands below use placeholders — `<arm-partition>`,
> `<account>`, `<qos>` — for Slurm identifiers. Fill them from your site's
> allocation (on Darwin these live in the git-ignored `slurm.yaml` and `dev/`,
> never in a tracked file). Replace `$USER` as needed.

## Per-engine build matrix (aarch64)

| Engine | aarch64 availability | Build effort | Needs LLVM? |
| --- | --- | --- | --- |
| **BNGsim** (system under test) | x86-64 PyPI wheel only; **aarch64 builds from the `v0.17.2` git checkout** | moderate CMake (SUNDIALS vendored) | no |
| **run_network** (BioNetGen) | no aarch64 prebuilt; **build from 2.9.3 source** | moderate CMake | no |
| **AMICI** | PyPI **sdist** on every platform → `uv pip install amici` compiles it (x86-64 and aarch64 alike) | moderate (SUNDIALS vendored); needs a C/C++ toolchain | no |
| **COPASI** (via BASICO) | **prebuilt `manylinux_2_28` wheels on x86-64 _and_ aarch64** (`python-copasi`); `copasi-basico` is pure Python | none — no build on any platform | no |
| **libRoadRunner** | **no aarch64 wheel and no sdist on PyPI**; **built from source (done)** | moderate CMake | yes — satisfied by a prebuilt conda-forge LLVM |

libRoadRunner is the only engine that needs LLVM, but LLVM does **not** have to
be built from source: conda-forge ships a prebuilt `llvmdev` for linux-aarch64
at the exact version libRoadRunner wants (13.0.x), and RoadRunner's
`FindLLVM.cmake` discovers it through `llvm-config`. Installing that prebuilt
LLVM turns what used to be a multi-hour build into an ordinary ~6-minute CMake
build. run_network builds from source with no LLVM dependency; AMICI compiles
from its sdist (no LLVM) on both platforms; COPASI needs no build at all — its
aarch64 wheel installs like any other.

## aarch64 cluster tooling

Verified on a ThunderX2 node (2026-11-13): aarch64, RHEL 8.10, glibc 2.28 (the
same floor as the x86 side), system gcc 8.5 with gcc up to 13.1 via Lmod; cmake,
ninja and git present; **no LLVM module** (only `llvm-compat-libs` runtime).

Grab an interactive ARM node:

```sh
srun -p <arm-partition> -A <account> -q <qos> -N1 -n1 -c 16 -t 00:25:00 bash -lc '...'
```

Two gotchas that will waste an afternoon if you hit them cold:

- **The `pixi`/`uv` on `PATH` are x86-64 binaries.** `~/.pixi/bin/pixi` and
  `~/.local/bin/uv` are x86-64 ELF and die with `Exec format error` on aarch64.
  Install a **native aarch64 pixi** to a separate prefix — e.g.
  `$HOME/.pixi-aarch64/bin/pixi` — and call it by full path on ARM. (Verified
  with pixi 0.84.2, static build.)

- **x86 user-site packages leak into the ARM env.** `~/.local/lib/python3.12`
  holds x86-64 numpy that Python picks up via user-site, so an ARM interpreter
  tries to load `_multiarray_umath.cpython-312-x86_64-linux-gnu.so` and fails
  with an `ImportError` about an incompatible C-extension. Always neutralize
  user-site on ARM:

  ```sh
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  ```

- **`/tmp` is node-local tmpfs and does not persist across allocations.** Put
  scratch builds and caches under `$HOME/.cache` (NFS, shared across
  architectures and allocations) so a second `srun` can still see them.

## BNGsim on aarch64 (from the git checkout, conda-forge toolchain)

PyPI ships only an x86-64 wheel for BNGsim, so on aarch64 the native core is
built locally. Two non-obvious requirements — miss either and the build
"succeeds" but the result is unusable:

1. **Build from the `v0.17.2` git checkout, not the PyPI sdist.** BNGsim's CMake
   bakes the build commit from `git rev-parse` into the compiled core, and that
   commit gates `bench doctor` READY, gates `bench submit`, and is part of the
   result freshness key. A PyPI sdist has no `.git`, so an sdist build reports a
   null commit — which both fails `doctor` and gives the ARM result a different
   fingerprint than the x86 wheel, defeating the "same build, two platforms"
   premise of the `docs/DESIGN.md` section 9.1 timing gate. Clone with
   `--recurse-submodules` at the tag so the commit matches the x86 wheel exactly.

2. **Build with the conda-forge compiler, never the node's system `g++`** (the
   ABI rule at the top of this file). A `uv pip install <path>` / `pip install
   <path>` picks up whatever `g++` is first on `PATH`, which on a RHEL 8 ARM node
   is the system **GCC 8.5** — a `std::filesystem` ABI that predates the GCC 9
   break. Run against the env's conda-forge `libstdc++` (GCC 13), that core
   segfaults the first time `Model.from_net` touches the `.net` path, with no
   traceback. Build inside the **same conda-forge toolchain** as the other
   engines and the ABI matches.

Pinned toolchain (native aarch64 pixi; mirrors the roadrunner one). The
`c-compiler`/`cxx-compiler` metapackages install the activation scripts that
export `$CC`/`$CXX`; `pip` must be present (scikit-build-core drives the build
through it):

```
channels = ["conda-forge"], platforms = ["linux-aarch64"]
python=3.12  pip  numpy  c-compiler  cxx-compiler  gcc=13.*  gxx=13.*
make  cmake>=3.18  ninja  pkg-config  scikit-build-core>=0.10  pybind11>=2.13
```

Build a wheel, then install **that wheel** (not a `{ path = <src> }` entry,
which would rebuild with the wrong compiler at install time):

```bash
git clone --branch v0.17.2 --recurse-submodules \
  https://github.com/lanl/bngsim.git <bngsim-src>

# Inside the conda-forge toolchain env, so $CC/$CXX point at the conda gcc 13.
# The build FetchContents SUNDIALS 7.2.1; pre-download it and name it with an
# absolute path so the build needs no network (its sha256 is checked against
# third_party/sundials/VENDOR.json):
export BNGSIM_SUNDIALS_ARCHIVE=<abs>/sundials-7.2.1.tar.gz
export BNGSIM_ENABLE_KLU=ON BNGSIM_REQUIRE_KLU=ON
export CMAKE_ARGS="-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"
python -m pip wheel <bngsim-src> --no-deps --no-build-isolation -w <wheel-out>
```

Confirm the result before trusting it:

- `readelf -p .comment _bngsim_core*.so` names the **conda-forge gcc**, not the
  system GCC 8.
- `bench doctor` prints `OK bngsim core loads a model (from_net)` (the native
  load self-test) and the baked `build_commit` matches the x86 wheel's.

## run_network on aarch64 (BioNetGen 2.9.3 from source)

The BioNetGen 2.9.3 **release tarball is x86-only and ships no source**, so the
source has to come from the tagged commit in the repository. The build vendors
its own solvers (CVODE 2.6.0 + muparser via `bng2/libsource`) — there is no
external SUNDIALS, no network access at build time, and no LLVM.

```sh
# 1. Source from the tag (NOT the release tarball).
curl -fsSL https://github.com/RuleWorld/bionetgen/archive/refs/tags/BioNetGen-2.9.3.tar.gz \
  | tar xz
cd bionetgen-BioNetGen-2.9.3/bng2/Network3

# 2. Unpack the bundled solver sources in place.
tar xzf ../libsource/cvode-2.6.0.tar.gz -C .
tar xzf ../libsource/muparser_v2_2_4.tar.gz -C .
```

**The one real gotcha.** CVODE 2.6.0 and muparser ship an autotools
`config.guess` too old to recognize aarch64, so `configure` aborts with
`configure: error: cannot guess build type`. The fix is to pass an explicit
`--build` triple (configure skips `config.guess` when `--build` is given). The
`Network3` Makefile **re-extracts cvode on every run** (`rm -rf … && tar -xzf …`),
so patch the two `./configure` invocations in the **Makefile**, not in the
extracted tree:

```
./configure  →  ./configure --build=aarch64-unknown-linux-gnu
```

Then build:

```sh
cp Makefile.cmake Makefile    # the aarch64-ready driver Makefile
make -j16                      # binary lands at bng2/bin/run_network
```

Notes:

- The `auto_ptr` line in the output is a benign C++17 deprecation warning, not a
  build error.
- **Verified numerically** on aarch64: a decay model integrates to
  `100·e⁻¹ = 36.78794` to seven digits, the solver reports
  `cvode using dense LU`, and the binary identifies as `run_network 3.0`.
- Unlike the x86-64 prebuilt binary, this native build **needs no glibc loader
  swap** — it links against the node's own glibc 2.28. (The loader swap in
  `pixi run fetch-glibc` exists only because the *prebuilt* 2.9.3 binary wants
  glibc ≥ 2.34; see the freshness note in `docs/DESIGN.md` section 6.)

## libRoadRunner on aarch64 (built from source)

No `linux-aarch64` wheel and no sdist exist on PyPI for 2.11.0, so it is built
from `sys-bio/roadrunner` at tag `v2.11.0`. This does not block the x86-64 suite,
which uses the published wheel.

The one piece that looks heavy — LLVM — does **not** need to be built from
source. conda-forge publishes a prebuilt `llvmdev` for linux-aarch64 at the
version RoadRunner wants, and RoadRunner finds it via `llvm-config`. With that,
the whole thing is two ordinary CMake builds and finishes in minutes, not hours.

### Pins (verified against roadrunner v2.11.0's own CI, `.github/workflows/main.yml`)

- **roadrunner** `v2.11.0`.
- **libroadrunner-deps** `v2.3.7` — must match the roadrunner tag. RoadRunner
  checks the SUNDIALS version it finds, so a mismatched deps revision fails the
  configure (upstream issue #1403). The deps repo vendors everything it needs as
  git submodules (libSBML 5.21.2, SUNDIALS, POCO, CLAPACK, nleq1/2, rr-libstruct,
  expat, zlib, bzip2, NuML, libSEDML, SBMLNetwork), so a `--recurse-submodules`
  clone is self-contained — no per-dependency hunting.
- **LLVM 13.0.x** — RoadRunner 2.11.0 is an LLVM-13 codebase (MCJIT/OrcJIT). Use
  conda-forge `llvmdev=13.0.1` for linux-aarch64 (prebuilt, PIC- and
  RTTI-enabled, so its static archives link into the Python extension). This is
  the step that would otherwise be an LLVM-from-source compile.
- **SWIG 4.x** for the Python bindings (the docs say "3.0.0" but that is stale;
  CI uses 4.x). **C++17.** **No Fortran compiler** anywhere — nleq1/2 and CLAPACK
  are f2c-translated C, and SuiteSparse/KLU is not part of deps `v2.3.7`.

### Toolchain (one env, for ABI consistency)

Build libroadrunner-deps *and* roadrunner with a single conda-forge aarch64
toolchain — the same family that built `llvmdev` — so libstdc++ ABI is
consistent across LLVM, the deps, and the final `.so`. Use the
**`c-compiler` / `cxx-compiler` metapackages**, not bare `gcc`/`gxx`: only the
metapackages install the conda activation scripts that export `$CC`/`$CXX`.
Without them CMake fails immediately with "No CMAKE_C_COMPILER could be found".
A pinned env (native aarch64 pixi) that works:

```
channels   = ["conda-forge"], platforms = ["linux-aarch64"]
python=3.12  c-compiler  cxx-compiler  gcc=13.*  make  cmake>=3.18  ninja
pkg-config  swig>=4.0.2  llvmdev=13.0.1.*  zlib  ncurses  libxml2  numpy
```

### Build

```bash
# 1. deps (self-contained; vendors libSBML/SUNDIALS/POCO/CLAPACK/nleq/... as submodules)
git clone --branch v2.3.7 --recurse-submodules \
  https://github.com/sys-bio/libroadrunner-deps.git
cmake -S libroadrunner-deps -B libroadrunner-deps/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<deps-install> \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build libroadrunner-deps/build --target install

# 2. roadrunner against the deps tree + the conda-forge LLVM
git clone --branch v2.11.0 https://github.com/sys-bio/roadrunner.git
cmake -S roadrunner -B roadrunner/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<rr-install> \
  -DRR_DEPENDENCIES_INSTALL_PREFIX=<deps-install> \
  -DLLVM_INSTALL_PREFIX=<conda-env-prefix> \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_PYTHON=ON -DPython_ROOT_DIR=<conda-env-prefix> \
  -DSWIG_EXECUTABLE=<conda-env-prefix>/bin/swig \
  -DBUILD_TESTS=OFF -DBUILD_RR_PLUGINS=OFF -DBUILD_EXAMPLES=OFF
cmake --build roadrunner/build --target install
```

The importable package lands at `<rr-install>/site-packages/roadrunner`; add
that `site-packages` to `PYTHONPATH`.

### Two things that trip up downstream use

- **dist-info.** The CMake install drops the `roadrunner` package but no
  `*.dist-info`, so `importlib.metadata.version("libroadrunner")` — how the suite
  reads the engine version — returns nothing. Write a minimal
  `libroadrunner-<ver>.dist-info/METADATA` (`Name: libroadrunner`,
  `Version: 2.11.0`) next to the package.
- **libpython.** The `_roadrunner.so` needs the env's `libpython3.12` at import;
  the suite already preloads it with `RTLD_GLOBAL` (`src/bench/env.py`). Outside
  the suite, ensure the conda env's `lib/` is reachable.

### Verified on aarch64 (ThunderX2, 2026-11-16)

`roadrunner.__version__ == 2.11.0`; `getVersionStr` reports
`JIT Compiler: LLVM-13.0`, `gcc 13.4.0`, `LibSBML Version: 5.21.2`; a first-order
decay integrates to `100·e⁻¹ = 36.78794` under CVODE. End-to-end build was
~6 minutes on 48 cores.

### Wiring the source build into the suite (`$BENCH_ROADRUNNER_PATH`)

The tracked environment pins `libroadrunner` as a wheel, which does not exist
for linux-aarch64, so on ARM the suite is pointed at the source build instead of
the (absent) wheel. Set **`BENCH_ROADRUNNER_PATH`** to the build's
`site-packages` directory — the one holding the `roadrunner` package and the
synthesized `libroadrunner-<ver>.dist-info`:

```sh
export BENCH_ROADRUNNER_PATH=<rr-install>/site-packages
```

The suite prepends it to `sys.path` in-process (`src/bench/env.py`), so it does
**not** go through `$PYTHONPATH` — on ARM that variable must stay empty (x86
user-site numpy leaks onto the ARM interpreter through it; see the tooling
gotchas above). `importlib.metadata.version("libroadrunner")` then resolves from
the synthesized dist-info, and `env.roadrunner_identity()` records a `source`
field so an ARM result is never confused with a wheel one. Two things make this
work without any `LD_LIBRARY_PATH` plumbing:

- The built `_roadrunner.so` carries an **absolute RPATH** into the build
  toolchain's `lib/`, so `libLLVM-13.so`, `libstdc++.so.6`, and
  `libpython3.12.so.1.0` all resolve from there automatically.
- Its `NEEDED libpython3.12.so.1.0` is satisfied by the run env's own
  Python 3.12 (the suite preloads libpython with `RTLD_GLOBAL`), so the run env
  and the build env share a Python minor version.

The run environment itself is an ordinary aarch64 conda/pixi env carrying what
the ODE capability needs — `python=3.12`, `perl`, `numpy`, `pyyaml`, and
`python-libsbml` (the SBML reader; it is on conda-forge for aarch64), plus the
locally built **BNGsim wheel** (see "BNGsim on aarch64" above — install the
prebuilt wheel, **not** a `{ path = <src> }` entry or the PyPI sdist, either of
which rebuilds the core with the wrong compiler and segfaults). Confirm it with
`bench doctor`, which now both imports roadrunner and runs the native load
self-test on BNGsim's core.

## COPASI and AMICI (prepared reference engines)

Both are drop-in `REFERENCE_ENGINES` entries per `docs/DESIGN.md` section 5.2.
The `CopasiODE` and `AmiciODE` adapters, their `pixi.toml` pins,
`env.copasi_identity()`/`env.amici_identity()`, and the `bench doctor` checks are
all in place; the registry entries (the one-line "flip" that re-ids and re-runs
the ODE corpus) are held back until the cross-platform timing leg finishes, so
the two platform legs stay comparable.

- **COPASI** needs **no build on any platform**. `python-copasi` (the SWIG
  bindings carrying the native engine) ships cp312 `manylinux_2_28` wheels for
  both x86-64 and aarch64, and `copasi-basico` (the `basico` wrapper the adapter
  drives) is pure Python. The aarch64 wheel's `manylinux_2_28` floor matches the
  cluster's glibc 2.28, so it installs on an ordinary aarch64 run environment
  with no compiler. Read **concentrations**, not particle numbers — BNG-derived SBML uses
  a unit compartment, so a species' concentration equals its amount numerically
  (what BNGsim reports), while COPASI's particle *number* is that amount times
  Avogadro.
- **AMICI** publishes a PyPI **sdist on every platform** (never a wheel), so
  `pixi install` compiles the C++/SWIG core locally on x86-64 and aarch64 alike
  (SUNDIALS is vendored; no LLVM). It also compiles a fresh C++ module per model
  at runtime via `sbml2amici`, using `$CC`/`$CXX` — so the conda-toolchain ABI
  discipline that the BNGsim core needs (build against the env's own `libstdc++`,
  not the system GCC 8.5) applies to AMICI's compiler too, on both platforms.
  Three things make its `pixi install` reproducible, all encoded in the manifest
  rather than needing a manual step:
    - **The build toolchain is pinned as conda dependencies** (`c-compiler`,
      `cxx-compiler`, `swig`, `cmake`, `ninja`). Pixi activates the conda
      environment before building an sdist, so these — and the `libstdc++` they
      target — are what the compile sees, not the node's system packages.
    - **`CC`/`CXX` are pointed at the conda-forge GCC** via
      `[target.linux-64.activation.env]` (the fully-qualified
      `x86_64-conda-linux-gnu-gcc`/`g++`). The env ships no compiler
      `activate.d` script, so without this the build would fall back to
      `/usr/bin/gcc` (8.5 on the cluster) — the exact `std::filesystem` ABI trap
      above. It is linux-64-scoped because the triple is x86_64-specific: macOS
      uses its ABI-stable clang, and the out-of-tree aarch64 env uses the
      `aarch64-conda-linux-gnu` triple (see `dev/BUILD_NOTES.md`).
    - **`AMICI_TRY_ENABLE_HDF5 = "OFF"`** in `[activation.env]` (global, both
      platforms). AMICI uses HDF5 only to read/write simulation data as files; the
      suite drives it in-process through numpy and never touches that path. Left
      to its "ON if available" default, AMICI's CMake half-finds HDF5 — the C
      headers resolve but the C++ header `H5Cpp.h` does not — and the build fails
      compiling `hdf5.cpp`. OFF skips an unused native dependency and makes the
      build self-contained. It must hold both at install time and when each
      model's `sbml2amici` reconfigures, hence global rather than linux-64-only.
  - **Version: pinned to `1.1.0`, one below the current `1.2.0`.** 1.2.0 has two
    open regressions that would corrupt results silently: a reused `Solver` leaks
    CVODES root-finding state so a warm re-run differs from a fresh one (breaks
    the reset-and-rerun timing contract), and an output timepoint within rounding
    of an event time returns NaN with `rdata.status == 0` (a success status on
    wrong output). `AmiciODE` already hardens against both — a fresh
    `create_solver()` per run, and an explicit finite-output check even on
    `status == 0` — but the pin is the real fix; do not bump until both are
    resolved upstream. If uv has a stale cached build of a previous attempt (it
    keys the sdist build dir and can hold an old `CMakeCache`), clear
    `~/.cache/rattler/cache/uv-cache/{sdists,builds}-*` for amici before a clean
    rebuild.

## Network-free references: standalone NFsim and RuleMonkey

The network-free capability (`docs/DESIGN.md` section 5.5) runs two standalone
engines beside BNGsim's embedded copies. Neither is in the pinned environment.

### NFsim: the binary BioNetGen bundles, under the loader swap

BioNetGen 2.9.3's `bin/NFsim` is NFsim 1.14.3, and like `run_network` it needs glibc
≥ 2.34 and `GLIBCXX_3.4.29`; on an older host (RHEL 8, glibc 2.28) it fails to start
with `GLIBC_2.34 not found`. It is run, unchanged, under the same newer loader
`pixi run fetch-glibc` installs for `run_network` (section "run_network" above). The
twist is that the capability runs NFsim through BNG2.pl, which execs
`$BNGPATH/bin/NFsim` itself, so the loader prefix cannot be put on its command line.
Each job therefore builds, in its own scratch directory, a mirror of the fetched
distribution (`env.bng_launch_tree`): every entry a symlink to the original, except
`bin/NFsim` and `bin/run_network`, which are two-line shell scripts that `exec` the
unchanged binary under the loader. BNG2.pl is run with `BNGPATH` naming the mirror.
The binaries' bytes, and so BioNetGen's sha256 in every fingerprint, are untouched.
Where no loader swap is configured (a host whose glibc is new enough, macOS), the
distribution is used as it is. Nothing to install beyond `fetch-bionetgen` and
`fetch-glibc`; `bench doctor` launches NFsim and checks that its banner reports
v1.14.3.

### RuleMonkey: built from its pinned source

RuleMonkey (MIT, <https://github.com/richardposner/RuleMonkey>) publishes source
releases only. `bench fetch-rulemonkey` builds its command-line driver `rm_driver`
reproducibly:

- **Source:** the v3.13.0 release tarball (the tag names commit `ad3bd755`),
  sha256 `5f5a7c599177aaf92561d3fbed339fc93fc1b3224ef07025d337f6dd2f89d160`, downloaded
  into `.cache/rulemonkey/downloads/` and unpacked into a new, empty build directory.
  bngsim 0.17.2 embeds `db7d1dff` = v3.13.0 plus five commits (a vendored
  expression-layer update, CI wording, documentation), so the driver is the same code
  lineage as BNGsim's `nf_exact` engine: a cost comparator and secondary reference.
- **Toolchain:** this environment's CMake (≥ 3.20 required), Ninja and conda-forge
  C++ compiler (`$CXX`, or the env's `*-conda-linux-gnu-g++`). A compiler outside the
  environment, such as the host's `/usr/bin/g++` (GCC 8.5 on Darwin), is refused —
  the ABI rule at the top of this file. `libstdc++` and `libgcc` are linked statically
  (`-static-libstdc++ -static-libgcc`), so the driver needs only the host's libc at
  run time, whatever environment runs it.
- **Configuration:** Release; the driver only (`RULEMONKEY_BUILD_TESTS=OFF`, no
  examples, no install rules); `RULEMONKEY_WARNINGS_AS_ERRORS=OFF`, since a newer
  compiler than upstream's CI may warn where it did not.
- **Install:** `.cache/rulemonkey/RuleMonkey-3.13.0/` holds `bin/rm_driver`, the build
  log, and `.bench-build.json` (source, recipe, compiler, CMake and Ninja versions,
  the CMake arguments, the binary's sha256). A build whose marker matches is reused.
  The harness finds it from `BNGPATH` (as it finds the glibc tree) or from
  `$BENCH_RULEMONKEY_PATH`, which names any directory holding `bin/rm_driver`.

On a cluster whose compute nodes lack the network, download and build apart:

```bash
pixi run bench fetch-rulemonkey --download-only   # login node: fetch and check the tarball
pixi run bench fetch-rulemonkey                   # compute node: build (about 2 min on 16 cores)
```

Verified on Darwin x86-64 (2026-11-23): conda-forge gcc 15.3.0, CMake 4.4.5,
Ninja 1.13.2, built in 1.5 min on a `general` node; `rm_driver` seed 1 on the CaMKII
`_nfsim` XML reproduces BNGsim's `RuleMonkeySession` seed 1 trajectory exactly. Not yet
built on aarch64: the compiler search matches the out-of-tree env's
`aarch64-conda-linux-gnu-g++`, so the same command should apply, but that is untested.
