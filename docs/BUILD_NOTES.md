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

> **Cluster specifics.** The commands below use placeholders — `<arm-partition>`,
> `<account>`, `<qos>` — for Slurm identifiers. Fill them from your site's
> allocation (on Darwin these live in the git-ignored `slurm.yaml` and `dev/`,
> never in a tracked file). Replace `$USER` as needed.

## Per-engine build matrix (aarch64)

| Engine | aarch64 availability | Build effort | Needs LLVM? |
| --- | --- | --- | --- |
| **BNGsim** (system under test) | PyPI wheel exists; `pixi install` works | none | no |
| **run_network** (BioNetGen) | no aarch64 prebuilt; **build from 2.9.3 source** | moderate CMake | no |
| **AMICI** | PyPI **sdist** → `uv pip install amici` compiles it | moderate (SUNDIALS vendored) | no |
| **COPASI CLI** | plain CMake C++ build if no aarch64 prebuilt | moderate CMake | no |
| **libRoadRunner** | **no aarch64 wheel and no sdist on PyPI** | heavy | **yes** — the long pole |

Only libRoadRunner needs LLVM from source: its `libroadrunner-deps` prebuilt
bundle covers x86-64 Linux, macOS-arm64 and Windows but not linux-aarch64, and
these nodes have no LLVM module. run_network, AMICI and COPASI CLI are ordinary
native builds with no LLVM dependency.

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

## libRoadRunner on aarch64 (the long pole)

No `linux-aarch64` wheel and no sdist exist on PyPI for 2.11.0, so it is built
from `sys-bio/roadrunner` at tag `v2.11.0`. The single heavy step is **LLVM from
source**, because the `libroadrunner-deps` prebuilt dependency bundle skips
linux-aarch64 and the ARM nodes carry no LLVM module. Everything else is an
ordinary CMake build. This is the remaining task before the section 9.1 ARM leg
can include libRoadRunner; it does not block the x86-64 suite, which uses the
published wheel.

## AMICI / COPASI CLI (future reference engines)

Neither is wired into the suite yet (they are drop-in `REFERENCE_ENGINES`
entries per `docs/DESIGN.md` section 5.2). For aarch64 when the time comes:

- **AMICI** publishes a PyPI **sdist**, so `uv pip install amici` compiles it
  locally (SUNDIALS is vendored; no LLVM). A native toolchain is enough.
- **COPASI CLI** is a plain CMake C++ build from source if no aarch64 prebuilt
  is available (no LLVM).
