# MUSIC4GPU — Kokkos Backend (portable: NVIDIA / AMD / Intel GPU + multicore CPU)

> **Stages 1–5 complete.** This is the performance-portable
> [Kokkos](https://github.com/kokkos/kokkos) back-end for MUSIC. All nine hydro
> kernels are ported from one source and validated on the **Serial / OpenMP /
> Cuda** execution spaces: on the GB10 GPU the `eps_max(τ)` trace agrees with
> the CPU reference **identically to the native CUDA build** (max rel err
> 6.44e-04) at **0.79× native-CUDA throughput**, and the three backends are
> cross-backend-consistent to ≤1.2e-5 (the D9 single-source gate,
> `tests/kokkos_consistency.sh`). Shear, bulk, and full-3D configs all pass.
> Optional `delta_qi+finalize` kernel fusion lives behind `MUSIC_KOKKOS_FUSE`
> (off by default — not a win on Blackwell). See
> **[PlanKokkosPort.md](PlanKokkosPort.md)** for the roadmap and
> **[Port_GPU_KoKKos.md](Port_GPU_KoKKos.md)** for the per-stage implementation +
> precision/throughput runs.

The Kokkos back-end targets **one kernel body that runs on NVIDIA, AMD, and
Intel GPUs plus multicore CPUs**, selected at build time by Kokkos execution
space. It slots in behind the *same* `GPUPipelines` seam used by the CUDA and
Metal back-ends (a `*Pipelines` singleton with seven `dispatch_*` methods +
`GPUGrid`), so the host evolution loop in `advance.cpp` is unchanged. The Metal
back-end stays separate (Kokkos has no Apple-Metal target); Kokkos subsumes CUDA
and adds AMD / Intel / CPU.

`USE_KOKKOS` is mutually exclusive with `USE_CUDA` and `USE_METAL` — pick one
back-end at configure time.

---

## Requirements

- **CMake ≥ 3.22** (required by Kokkos 5.x; 3.16 is fine for the MUSIC side).
- A **C++20** compiler (Kokkos 5.x requires C++20) — e.g. GCC ≥ 10 or Clang ≥ 13.
  The `USE_KOKKOS` configuration raises MUSIC to C++20 automatically.
- **git** (used by `get_kokkos.sh` to clone Kokkos and resolve the latest tag).
- For a GPU build (Stage 2): the vendor toolchain — CUDA (NVIDIA), ROCm/HIP
  (AMD), or oneAPI/SYCL (Intel). Stage 0 builds on the **Serial / OpenMP** host
  back-end and needs none of these.

---

## Installing Kokkos

Fetch Kokkos with the helper script (run from the music4gpu repo root). It
clones into `external/kokkos/` (git-ignored), where the build looks for it:

```bash
bash get_kokkos.sh            # latest Kokkos release (default)
bash get_kokkos.sh 5.1.1      # or pin a specific release tag
```

- **Default = latest release**, resolved with `git ls-remote --sort=-v:refname`
  (no GitHub API / `jq` needed). **Pin a version** for reproducible CI / HPC
  builds.
- The script is **idempotent** — re-running with `external/kokkos` present is a
  no-op (`rm -rf external/kokkos` to re-clone).

**HPC / system Kokkos.** On clusters that already ship a tuned Kokkos (Spack or
a module — Frontier, Aurora, Perlmutter, …), **skip the script** and point CMake
at the install:

```bash
cmake -S . -B build_kokkos -DUSE_KOKKOS=ON -DKokkos_ROOT=/path/to/kokkos/install ...
```

The build's discovery order is: `-DKOKKOS_SOURCE_DIR=<dir>` override →
in-repo `external/kokkos` → sibling `../kokkos` → `find_package(Kokkos)`.

---

## Building

### Stand-alone MUSIChydro

```bash
bash get_kokkos.sh
cmake -S . -B build_kokkos -DUSE_KOKKOS=ON -DKokkos_ENABLE_SERIAL=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build_kokkos -j$(nproc)
# -> build_kokkos/src/MUSIChydro  (+ libmusic.so)
```

Add `-DKokkos_ENABLE_OPENMP=ON` for the multicore-CPU host back-end.

CPU reference build (for the verification scripts below):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Inside X-SCAPE (the primary, intended workflow)

X-SCAPE builds the GPU MUSIC by `add_subdirectory`-ing this package, so the same
music4gpu CMake drives Kokkos there too — the parent only adds the `USE_KOKKOS`
option. From the **X-SCAPE root**:

```bash
./external_packages/get_music4gpu.sh                 # clones music4gpu (if not present)
bash external_packages/music4gpu/get_kokkos.sh       # fetch Kokkos into the package
cmake -S . -B build -DUSE_MUSIC=ON -DUSE_KOKKOS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## Selecting the execution-space back-end

Kokkos picks its execution spaces at configure time via standard `Kokkos_*`
options passed to the `cmake` line:

| Backend | Flags | Status |
|---|---|---|
| Serial (host) | `-DKokkos_ENABLE_SERIAL=ON` | **Stage 0 — works** |
| OpenMP (multicore host) | `-DKokkos_ENABLE_OPENMP=ON` | **Stage 0 — works** |
| CUDA (NVIDIA) | `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<GPU>=ON` (e.g. `Kokkos_ARCH_HOPPER90`) | Stage 1–2 |
| HIP (AMD) | `-DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON` (MI250X) | Stage 2 |
| SYCL (Intel) | `-DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON` (Aurora) | Stage 2 |

Until Stage 1 lands the kernels, every back-end build runs the CPU reference
path regardless of the execution space selected.

---

## Runtime lifecycle (Kokkos initialize / finalize)

Kokkos is brought up by a host-safe RAII guard, `KokkosRuntimeGuard`
(`src/gpu/kokkos_runtime.h`), declared once at the top of the driver `main()`,
before anything that allocates Kokkos Views:

- **Stand-alone:** in `src/main.cpp` (added under `#ifdef USE_KOKKOS`).
- **X-SCAPE (primary):** in the framework driver `main()`, before the `JetScape`
  object — **not** a `JetScape` member (the module list is destroyed last, which
  would finalize Kokkos while module Views are still alive). See
  [PlanKokkosPort.md](PlanKokkosPort.md) **D2**.

The guard is collision-safe: if Kokkos is already initialized (e.g. another
module brought it up), it does nothing and does not finalize. The header is
Kokkos-free so MUSIC's other TUs stay plain host TUs (PIMPL boundary, D7); only
`kokkos_runtime.cpp` includes `<Kokkos_Core.hpp>`.

You can pass Kokkos runtime flags on the command line, e.g.
`./MUSIChydro input --kokkos-num-threads=8`.

---

## Verification

Use the standard `tests/` harness (the same one that validated the CUDA and
Metal ports). Run from the repo root; the EOS test needs the hotQCD table
(`cd EOS && bash download_hotQCD.sh SMASH_binary`). Point it at the Kokkos build
with `GPU_BIN=`:

```bash
# build the CPU reference and the Kokkos build first, then:
GPU_BIN=$PWD/build_kokkos/src/MUSIChydro bash tests/eos_gpu_vs_cpu.sh
```

- `tests/eos_gpu_vs_cpu.sh` — **correctness** oracle: compares `eps_max(τ)` of
  the CPU and GPU binaries within `TOL` (1e-3), default EOS 91 (hotQCD).
- `tests/cuda_vs_cpu_bench.sh`, `tests/cuda_perstep_bench.sh` — **throughput**
  and **per-step** benchmarks (meaningful once Stage 1 dispatches to the GPU;
  to be renamed to backend-explicit variants, e.g. `kokkos_vs_cpu_bench.sh`).

**Stage-0 result on this branch:** `eos_gpu_vs_cpu.sh` reports **[3/3] PASS,
max rel error `0.00e+00`** (the `USE_KOKKOS` build reproduces the CPU reference
bit-for-bit) and **[2/3] PASS**. Check **[1/3] ("GPU dispatch") FAILs by
design** — the skeleton's `KokkosPipelines::initialize()` returns false, so
MUSIC runs the CPU path; [1/3] flips to PASS at Stage 1.

---

## Source layout

| File | Role |
|------|------|
| `get_kokkos.sh` | Clone Kokkos into `external/kokkos` (latest by default, pin via arg) |
| `CMakeLists.txt` | `USE_KOKKOS` option, mutual exclusion, C++20 + PIC, Kokkos discovery (`add_subdirectory` / `find_package`) |
| `src/CMakeLists.txt` | Kokkos source branch; links `Kokkos::kokkos` into `libmusic` |
| `src/advance.h` | Third `GPUPipelines` alias branch + `MUSIC_USE_GPU` guard for `USE_KOKKOS` |
| `src/gpu/KokkosPipelines.{h,cpp}` | Singleton mirroring `CUDAPipelines.h` (7 `dispatch_*`, `reduce_max`, `wait`, …); PIMPL, Kokkos-free header |
| `src/gpu/GPUGrid_kokkos.cpp` | Kokkos backing for the existing `GPUGrid` class (counterpart of `GPUGrid_cuda.cu`) |
| `src/gpu/kokkos_runtime.{h,cpp}` | Host-safe `KokkosRuntimeGuard` (the only TU that includes Kokkos headers) |
| `src/main.cpp` | Stand-alone `KokkosRuntimeGuard` bracketing the run |

(Under X-SCAPE, `CMakeLists.txt` and `external_packages/get_music4gpu.sh` gain a
matching `USE_KOKKOS` option and a Kokkos-fetch hint.)

---

## Notes & gotchas

- **C++20 is required** by Kokkos 5.x (4.x needed C++17). The `USE_KOKKOS`
  configuration sets C++20 for the whole `libmusic`; the CUDA/Metal/CPU builds
  are untouched (still C++11/17).
- **Position-independent code:** `libmusic` is a shared library, so Kokkos'
  static archive is built with `-fPIC` (`CMAKE_POSITION_INDEPENDENT_CODE ON`).
  Without it the link fails on AArch64 (`relocation R_AARCH64_* … recompile with
  -fPIC`).
- **Reproducible builds:** pin the Kokkos tag (`get_kokkos.sh <tag>`); the
  default pulls whatever the latest release is at clone time.
- **No Apple-Metal target:** on macOS use `-DUSE_METAL=ON` instead; Kokkos does
  not cover Apple GPUs.

---

## Status & roadmap

Stage 0 (this branch) delivers the build integration, the runtime lifecycle, the
back-end seam, and skeleton TUs — a green build that falls through to the CPU
reference. Stages 1–5 (kernel port → performance parity → AMD/Intel bring-up →
kernel fusion → single-source unification) are detailed in
**[PlanKokkosPort.md](PlanKokkosPort.md)**.
