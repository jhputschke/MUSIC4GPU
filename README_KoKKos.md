# MUSIC4GPU — Kokkos Backend (portable: NVIDIA / AMD / Intel GPU + multicore CPU)

> **Stages 1–5 complete.** This is the performance-portable
> [Kokkos](https://github.com/kokkos/kokkos) back-end for MUSIC. All nine hydro
> kernels are ported from one source and validated on the **Serial / OpenMP /
> Cuda** execution spaces: on the GB10 GPU the `eps_max(τ)` trace agrees with
> the CPU reference **identically to the native CUDA build** (max rel err
> 6.44e-04) at **~0.8× native-CUDA throughput** (native CUDA stays the faster
> NVIDIA backend — see the Performance section), and the three backends are
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
- For a GPU build: the vendor toolchain — CUDA (NVIDIA), ROCm/HIP (AMD), or
  oneAPI/SYCL (Intel). The **Serial / OpenMP** host back-ends need none of these.

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
| Serial (host) | `-DKokkos_ENABLE_SERIAL=ON` | ✅ validated (6.44e-04 vs CPU) |
| OpenMP (multicore host) | `-DKokkos_ENABLE_OPENMP=ON` | ✅ validated (6.44e-04 vs CPU) |
| CUDA (NVIDIA) | `-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON -DKokkos_ARCH_<GPU>=ON` (GB10 = `Kokkos_ARCH_BLACKWELL121`; also `HOPPER90`, …) | ✅ validated on GB10 (6.44e-04; ~0.8× native CUDA — Performance §) |
| HIP (AMD) | `-DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON` (MI250X) | wired; not run (no AMD HW here) |
| SYCL (Intel) | `-DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON` (Aurora) | wired; not run (no Intel HW here) |

The CUDA build uses the **CMake CUDA-language** path
(`-DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON`) so only the Kokkos device TUs go
through `nvcc` and the rest of MUSIC stays plain host `g++`. The full GB10 line is
in **[Port_GPU_KoKKos.md](Port_GPU_KoKKos.md)**.

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
- `tests/kokkos_consistency.sh` — **D9 single-source gate**: runs the same input
  on every built Kokkos backend (Serial/OpenMP/Cuda) + CPU and checks each vs CPU
  (1e-3) and the Kokkos backends vs each other (1e-4).
- `tests/cuda_vs_cpu_bench.sh`, `tests/cuda_perstep_bench.sh` — throughput / per-step.

**Current result (Stages 1–5):** all three Kokkos backends PASS `eos_gpu_vs_cpu.sh`
**[1/3][2/3][3/3]** at **max rel error 6.44e-04** vs CPU — *identical to the native
CUDA build*. `kokkos_consistency.sh` is green: Serial ≡ OpenMP (0.00e+00), Cuda
within 1.2e-5. Shear, bulk, and full-3D configs all pass. On GB10 throughput is
**~0.8× native CUDA** (Performance section below). See
**[Port_GPU_KoKKos.md](Port_GPU_KoKKos.md)** for the per-stage precision /
throughput tables.

---

## Performance — Kokkos/Cuda vs native CUDA

On the GB10 the Kokkos CUDA backend reaches **precision parity but *not*
throughput parity** with the hand-written native CUDA backend. Per-step compute
(128×128×1, `(T_long − T_short)/Δsteps`, best of 5):

| backend | per-step | ratio |
|---|---|---|
| native CUDA | 3.07 ms | 1.0× |
| Kokkos / Cuda | 3.54 ms | **0.87× native** |

So **native CUDA is the faster backend — by ~1.15×** (run-to-run the ratio sits
in the **0.79–0.87× native** band). **Precision is identical** (both 6.44e-04 vs
CPU); the gap is throughput only, and it is **closable, not structural** — three
deferred items, all things the native backend does and the port does not yet:

1. **Per-kernel occupancy tuning** — native CUDA picks a block size per kernel
   (`cudaOccupancyMaxPotentialBlockSize`); the port uses one MDRange tile for all
   nine. The register-heavy `delta_qi` (~60% of runtime) likely wants its own.
2. **`__ldg` / RandomAccess EOS** — native routes the data-dependent EOS lookups
   through the read-only cache; the Stage-1 port uses plain global loads.
3. **Shared-memory tiled `w_source`** — native has a hand-tiled stencil kernel;
   the port runs the plain version (the `TeamPolicy`+scratch port is the Stage-3b
   item in [PlanKokkosPort.md](PlanKokkosPort.md)).

The work stopped at ~0.8–0.87× because the goal was *precision* parity; the
Stage-2/3 throughput targets (≈1.0× CUDA, then `>`CUDA via fusion/scratch) remain
open. (The `delta_qi+finalize` fusion behind `MUSIC_KOKKOS_FUSE` was tried and is
**not** a win on Blackwell — `delta_qi` is register-bound — so it stays off; see
Port_GPU_KoKKos.md Stage 3.)

**The trade-off.** You give up ~15–25% of peak NVIDIA throughput and get, from
**one** kernel source: NVIDIA **plus** the same physics on AMD / Intel GPUs and
multicore CPU, the removal of the CUDA-vs-CPU physics duplication (and its drift),
and insulation from vendor-API churn. Whether that is worth it depends on how much
the portability is worth against the last ~15% on NVIDIA specifically.

Two caveats on the number: per-step **isolates kernel cost** — end-to-end
wall-time narrows the gap, since both backends share the identical host packing,
EOS load, and I/O; and this is a **GB10 (coherent-memory)** result — on a
*discrete* GPU the port still lacks the native discrete memory path, so re-measure
there (see **[PlanKoKKosDiscrete.md](PlanKoKKosDiscrete.md)**).

---

## Source layout

| File | Role |
|------|------|
| `get_kokkos.sh` | Clone Kokkos into `external/kokkos` (latest by default, pin via arg) |
| `CMakeLists.txt` | `USE_KOKKOS` option, mutual exclusion, C++20 + PIC, Kokkos discovery (`add_subdirectory` / `find_package`) |
| `src/CMakeLists.txt` | Kokkos source branch; links `Kokkos::kokkos` into `libmusic` |
| `src/advance.h` | Third `GPUPipelines` alias branch + `MUSIC_USE_GPU` guard for `USE_KOKKOS` |
| `src/gpu/music_kernels_kokkos.hpp` | The 9 ported hydro kernels + device helpers (`KOKKOS_INLINE_FUNCTION`); single source for all backends (counterpart of `music_kernels.cu`) |
| `src/gpu/KokkosPipelines.{h,cpp}` | Singleton mirroring `CUDAPipelines.h` (7 `dispatch_*`, `reduce_max`, `wait`, …); PIMPL, Kokkos-free header; `parallel_for` over `MDRangePolicy<Rank<3>>` |
| `src/gpu/GPUGrid_kokkos.cpp` | Kokkos backing for the existing `GPUGrid` class (`Kokkos::View<float*, SharedSpace>`; counterpart of `GPUGrid_cuda.cu`) |
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

**Stages 0–5 are done** (roadmap in **[PlanKokkosPort.md](PlanKokkosPort.md)**,
per-stage results in **[Port_GPU_KoKKos.md](Port_GPU_KoKKos.md)**):

- **0** infrastructure · **1** all 9 kernels ported (Serial/OpenMP/Cuda parity) ·
  **2** perf parity on GB10 (fast-math + occupancy tile → 0.79× native CUDA) ·
  **3** `delta_qi+finalize` fusion behind `MUSIC_KOKKOS_FUSE` (off by default —
  not a win on Blackwell) · **4** D9 single-source consistency gate green ·
  **5** coverage matrix + shear/bulk/3D breadth validated.

The Kokkos build reproduces the CPU reference to the **same precision as native
CUDA** (6.44e-04), from one kernel source running on three execution spaces.

### Next steps

- **Discrete GPUs (A100 / H100 / RTX / MI250X / PVC).** The port was tuned on the
  GB10's *coherent* unified memory and uses `Kokkos::SharedSpace` (managed) for
  all buffers — correct on a discrete GPU but not optimal (demand-paged migration
  instead of pinned bulk DMA + copy/compute overlap). The native CUDA backend's
  discrete-specific memory path is **not yet carried over**; the pick-up plan —
  what already carries over, the native-CUDA→Kokkos mapping, a DG0–DG4 roadmap,
  file-level hooks, and the verification recipe — is in
  **[PlanKoKKosDiscrete.md](PlanKoKKosDiscrete.md)**. (Needs real discrete
  hardware to benchmark; precision is unaffected.)
- **AMD (HIP) / Intel (SYCL).** Backend-agnostic by construction — a configure
  flip plus the same `eos_gpu_vs_cpu.sh` / `kokkos_consistency.sh` gates; not
  runnable on this NVIDIA-only box.
- **Single-source cleanup (deferred Stage-4 tail).** View-back `Fields` and retire
  the legacy CPU per-cell loops once Stage-5 coverage (baryon diffusion, finite-µB
  EOS, multi-charge) lands per-feature (D8).
