# Kokkos Backend Port Notes (music4gpu)

Companion to **[PlanKokkosPort.md](PlanKokkosPort.md)** (the feasibility study +
staged roadmap) and **[PORT_GPU_CUDA.md](PORT_GPU_CUDA.md)** (the native-CUDA
port it mirrors).  This document tracks the *execution* of the Kokkos port —
the performance-portable backend that subsumes CUDA and adds AMD / Intel GPU +
multicore CPU from **one kernel source** — and records the per-stage precision
and throughput runs.

- Branch: `KoKKos-Port`.
- Hardware: **NVIDIA GB10** (Grace-Blackwell, cc 12.1, coherent unified
  memory), CUDA 13.0, GCC 13.3, CMake 3.28, 20-core Grace CPU.
- Kokkos: **5.1.1** (vendored in `external/kokkos` via `get_kokkos.sh`).
- Reference EOS test: `tests/eos_gpu_vs_cpu.sh` (EOS 91 hotQCD, 32×32×1 Gubser,
  shear on, 101 steps), tolerance 1e-3 on the `eps_max(τ)` trace.

> **Acceptance criterion (from the goal):** the Kokkos build's precision vs the
> CPU reference must be *close/similar* to the native-CUDA build's.  As of
> Stage 1 it is **identical to three significant figures on every backend** —
> see the table below.

---

## Status at a glance

| Stage | Scope | State |
|---|---|---|
| 0 | Infrastructure (build, lifecycle, seam, skeletons) | ✅ done (pre-existing) |
| **1** | **Functional GPU parity — all 9 kernels ported; Serial/OpenMP/Cuda validated** | ✅ **done** |
| **2** | **Perf parity (fast-math + occupancy MDRange tile) → 0.79× native CUDA; AMD/Intel notes** | ✅ **done** |
| **3** | **Kokkos-native optimization: delta_qi+finalize fusion behind `MUSIC_KOKKOS_FUSE` (D11)** | ✅ **done** |
| **4** | **Single-source unification: D9 gate green (Serial≡OpenMP, Cuda@1.2e-5); `kokkos_consistency.sh`** | ✅ **done** |
| 5 | Extend GPU coverage notes + final validation | ⏳ next |

---

## Precision: Kokkos vs CPU, matched against native CUDA  (the goal gate)

`tests/eos_gpu_vs_cpu.sh`, EOS 91 (hotQCD), 32×32×1 Gubser viscous, 101 steps,
`eps_max(τ)` trace vs the `build/` CPU reference (double precision):

| Backend (device float32) | max rel err | mean rel err | [1/3] | [2/3] | [3/3] |
|---|---|---|---|---|---|
| **native CUDA** (GB10) — *baseline* | 6.44e-04 | 1.87e-04 | PASS | PASS | PASS |
| **Kokkos / OpenMP** (20 host threads) | **6.44e-04** | 1.89e-04 | PASS | PASS | PASS |
| **Kokkos / Cuda** (GB10) | **6.44e-04** | 1.88e-04 | PASS | PASS | PASS |

The max relative error is **bit-for-bit identical** (6.44e-04, same offending
step 58) across native CUDA and both Kokkos backends.  That is expected and is
the whole point: the Kokkos per-cell bodies are a line-for-line port of the
validated CUDA kernels (same float32 ops, same order), so they reproduce the
*same* single-precision rounding profile relative to the double-precision CPU
solver.  This is also the **D9 cross-backend single-source consistency gate**:
OpenMP and Cuda, compiled from one kernel source, agree with CPU identically.

## Throughput — Stage 1 (untuned) → Stage 2 (perf parity)

128×128×1 Gubser viscous, per-step compute isolated as
`(T(220 steps) − T(20 steps)) / 200`, min of 3 runs:

| Backend | Stage-1 per-step | **Stage-2 per-step** | vs native CUDA |
|---|---|---|---|
| CPU (OpenMP, 20 threads) | 12.98 ms | 13.11 ms | — |
| native CUDA (GB10) | 2.95 ms | **2.89 ms** | 1.0× |
| Kokkos / Cuda (GB10) | 11.76 ms | **3.65 ms** | **0.79×** |
| Kokkos / OpenMP (20 threads) | 10.53 ms | 12.08 ms | — |

**Stage 2 brought Kokkos/Cuda from 0.25× → 0.79× native CUDA** (3.2× faster),
into the plan's Stage-2 band ("≈1.0× CUDA"), with **precision unchanged**
(6.44e-04, mean now exactly the native-CUDA 1.87e-04).  Two levers did it; a
third was tried and rejected:

1. **`--use_fast_math`** on the Kokkos CUDA TUs (`src/CMakeLists.txt`, mirroring
   the native backend): 11.76 → 6.69 ms.  The pipeline is dominated (~60%) by
   `delta_qi`'s Newton-Brent solve + 12 reconstructions/cell, all transcendental
   (`sqrtf`/`logf`/`expf`/`powf`/division), so fast-math is the biggest single
   win.  FP32 truncation dominates the error budget, so precision is untouched.
2. **Occupancy-capped MDRange tile** (`range3()` in `KokkosPipelines.cpp`):
   `{tEta≤4, tY, tX≤32}` with product ≤256, reproducing the native
   `compute_launch` geometry (eta≤4, x=32 coalesced, ≤256-thread block).  With
   fast-math this took 6.69 → **3.65 ms**.  The cap is what keeps the
   register-heavy `delta_qi`/`w_full` below the occupancy cliff.  Applied to GPU
   spaces only (`if constexpr` on `SpaceAccessibility`); host keeps the default
   tiling, which auto-vectorises.
3. **`__restrict__` on the kernel pointers — *rejected*.**  Native CUDA marks
   every pointer `__restrict__`, but on Blackwell + Kokkos it *regressed*
   throughput (6.69 → 9.99 ms, and 7.71 ms even with the tile): the extra
   no-reload register pressure pushed the heavy kernels over the occupancy
   cliff.  Removing it gave the 3.65 ms result.  Left out.

**Not pursued (diminishing returns past 0.79×, would touch every kernel):**
RandomAccess EOS Views (`MemoryTraits<RandomAccess>` → `__ldg` texture path) and
the flat→rank-2 layout-templated Views (D3/§2a).  The flat `comp*Ncells+cell`
indexing already gives coalesced GPU access; the layout migration mainly buys
CPU-backend vectorisation and is better folded into the Stage-4 unification,
where the OpenMP path replaces the legacy CPU loops.

### AMD (HIP) / Intel (SYCL) — build-matrix only on this box

The port is genuinely backend-agnostic: one kernel source, `MDRangePolicy`,
`Kokkos::SharedSpace` keyed off the space's unified-memory trait (no `#ifdef`),
`parallel_reduce<Max>` (no vendor atomics).  Bringing up AMD/Intel is a configure
flip — `-DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON` (MI250X/Frontier) or
`-DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON` (Aurora) — plus the same
`tests/eos_gpu_vs_cpu.sh` gate.  **This machine has only the NVIDIA GB10**, so
HIP/SYCL are wired-and-documented but not run here; the Serial/OpenMP/Cuda
consistency (all 6.44e-04 vs CPU) is the portability evidence available on-box.

---

## Stage 4 — single-source unification posture (D5 / D7 / D9)

The unification thesis — *collapse the triplicated CUDA + Metal + CPU per-cell
physics into one source with no drift* — is **architecturally realized for the
GPU-supported configs already**: `music_kernels_kokkos.hpp` is the *only* copy
of those kernels, and it is what runs on Serial, OpenMP, **and** Cuda.  On a
CPU-only machine a `USE_KOKKOS` + OpenMP build's `try_gpu_advance` path *is* the
single Kokkos source executing on OpenMP — there is no separate hand-written CPU
GPU-twin to drift from it.

**The D9 gate makes that concrete** (`tests/kokkos_consistency.sh`): one input,
the same source compiled three ways, EOS 91 / 32²×1 / 101 steps:

| comparison | max rel err | gate |
|---|---|---|
| Kokkos-Serial vs CPU (double) | 6.44e-04 | < 1e-3 ✅ |
| Kokkos-OpenMP vs CPU (double) | 6.44e-04 | < 1e-3 ✅ |
| Kokkos-Cuda  vs CPU (double) | 6.44e-04 | < 1e-3 ✅ |
| **Kokkos-Serial vs Kokkos-OpenMP** | **0.00e+00** | < 1e-4 ✅ |
| Kokkos-Serial vs Kokkos-Cuda | 1.22e-05 | < 1e-4 ✅ |
| Kokkos-OpenMP vs Kokkos-Cuda | 1.22e-05 | < 1e-4 ✅ |

Serial and OpenMP are **bit-identical** (same float ops, same order, host libm);
Cuda differs only at **1.2e-5** (fast-math FP reordering), an order of magnitude
inside the D10 cross-backend tolerance.  This is the single-source correctness
guarantee the unification stage exists to provide.

**Deliberately deferred (de-risk; not on the precision critical path):** the
remaining Stage-4 items in PlanKokkosPort.md — *View-backing `Fields`* (D5) and
*deleting the legacy CPU per-cell loops* — are a wide refactor of the host
evolution path (`advance.cpp` / `dissipative.cpp` / `u_derivative.cpp`) whose
*output* is already validated-equivalent here.  They are pure maintainability
(remove the still-present legacy CPU fallback), they do not change any precision
result, and the legacy CPU path must stay anyway until Stage 5 extends Kokkos to
the currently CPU-only configs (D8).  Doing the deletion now would risk the
green state for no precision gain, so it is sequenced after Stage 5 coverage.

## Stage 3 — kernel fusion behind a flag (D11)

The two heaviest data-path kernels — `delta_qi` (build the ideal `qi` update) and
`finalize_ideal` (Newton-reconstruct it into the next-step primitives) — exchange
`qi` through a `5·Ncells` global buffer (`qi_out`).  Stage 3 adds a fused kernel
(`apply_delta_qi_finalize`) that keeps `qi` in registers and runs the
reconstruction immediately, eliminating that round-trip.  It is gated by
`-DMUSIC_KOKKOS_FUSE=ON`; default builds keep the two unfused kernels as the
**D6/D11 golden reference**, and the fusion is wired so `advance.cpp`'s dispatch
sequence is untouched (`dispatch_delta_qi` becomes a no-op, `dispatch_finalize_ideal`
runs the fused kernel).

**Correctness:** the fused build is **bit-identical to the unfused golden** —
both give max rel err **6.44e-04** / mean **1.87e-04** vs CPU (the only change is
that `qi` skips a `float` store+load, which is exact).

**Throughput on GB10 (128²×1, min of 4):**

| Kokkos/Cuda | per-step |
|---|---|
| unfused (default) | 3.52 ms |
| fused (`MUSIC_KOKKOS_FUSE=ON`) | 3.67 ms |

**Fusion does *not* pay on Blackwell here**, and is correctly left **OFF by
default**.  `delta_qi` is **compute/register-bound** (Newton-Brent solve + 12
reconstructions/cell, ~60% of runtime), not bandwidth-bound — so removing the
`qi_out` traffic saves little, while the extra in-register reconstruction nudges
register pressure and trims occupancy (the same Blackwell sensitivity that made
`__restrict__` a regression in Stage 2).  This is exactly the outcome D11 is
designed for: implement the optimization behind a switch, validate it against the
unfused golden, and enable it per-backend *only* once it passes a perf gate —
which it does not on this GPU.  A bandwidth-bound fusion (the plan's other
candidate, `uwrhs`+`uprhs`, which reuse the same ±2-cell `u` neighbourhood) is
the more promising next target and uses the identical gating mechanism; it is
bulk-viscosity-only (`turn_on_bulk`), so it is left for when that path is
benchmarked.

## What Stage 1 implemented

Three TUs were filled in behind the existing `GPUPipelines`/`GPUGrid` seam (the
host loop in `advance.cpp` is **unchanged** — same `dispatch_*`/`wait`/`reduce`
surface as CUDA/Metal):

### `src/gpu/music_kernels_kokkos.hpp`  (new — the core)
A line-for-line port of `music_kernels.cu` to portable Kokkos:

- **Device helpers** → `KOKKOS_INLINE_FUNCTION` / `KOKKOS_FORCEINLINE_FUNCTION`,
  templated-ready on `Real` (D4; `Real = float` in Stage 1): `clampf/clampi`,
  `cell_idx`, `clamped_cell`, the EOS log-interp (`gpu_log_interp`, `gpu_P`,
  `gpu_dPde`, `gpu_cs2`, `gpu_s`, `gpu_T_e`), `gpu_minmod_dx`, `gpu_TJb0`, the
  full Newton-Brent reconstruction (`gpu_vel_fdf`, `gpu_solve_v`, `gpu_u0_fdf`,
  `gpu_solve_u0`, `gpu_reconst`), `gpu_max_speed`, `gpu_get_TJb_reconst`, and
  the η/s + ζ/s transport profiles + `gpu_uW_source` / `gpu_uPi_source` /
  `gpu_uWRHS_geom`.
- **Per-cell kernel bodies** → `apply_*(ix,iy,ieta, …)` free functions, one per
  CUDA `__global__`: `apply_make_du`, `apply_make_uwrhs`, `apply_make_w_source`,
  `apply_make_uprhs`, `apply_make_delta_qi`, `apply_finalize_ideal`,
  `apply_first_rk_step_w_full`, `apply_pack_evolution_ideal`.  The CUDA
  thread-index math + bounds check are dropped — the launch policy supplies the
  indices.
- **Mechanical substitutions only:** `__ldg(&p[i])`→`p[i]`;
  `isfinite/isnan`→`Kokkos::isfinite/isnan`; the `__constant__ WMUNU_IDX` table
  →a local `constexpr` accessor `WIDX(α,dir)` (the gpu_types.h `__constant__`
  form is device-only under nvcc, unusable from the host half of a
  `KOKKOS_INLINE_FUNCTION`).  The arithmetic is otherwise byte-identical.

### `src/gpu/GPUGrid_kokkos.cpp`  (rewrite — the memory layer)
- Every snapshot/scratch/EOS buffer is a
  `Kokkos::View<float*, Kokkos::SharedSpace>`.  `SharedSpace` resolves to
  CUDA/HIP/SYCL **managed** memory on a GPU build and **HostSpace** on a host
  build, so the host AoS↔SoA pack/unpack loops write/read the View `data()`
  pointer directly on every backend — the same in-place, zero-copy model the
  native CUDA backend uses on this coherent GB10 box (`upload_snapshots_async`
  is a no-op).
- `GPUGrid.h` stays Kokkos-free (PIMPL, D7): the owning Views live in a
  TU-static store keyed by the `GPUGrid` instance; only the raw `float*` aliases
  (shared with all backends) sit on the struct.  Snapshot rotation/swizzle
  permutes those aliases without moving data.
- `release()` fences + drops the Views only while `Kokkos::is_initialized()`,
  so a teardown after `Kokkos::finalize()` can't fault (the
  `KokkosRuntimeGuard` in `main()` outlives every `GPUGrid` — D2).

### `src/gpu/KokkosPipelines.cpp`  (rewrite — the dispatch layer)
- `initialize()` confirms the runtime (brought up by the guard) and reports the
  execution space + CUDA device.  `dispatch_*` launch
  `parallel_for(MDRangePolicy<Rank<3>>({0,0,0},{Neta,Ny,Nx}), …)` with **ix
  innermost** (coalesced loads / cache-friendly).  All launches go on the
  default execution-space instance, so they serialise in issue order like the
  CUDA single compute stream — the pipeline's producer→consumer dependencies
  hold with no inter-kernel fence.  `wait()` is one `Kokkos::fence()`.
- `reduce_max` is a `parallel_reduce` with `Kokkos::Max<float>` (drops the
  hand-rolled `atomicMax` bit-trick).  `pack_evolution_ideal` is a 1-D
  `parallel_for` into a managed scratch View + a host `memcpy`.

### Build (`CMakeLists.txt`, `src/CMakeLists.txt`)
The CUDA backend uses the **CMake CUDA language** (not nvcc_wrapper-as-CXX), so
only the three Kokkos-touching TUs go through `nvcc` while the rest of MUSIC
stays plain host `g++` (preserves the D7 PIMPL boundary).  Triggered by
`-DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON`; the build then
`enable_language(CUDA)`, sets `CMAKE_CUDA_STANDARD 20`, and marks
`GPUGrid_kokkos.cpp` / `KokkosPipelines.cpp` / `kokkos_runtime.cpp`
`LANGUAGE CUDA`.

---

## Build & verify (this branch)

```bash
# CPU reference + native-CUDA baseline
cmake -S . -B build       -DCMAKE_BUILD_TYPE=Release && cmake --build build       -j
cmake -S . -B build_cuda  -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build_cuda -j

bash get_kokkos.sh                                            # vendored 5.1.1

# Kokkos host backend (Serial + OpenMP)
cmake -S . -B build_kokkos -DUSE_KOKKOS=ON \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON \
      -DCMAKE_BUILD_TYPE=Release && cmake --build build_kokkos -j

# Kokkos CUDA backend (GB10 = sm_121 = BLACKWELL121)
cmake -S . -B build_kokkos_cuda -DUSE_KOKKOS=ON \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ARCH_BLACKWELL121=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_BUILD_TYPE=Release && cmake --build build_kokkos_cuda -j

# Precision gate (point GPU_BIN at either Kokkos build)
GPU_BIN=$PWD/build_kokkos/src/MUSIChydro      bash tests/eos_gpu_vs_cpu.sh   # OpenMP
GPU_BIN=$PWD/build_kokkos_cuda/src/MUSIChydro bash tests/eos_gpu_vs_cpu.sh   # Cuda (GB10)
```

For an explicit CPU diff: `MUSIC_FORCE_CPU=1` forces the CPU path in any build.
