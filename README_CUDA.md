# MUSIC4GPU — CUDA Port

CUDA (NVIDIA) back-end for the GPU-accelerated MUSIC 3+1D viscous
hydrodynamics solver, ported from the Metal 3 implementation. The CUDA layer
preserves the same public API (`GPUGrid` + a `*Pipelines` singleton with seven
`dispatch_*` methods) so the host evolution loop in `advance.cpp` is shared
between the Metal and CUDA back-ends.

## Building

```bash
cmake -S . -B build_cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda -j$(nproc)
```

- `CMAKE_CUDA_ARCHITECTURES` defaults to `native` (auto-detects the local GPU).
  Override for a target GPU, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` (A100).
- `USE_CUDA` and `USE_METAL` are mutually exclusive.
- If the GSL headers are not on the compiler's include path, add
  `-DCMAKE_DISABLE_FIND_PACKAGE_GSL=ON` (GSL is only used for Cooper-Frye
  freeze-out, not the hydro evolution exercised by the benchmarks).

CPU reference build (for comparison):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Source layout

| File | Role |
|------|------|
| `src/gpu/gpu_types.h` | Shared C++/MSL/CUDA structs; `__CUDACC__` branch puts the index tables in `__constant__` memory |
| `src/gpu/music_kernels.cu` | All 7 kernels + device helpers (port of `music_kernels.metal`) |
| `src/gpu/music_kernels.cuh` | `__global__` prototypes shared by the kernels and the dispatcher |
| `src/gpu/CUDAPipelines.{h,cu}` | Singleton: device init, stream, 7 `dispatch_*`, `wait()` |
| `src/gpu/GPUGrid_cuda.cu` | `cudaMallocManaged` allocation + prefetch; AoS↔SoA converters |
| `src/advance.{h,cpp}` | `GPUPipelines` alias + `MUSIC_USE_GPU` guard select the back-end |

## Verification

```bash
OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench.sh      # 2D boost-invariant
OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench_3d.sh   # 3+1D production
```

Acceptance threshold: **max relative `eps_max` error < 1e-3** across all steps,
versus the CPU build. This is re-checked after every phase as a regression
guard.

---

## Phase 1 — Direct port (correctness first)

A line-for-line translation of the Metal Shading Language kernels to CUDA C++,
using `cudaMallocManaged` unified memory (the natural analogue of Metal's
shared storage mode). No CUDA-specific optimizations yet — the goal is a
correct baseline.

### Translation map (MSL → CUDA)

| Metal | CUDA |
|-------|------|
| `kernel void f(device const float* b [[buffer(N)]], …)` | `__global__ void f(const float* __restrict__ b, …)` |
| `[[thread_position_in_grid]]` (uint3) | `blockIdx*blockDim + threadIdx` |
| `constant T& p [[buffer(N)]]` | by-value kernel argument `T p` |
| `threadgroup` / `threadgroup_barrier` | `__shared__` / `__syncthreads()` (unused in Phase 1) |
| `clamp(x,lo,hi)` | `clampf` / `clampi` helpers (`fminf/fmaxf`, `min/max`) |
| `abs/min/max/sqrt/exp/log/pow` (float) | `fabsf/fminf/fmaxf/sqrtf/expf/logf/powf` |
| `MTLResourceStorageModeShared` | `cudaMallocManaged` + `cudaMemPrefetchAsync` |
| `[cb commit]` + `[cb waitUntilCompleted]` | `cudaStreamSynchronize(stream)` |

`cudaMemPrefetchAsync` uses the CUDA 13 `cudaMemLocation` signature.

### Correctness (vs CPU, Initial_profile 0, Δτ=0.005)

All grids **PASS** (< 1e-3):

| Grid | steps | max rel `eps_max` err |
|------|-------|------------------------|
| 32×32×1 (2D) | 101 | 5.3e-05 |
| 64×64×1 (2D) | 101 | 9.7e-05 |
| 128×128×1 (2D) | 101 | 8.2e-05 |
| 32×32×8 (3D) | 61 | 4.7e-05 |
| 32×32×32 (3D) | 41 | 3.7e-05 |
| 64×64×16 (3D) | 41 | 3.1e-05 |
| 64×64×32 (3D) | 41 | 2.7e-05 |

The residual ~1e-5 error is the expected FP32 GPU vs FP64 CPU divergence
(the kernels compute in single precision), not an algorithmic difference.

### Performance — Phase 1 baseline

Hardware: **NVIDIA GB10** (Grace-Blackwell, cc 12.1, 121.6 GB unified) vs
20-thread CPU on the same node. Wall time for the full run (init + evolution).

**2D boost-invariant** (`tests/cuda_vs_cpu_bench.sh`, 100 steps):

| Grid | CPU 20T (s) | CUDA (s) | Speedup |
|------|------------:|---------:|--------:|
| 32×32×1 | 0.11 | 0.51 | 0.22× |
| 64×64×1 | 0.73 | 0.59 | 1.24× |
| 128×128×1 | 2.20 | 1.32 | 1.67× |

**3+1D production** (`tests/cuda_vs_cpu_bench_3d.sh`, 40 steps):

| Grid | CPU 20T (s) | CUDA (s) | Speedup |
|------|------------:|---------:|--------:|
| 32×32×8 | 0.32 | 0.54 | 0.59× |
| 32×32×32 | 0.63 | 0.56 | 1.12× |
| 64×64×16 | 1.40 | 0.99 | 1.41× |
| 64×64×32 (131k) | 3.02 | 1.26 | **2.40×** |

### Comparison to the Metal baseline

The README Metal baseline (M3 Max, 64×64×32, 40 steps) was **GPU 1.62 s vs CPU
12T 2.43 s → 1.50×**. The CUDA Phase-1 baseline reaches **2.40×** at the same
grid (GB10 vs 20-thread CPU). Absolute GPU wall time drops from 1.62 s (Metal,
M3 Max) to 1.26 s (CUDA, GB10) — a different machine, but the same production
problem.

**Observations carried into later phases:**
- Small grids (≤ 32² cells) are launch-/transfer-bound — speedup grows with
  grid size, so the optimizations target the large-grid regime.
- Phase 1 leaves the obvious CUDA wins on the table: EOS tables still live in
  global memory (Phase 2 → `__constant__`/`__ldg`), block dims are a fixed
  8×8×4 (Phase 2 tuning), stencils re-read global memory (Phase 3 tiling), and
  host↔device packing is serialized with compute (Phase 4 streams).

---

## Phase 2 — Cache & launch-configuration tuning

### Read-only data cache (`__ldg`) instead of `__constant__`

The plan proposed copying the `eos_P` / `eos_dPde` tables into `__constant__`
memory. In practice that is the wrong tool here, for two reasons:

1. **Capacity.** `GPU_EOS_N = 8192` → 32 KB per table; the two tables alone
   are exactly 64 KB, leaving no room for the index tables already resident in
   constant memory (the module would overflow the 64 KB constant bank).
2. **Access pattern.** `__constant__` is fast only for *broadcast* reads (all
   lanes in a warp read the same address). EOS lookups are indexed by each
   cell's local energy density, so neighbouring lanes hit *different* table
   entries — exactly the non-broadcast pattern that serializes constant memory.

Instead, every EOS and stencil load now goes through the **read-only data
cache** via `__ldg()` (in `gpu_eos_interp`, `gpu_log_interp`, `gpu_TJb0`, and
the `get_Wmunu` / `get_u` / `get_pi_b` halo helpers). This caches the tables
and tolerates the per-lane index divergence. The index tables
(`WMUNU_IDX`, `GMUNU_DIAG`) do stay in `__constant__` (the `__CUDACC__` branch
in `gpu_types.h`) — those *are* uniform broadcasts.

### Adaptive block dimensions + occupancy tuning

Phase 1 used a fixed `dim3(8,8,4)` block. On a 2-D boost-invariant grid
(`Neta == 1`) that idles 3 of every 4 eta lanes — a 4× waste. `compute_launch`
now factors the block from an occupancy-tuned thread budget
(`cudaOccupancyMaxPotentialBlockSize` on the heavy `gpu_make_delta_qi` kernel),
adapting the eta extent to `min(Neta,4)` and sizing x to 32 for coalesced SoA
loads:

| Grid | Phase 1 block | Phase 2 block |
|------|---------------|---------------|
| Neta = 1 (2D) | 8×8×4 (64/256 active) | 32×8×1 (256 active) |
| Neta ≥ 4 (3D) | 8×8×4 | 32×2×4 (coalesced x) |

### Per-step benchmark methodology

Wall-clock totals include ~0.5 s of fixed overhead (CUDA context creation + EOS
sampling) that dominates short runs. `tests/cuda_perstep_bench.sh` removes it by
differencing two step counts (10 vs 110 steps) and taking the min of 3 runs:
`per_step = (T_long − T_short) / 100`.

### Performance — per-step (GB10 vs 20-thread CPU)

| Grid | GPU ms/step | CPU ms/step | Speedup |
|------|------------:|------------:|--------:|
| 128×128×1 (2D) | 6.21 | 14.77 | 2.38× |
| 64×64×16 (3D) | 7.97 | 30.64 | 3.84× |
| 64×64×32 (3D, 131k) | 11.84 | 61.71 | **5.21×** |

Correctness unchanged (max rel `eps_max` error still ~3e-5 at 64×64×32). The
per-step speedup at the production grid is **5.21×**, versus the ~2.4× wall-clock
figure of Phase 1 that was diluted by fixed init overhead — both the cleaner
methodology and the cache/launch tuning contribute.

---

## Phase 3 — Shared-memory tiling

### Profile-driven retargeting

Before tiling anything, `nsys` was used to find where the time actually goes
(64×64×32, shear on):

| Kernel | % runtime | Nature |
|--------|----------:|--------|
| `gpu_make_delta_qi` | 31.5% | compute-bound (Newton-Brent reconstruction) |
| `gpu_first_rk_step_w_full` | 25.2% | compute-bound (per-cell algebra, **no stencil**) |
| `gpu_make_w_source` | **23.7%** | bandwidth-bound radius-1 stencil (14 Wmunu comps) |
| `gpu_make_du` | 7.5% | radius-1 stencil |
| `gpu_make_uwrhs` | 6.2% | radius-2 stencil |
| `gpu_finalize_ideal` | 5.8% | per-cell |

This contradicts the plan's tiling priority (which led with `gpu_make_delta_qi`).
The two largest kernels are **compute-bound** — `delta_qi` spends its time in the
Newton-Brent velocity solve (up to 60 iterations × EOS evaluations per
reconstruction, 12 reconstructions per cell) and `w_full` reads only per-cell
buffers with no neighbour stencil at all. Shared-memory tiling cannot help
either; they are addressed by Phase 5 (warp-level Newton early exit) instead.

The genuine tiling target is **`gpu_make_w_source`** (23.7%, a bandwidth-bound
radius-1 stencil with high neighbour reuse), so that is what Phase 3 tiles.

### Implementation

`gpu_make_w_source_tiled` cooperatively stages the current-snapshot
`Wmunu[14] + u[4] + pi_b` into a `(bx+2)(by+2)(bz+2)` halo tile in **dynamic
shared memory**, then serves every stencil neighbour read from shared instead of
global. The previous snapshot is sampled only at the cell centre (the time
derivative), so it stays in global memory. The arithmetic is bit-for-bit
identical to the untiled kernel. Launch uses a balanced 8×8×`min(Neta,4)` block
(small, low-halo-overhead) — all configurations stay under the 48 KB default
shared carveout (45.6 KB at the 3-D block).

### Performance

Kernel-level (`nsys`, 64×64×32, avg over 42 launches):

| `gpu_make_w_source` | Phase 2 (untiled) | Phase 3 (tiled) |
|---------------------|------------------:|----------------:|
| avg per launch | 228.6 µs | 192.9 µs (**−16%**) |
| share of runtime | 23.7% | 20.8% |

Per-step (vs 20-thread CPU), correctness still PASS (~3e-5):

| Grid | Phase 2 ms/step | Phase 3 ms/step | Speedup vs CPU |
|------|----------------:|----------------:|---------------:|
| 64×64×32 (131k) | 11.84 | 10.79 | **5.50×** |

The ~16% kernel win translates to ~9% at the production grid — bounded by
Amdahl, since the two compute-bound kernels (`delta_qi` + `w_full` ≈ 57%) are
untouched by tiling. This is the honest ceiling for tiling on this workload;
the remaining headroom is in the Newton solver (Phase 5) and host↔device
overlap (Phase 4).
