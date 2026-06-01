# Technical Analysis: MUSIC4GPU CUDA Speedup Opportunities

## Context

MUSIC4GPU is a 3+1D viscous relativistic hydrodynamics code (heavy-ion collisions) with a hybrid CPU/GPU execution model. The GPU path handles all core hydro evolution; the CPU retains diagnostics, freeze-out surface extraction, and I/O.

---

## Current GPU Pipeline (per RK substep — 7 kernel launches × 2 RK stages = 14/timestep)

| # | Kernel | Purpose | Limiting Factor |
|---|--------|---------|----------------|
| 1 | `gpu_make_du` | Velocity gradients: θ, a^μ, σ^{μν} | Global memory bandwidth |
| 2 | `gpu_make_delta_qi` | KT flux divergence + Newton-Brent reconstruction | **~200 registers/thread → ~25-35% occupancy** |
| 3 | `gpu_make_w_source_tiled` | Viscous stress source ∂_α W^{αμ} | Shared memory (tiled — good) |
| 4 | `gpu_make_uwrhs` | Wmunu spatial flux RHS (5 components) | Global memory bandwidth |
| 5 | `gpu_make_uprhs` | Bulk π_b spatial flux RHS | Global memory bandwidth |
| 6 | `gpu_first_rk_step_w_full` | Advance Wmunu, π_b with IS source | Branchy transport coeff. |
| 7 | `gpu_finalize_ideal` | Reconstruct primitives (ε, rhob, u) | Register pressure |

Additional: `gpu_reduce_max_eps_rhob` — 1×/timestep for CFL check.

**Key data structures:** SoA layout `field[comp*Ncells + cell]`; 21 floats/cell; 3 snapshots + ~45×Ncells scratch. EOS: 4 tables × 8192 entries stored in global memory with `__ldg()`. Two CUDA streams: compute + copy (async H2D overlap).

---

## Current Optimizations Already In Place

- SoA component-major layout (coalesced warp access)
- `--use_fast_math` (FP32 reciprocal, sqrt, log via hardware)
- Shared memory tiling on `gpu_make_w_source_tiled` (45 KB / 48 KB shared)
- Occupancy-driven block sizing via `cudaOccupancyMaxPotentialBlockSize` for register-heavy kernels
- Dual-stream: async H2D on `copy_stream_`, gated by event before compute
- GPU state residency flags (`gpu_owns_state_`, `gpu_state_authoritative_`) to skip redundant H2D uploads between RK substeps
- Pinned staging buffers on discrete GPUs (`cudaHostAlloc`)
- Managed memory on coherent hardware (GB10 / NVLink-C2C) with prefetch hints

---

## Identified Bottlenecks

### GPU-Side

**1. `gpu_make_delta_qi` register pressure** (`src/gpu/music_kernels.cu:729-846`)
- ~200 registers/thread from `ReconstResult` struct + Newton-Brent solver state + minmod reconstruction across 3 directions × 5 stencil points
- Occupancy capped at 25-35% despite `cudaOccupancyMaxPotentialBlockSize`; this is the dominant kernel
- Block capped at 256 threads max

**2. Untiled stencil kernels waste L2 bandwidth**
- `gpu_make_du`, `gpu_make_uwrhs`, `gpu_make_uprhs` all do radius-1 stencil reads from global memory
- Only `gpu_make_w_source_tiled` uses shared memory; the others re-fetch the same neighbor data independently

**3. EOS lookups in global memory**
- `gpu_log_interp()` called ~12× per cell in `gpu_make_delta_qi` (once per reconstruction per direction)
- `__ldg()` uses read-only L1 cache but 1D texture objects have dedicated texture cache hardware and support hardware interpolation

**4. Independent kernels not fused**
- `gpu_make_uwrhs` and `gpu_make_uprhs` both read the same velocity and Wmunu fields; running back-to-back doubles L2 fetches
- `gpu_make_du` and `gpu_make_w_source` both read `u_curr` from global memory

**5. Hand-rolled reduction**
- `gpu_reduce_max_eps_rhob` uses shared memory + `atomicMax` (IEEE float trick); CUB `DeviceReduce::Max` is typically 2-3× faster on modern GPUs

### Host-Side

**6. Freeze-out D2H sync fires every step, not every `facTau` steps** (`src/evolve.cpp:330-355`)
- `advance.sync_arena_from_gpu_readonly()` at line 334 is unconditional inside `if (freezeout_flag == 1)` — it runs every timestep
- `FindFreezeOutSurface_Cornelius()` only runs every `facTau` steps (line 340), but the full arena D2H happens regardless on every intermediate step
- Fix: move the sync inside the `facTau` condition — `store_previous_step_for_freezeout` (lines 349-350) is the only caller that actually needs the host arena data
- **2-line host change; no GPU work required; reduces D2H by factor of `facTau`**

**7. Forced D2H every timestep for diagnostics** (`src/evolve.cpp:199-265`)
- Default `Nskip_diag=1` forces full arena D2H every single step for momentum anisotropy, conservation laws, vorticity
- On large grids this serializes CPU and GPU completely once per step

**8. Cornelius freeze-out is CPU-only and single-threaded** (`src/evolve.cpp:668-950`)
- Serial loop over η-slices (OMP pragma commented out due to file I/O contention)
- Even after fix 6, every `facTau` steps: GPU idles, Cornelius runs single-threaded, GPU resumes

---

## Ranked Speedup Opportunities

### Priority 0 — Trivial Wins, Immediate

**A. Gate freeze-out D2H inside `facTau` check** (`src/evolve.cpp:330-355`) — all platforms

```cpp
// BEFORE — sync fires every step
if (freezeout_flag == 1) {
    advance.sync_arena_from_gpu_readonly(*fpPrev, *fpCurr);  // every step!
    if (freezeout_lowtemp_flag == 1 && it == iFreezeStart) { ... }
    if ((it - iFreezeStart)%facTau == 0 && it > iFreezeStart) {
        FindFreezeOutSurface_Cornelius(...);
        store_previous_step_for_freezeout(...);
    }
}

// AFTER — sync only when data is actually needed
if (freezeout_flag == 1) {
    if (freezeout_lowtemp_flag == 1 && it == iFreezeStart) {
        advance.sync_arena_from_gpu_readonly(*fpPrev, *fpCurr);
        FreezeOut_equal_tau_Surface(...);
    }
    if ((it - iFreezeStart)%facTau == 0 && it > iFreezeStart) {
        advance.sync_arena_from_gpu_readonly(*fpPrev, *fpCurr);  // every facTau
        FindFreezeOutSurface_Cornelius(...);
        store_previous_step_for_freezeout(...);
    }
}
```

Eliminates `(facTau - 1) / facTau` of freeze-out sync calls. On discrete GPU each avoided sync also eliminates a PCIe transfer — large gain. On GB10/Metal the sync is just a `cudaStreamSynchronize` / `waitUntilCompleted` with no data movement — still removes unnecessary CPU-GPU serialisation but impact is smaller.

**B. CPU OpenMP parallelisation of Cornelius over η-slices** (`src/evolve.cpp:679-695`) — all platforms

The infrastructure is **already written** — the pragma is just commented out and `thread_id`-based per-thread output files are already in place. Three minimal changes unlock full CPU parallelism:

1. **Uncomment the pragma** (`evolve.cpp:679`):
```cpp
// BEFORE
//#pragma omp parallel for reduction(+:intersections)
for (int ieta = 0; ieta < (neta-fac_eta); ieta += fac_eta) {
    int thread_id = omp_get_thread_num();   // always 0 in serial

// AFTER
#pragma omp parallel for reduction(+:intersections)
for (int ieta = 0; ieta < (neta-fac_eta); ieta += fac_eta) {
    int thread_id = omp_get_thread_num();   // 0..nthreads-1
```

2. **Replace `return` inside the parallel loop** (`evolve.cpp:687`) — `return` in an OMP parallel region is UB:
```cpp
// BEFORE
if (DATA.reRunHydro) { return(0); }
// AFTER
if (DATA.reRunHydro) { continue; }   // caller checks reRunHydro after loop
```

3. **Protect the static warning flag** (`evolve.cpp:799`) — static local written from multiple threads:
```cpp
// BEFORE
static bool warned_nonfinite = false;
if (!warned_nonfinite) { ...; warned_nonfinite = true; }
// AFTER
static std::atomic<bool> warned_nonfinite{false};
if (!warned_nonfinite.exchange(true)) { ...; }
```

Everything else in `FindFreezeOutSurface_Cornelius_XY` is already thread-safe: `cornelius_ptr`, `cube`, `fluid_cube`, and `u_derivative_helper` are all freshly heap-allocated per call; the arena inputs are read-only; output goes to separate per-thread files.

- **Effort**: ~1 hour  
- **Speedup**: scales with min(CPU cores, Neta); for Neta=32 and 16 cores → up to ~16× on the Cornelius step  
- **Status**: natural intermediate step before the GPU port (item H); per-thread file design already anticipates this

### Priority 1 — High Impact, Moderate Effort

**C. Compute diagnostics on GPU; transfer only scalars** (`src/evolve.cpp:199-265`, new GPU kernels)
- Momentum anisotropy `ε_x/ε_y` and conservation checks are O(Ncells) reductions
- Replace full D2H + CPU loop with GPU reduction kernels using `cub::DeviceReduce`; only transfer 2-4 floats/step
- Host change: call new `dispatch_diagnostics()` on compute stream; `wait()` only for output
- Eliminates the dominant serial stall when `Nskip_diag=1`
- **Discrete GPU only** (A100, H100, RTX, V100). On GB10 (NVLink-C2C, `g_cuda_coherent=true`) and Metal (Apple Silicon), `sync_arena_from_gpu_readonly` is just a stream sync + CPU read of managed/shared memory — no PCIe transfer, so the benefit disappears

**D. Replace hand-rolled reduction with CUB** (`src/gpu/music_kernels.cu:1873-1909`)
- `cub::DeviceReduce::Max` on the epsilon array; 2-3× faster, single call
- Eliminates `atomicMax` across 1024 blocks; CUB uses a tuned two-pass approach
- 2-3 lines replacing the current kernel launch

### Priority 2 — High Impact, Higher Effort

**E. Shared memory tiling for `gpu_make_du` and `gpu_make_uwrhs`** (`src/gpu/music_kernels.cu:1131, 928`)
- Both kernels do radius-1 stencil reads but don't use shared memory (unlike `gpu_make_w_source_tiled`)
- `gpu_make_du`: tile u (4 components); at 8×8×4 block: 4 × 600 × 4 = 9.6 KB → well within 48 KB budget
- `gpu_make_uwrhs`: tile 5 Wmunu components; 5 × 600 × 4 = 12 KB
- Apply same pattern as `gpu_make_w_source_tiled` lines 210-360
- Expect 2-3× bandwidth reduction for these kernels

**F. EOS texture memory** (`src/gpu/music_kernels.cu:386-410`, `src/gpu/GPUGrid_cuda.cu:upload_eos`)
- Convert EOS tables to `cudaTextureObject_t` 1D textures with hardware linear interpolation
- Dedicated texture cache separate from L1/L2; won't evict stencil data in `gpu_make_delta_qi`
- `cudaCreateTextureObject()` with `cudaResourceTypeLinear`; replace `gpu_log_interp()` calls with `tex1D<float>(tex, coord)`

**G. Fuse `gpu_make_uwrhs` + `gpu_make_uprhs`** (`src/gpu/CUDAPipelines.cu:278-302`)
- Both kernels read the same u and Wmunu inputs; back-to-back launches double the global memory sweep
- Fusing: combine kernel bodies, single launch with same block/grid as uwrhs; write both output arrays

**H. GPU Cornelius Freeze-Out Port** (new kernel, `src/evolve.cpp:668-950`)
- Natural follow-on to item B (CPU OMP parallelisation); do B first, then replace with GPU kernel when warranted
- Hypercube check is embarrassingly parallel: each (ix, iy, ieta) cell is independent
- GPU kernel over all cells; surface elements written via atomic counter or scan-based compaction
- Eliminates the remaining D2H + any residual CPU bottleneck at every `facTau` step
- I/O write remains on CPU after D2H of surface element array

### Priority 3 — Medium Impact

**I. `gpu_make_delta_qi` register pressure reduction** (`src/gpu/music_kernels.cu:729-846`)
- Option A: Split into per-direction sub-kernels (x, y, η) — each needs only 1-direction reconstruction
- Option B: `__launch_bounds__(256, 2)` to cap registers at cost of L1 spills
- Option C: `__noinline__` on `gpu_reconst()` device function to reduce live register scope
- Profile first: `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`

**J. Async diagnostic stream**
- Third `diag_stream_` running `dispatch_diagnostics()` in parallel with RK compute
- Diagnostics read `snap_curr` (stable during diagnostics); output scalars only

**K. Pinned host arena fields** (`src/fields.h`, `src/advance.cpp`)
- `cudaHostRegister` on `arenaFieldsCurr_`/`arenaFieldsPrev_` storage
- Eliminates page-fault overhead on H2D for discrete GPUs

### Priority 4 — Architectural

**L. Multi-GPU domain decomposition in η**
- η-slices are nearly independent (radius-1 longitudinal coupling only)
- Halo exchange: 1 slice × 21 floats/cell × Nx × Ny per substep via `cudaMemcpyPeer` or NCCL
- ~1.8-1.9× speedup for 2 GPUs on 3D grids

---

## Recommendation: Where the Majority of Gains Come From

**A + B + C eliminate the host-side bottlenecks; I tackles the dominant GPU-side bottleneck.**

- **A and B** are pure CPU code and benefit all backends (CUDA discrete, CUDA coherent/GB10, Metal, CPU-only). **A** removes unnecessary GPU sync calls every step; impact is large on discrete GPU (avoids PCIe transfer) and smaller but non-zero on coherent hardware (avoids CPU-GPU serialisation). **B** parallelises the Cornelius CPU step regardless of GPU backend.

- **C** eliminates expensive PCIe D2H transfers forced by diagnostics every step — **discrete GPU only**. On GB10 (NVLink-C2C, `g_cuda_coherent=true`) and Metal (Apple Silicon), memory is coherent/unified and the sync is just a stream wait with no data movement, so C provides no benefit on those platforms.

- **D (CUB reduction)** is trivial to implement but negligible in practice. `gpu_reduce_max_eps_rhob` runs once per timestep and is a tiny fraction of total runtime (~0.1–0.5%). It stays in the list because the change is free, not because it moves the needle.

- **I (`gpu_make_delta_qi` register pressure)** is the main remaining GPU-side gain. This kernel dominates the compute pipeline at 25–35% occupancy due to ~200 registers/thread. A successful register reduction gives back 10–20% of total GPU kernel time — more than items E, F, and G combined. It is harder to implement than A–D but is the right next target once host bottlenecks are resolved.

**Suggested implementation order:** A → B → C → I → (E, F, G, H as time permits).

---

## Quick-Win Summary

| Change | Est. Speedup | Effort | Location |
|--------|-------------|--------|----------|
| A — Gate freeze-out sync in `facTau` | `facTau`× fewer D2H | 30 min | `src/evolve.cpp:334` |
| B — OMP Cornelius (uncomment pragma) | up to ~Neta× on FO step | 1 hour | `src/evolve.cpp:679,687,799` |
| C — GPU diagnostics | 10-30% wall-time | 2 days | `src/evolve.cpp`, new kernel |
| D — CUB reduction | 2-3× for that kernel | 0.5 day | `src/gpu/music_kernels.cu:1873` |
| E — Tile make_du, uwrhs | 10-20% for those kernels | 2 days | `src/gpu/music_kernels.cu:1131,928` |
| F — EOS textures | 5-10% on delta_qi | 1 day | `src/gpu/music_kernels.cu:386` |
| G — Fuse uwrhs+uprhs | 5-10% | 1 day | `src/gpu/music_kernels.cu:928` |
| H — GPU Cornelius (after B) | Eliminates FO stall fully | 3-5 days | new kernel, `src/evolve.cpp` |
| I — delta_qi registers | 10-20% dominant kernel | 2-3 days | `src/gpu/music_kernels.cu:729` |

---

## Implementation Log

| Item | Commit | Status | Notes |
|------|--------|--------|-------|
| A — Gate FO sync in `facTau` | `199531d` | Done | 2-line change; all platforms |
| B — OMP Cornelius | `c5bab7a` | Done | Uncomment pragma + atomic flag + `<atomic>` include |
| D — CUB reduction | `f405d38` | Done | CUB scratch buffer in GPUGrid; old kernel retained but unused |
| C — GPU conservation sums | `2be00b3` | Done | Discrete GPU only; coherent HW (GB10/Metal) falls back to CPU |

---

## Verification Approach

1. `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum` — per-kernel before/after
2. `cuda_vs_cpu_bench.sh` for end-to-end wall-time validation
3. Numeric check: `max_epsilon` evolution should agree to ~1e-4 between reference and optimized runs (FP32)
4. For GPU Cornelius: compare surface cell count and d³σ_μ values against CPU Cornelius output
