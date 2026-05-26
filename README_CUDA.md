# MUSIC4GPU — CUDA Port

> **Experimental — port only, no new physics.** This is a direct GPU port of the
> CPU algorithms in **MUSIC release 3.1**, produced with **Claude Opus 4.7**. It
> is **not yet validated for physics production** — see the scope-and-status
> disclaimer at the top of the main [README.md](README.md) before use.

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
| `src/gpu/music_kernels.cu` | All 7 hydro kernels + `gpu_reduce_max_eps_rhob` (Phase 6) + device helpers (port of `music_kernels.metal`) |
| `src/gpu/music_kernels.cuh` | `__global__` prototypes shared by the kernels and the dispatcher |
| `src/gpu/CUDAPipelines.{h,cu}` | Singleton: device init, stream, 7 `dispatch_*` + `reduce_max`, `wait()` |
| `src/gpu/GPUGrid_cuda.cu` | Two-memory-mode allocation (managed or device+pinned); AoS↔SoA converters; `rotate_snapshots` / `swap_curr_future` (Phase 6) |
| `src/advance.{h,cpp}` | `GPUPipelines` alias + `MUSIC_USE_GPU` guard select the back-end; `gpu_owns_state_` flag (Phase 6) |
| `src/evolve.cpp` | `AdvanceRK` mirrors host swap → `swap_curr_future_gpu()`; `EvolveIt` calls `reduce_max_gpu()` when residency active (Phase 6) |

## Verification

```bash
OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench.sh      # 2D boost-invariant
OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench_3d.sh   # 3+1D production
```

Acceptance threshold: **max relative `eps_max` error < 1e-3** across all steps,
versus the CPU build. This is re-checked after every phase as a regression
guard.

---

## Physics support matrix (CUDA)

The CUDA back-end is a direct port of the Metal kernels and shares the same host
dispatch gates in `advance.cpp` (via the `MUSIC_USE_GPU` / `GPUPipelines`
indirection), so its physics coverage is **identical to the Metal back-end**.
The hydro evolution runs entirely on the GPU when the configuration falls inside
the supported matrix; anything outside it transparently falls back to the
existing CPU code path (per-cell), so results stay correct — just slower.

**Fully on the GPU (no per-cell CPU loop):**

| Capability | GPU kernel(s) | Notes |
|------------|---------------|-------|
| Ideal hydro: KT flux + conserved→primitive Newton reconstruction | `gpu_make_delta_qi`, `gpu_finalize_ideal` | the per-cell `FirstRKStepT` is skipped entirely |
| Viscous source divergence ∂(τWᵐⁿ) | `gpu_make_w_source` (tiled) | |
| Shear stress πᵐⁿ full 2nd-order IS update | `gpu_make_uwrhs`, `gpu_make_du`, `gpu_first_rk_step_w_full` | `turn_on_shear == 1` |
| Bulk pressure Π update | `gpu_make_uprhs`, `gpu_first_rk_step_w_full` | `turn_on_bulk == 1` |
| Velocity gradients θ, aᵘ, σᵘᵛ | `gpu_make_du` | |
| T-dependent η/s | in `gpu_first_rk_step_w_full` | modes **0, 1, 2, 3, 11** |
| T-dependent ζ/s | in `gpu_first_rk_step_w_full` | modes **0, 1, 2, 3, 7, 8, 9, 10** |
| 2nd-order coupling terms (W·σ, W·W, π↔Π) | in `gpu_first_rk_step_w_full` | `include_second_order_terms == 1` |
| QuestRevert regulator | in `gpu_first_rk_step_w_full` | active when `Initial_profile ∉ {0,1}` |
| Hydro source terms jᵘ (energy + baryon) | `gpu_finalize_ideal` | source **evaluated on CPU** into `qi_source_buf`, integrated on GPU |
| Boost-invariant (2D) and full 3+1D | all | |

**Falls back to the CPU path (correct, not yet ported):**

| Capability | Gate that forces CPU | What it would take to port |
|------------|----------------------|----------------------------|
| Baryon diffusion qᵘ | `turn_on_diff == 1` (disables `gpu_make_du`/`w_full`) | port `Make_uqRHS`/`Make_uqSource`; the diffusion components (idx 10–13) are currently zeroed on the GPU |
| Vorticity terms | `include_vorticity_terms == 1` | port the kinetic-vorticity tensor (`dUoverTsup`/`dUTsup`) into `gpu_make_du` |
| Finite net-baryon EOS / μ_B-dependent shear | `muB_dependent_shear_to_s != 0` | the GPU EOS tables are sampled at **rhob = 0**; needs a 2-D P(e, ρ_B) table + μ_B(e, ρ_B) and the diffusion sector |
| η/s modes outside {0,1,2,3,11} | `T_dependent_shear_to_s` other | add the profile to `gpu_eta_over_s` |

All temperature-dependent η/s **and** ζ/s profiles are now ported (ζ/s mode 7,
bigbroadP, was the last gap — added on the CUDA side beyond the Metal baseline,
validated CPU-vs-GPU at 4.9e-5). The only remaining transport gap is the
**μ_B-dependent** shear multiplier, which is intrinsically tied to the finite-μ_B
EOS work above.

**Not GPU work at all (host, by design):** initial-condition construction
(`init.cpp`), Cooper–Frye freeze-out / Cornelius surface finding, evolution
output, and the per-step diagnostics. The per-step `eps_max`/`T_max` reduction
in particular is the host dependency targeted by the GPU-resident-state plan
below.

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

---

## Phase 4 — Dual-stream parallelism (and the host-bound finding)

### What the profile actually says

The most important measurement of the whole port: at 64×64×32, summing the
kernel times gives **~39 ms of GPU work over a ~710 ms run — the GPU is busy
only ~5.5% of wall time** (≈22% of the evolution phase once the ~0.53 s fixed
init is excluded). The run is **host-bound**: the AoS↔SoA conversion
(`copy_to_gpu` / copy-back, FP64↔FP32 over the whole grid) and the per-step CPU
diagnostics dominate, not the kernels.

`nsys` also reports **no GPU memory-transfer data at all** — GB10 is a
Grace-Blackwell superchip with a **coherent** CPU/GPU address space over
NVLink-C2C, so `cudaMallocManaged` buffers are read in place with no discrete
PCIe migration.

### Consequence for the plan's Phase 4

The plan's Phase 4 — pinned host staging + `cudaMemcpyAsync` H2D overlapped with
compute — is designed for a **discrete** GPU. On GB10 there is no transfer to
overlap, and an early experiment confirmed the hazard: issuing a
`cudaMemPrefetchAsync` + `cudaStreamWaitEvent` gate before each substep
*regressed* the production grid (13.1 vs 10.8 ms/step) by forcing a migration
onto the critical path that coherent access did not need.

### What was implemented

The dual-stream architecture is implemented as the plan specifies — a dedicated
`copy_stream_` + completion event, and `upload_snapshots_async()` which
prefetches the freshly-packed `snap_curr`/`snap_prev` onto the copy stream and
gates the compute stream — but it is **guarded by a runtime memory-model check**
(`cudaDevAttrPageableMemoryAccess` / `prop.integrated`). On a coherent part
(GB10 → "coherent host memory: yes") the prefetch+gate is skipped, so Phase 4 is
a correctness- and performance-neutral no-op there; on a discrete GPU (A100, the
plan's target) the path stays active and moves the SoA upload onto its own
stream so it can overlap the previous substep's compute.

### Performance

| Grid | Phase 3 ms/step | Phase 4 ms/step | Speedup vs CPU |
|------|----------------:|----------------:|---------------:|
| 64×64×32 (131k) | 10.79 | 11.48 | 4.90× |

Within run-to-run noise (≈±10%), Phase 4 is neutral on GB10 — the intended
outcome given coherent memory. The takeaway recorded for future work: on this
class of hardware the lever is **not** transfer overlap but reducing host-side
per-step cost (keeping evolving state GPU-resident across timesteps, moving the
`eps_max` reduction onto the GPU) and speeding the dominant compute-bound
kernel (Phase 5).

---

## Phase 5 — Compute-bound kernel acceleration

### Why not warp-vote Newton convergence (the plan's idea)

The plan proposed `__ballot_sync` / `__all_sync` to break the Newton loop once
all 32 lanes converge. On analysis this gives **nothing** here: the kernels
already `break` per-thread, which deactivates a converged lane while the warp
continues for the stragglers — so the warp's iteration count is already
`max_lane(k)`. A warp vote cannot lower that. Worse, *removing* the per-thread
break to rely only on the vote would keep converged lanes evaluating the EOS
every iteration (strictly more work). So the per-thread `break` already in the
solver is optimal; the warp-vote version was not implemented.

### What actually helps: fast-math intrinsics

The dominant kernel `gpu_make_delta_qi` (Newton-Brent reconstruction) and the
relaxation kernels are dense with divisions, `sqrtf`, and — in the transport
profiles — `expf`/`powf`. `--use_fast_math` maps these to the hardware
approximate units. The hydro state is already FP32 (~1e-5 vs the FP64 CPU), and
the EOS pressure table is *linear* (no transcendental on the hot path), so
fast-math stays comfortably inside the 1e-3 threshold.

### Performance — kernel level (`nsys`, 64×64×32, avg of 42 launches)

| Kernel | Phase 4 (µs) | Phase 5 (µs) | Change |
|--------|-------------:|-------------:|-------:|
| `gpu_make_delta_qi` | 302 | 148 | **−51% (2.04×)** |
| `gpu_finalize_ideal` | 55 | 49 | −11% |
| `gpu_make_uwrhs` | 61 | 46 | −25% |
| total GPU kernel time | ~39 ms | ~33.5 ms | −14% |

Correctness unchanged: max rel `eps_max` error **2.7e-5** at 64×64×32 (PASS).

The dominant compute kernel is **halved** with no accuracy cost. Per-step *wall*
time barely moves (11.0 ms at 64×64×32) because — as Phase 4 established — the
GPU is only ~5.5% of wall on this coherent host-bound run; shrinking the kernels
shrinks an already-small slice. On a discrete GPU, or once the host-side
AoS↔SoA / diagnostics cost is addressed, this 2× on the heaviest kernel is the
one that matters.

FP16 (the plan's other Phase-5 idea) was not pursued: it trades accuracy for
*bandwidth*, but the kernels that dominate here are compute-bound, and the run
is host-bound — so FP16 would add accuracy risk for no relevant gain on this
hardware.

---

## Phase 6 — GPU-resident state across timesteps

### Motivation

Phases 1–5 left the per-step pipeline still re-uploading the evolving state on
every rk0:

```
rk0 entry → copy_to_gpu(arena_current, snap_curr)   ≈ 10.5 MB AoS→SoA pack + H2D
            copy_to_gpu(arena_prev,    snap_prev)   ≈ 10.5 MB AoS→SoA pack + H2D
            kernels…                                ≈   13 ms compute
rk1 entry → rotate_snapshots (intra-step, OK)
            kernels…
            copy_back snap_future → arena_future    ≈ 11 MB SoA→AoS unpack + D2H
            host swap(current, future)
```

That `copy_to_gpu` at the start of every rk0 is **pure waste on a discrete GPU**:
the data being uploaded is exactly the data the GPU produced one step earlier and
sent back via the D2H copy-back at the previous rk1.  On RTX 3090 this is
~21 MB of redundant PCIe H2D traffic per step (plus the AoS↔SoA pack work).

The plan that targets this restructure is
[`Plan-GPU-resident-state.md`](Plan-GPU-resident-state.md).

### What was implemented (Phase A + Phase B of the plan)

**Phase A — eliminate the per-step H2D upload.**
A new `GPUGrid::swap_curr_future()` (pointer-alias swap of `snap_curr ↔
snap_future`, no data move) is called from `Evolve::AdvanceRK` immediately after
the rk1 `std::swap(arena_current, arena_future)`.  Combined with the existing
`rotate_snapshots()` at the rk0→rk1 transition, this keeps the GPU snapshot
roles in lockstep with the host arena pointers across the timestep boundary —
so at the next rk0, `snap_curr` and `snap_prev` are already the correct
arenas.  A run-level flag `gpu_owns_state_` (set after the first complete
full-GPU step) then gates `AdvanceIt` to **skip the `copy_to_gpu` calls** at
rk0.  The D2H copy-back is **kept** so that the every-step host diagnostics
(`output_momentum_anisotropy_vs_tau`, `check_conservation_law`,
`get_maximum_energy_density`) continue to read fresh arena data — see Phase C
below for why this isn't free yet.

**Phase B — GPU max-reduction for `eps_max` / `rhob_max`.**
A new `gpu_reduce_max_eps_rhob` kernel (256-thread blocks, shared-memory tree
reduction, `atomicMax` via the IEEE-754 non-negative-float trick) replaces the
CPU O(N) scan in `get_maximum_energy_density` when residency is active.  Only
two `float` scalars cross the PCIe boundary per step.  `T_max` is computed on
the host as `eos.get_temperature(eps_max, 0)` — exact for the rhob=0 EOS used
in standard runs, per §4.2 of the plan.  Metal gets a CPU-scan fallback
(coherent unified memory, ≤ 1 MB scan, negligible).

### Performance — before vs after Phase 6, on RTX 3090 (discrete PCIe)

Hardware: **NVIDIA RTX 3090** (cc 8.6, 24 GB, discrete PCIe, `coherent host
memory: no`) vs 48-thread CPU on the same node.  `tests/cuda_perstep_bench.sh`
(min of 3 runs, 10-vs-110 step differencing — see Phase 2 methodology):

| Grid | **Before** GPU ms/step | **After** GPU ms/step | Per-step speedup (after / before) | Speedup vs CPU (before → after) |
|------|----------------------:|----------------------:|----------------------------------:|-------------------------------:|
| 128×128×1 (2D) | 12.30 | 11.15 | **1.10×** | 1.57× → 1.77× |
| 64×64×16 (3D) | 15.18 | 13.36 | **1.14×** | 2.25× → 2.55× |
| 64×64×32 (3D, 131k) | 17.01 | 13.44 | **1.27×** | 3.22× → **4.17×** |

The 3D production grid (64×64×32) gains **27% per-step** purely from the
residency change — that is the per-step cost of the redundant H2D + AoS pack
the previous pipeline was paying every step.  The 2D boost-invariant case
gains less because its per-step transfer (~5 MB total) is a smaller fraction
of step time.

### Correctness

Both regression tests pass unchanged on the new pipeline:

| Test | Max rel `eps_max` error |
|------|------------------------:|
| `tests/cuda_vs_cpu_bench.sh` (2D, 100 steps) | 8.8e-05 |
| `tests/cuda_vs_cpu_bench_3d.sh` (3D, 40 steps, 64×64×32) | 2.7e-05 |

Identical to the pre-Phase-6 numbers — the residency change is a transport
reshuffle, not a numerical change.

### Profile-driven follow-up: parallelize the every-step host diagnostic

After Phase A+B landed, instrumenting the EvolveIt loop with a small
opt-in profiler (`src/bench_timer.h`, activated by `MUSIC_PROFILE=1`)
showed a surprise on RTX 3090, 64×64×32:

| Section | ms/step | % of step |
|---------|---:|---:|
| `evolve.output_momentum_anisotropy_vs_tau` (host, every step) | 8.6 | **~52%** |
| `evolve.AdvanceRK` (host-side GPU pipeline) | 4.7 | ~28% |
| ├── `advance.d2h_copyback` | 2.0 | ~12% |
| ├── `advance.dispatch_wait` (GPU kernels) | 0.9 | ~5% |
| └── `advance.upload_or_rotate` | ~0 | (Phase A eliminated) |
| `evolve.check_conservation_law` | 1.0 | ~6% |
| `evolve.max_energy_density` (GPU reduce) | 0.08 | ~0.5% |

The biggest single per-step cost was a **serial CPU loop** —
`output_momentum_anisotropy_vs_tau` was the one every-step host diagnostic
in `grid_info.cpp` that hadn't been OpenMP-reduced (unlike
`check_conservation_law` and `compute_angular_momentum` which already were).
A one-pragma fix — `#pragma omp parallel for collapse(2) reduction(+:…)` on
both passes (centroid + moments), with the three `std::vector<double>`
accumulators converted to plain C arrays so OpenMP 4.5 array-section
reduction applies — drops the function from **8.6 ms → 0.74 ms per step
(11.6×)** on 48 CPU threads, with the 1-thread vs 48-thread output
bit-identical (2109 values, max abs diff 0.0).

| Grid | Phase 6 only ms/step | + OpenMP fix ms/step | vs 48-thread CPU |
|------|---------------------:|---------------------:|-----------------:|
| 128×128×1 (2D) | 11.15 | **3.61** | 1.77× → **2.87×** |
| 64×64×16 (3D) | 13.36 | **3.28** | 2.55× → **7.65×** |
| 64×64×32 (3D, 131k) | 13.44 | **5.26** | 4.17× → **9.13×** |

The CPU reference also benefited (56 → 48 ms/step at 64×64×32) since the
same serial loop was the bottleneck there too; the GPU speedup ratio grows
because the GPU side improved more than the CPU side did.

### Knob: `output_diagnostics_every_N_timesteps`

In addition to the OpenMP fix above, the per-step summary diagnostics in
`EvolveIt` (the parallelized `output_momentum_anisotropy_vs_tau`, plus
`check_conservation_law` on 3D runs and the vorticity-block outputs if
`output_vorticity == 1`) are now gated by a new input parameter:

```
output_diagnostics_every_N_timesteps 1    # default — every step, legacy behavior
output_diagnostics_every_N_timesteps 10   # cuts the host-side diagnostic cost ~10×
```

These outputs are summary statistics (single line per τ to a few text
files), so the τ-resolution is much finer than physically meaningful for
most analyses.  Setting `N = 10` is a no-risk way to recover the remaining
per-step host cost when high-cadence diagnostic sampling is not needed.

Measured on RTX 3090, init-corrected (same methodology as
`tests/cuda_perstep_bench.sh`):

| Grid | N=1 (default) | N=10 | Additional saving | vs 48-thread CPU at N=10 |
|------|--------------:|-----:|------------------:|-------------------------:|
| 64×64×16 (3D) | 2.97 ms/step | **2.53 ms/step** | ~15% | 9.6× |
| 64×64×32 (3D, 131k) | 5.00 ms/step | **3.54 ms/step** | ~29% | **13.6×** |
| 128×128×1 (2D, boost-inv) | 3.49 ms/step | (at noise floor) | — | — |

The 2D boost-invariant grid sees no measurable gain because
`check_conservation_law` is skipped on boost-invariant runs (only the
~0.74 ms/step momentum-anisotropy call is gated, and that's already inside
the bench script's per-step noise band on the small 2D grid).

The evolution itself is unaffected; this only changes how often the
diagnostic files are written.  Default value (1) preserves the legacy
every-step behavior so existing inputs are unchanged.

### When to combine the two

- **Default config (N=1):** OpenMP parallelization already gives the
  ~11.6× drop on the diagnostic; per-step is 5.26 ms at 64×64×32.
  Recommended for development / validation runs where every-step physics
  monitoring is wanted.
- **Production runs (N=10):** combines OpenMP + frequency-gate; the
  diagnostic essentially disappears from the per-step budget.  The
  evolution output (controlled by the separate
  `output_evolution_every_N_timesteps`, see below) is independent.

### Phase C — full D2H elimination (future)

The plan's full vision (§3) is to drop the per-step D2H copy-back and sync
host arenas only when needed (output cadence, freeze-out).  After the
OpenMP fix above, the profile says this is now worth roughly:

- D2H copy-back per step: ~1.0 ms — i.e. an additional ~20% per-step win at
  64×64×32 (5.26 → ~4.3 ms/step) if the every-step host arena reads can
  also be eliminated.
- `output_momentum_anisotropy_vs_tau` is no longer the blocker (0.74 ms/step
  serial-equivalent, fits easily under a per-step sync budget).
- `check_conservation_law` (3D-only): 0.45 ms/step, similar story.

So Phase C is now a modest, well-scoped optimization rather than an
end-to-end blocker.  All the *plumbing* needed for it is already in place —
`gpu_owns_state()`, the GPU max-reduction, and the `swap_curr_future()`
mirror — so the change reduces to wiring an `Advance::sync_host_from_gpu(current[, prev])`
call into the relevant `EvolveIt` blocks and removing the unconditional
copy-back in `AdvanceIt`.  It remains deferred to a follow-up.

---

## Summary across phases (64×64×32, GB10 vs 20-thread CPU)

| Phase | Change | Per-step speedup | Key kernel-level result |
|-------|--------|-----------------:|-------------------------|
| 1 | Direct port, managed memory | 2.40× (wall) | baseline |
| 2 | `__ldg` read-only cache + adaptive/occupancy block dims | 5.21× (per-step) | fixed 4× 2D eta-lane waste |
| 3 | Shared-memory tiling of `gpu_make_w_source` | 5.50× | w_source −16% |
| 4 | Dual-stream scaffolding (memory-model gated) | 4.90× (neutral) | no-op on coherent GB10 |
| 5 | `--use_fast_math` | 5.29× | `delta_qi` −51% (2.04×) |
| 6 | GPU-resident state (skip H2D re-upload) + GPU max-reduction | 4.17× (RTX 3090) | 27% per-step on discrete; ≈neutral on coherent GB10 |
| 6b | OpenMP-parallelize `output_momentum_anisotropy_vs_tau` (profile-driven) | **9.13× (RTX 3090)** | host function 8.6 → 0.74 ms/step (11.6×); 1T=48T bit-identical |
| 6c | Knob: `output_diagnostics_every_N_timesteps` (default 1) | **13.6× at N=10 (RTX 3090)** | additional 29% per-step at 64×64×32 when per-step diagnostic resolution isn't needed |

(Per-step figures carry ≈±10% run-to-run noise; the kernel-level `nsys` numbers
are the reliable per-optimization signal.) Correctness holds throughout: max
rel `eps_max` error ≈ 3e-5, far inside the 1e-3 regression threshold.

### Comparison to the Metal baseline

The README Metal baseline (M3 Max, 64×64×32, 40 steps) was **GPU 1.62 s vs CPU
12T 2.43 s → 1.50×**. The CUDA port on GB10 reaches **~5.3× per-step vs a
20-thread CPU**, with the dominant reconstruction kernel running in ~148 µs.
The two machines and CPU thread counts differ, so this is not a controlled
Metal-vs-CUDA comparison — but the CUDA back-end clearly clears the Metal
baseline's speedup ratio.

### The headline engineering finding

At production grid on a coherent Grace-Blackwell superchip, **MUSIC's hydro
kernels are not the bottleneck** — they are ~5.5% of wall time.  The port is
correct and the kernels are well-optimized (delta_qi halved, w_source tiled),
so further end-to-end speedup required attacking the **host-side** cost.
Phase 6 implemented Phases A+B of the GPU-resident-state plan
([`Plan-GPU-resident-state.md`](Plan-GPU-resident-state.md)): the H2D
re-upload at the start of each rk0 is eliminated, and `eps_max`/`rhob_max` are
reduced on the GPU.  On the discrete RTX 3090 this delivers a 27% per-step
win at 64×64×32 (3.22× → 4.17× vs 48-thread CPU).  On coherent GB10 the same
change is correctness-neutral and ≈ performance-neutral, because no PCIe
transfer ever happened.

The bigger lesson from instrumenting the loop after Phase 6 was that the
*next* bottleneck wasn't any GPU detail at all — it was a single serial
host function, `output_momentum_anisotropy_vs_tau`, running ~8.6 ms/step
(half the loop) while every other every-step diagnostic in the same file
already had an OpenMP reduction.  Adding the same pragma there (and
converting three `std::vector<double>` accumulators to plain arrays so
OpenMP 4.5 array-section reduction applies) is a one-day-of-a-PhD-student
change that drops the function 11.6× and the whole step 2.5× to **5.26
ms/step, 9.13× vs CPU**.  Worth recording as a process point: GPU porting
buys you nothing while a CPU diagnostic is single-threaded.

Phase C — eliminating the D2H copy-back too — is now a modest follow-up
worth ~20% per-step on top of the above, with all required plumbing
(`gpu_owns_state()`, GPU reduction, `swap_curr_future()`) already in place.
It is deferred but no longer load-bearing.

**Multiple GPUs.** Scaling to several discrete GPUs is feasible — the structured
grid suits η-slab domain decomposition with a 2-cell halo exchange — but it is
gated on the above: a single GPU must first be compute-bound (otherwise N GPUs
sit idle), and for event-by-event production, one-event-per-GPU is simpler and
scales better than decomposing a single grid. The design and prerequisite
ordering are recorded in **[`Plan-multi-GPU.md`](Plan-multi-GPU.md)**.

---

## Running on a discrete GPU (A100 / RTX / H100)

The port was developed and benchmarked on a coherent-memory GB10. Everything
here also targets ordinary **discrete** NVIDIA GPUs. **No source changes are
required to run correctly** — the kernels are architecture-agnostic (sm_70+)
and all buffers use `cudaMallocManaged`, which is supported on every CUDA GPU.

### Required: pick the target architecture

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which detects the GPU **at
configure time** — correct only if you build *on* the discrete-GPU machine. If
you cross-compile (e.g. on a login node without the GPU attached), set it
explicitly:

```bash
cmake -S . -B build_cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80      # 80=A100, 86=RTX30xx, 89=RTX40xx, 90=H100
cmake --build build_cuda -j$(nproc)
```

(The `-DCMAKE_DISABLE_FIND_PACKAGE_GSL=ON` flag used elsewhere in this document
was only a workaround for this machine's conda include path — drop it if GSL
resolves normally on the target.)

### Works automatically — nothing to change

- **Phase 4 dual-stream activates by itself.** The runtime coherence check
  (`cudaDevAttrPageableMemoryAccess` / `prop.integrated`) returns *false* on a
  discrete GPU, so `coherent_memory_ = false` and `upload_snapshots_async()`
  runs its prefetch + event-gate path (skipped on GB10). This is exactly the
  case the dual-stream scaffolding was built for.
- **Shared-memory tiling fits.** The tiled `gpu_make_w_source` caps at 45.6 KB
  of dynamic shared memory (3-D worst case, `(8+2)(8+2)(4+2)` × 19 floats),
  under the 48 KB default carveout on every architecture — no opt-in needed.
- `__ldg`, `--use_fast_math`, the occupancy-tuned block sizing, and the
  256-thread blocks are all architecture-agnostic.

### Memory back-end: explicit device buffers + pinned staging (implemented)

The SoA snapshot buffers use **two memory back-ends, chosen at runtime** from
the device's coherence capability (`cudaDevAttrPageableMemoryAccess` /
`prop.integrated`, surfaced as `g_cuda_coherent`):

- **Coherent (GB10):** snapshots are `cudaMallocManaged`; `copy_to_gpu` packs
  AoS→SoA straight into them and the kernels read in place — zero copy.
- **Discrete (A100/RTX/H100):** snapshots are device-resident `cudaMalloc`;
  `copy_to_gpu` packs into **pinned** host staging (`cudaHostAlloc`), and the
  data moves by explicit `cudaMemcpyAsync` — H2D on `copy_stream_` in
  `upload_snapshots_async` (gated to the compute stream via the event), D2H in
  the copy-back. This avoids the per-fault managed-memory migration over PCIe
  that the all-managed design would otherwise incur every substep.

The non-snapshot scratch/output buffers (`dwmn`, `qi_out`, `uwrhs_out`,
`uprhs_out`, `qi_source_buf`, `theta`/`a`/`sigma`, EOS tables) stay
`cudaMallocManaged` in both modes: they are device-resident in the fully-GPU
path (never host-touched, so they migrate once and stay) yet remain
host-accessible for the partial-GPU fallback configurations. The snapshots —
the bulk of the per-step host↔device traffic — are what the discrete path
optimizes.

The selection is automatic; **no flag or code change is needed on a discrete
GPU**. The hydro-source pre-pass (which reads the cell 4-velocity on the host)
uses `host_readable_u_curr()`, returning the pinned staging on the discrete path
and refreshing it after a snapshot rotation (`refresh_u_curr_stage`).

**Validation.** A diagnostic override `MUSIC_CUDA_FORCE_DISCRETE=1` forces the
discrete path on coherent hardware so it can be tested where no discrete GPU is
available. On GB10 the forced-discrete path is **bit-identical** to the
zero-copy path (max rel `eps_max` error vs coherent = 0.0e0; vs CPU = 2.7e-5,
PASS across all 3D grids) — confirming the device-buffer + DMA plumbing is
correct. On a true discrete GPU this is the path that runs by default.

```bash
# exercise the discrete memory path explicitly (e.g. for testing on GB10):
MUSIC_CUDA_FORCE_DISCRETE=1 OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench_3d.sh
```

> **Caveat — performance is unvalidated on real discrete hardware.** Only the
> *correctness* of the discrete path has been verified, on GB10. Its *timing*
> has **not** been measured on an actual discrete GPU. The forced-discrete runs
> above still execute on GB10, where the `cudaMemcpyAsync` transfers traverse
> the fast coherent NVLink-C2C link rather than PCIe — so their wall times are
> **not representative** of discrete-GPU behavior and are not reported here. The
> design avoids per-fault managed-memory migration by construction, but the
> real PCIe-bound speedup on an A100/RTX/H100 remains to be benchmarked on that
> hardware.

### Recommended workflow on a discrete GPU

1. Build with the correct `CMAKE_CUDA_ARCHITECTURES` and run the correctness
   guard: `OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench.sh`
   (accept < 1e-3).
2. Measure per-step cost: `bash tests/cuda_perstep_bench.sh`.
3. Profile to confirm where the time goes:
   `nsys profile --stats=true ./build_cuda/src/MUSIChydro <input>`.
   The explicit device-buffer + pinned-staging path is already selected
   automatically on a discrete GPU (no managed-memory migration on the
   snapshots). A discrete GPU also has far more FP32 throughput than GB10, so
   the Phase-5 fast-math win on `delta_qi` (and the compute-bound kernels
   generally) should translate into a larger share of the end-to-end speedup
   there than it does on the host-bound GB10. If profiling still shows host↔
   device cost dominating, the next lever is the scratch/EOS buffers (currently
   managed) and keeping evolving state GPU-resident across timesteps.

---

## Running independent events, one per GPU (event-level parallelism)

For event-by-event production the simplest and best-scaling way to use multiple
GPUs is **one independent MUSIC process per GPU** — no halo exchange, near-linear
scaling, and **no code change**. Device selection uses the standard CUDA
`CUDA_VISIBLE_DEVICES` variable: it exposes one physical GPU per process, which
the binary's `cudaSetDevice(0)` then picks up. (Decomposing a *single* event
across GPUs is a separate, heavier effort — see
[`Plan-multi-GPU.md`](Plan-multi-GPU.md).)

```bash
# 1. See what GPUs are available
nvidia-smi -L                 # lists "GPU 0: ...", "GPU 1: ...", with UUIDs

# 2. Launch two independent events, pinned to different GPUs.
#    - separate working dirs so their output files don't collide
#    - split CPU cores so the host-side OpenMP work doesn't oversubscribe
mkdir -p run_A run_B

( cd run_A && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=10 \
    ../build_cuda/src/MUSIChydro ../input_eventA > log_A.txt 2>&1 ) &

( cd run_B && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10 \
    ../build_cuda/src/MUSIChydro ../input_eventB > log_B.txt 2>&1 ) &

wait
```

Each process sees exactly one GPU (its assigned device, reported as logical
"device 0") and runs a full, independent hydro evolution; the two never
interact.

**Notes**
- **Choosing the GPU:** `CUDA_VISIBLE_DEVICES=N` makes physical GPU `N` the only
  one visible to that process; the hardcoded `cudaSetDevice(0)` then selects it,
  and `cudaGetDeviceCount` returns 1 inside the process. To target a specific
  card by identity rather than index, use its UUID:
  `CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-...`.
- **Verify the pinning:** each process prints `[MUSIC-GPU] CUDA device: <name>`
  at startup; `nvidia-smi` during the run shows one process on each GPU.
- **Separate output directories** (`run_A` / `run_B` above): MUSIC writes output
  into the working directory, so two instances in the same CWD would overwrite
  each other.
- **CPU oversubscription:** both processes use OpenMP for the host-side pack and
  diagnostics. On an `M`-core node running `P` processes, set
  `OMP_NUM_THREADS = M / P` (e.g. 20 cores, 2 processes → 10 each).
- **More events than GPUs:** assign round-robin,
  `CUDA_VISIBLE_DEVICES=$((i % NGPU))`, and keep `NGPU` events in flight (e.g.
  via GNU `parallel` or any job queue). CUDA **MPS** is an alternative if you
  want several events to share one GPU concurrently.

> The hydro evolution does not saturate a single GPU at production grid sizes
> (it is host-bound — see the host-bound finding above), so packing one event
> per GPU is also a good way to raise aggregate GPU utilization on a multi-GPU
> node.

### Packing several events onto one GPU with CUDA MPS

Because a single event leaves the GPU ~83% idle (kernels are ~17% of the step —
see Phase 6), more than one event can share a GPU.  Two ways to do it:

1. **Just launch concurrent processes** (no MPS).  The OS time-slices the GPU
   between each process's CUDA context.  For this host-bound workload that
   already fills most of the idle gaps (see the measurements below).
2. **CUDA MPS (Multi-Process Service).**  A per-user daemon that lets multiple
   processes share *one* GPU context, so their kernels can run *concurrently*
   on the SMs instead of being time-sliced.  It pays off when GPU kernels are
   the contended resource; for MUSIC at production grid they are not (below).

#### Step-by-step: MPS

MPS is a per-user daemon; **no root and no compute-mode change is required** for
basic use (a single cooperating set of processes in the GPU's `Default` compute
mode).  From a shell:

```bash
# 1. Point MPS at per-user pipe/log dirs (avoids clashing with other users).
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$USER/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_$USER/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# 2. Start the control daemon (forks into the background).
nvidia-cuda-mps-control -d

# 3. (optional) sanity-check it is up.
echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control   # prints 100.0

# 4. Launch the events normally.  Any process whose CUDA_MPS_PIPE_DIRECTORY
#    matches the daemon auto-routes through MPS.  Separate working dirs
#    (MUSIC writes output into CWD); split the cores; pin the GPU.
CORES=$(nproc); N=8
for i in $(seq 1 $N); do
  mkdir -p run_$i
  ( cd run_$i && CUDA_VISIBLE_DEVICES=$(( (i-1) % 2 )) \
        OMP_NUM_THREADS=$(( CORES / N )) \
        ../build_cuda/src/MUSIChydro ../input_$i > log.txt 2>&1 ) &
done
wait

# 5. Shut the daemon down when done.
echo quit | nvidia-cuda-mps-control
```

While running, `nvidia-smi` shows a single `M+C` (MPS server) process holding
the GPU with the clients underneath it.  For strict single-tenant nodes you can
additionally `sudo nvidia-smi -c EXCLUSIVE_PROCESS` so *all* work is forced
through MPS — not needed for a cooperating batch in `Default` mode.

#### Measured throughput (2× RTX 3090, 48 cores)

Each "event" is the 64×64×32 / 110-step benchmark.  Single event (48 threads) =
**0.91 s**, of which init (CUDA context + EOS sampling) is ~0.56 s — so these
short events are **62% init**, which *understates* the steady-state gain a real
(thousands-of-steps) event would see.  Throughput gain = (N × single-event
time) / (wall time for N concurrent):

| Config | wall (N events) | throughput gain |
|--------|----------------:|----------------:|
| N=4, 1 GPU, no MPS | 1.59 s | 2.29× |
| N=4, 1 GPU, **MPS** | 1.64 s | 2.21× |
| N=8, 1 GPU, no MPS | 2.52 s | 2.89× |
| N=8, 1 GPU, **MPS** | 2.48 s | 2.93× |
| N=8, **2 GPUs** (4+4), MPS | 2.62 s | 2.78× |

Two honest findings the measurement forces:

- **MPS adds essentially nothing here** (2.21× vs 2.29×, 2.93× vs 2.89× — within
  run-to-run noise).  The throughput win comes from *concurrency itself*, which
  plain process launching already delivers.  MPS's specialty — concurrent
  kernel co-residency — only helps when GPU kernels are the contended resource,
  and for MUSIC at this grid they are not.
- **The second GPU did not help at N=8** (2.78× on 2 GPUs vs 2.93× on 1).  This
  is direct confirmation that the GPU is *not* the bottleneck even when packed:
  the binding resources are the **48 CPU cores** (host-side pack + diagnostics)
  and the **per-process CUDA init** (largely driver-serialized).  Spreading the
  kernels over two GPUs cannot speed up work that is waiting on the CPU.

#### When MPS / the second GPU *do* pay off

- **Longer events.** At thousands of steps, init (~0.5 s) is negligible and the
  steady-state evolution dominates, so the per-event GPU-idle gaps are what's
  being filled — the gain rises toward the GPU-utilization ceiling (~1/0.17 ≈
  6× per GPU) until the CPU-core budget binds.
- **Bigger grids / heavier configs** (128³, full shear+bulk+diffusion): the
  kernels become a larger share of the step, GPU-kernel contention becomes real,
  and MPS's concurrent execution + the second GPU both start to matter.
- **More CPU cores per GPU.** The host-bound ceiling here is set by 48 cores
  feeding 2 GPUs; a node with a higher core-to-GPU ratio reaches GPU saturation
  (where MPS helps) before the CPU saturates.

**Bottom line for production today:** on this box, just launch `N≈6–8` concurrent
events round-robin across the two GPUs with `OMP_NUM_THREADS = 48/N` and separate
working dirs — that already gives ~3× aggregate throughput. MPS is a no-cost
add-on (start the daemon, no code change) that becomes worthwhile as event
length and grid size grow, not at this short-event / production-grid operating
point.
