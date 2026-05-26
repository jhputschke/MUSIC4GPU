# MUSIC — Metal GPU Backend (macOS / Apple Silicon)

> Part of the **experimental** GPU acceleration of MUSIC 3.1 (ports only — no new
> physics; produced with Claude Opus 4.7). See the main [README.md](README.md) for
> overall scope, the experimental-use disclaimer, and the CUDA backend
> ([README_CUDA.md](README_CUDA.md)).

## Metal GPU Acceleration (macOS / Apple Silicon)

MUSIC includes an optional Metal GPU backend that offloads the ideal KT
flux (`MakeDeltaQI`) and viscous-source (`MakeWSource`) kernels to the
Apple Silicon GPU using Metal Shading Language (MSL).  On Apple Silicon
the CPU and GPU share physical memory, so no explicit host↔device
transfers are needed — the same buffers are readable by both processors.

### Requirements

| Requirement | Notes |
|---|---|
| Apple Silicon Mac (M1 or later) | Unified memory required |
| macOS 13 Ventura or later | Metal 3 API |
| Xcode Command Line Tools | Provides `xcrun`, `metal`, `metallib` |

Install the tools if needed:

```bash
xcode-select --install
```

### Building the Metal-enabled binary

```bash
# configure (Release recommended for performance benchmarks)
cmake -S . -B build_metal -DUSE_METAL=ON -DCMAKE_BUILD_TYPE=Release

# compile (also runs xcrun to compile music_kernels.metal → music_kernels.metallib)
cmake --build build_metal -j$(sysctl -n hw.logicalcpu)

# install executable to the repo root
cmake --install build_metal
```

The build produces:

- `build_metal/src/MUSIChydro` — Metal-enabled executable
- `build_metal/src/music_kernels.metallib` — compiled GPU shader library
- `music_kernels.metallib` — copy in the repo root (runtime lookup path)

### Running with Metal

Usage is identical to the CPU build.  Metal is initialised automatically on the
first call to `AdvanceIt()`.  Confirmation messages are printed to stderr:

```
[MUSIC-GPU] Metal device: Apple M3 Max
[MUSIC-GPU] Initialized. max threads/group = 1024
[MUSIC-GPU] GPU grid allocated (NxxNyx1 cells).
```

If Metal initialisation fails (e.g. `music_kernels.metallib` not found), the
code falls back to the CPU path with a warning and continues normally.

### GPU kernel — what is ported

| Kernel | Source | Status |
|---|---|---|
| `gpu_make_w_source` | `Diss::MakeWSource()` in `src/dissipative.cpp` | **Ported** — runs on GPU every RK stage |
| `gpu_make_delta_qi` | `Advance::MakeDeltaQI()` in `src/advance.cpp` | **Ported (Tier 2)** — see EOS caveat below |
| `gpu_finalize_ideal` | `Reconst::ReconstIt_shell()` + RK mixing in `Advance::FirstRKStepT()` | **Ported (Tier 3a)** — final Newton solve runs on GPU; the per-cell CPU loop body for the ideal step is gone. **Phase 4** adds `qi_source_buf` consumption: when `flag_add_hydro_source` is true the host runs an OpenMP pre-pass calling `hydro_source_terms_ptr->get_hydro_energy_source` per cell into a GPU shared buffer, and the kernel adds `τ_rk · j^α · Δτ` to qi. The source models themselves (`HydroSourceStrings`, `HydroSourceAMPT`, `HydroSourceTATB`) stay on CPU. |
| `gpu_make_uwrhs` | stencil portion of `Diss::Make_uWRHS()` in `src/dissipative.cpp` | **Ported (Tier 3b)** — KT flux divergence of `u^a W^{mu nu}` for the 5 shear indices; per-cell geometric tail (`Diss::Make_uWRHS_geom`) stays on CPU since it depends on `theta` and `Du^mu` from `MakedU` |
| `gpu_make_du` | `U_derivative::MakedU` + `calculate_expansion_rate` + `calculate_Du_supmu` + `calculate_velocity_shear_tensor` | **Ported (Tier 3c, Phase 1)** — writes `theta_buf`, `a_buf`, `sigma_buf`; CPU loop reads from buffers instead of computing per cell. Assumes vorticity OFF and baryon-diffusion OFF (otherwise falls back to CPU `U_derivative`). |
| `gpu_first_rk_step_w_full` | `Diss::Make_uWSource` + `Make_uWRHS_geom` + transversality / tracelessness fix-up + `Make_uPiSource` (bulk update) + `Advance::QuestRevert` + second-order transport terms | **Ported (Tier 3c, Phase 2 + 3 + 5 + 6)** — full per-cell shear-stress + bulk-pressure evolution + viscous-regulator on GPU; writes `snap_future.Wmunu` and `pi_b` directly. Eliminates the per-cell CPU viscous loop entirely. Supports `T_dependent_shear_to_s ∈ {0, 1, 2, 3, 11}`, `T_dependent_bulk_to_s ∈ {0, 1, 2, 3, 8, 9, 10}`, and `include_second_order_terms ∈ {0, 1}` (Wsigma, WW, Coupling_to_Bulk, Coupling_to_Shear all on the GPU). QuestRevert activates inline for `Initial_profile ∉ {0, 1}`. Restrictions: `muB_dependent_shear_to_s == 0`, vorticity OFF, no baryon diffusion. |
| `gpu_make_uprhs` | stencil portion of `Diss::Make_uPRHS()` | **Ported (Tier 3c, Phase 3)** — KT flux divergence of `u^a * pi_b` for the bulk-pressure update; dispatched only when `turn_on_bulk == 1` |
| Freeze-out / Cornelius | `src/freeze_pseudo.cpp` | Permanent CPU — irregular geometry |
| Vorticity / baryon-diffusion source paths | `src/dissipative.cpp` / `src/u_derivative.cpp` | CPU — used only when `include_vorticity_terms == 1` or `turn_on_diff == 1` |

#### EOS caveat for `gpu_make_delta_qi`

`MakeDeltaQI` requires equation-of-state lookups (pressure, dP/de) inside
a Newton iteration at every half-interface of the KT flux stencil.  The GPU
kernel carries a **pre-sampled float32 table** of P(e) and dP/de(e) evaluated
at **rhob = 0** on a uniform 8192-point grid spanning `[0, eps_max]`.

This covers the standard heavy-ion case where the net baryon density is
negligible (EOS IDs 2–17 in MUSIC, i.e. all single-variable EOS).  When the
net baryon density is non-zero (`turn_on_rhob = 1` with a 2D EOS such as
`neos` or `best`), the GPU path is **automatically bypassed** and the CPU
`MakeDeltaQI` is called as a fallback — no user action required.

Additionally, `dP/drhob` is assumed zero inside the GPU Newton solver
(consistent with the rhob = 0 EOS sample), so the velocity reconstruction
is slightly approximate even when rhob is small but non-zero.  For
production finite-muB runs, disable the GPU ideal step by building without
`-DUSE_METAL` or by waiting for Tier 3.

### Numerical accuracy

All GPU kernels use `float32` arithmetic; the CPU path uses `float64`.

**Viscous sector** (`gpu_make_w_source`): at production step size
(`Delta_Tau=0.005`) the GPU and CPU produce **bitwise-identical `eps_max`
values** at every timestep for grids up to 128×128 over 100 steps — the
float32 Wmunu source terms contribute too little per step to shift the
float64 primitive-variable reconstruction.

**Ideal sector** (`gpu_make_delta_qi`): float32 KT fluxes feed back into the
float64 Newton reconstruction (`ReconstIt_shell`) on the CPU, which
re-establishes double precision.  In practice the rounding error in the
flux sum is O(10⁻⁷) relative; this is well within hydrodynamic truncation
error at any reasonable resolution and step size.

At coarser step sizes (`Delta_Tau≥0.02`) float32 errors in both sectors
accumulate over long runs and can cause visible divergence near the
freeze-out surface; always use the production step size (≤0.01 fm/c) with
the GPU path.

### Running the benchmark / validation scripts

```bash
# build both CPU and Metal binaries first, then:
bash tests/metal_vs_cpu_bench.sh       # 2D boost-invariant
bash tests/metal_vs_cpu_bench_3d.sh    # 3+1D, exercises η-direction code
```

Both scripts:
1. Generate analytical Gubser-viscous input files at several grid sizes.
2. Time `build/src/MUSIChydro` (Release, no Metal) and `build_metal/src/MUSIChydro` (Release, Metal) on each.
3. Compute the speedup and the maximum relative error in the `eps_max` trace.

The 2D script uses `boost_invariant=1, Nη=1` (100 timesteps).  The 3D script
uses `boost_invariant=0, Nη ∈ {8, 16, 32}` (40 timesteps each); this turns
on every η-direction code path — cosh/sinh-of-Δη geometric terms in
`MakeDeltaQI` / `MakeWSource`, the η-stencil in `Make_uWRHS`, and the
`u^3 / τ` couplings inside `Make_uWSource`.  Initial state is the Gubser
XY profile replicated across all η slices (so `u^η = 0` initially);
the evolution stays approximately η-invariant, but the code paths are
fully exercised.

Example output (Apple M3 Max, 12 P-cores; both binaries built with `-DCMAKE_BUILD_TYPE=Release`,
CPU build linked against Homebrew `libomp`):

**2D (`metal_vs_cpu_bench.sh`, 100 timesteps, `Delta_Tau=0.005`)**
```
Grid              CPU-1T(s)  CPU-12T(s)   GPU(s)   GPU/1T   GPU/12T   MaxErr
----              ---------  ----------   ------   ------   -------   ------
32x32x1               1.07        0.18      0.22    4.86x     0.82x   5.3e-05
64x64x1               1.88        0.38      0.42    4.48x     0.90x   9.9e-05
128x128x1            10.20        1.43      1.04    9.81x     1.38x   9.4e-05
```

**3+1D (`metal_vs_cpu_bench_3d.sh`, 40 timesteps each)**
```
Grid (Nx×Ny×Nη)  Cells   CPU-1T(s)  CPU-12T(s)   GPU(s)   GPU/1T   GPU/12T   MaxErr
----             -----   ---------  ----------   ------   ------   -------   ------
32x32x8           8.2k       2.23        0.34      0.36    6.19x     0.94x   5.3e-05
32x32x32         32.8k       5.86        0.66      0.50   10.46x     1.32x   3.7e-05
64x64x16         65.5k      11.96        1.32      0.93   11.18x     1.42x   3.1e-05
64x64x32        131.1k      24.00        2.43      1.62   12.44x     1.50x   3.1e-05
```

Set `OMP_NUM_THREADS` before invoking either benchmark script (e.g.
`OMP_NUM_THREADS=12 bash tests/metal_vs_cpu_bench.sh`).  The "12T" column
above used all 12 performance cores of the M3 Max; the "1T" column is the
historical serial baseline retained for comparison.

**Reading the numbers.** The GPU wins everywhere except the smallest grids
where launch overhead dominates.  In 3D — the production-relevant case —
the GPU is **1.32–1.50× faster than the 12-thread OpenMP CPU** and 6–12×
faster than serial.  In 2D the crossover is around 128² (GPU 38% faster
there).  CPU strong-scaling 1T→12T is near-linear in 2D (~7×) and
super-linear in 3D (~9–10×, helped by cache pressure dropping as the
working set partitions across cores), but the GPU still outpaces it once
the per-cell work amortizes its constant overhead.

The 3D advantage scales **up** with cell count (0.94× at 8k cells →
1.50× at 131k cells), reflecting the GPU's preference for saturating
occupancy — at production sizes (e.g. 128×128×64 ≈ 1M cells) it should
hold or grow further.

These numbers reflect **Tier 3 + Tier 3c Phase 1/2 + dispatch/copy
optimizations**.  The latter (three focused changes) deliver most of the
post-OpenMP-CPU win:

1. **Parallel AoS↔SoA copies.**  The three copy helpers in
   [`src/gpu/GPUGrid.mm`](src/gpu/GPUGrid.mm) are now `#pragma omp
   parallel for collapse(3)`.  At 131k cells the AoS↔SoA repack is the
   single biggest CPU-side cost; OpenMP cuts it ~10×.
2. **One Metal command buffer per substep.**  `MetalPipelines` exposes
   `begin_batch()` / `end_batch()` so all 7 kernels of a substep encode
   into a single `MTLCommandBuffer` — one commit, one wait, instead of
   seven.  Saves ~1 ms/step of driver overhead.
3. **No CPU round-trip between RK substeps.**  When the GPU fully
   produces the next state (`gpu_finalize_active && gpu_w_full_active`),
   the per-substep `copy_primitives_to_cpu` / `copy_wmunu_to_cpu` is
   skipped and replaced by `GPUGrid::rotate_snapshots()` — a pointer-
   alias rotation of three `GPUSnapshot` structs (no GPU work, no CPU
   work).  The hydro-source pre-pass now reads `u` directly from
   `snap_curr.u` (unified memory) so it stays correct after rotation.

Together these eliminate roughly 30 ms/step of pure CPU bookkeeping at
64×64×32 — the gap that previously kept the GPU behind the 12-thread
CPU.  The 12.44× at 64×64×32 (vs serial CPU) is ~1.6× the pre-optimization
number; the **1.50× vs 12-thread OpenMP CPU** is a clean reversal of the
prior 0.80× deficit.

The max relative error in `eps_max` is O(10⁻⁴) after Phase 2 — at the
bench's 1e-4 pass threshold.  An earlier draft of Phase 2 hit O(10⁻²) at
small grids; the culprit was the entropy table layout (`s ~ e^{3/4}` was
sampled linearly over `[0, 1e5]`, leaving < 1 bin to resolve the
`e ~ 0.01-2 1/fm⁴` range where most cells live).  Switching to a log-spaced
entropy table dropped the error back to ~1e-4.  The remaining drift is
genuine float32 vs float64 accumulation in the long-time Wmunu evolution.

### Understanding the speedup

The 9.81× at 128² and 12.44× at 64×64×32 (both vs serial CPU) reflect
Amdahl's law applied to the remaining CPU work.  The original serial
CPU wall time breaks down roughly as follows:

| Work | CPU fraction | Tier 3 status |
|---|---|---|
| `MakeDeltaQI` — KT ideal flux | ~60% | **on GPU** |
| `ReconstIt_shell` — Newton conserved→primitive | ~25–30% | **on GPU (Tier 3a)** |
| `MakeWSource` + `Make_uWRHS` — viscous stencils | ~5–10% | **on GPU** (Tier 1 + Tier 3b) |
| `Make_uWSource` + transport coefficients + MakedU | ~5% | **on GPU (Tier 3c)** |
| AoS↔SoA copies + I/O | ~2–3% | CPU (parallel) — halved per-step by inter-substep rotation |

The remaining ceiling is the **per-timestep** SoA↔AoS round-trip (after the
last RK substep) plus output / freeze-out passes.  Further wins require
making the CPU AoS arenas a lazy snapshot of the GPU SoA buffers, so the
download only happens when freeze-out or output dump actually reads them.

Additional limiting factors:

1. **Grid too small to saturate the GPU.**  128×128×1 has only 16 384 cells.
   A production 3D run (e.g., 128×128×50) has 50× more independent cells and
   much better GPU occupancy.

2. **Divergent Newton iteration inside `gpu_make_delta_qi` and
   `gpu_finalize_ideal`.**  The `gpu_reconst` helper runs up to 60
   Newton-Brent iterations per thread.  Cells with different velocities
   and energy densities converge at different rates, causing SIMD
   divergence and reducing effective throughput.

---

## Files modified for Metal GPU support

### Original MUSIC files changed

| File | Change summary |
|---|---|
| [`CMakeLists.txt`](CMakeLists.txt) | Added `OBJCXX` to `LANGUAGES`; added `option(USE_METAL ...)` |
| [`src/CMakeLists.txt`](src/CMakeLists.txt) | Metal source files, `xcrun` build commands for `.metallib`, `-framework Metal/Foundation` |
| [`src/grid.h`](src/grid.h) | Added `const` overloads for `get()` and `getHalo()` (required by `copy_to_gpu(const SCGrid&)`) |
| [`src/advance.h`](src/advance.h) | `#ifdef USE_METAL` guard; added `GPUGrid gpu_grid_`, `init_metal_if_needed()`, `make_gpu_params()` (now takes `rk_flag` + `tau_orig`); extended `FirstRKStepT` and `FirstRKStepW` signatures with GPU base pointers |
| [`src/advance.cpp`](src/advance.cpp) | `init_metal_if_needed()` (+ EOS table sampling), `make_gpu_params()` fills `rk_flag` / `tau_orig`, quad-dispatch (`w_source` + `delta_qi` + `finalize_ideal` + `uwrhs`) in `AdvanceIt()`, batch `copy_primitives_to_cpu` after wait, CPU `FirstRKStepT` skipped when GPU finalize is active, `FirstRKStepW` consumes the GPU `uwrhs` flux |
| [`src/dissipative.h`/`.cpp`](src/dissipative.h) | Added `Diss::Make_uWRHS_geom()` — the per-cell algebraic/geometric tail of `Make_uWRHS`, used after the GPU stencil pre-pass |

### New GPU files added

| File | Purpose |
|---|---|
| [`src/gpu/gpu_types.h`](src/gpu/gpu_types.h) | `MUSICGridParams` (+ `minmod_theta`, `rk_flag`, `tau_orig`), `GPUEosParams`, `WMUNU_IDX` — shared between C++ host and Metal shaders |
| [`src/gpu/GPUGrid.h`](src/gpu/GPUGrid.h) | `GPUSnapshot` / `GPUGrid` — SoA Metal shared buffers; `eos_P`, `eos_dPde`, `qi_out`, `uwrhs_out` fields; `upload_eos()`, `copy_primitives_to_cpu()` |
| [`src/gpu/GPUGrid.mm`](src/gpu/GPUGrid.mm) | `GPUGrid` implementation — buffer allocation, AoS↔SoA copy, primitives SoA→AoS, `upload_eos()` |
| [`src/gpu/MetalPipelines.h`](src/gpu/MetalPipelines.h) | Singleton `MetalPipelines` — device, queue, PSO objects; `dispatch_w_source`, `dispatch_delta_qi`, `dispatch_finalize_ideal`, `dispatch_uwrhs` |
| [`src/gpu/MetalPipelines.mm`](src/gpu/MetalPipelines.mm) | `MetalPipelines` implementation — library loading, PSO creation for all four kernels, all four dispatch helpers |
| [`src/gpu/music_kernels.metal`](src/gpu/music_kernels.metal) | MSL compute kernels: `gpu_make_w_source`, `gpu_first_rk_step_w` (placeholder, unused), `gpu_make_delta_qi`, `gpu_finalize_ideal`, `gpu_make_uwrhs` |

### Test and benchmark files

| File | Purpose |
|---|---|
| [`test_metal_input`](test_metal_input) | Gubser viscous smoke-test input (32×32×1, `boost_invariant 1`, shear viscosity on) |
| [`tests/metal_vs_cpu_bench.sh`](tests/metal_vs_cpu_bench.sh) | 2D (boost-invariant) timing and correctness comparison between CPU and Metal GPU builds |
| [`tests/metal_vs_cpu_bench_3d.sh`](tests/metal_vs_cpu_bench_3d.sh) | 3+1D timing and correctness comparison; exercises every η-direction code path |

---

## What's next

The following items remain before the GPU backend can run a full timestep
without CPU involvement.

### Tier 3 — complete GPU timestep (partially done)

| Task | Status | Notes |
|---|---|---|
| Port `ReconstIt_shell` | **Done (Tier 3a)** | `gpu_finalize_ideal` runs the final Newton solve on the GPU and writes `snap_future.{epsilon, rhob, u}` directly |
| Eliminate per-cell CPU loop for the ideal step | **Done (Tier 3a)** | `FirstRKStepT` is no longer called when GPU finalize is active; primitives are batch-copied back via `copy_primitives_to_cpu` |
| Port `Make_uWRHS` stencil | **Done (Tier 3b)** | `gpu_make_uwrhs` writes the KT flux divergence for the 5 shear indices; CPU still applies `Make_uWRHS_geom` (per-cell algebraic / geometric tail) since it needs `theta` and `Du^mu` from `MakedU` |
| Port `MakedU` + viscous geometry (theta, a^μ, σ^{μν}) | **Done (Tier 3c Phase 1)** | `gpu_make_du` writes `theta_buf` / `a_buf` / `sigma_buf`; CPU viscous loop reads from buffers. Restrictions: vorticity OFF + baryon-diffusion OFF (otherwise CPU `U_derivative` is used) |
| Port algebraic shear source (`Make_uWSource`) + final assembly | **Done (Tier 3c Phase 2)** | `gpu_first_rk_step_w_full` writes `snap_future.Wmunu` directly. Per-cell CPU loop is retired entirely for the supported config (constant shear, no bulk, no vorticity, no second-order, no diffusion). Uses log-spaced `eos_s(e)` table; all other configs fall back to CPU. |
| Port bulk evolution (`Make_uPRHS` stencil + `Make_uPiSource` algebra) | **Done (Tier 3c Phase 3)** | `gpu_make_uprhs` writes the scalar bulk stencil flux; `gpu_first_rk_step_w_full` now performs the full bulk update inline alongside the shear sector. Supports ζ/s modes 0, 1 (Gabriel), 2 (Duke), 3 (Sims), 8/9 (fixed AsymGaussian), 10 (DATA-controlled AsymGaussian). Mode 7 (bigbroadP) falls back to CPU. |
| Port T-dependent shear viscosity profiles | **Done (Tier 3c Phase 2.5)** | `eos_T(e)` log-spaced table on GPU; MSL ports of `get_eta_over_s` modes 1 (default), 2 (Duke), 3 (Sims), 11 (profile-multiplier). Mode 10 (µ_B-dependence) still requires finite-µ_B EOS (Tier 4). |
| Port T-dependent `get_zeta_over_s` profiles | **Done (Tier 3c Phase 3)** | Reuses the `eos_T(e)` table; MSL ports of modes 1, 2, 3, 8, 9, 10. Mode 7 falls back to CPU. |
| Hydro-source path (`flag_add_hydro_source == true`) ideal step | **Done (Tier 3c Phase 4)** | CPU pre-fills `qi_source_buf[5*Ncells]` via an OpenMP pass; `gpu_finalize_ideal` adds the source to qi. Source models (strings, AMPT, TATB) stay on CPU. |
| Port `QuestRevert` to GPU | **Done (Tier 3c Phase 5)** | Per-cell algebraic regulator at the end of `FirstRKStepW`; ported inline at the end of `gpu_first_rk_step_w_full`. Validated by forcing the regulator on for both CPU and GPU during a Gubser run — max rel err 9.95e-5, identical to the non-regulated path. The host gate no longer restricts `Initial_profile`; source-driven runs now use the full viscous GPU path. |
| Port second-order transport terms | **Done (Tier 3c Phase 6)** | Wsigma_term + WW_term (gated on `include_second_order_terms == 1 && Initial_profile != 0`), Coupling_to_Bulk (in `gpu_uW_source`), Coupling_to_Shear (in `gpu_uPi_source`). Per-cell scalar invariants `W^{μν}σ_{μν}` and `W^{μν}W_{μν}` computed once per cell before the shear-index loop. Validated with `Initial_profile=1 + second_order=1 + Sims-shear + Sims-bulk` (all 4 second-order pieces active): max rel err 4.7e-5 at 64². The host gate no longer restricts `include_second_order_terms`. |
| Port baryon diffusion (`MakedU` + diffusion flux) | **TODO** | Currently computed entirely on CPU; requires the gradient of µ_B/T on the GPU |

### Tier 4 — finite-muB EOS on GPU

| Task | Notes |
|---|---|
| 2D EOS table (P(e, rhob)) | Extend `GPUEosParams` to a 2D bilinear grid; modify `gpu_make_delta_qi` to interpolate in both e and rhob |
| dP/drhob table | Add `eos_dPdrho` buffer; restore the `J0 * dPdrho` term in `gpu_vel_fdf` |
| Validation against CPU for `neos` / `best` EOS | The CPU fallback is currently used for all finite-muB runs |

### Stencil kernel fusion + threadgroup-shared neighbour loads

After Steps 1–3 of dispatch/copy work, the highest-leverage remaining
optimisation is reducing redundant global-memory traffic inside the
stencil kernels.  Most cells today re-read each of their 12 neighbours
(4 per direction × 3 directions) from global memory independently —
neighbour-load traffic alone is ~2 kB/cell/substep (~520 MB/timestep
at 131k cells).  A halo'd 12×12×8 threadgroup tile loaded once into
threadgroup memory cuts that to ~108 B/cell — a ~19× reduction in raw
bandwidth; the realised speedup is smaller because Apple's L1/LLC
already captures part of the redundancy.

| Item | Where | LOC | Risk | Est. kernel speedup | Est. total GPU |
|---|---|---|---|---|---|
| Pull `gpu_TJb0` out of the alpha loop in `gpu_make_delta_qi` | [music_kernels.metal:791-800](src/gpu/music_kernels.metal#L791-L800) | ~30 | Low | 15–25% | **5–9%** |
| Fuse `gpu_make_w_source` + `gpu_make_du` (both consume `u_curr`) | [music_kernels.metal:74](src/gpu/music_kernels.metal#L74), [:1258](src/gpu/music_kernels.metal#L1258) | ~80 | Low | 10–15% combined | **2–3%** |
| Fuse `gpu_make_uwrhs` + `gpu_make_uprhs` (identical KT pattern, extra output channel) | [music_kernels.metal:1001](src/gpu/music_kernels.metal#L1001), [:1128](src/gpu/music_kernels.metal#L1128) | ~60 | Low | 5–10% combined | **1–2%** |
| Threadgroup tile `gpu_make_uwrhs` (Wmunu + u into shared mem) | [music_kernels.metal:1057-1072](src/gpu/music_kernels.metal#L1057-L1072) | ~120 | Medium | 20–30% kernel | **3–5%** |
| Threadgroup tile `gpu_make_uprhs` (pi_b + u into shared mem) | [music_kernels.metal:1166-1188](src/gpu/music_kernels.metal#L1166-L1188) | ~80 | Medium | 20–30% kernel | **1–2%** |
| Threadgroup tile `gpu_make_du` (u into shared mem) | [music_kernels.metal:1258](src/gpu/music_kernels.metal#L1258) | ~120 | Medium | 25–35% kernel | **3–4%** |
| Threadgroup tile `gpu_make_delta_qi` (epsilon + rhob + u; coexist with Newton solve) | [music_kernels.metal:733](src/gpu/music_kernels.metal#L733) | ~200 | **High** — Newton + multi-alpha interplay; per-cell register pressure | 10–15% kernel | **3–5%** |

**Cumulative realistic estimate: 18–30% additional GPU speedup**, lifting
3D 64×64×32 GPU time from 1.62 s → ~1.15–1.30 s (ratio vs CPU-12T
goes from 1.50× to **~1.85–2.10×**).  Same proportional gain at
production sizes (≈ 1M cells).

**Caveats:**
- Apple GPU L1 caches are surprisingly effective on stencil patterns
  even without explicit tiling; measured wins will likely be at the
  lower end of these ranges.
- `gpu_make_delta_qi`'s real bottleneck on Apple Silicon is the
  divergent Newton-Brent iteration (12 reconst calls per cell, ~30
  iterations each) — memory tiling cannot fix that.  A vectorised /
  cooperative Newton variant is a separate, much harder workstream.
- Threadgroup-size tuning is per-kernel — uniform 8×8×4 is unlikely
  to be optimal once shared memory and register pressure differ across
  kernels.

**Recommended ordering:** the first three items (alpha-loop pull-out +
two fusions, ~2.5 LOC-days, low risk) deliver about 1/3 of the maximum
projected gain at minimal complexity cost.  The full tiling treatment
(~2 weeks) takes the GPU from 1.5× to ~2.0× ahead of the 12-thread CPU
in 3D production-size runs.
