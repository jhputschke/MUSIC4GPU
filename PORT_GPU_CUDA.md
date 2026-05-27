# CUDA Backend Port Notes (XSCAPE)

Companion to `PORT_GPU.md`.  That document tracks making MUSIC's GPU
backends reachable from the XSCAPE/JETSCAPE `Fields` entry point and was
written and verified on **Metal** (Apple Silicon).  This document tracks
making the **CUDA** backend work on that same Fields path and confirming
it matches the `main_gpu` branch on discrete NVIDIA hardware.

Started: 2026-05-26, on branch `XSCAPE_CUDA` (forked from the tip of
`XSCAPE`, commit `0a35b57`).  Hardware: 2× NVIDIA RTX 3090 (cc 8.6),
CUDA 12.4, driver 550.

---

## 1. Starting symptom

On `XSCAPE`/`XSCAPE_CUDA`, a `-DUSE_CUDA=ON` build *compiled and ran* —
it initialised the GPU correctly:

```
[MUSIC-GPU] CUDA device: NVIDIA GeForce RTX 3090 (cc 8.6, 23.7 GB)
[MUSIC-GPU] coherent host memory: no (device buffers + pinned-staging DMA)
```

…but the physics was garbage: on a 64×64×1 Gubser viscous run the
`eps_max` trace matched the CPU only at the initial condition, then
collapsed:

```
step   CPU eps_max     CUDA eps_max
   0   5.669710e-02    5.669710e-02      <- initial condition (no GPU step yet)
   1   5.501400e-02    1.973000e-09      <- first GPU step: collapse
   2   5.345790e-02    0.000000e+00
```

The Metal build of the *same* Fields path was correct.  This document
explains why, and what was ported/fixed.

PORT_GPU.md §8.3 had already flagged the risk:
> "A CUDA bench on Linux hasn't been re-run on this branch."

The CUDA Fields overloads had been written blind-by-analogy to Metal
and never executed on hardware.

## 2. Root causes

Two bugs.  Both are explained by one fact: **Metal uses coherent unified
memory** (the `GPUSnapshot` `float*` pointers are host-writable, kernels
read them in place), while **a discrete CUDA GPU does not** (snapshot
buffers are `cudaMalloc` device memory; host packing goes into pinned
staging and must be DMA'd across PCIe).  Any code that *only* exercised
the coherent path slipped through Metal testing.

### Bug 1 — missing host→device upload (CUDA-only) — the collapse

`main_gpu`'s SCGrid dispatch packed the host arena into pinned staging
with `copy_to_gpu(...)`, then **moved it to the device**:

```cpp
gpu_grid_.copy_to_gpu(arena_current, gpu_grid_.snap_curr);
gpu_grid_.copy_to_gpu(arena_prev,    gpu_grid_.snap_prev);
#if defined(USE_CUDA)
    GPUPipelines::instance().upload_snapshots_async(gpu_grid_);  // H2D DMA
#endif
```

When the SCGrid `AdvanceIt` was dropped in the `main_gpu`→`XSCAPE`
merge, the new Fields dispatch in `Advance::try_gpu_advance`
(`src/advance.cpp`) kept the two `copy_to_gpu` calls but **lost the
`upload_snapshots_async`**.  On Metal that is a no-op (the copy already
wrote device-visible memory), so nothing broke.  On a discrete CUDA GPU
the device snapshot buffers were never written — the kernels read
uninitialised `cudaMalloc` memory on the very first substep, and the
state collapsed to zero immediately.

`copy_to_gpu(const Fields&, GPUSnapshot&)` itself was correct: it packs
into `epsilon_stage` / `u_stage` / … exactly like the SCGrid version.
The DMA that pushes staging to the device was simply never invoked.

### Bug 2 — RK2 corrector dropped (both backends) — the accuracy loss

After fixing Bug 1 the run no longer collapsed, but the error vs CPU was
**3.6 × 10⁻³** and grew *monotonically from step 1* — the signature of a
systematic first-vs-second-order discretisation error, not random
float32 noise.

The inter-step GPU residency keeps state on the device across substeps
by swapping `GPUSnapshot` pointer aliases that mirror the host arena
rotation in `Evolve::AdvanceRK`:

| host (AdvanceRK)            | GPU mirror                       |
|-----------------------------|----------------------------------|
| rk0 → 3-way pointer rotate  | `Advance::rotate_snapshots_gpu()` |
| rk1 → fpCurr↔fpNext swap    | `Advance::swap_curr_future_gpu()` |

`try_gpu_advance` sets the residency flags at substep end:
- non-last substep (rk0): `gpu_state_authoritative_ = true`
- last substep (rk1):     `gpu_state_authoritative_ = false`,
                          `gpu_owns_state_ = true`

`main_gpu` gated the rk1 swap on `gpu_owns_state_` (true after rk1, so
the swap runs).  Commit `0cecdf5` ("Add intra-substep GPU residency")
re-gated it on `gpu_state_authoritative_`:

```cpp
// BEFORE (main_gpu, correct):
void Advance::swap_curr_future_gpu() {
    if (gpu_owns_state_) gpu_grid_.swap_curr_future();
}
// AFTER (0cecdf5, regression):
void Advance::swap_curr_future_gpu() {
    if (gpu_state_authoritative_) gpu_grid_.swap_curr_future();
}
```

But `try_gpu_advance` clears `gpu_state_authoritative_` to **false** on
the last substep, *before* `AdvanceRK` calls `swap_curr_future_gpu()`.
So the gate was always false at the rk1 boundary and **the swap never
ran**: the rk1 corrector result was stranded in `snap_future`, and the
next step evolved from the rk0 *predictor* in `snap_curr`.  That turns
the 2nd-order Runge–Kutta scheme into forward Euler.

This bug is **backend-agnostic** — it affected Metal too.  It is the
real explanation for PORT_GPU.md §8's unexplained observation that
XSCAPE-Metal's error (~4 × 10⁻³) was ~100× worse than `main_gpu`-Metal's
(~5 × 10⁻⁵) *using identical kernels*.  §8.4/§9.7 attributed that gap to
"float32 over 100 steps"; it was actually the dropped corrector.  See
the retraction note appended to PORT_GPU.md.

## 3. Fixes applied

Both in `src/advance.cpp` (only GPU builds compile the affected code;
the CPU build is untouched).

1. **Restore the H2D upload** inside `try_gpu_advance`, right after the
   `copy_to_gpu` calls, guarded `#if defined(USE_CUDA)` so it compiles
   only on CUDA (no-op / absent on Metal, whose `MetalPipelines` has no
   such method):

   ```cpp
   #if defined(USE_CUDA)
       GPUPipelines::instance().upload_snapshots_async(gpu_grid_);
   #endif
   ```

2. **Re-gate `swap_curr_future_gpu()` on `gpu_owns_state_`** (reverting
   the `0cecdf5` regression), matching `main_gpu`.  `rotate_snapshots_gpu()`
   stays gated on `gpu_state_authoritative_`, which *is* the correct flag
   for the rk0→rk1 transition.

No CUDA kernel, pipeline, or `GPUGrid` source changed — they were
verified byte-identical to `main_gpu`
(`music_kernels.cu`, `music_kernels.cuh`, `CUDAPipelines.{cu,h}`,
`gpu_types.h`).  The bugs were entirely in the `advance.cpp` dispatch
glue lost/altered during the merge and a later residency commit.

## 4. Verification — correctness is bit-identical to `main_gpu`

Built `main_gpu` in a sibling worktree and ran the repository's own
`tests/cuda_*` scripts on **both** branches on the same RTX 3090,
`OMP_NUM_THREADS=12`.  The max relative error on the `eps_max` trace is
**identical at every grid size** — XSCAPE_CUDA and `main_gpu` produce
the same float32 results because they run the same kernels with the same
(now correct) RK2 integration.

### 4.1 `tests/cuda_vs_cpu_bench.sh` (2D boost-invariant, 100 steps)

| Grid       | XSCAPE_CUDA err | `main_gpu` err |
|------------|-----------------|----------------|
| 32×32×1    | 4.7 × 10⁻⁵      | 4.7 × 10⁻⁵     |
| 64×64×1    | 9.9 × 10⁻⁵      | 9.9 × 10⁻⁵     |
| 128×128×1  | 8.8 × 10⁻⁵      | 8.8 × 10⁻⁵     |

### 4.2 `tests/cuda_vs_cpu_bench_3d.sh` (3+1D, ~40 steps)

| Grid       | XSCAPE_CUDA err | `main_gpu` err |
|------------|-----------------|----------------|
| 32×32×8    | 4.7 × 10⁻⁵      | 4.7 × 10⁻⁵     |
| 32×32×32   | 3.7 × 10⁻⁵      | 3.7 × 10⁻⁵     |
| 64×64×16   | 2.7 × 10⁻⁵      | 2.7 × 10⁻⁵     |
| 64×64×32   | 2.72 × 10⁻⁵     | 2.72 × 10⁻⁵    |

All cases **PASS** (threshold 1 × 10⁻³).  Before the fixes the CUDA
build collapsed to 0 (Bug 1) or, with only Bug 1 fixed, drifted at
~3.6 × 10⁻³ (Bug 2).

## 5. Performance

### 5.1 Official bench scripts (wall-clock, `OMP_NUM_THREADS=12`)

`tests/cuda_vs_cpu_bench_3d.sh`, GPU wall-clock seconds / speedup-vs-CPU:

| Grid       | XSCAPE_CUDA      | `main_gpu`       |
|------------|------------------|------------------|
| 32×32×8    | 0.29 s / 1.69×   | 0.25 s / 2.08×   |
| 32×32×32   | 0.39 s / 3.03×   | 0.31 s / 4.19×   |
| 64×64×16   | 0.59 s / 4.12×   | 0.48 s / 5.52×   |
| 64×64×32   | 0.91 s / 5.18×   | 0.67 s / 7.64×   |

XSCAPE_CUDA is correct and gives a solid GPU speedup, but the *total*
wall-clock is ~25–35 % higher than `main_gpu`.  Profiling shows this is
**not** in the GPU work.

### 5.2 Where the time goes (`MUSIC_PROFILE=1`, 64×64×32, 111 steps)

Default diagnostics (`output_diagnostics_every_N_timesteps = 1`, every step):

| Section (avg ms/step)               | XSCAPE_CUDA | `main_gpu`       |
|-------------------------------------|-------------|------------------|
| `evolve.step_total`                 | 10.04       | 7.31             |
| `evolve.AdvanceRK` (GPU kernels+sync)| **2.45**   | 3.03             |
| D2H per step                        | 2.13 (curr 1.00 + arena 1.13) | 0.95 (`d2h_copyback`) |
| `output_momentum_anisotropy_vs_tau` | 1.00        | 0.82             |
| `check_conservation_law`            | 0.65        | 0.97             |
| `max_energy_density` (reduce_max_gpu)| 0.085      | 0.060            |

**The GPU compute itself — `AdvanceRK`, which contains the kernel
dispatch and intra-step sync — is *faster* on XSCAPE_CUDA (2.45 ms) than
on `main_gpu` (3.03 ms).**  The port is not the bottleneck.

The total-step gap comes from the CPU side:

1. **Untimed evolve-loop overhead.**  Summing the timed sections leaves
   ~3.7 ms/step unaccounted on XSCAPE vs ~2.5 ms on `main_gpu`.  This is
   host code in `Evolve::EvolveIt` outside any GPU touch — the XSCAPE
   evolve driver differs from `main_gpu`'s by ~1200 lines (adaptive
   timestep, Fields management, different diagnostics) and is heavier
   per step *regardless of backend* (it runs on the CPU path too).  It
   is not part of the CUDA port.

2. **On-demand D2H syncs fire every step at the default diagnostic
   frequency.**  XSCAPE's residency design deliberately omits the
   per-substep copy-back and instead syncs lazily at the diagnostic call
   sites in `evolve.cpp` (lines 199–229), all gated by
   `output_diagnostics_every_N_timesteps`.  At the default of 1, every
   step needs `fpCurr` (anisotropy) and `fpPrev`+`fpCurr` (conservation
   check), so it pays ~2.1 ms/step of sync.  `main_gpu` copies back
   ~0.95 ms/step unconditionally inside `AdvanceRK` and cannot skip it
   (no residency).

### 5.3 With diagnostics gated (`output_diagnostics_every_N_timesteps = 10`)

| Section (avg ms/step) | XSCAPE_CUDA | `main_gpu` |
|-----------------------|-------------|------------|
| `evolve.step_total`   | 6.51        | 5.57       |
| `evolve.AdvanceRK`    | **2.06**    | 2.85       |

Gating amortises XSCAPE's syncs to ~every 10th step (~0.33 ms/step) and
skips the host diagnostic compute.  `main_gpu` still copies back every
substep.  The residual ~0.9 ms/step is the untimed CPU evolve overhead
(point 1 above) — branch-inherent, not GPU.

### 5.4 Per-step GPU compute (`tests/cuda_perstep_bench.sh`, GPU ms/step)

| Grid       | XSCAPE_CUDA | `main_gpu` |
|------------|-------------|------------|
| 128×128×1  | 4.19        | 3.27       |
| 64×64×16   | 4.40        | 3.48       |
| 64×64×32   | 6.54        | 5.17       |

This difference method does not cancel the per-step CPU diagnostics
(which run at the default frequency in the bench input), so it tracks
`step_total`, not the GPU kernels.  The profiled `AdvanceRK` numbers in
§5.2/§5.3 are the apples-to-apples GPU comparison.

### 5.5 Bottom line (initial port)

The CUDA backend on XSCAPE is **correct (bit-identical to `main_gpu`)**
and its **GPU compute matches or beats `main_gpu`**.  The wall-clock gap
in the default bench is CPU-side.  §8 dissects that gap and removes the
GPU-attributable part of it; the genuinely-remaining piece is XSCAPE's
heavier `EvolveIt` driver, which is not the CUDA backend and runs on the
CPU path too.

> **Note (superseded):** an earlier version of this section claimed that
> setting `output_diagnostics_every_N_timesteps` to a production value
> "closes most of" the gap.  §8.3 measures this and walks it back —
> gating removes the per-step *sync* cost but **not** the untimed
> evolve-loop overhead, which is the dominant residual at any diagnostic
> frequency.

## 6. Reproducing

```bash
# CPU + CUDA builds (bench scripts expect these exact dir names)
cmake -S . -B build               -DCMAKE_BUILD_TYPE=Release && cmake --build build      -j
cmake -S . -B build_cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build_cuda -j

OMP_NUM_THREADS=12 bash tests/cuda_vs_cpu_bench.sh
OMP_NUM_THREADS=12 bash tests/cuda_vs_cpu_bench_3d.sh
OMP_NUM_THREADS=12 bash tests/cuda_perstep_bench.sh

# per-section profile
OMP_NUM_THREADS=12 MUSIC_PROFILE=1 build_cuda/src/MUSIChydro <input>   # dump on stderr
```

Head-to-head against `main_gpu`: `git worktree add ../mg main_gpu`,
build the same two dirs there, run the same scripts.

## 7. Known limitations / follow-ups

- **Source-term residency (untested path).**  The hydro-source CPU
  pre-pass (`prefill_hydro_source_on_cpu`) reads the 4-velocity from the
  host `Fields` arena.  Under inter-step residency with diagnostics
  *gated* (`output_diagnostics_every_N_timesteps > 1`) in the batch
  `EvolveIt` driver, the host arena can be one step stale at pre-pass
  time.  In the actual XSCAPE/JETSCAPE source-term workflow this does
  **not** bite: that path uses `EvolveOneTimeStep`, which syncs and
  clears `gpu_owns_state_` at the end of every call, so the host arena
  is always fresh before the next `AdvanceIt`.  `main_gpu` avoided the
  issue structurally by reading `u` from the GPU snapshot
  (`refresh_u_curr_stage()` + `host_readable_u_curr()`); XSCAPE reads
  the host arena instead.  A clean fix would sync `fpCurr` before the
  pre-pass when `gpu_owns_state_` is set, but it could not be exercised
  in-session (no working source input — see PORT_GPU.md §9.3), so it was
  left as a documented follow-up rather than shipped untested.

- **Per-arena D2H cost.**  XSCAPE's `copy_primitives_to_cpu` +
  `copy_wmunu_to_cpu` issue five separate `cudaMemcpy` D2H calls per
  arena (eps, rhob, u, Wmunu, pi_b), ~1 ms/arena.  The *prev*-arena half
  of the per-step cost was eliminated in §8.2 (cross-step residency); the
  remaining *curr*-arena sync (~1 ms) is still a bit heavier than
  `main_gpu`'s AoS copy-back.  Batching the five transfers into one or
  speeding the SoA float→double unpack would help, but only at
  `output_diagnostics_every_N_timesteps = 1`.

- **CUDA on coherent hardware (GB10 / Grace-C2C / integrated).**
  Coherence is detected from `cudaDevAttrPageableMemoryAccess` (or
  `prop.integrated`) in `CUDAPipelines::initialize()` — the standard
  attribute that reports 1 on Grace-coherent parts (GH200/GB10/GB200)
  and integrated GPUs, so GB10 resolves to the zero-copy managed path.
  On that path `copy_to_gpu` packs straight into the `cudaMallocManaged`
  buffers the kernels read, `upload_snapshots_async` early-returns (the
  Bug-1 fix is a no-op), and the D2H copy-backs read the managed buffers
  in place (no `cudaMemcpy`); the Bug-2 swap fix is a pure pointer-alias
  rotation, identical on both memory models.

  No GB10 silicon was available, but the coherent *code path* was
  validated on the discrete RTX 3090 via the new
  `MUSIC_CUDA_FORCE_COHERENT=1` override (symmetric to
  `MUSIC_CUDA_FORCE_DISCRETE`).  Forced-coherent results are
  **bit-identical** to the discrete path — 9.9 × 10⁻⁵ (2D 64×64×1) and
  exactly 0 difference vs discrete in 3D (32×32×8) — and match the CPU
  reference at the float32 floor.  This confirms the coherent branches
  are logically correct; it does not measure true-coherence performance
  (the 3090's managed memory migrates over PCIe, whereas GB10's is
  hardware-coherent and zero-copy).

- **`rhoq`/`rhos`, finite-µB EOS, baryon diffusion, vorticity** — same
  CPU-fallback guards as the Metal path (PORT_GPU.md §4); unchanged.

## 8. Post-port CPU-side optimizations & final decomposition

After the port was correct, two targeted optimizations chipped at the
default-config (`output_diagnostics_every_N_timesteps = 1`) wall-clock
gap.  Both are CPU-side (the GPU kernels were already as fast as
`main_gpu`).  Numbers are 64×64×32, RTX 3090.

### 8.1 Per-thread `thermalVec` in the anisotropy diagnostic (commit `615527d`)

`output_momentum_anisotropy_vs_tau` declared its `std::vector<double>
thermalVec` **inside** the OpenMP reduction loop — a heap malloc/free per
cell.  `eos.getThermalVariables()` does `clear()`+`push_back`, and
`clear()` keeps capacity, so a thread that keeps its buffer refills it in
place.  Hoisted to `firstprivate` (one buffer per thread, reused).
Bit-identical output at OMP=1; ~16 % faster on this function at OMP=12
(1.00 → 0.84 ms/step); neutral at OMP=48 (the 25-variable reduction tree
dominates a 64×64 η-slice, not the allocator).  `check_conservation_law`
already used the scalar `eos.get_pressure` (no vector); the other
`getThermalVariables` callers are serial — both left as-is.

### 8.2 Cross-step prev residency (commit `aee1736`)

At Nskip_diag=1 the diagnostics re-synced **both** host arenas every step
(`curr` for the anisotropy, `prev` for conservation/freezeout), ~2.1 ms,
vs `main_gpu`'s ~0.95 ms — `main_gpu` gets `prev` "for free" because its
per-substep copy-back leaves the previous step resident on the host.

But MUSIC's RK rotation makes this step's `fpPrev` the *same host buffer*
that was `fpCurr` last step (net per-step rotation: `fpPrev ← old
fpCurr`), and the GPU snapshot rotation makes `snap_prev` == last step's
`snap_curr` — so when `curr` was synced the immediately-preceding step,
the `prev` D2H is redundant.  Tracked safely with a step counter
(`gpu_step_count_`, `curr_synced_{buf,step}_`, `prev_fresh_buf_`):
the `prev` copy is skipped **iff** `&prev` equals the buffer synced as
`curr` exactly one step earlier; otherwise it copies as before.
Conservative — never skips unless proven fresh, so it is correct at any
diagnostic frequency.

Effect: `prev` sync 1.13 → 0.02 ms/step; per-step bench (OMP=48) 7.30 →
6.51 ms/step; gap to `main_gpu` 1.52× → 1.35×.  **Validated
bit-identical** at OMP=1 vs the pre-change build — eps trace,
`global_conservation_laws.dat` (reads `prev`), and the Cornelius
freezeout surface (reads `prev`) — at Nskip_diag = 1 and 3.

### 8.3 Final decomposition — what the residual gap actually is

Per-step profile after both optimizations (OMP=12, 64×64×32, Nskip=1):

| component                  | XSCAPE (now) | `main_gpu` |
|----------------------------|--------------|------------|
| `evolve.AdvanceRK` (kernels)| **2.04 ms** | 3.03 ms    |
| D2H sync (curr + prev)     | **1.16 ms**  | 0.95 ms    |
| untimed `EvolveIt` CPU      | 3.75 ms     | 2.43 ms    |
| `evolve.step_total`        | 8.52 ms      | 7.31 ms    |

So after §8.1–8.2: **the GPU compute is faster on XSCAPE, and the D2H
sync is essentially matched.**  The entire residual (~1.3 ms/step) is
**untimed CPU code in `EvolveIt`** — XSCAPE's evolve driver differs from
`main_gpu`'s by ~1200 lines (adaptive timestep, `Fields` bookkeeping,
gated checks) and is heavier per step regardless of backend (it runs on
the CPU path too).

Gating diagnostics (`Nskip_diag = 10`) was measured at XSCAPE 6.51 vs
`main_gpu` 5.57 ms (OMP=12) — still ~1.17×, confirming the residual is
the evolve-loop overhead, **not** the sync (which gating already
amortizes).  This is why the §5.5 "gating closes most of it" claim was
walked back.

**Conclusion for the CUDA port:** correct (bit-identical to `main_gpu`),
GPU + sync costs matched or better.  The last ~1.3 ms/step is the XSCAPE
`EvolveIt` CPU driver — a separate optimization target from the CUDA
backend, profiled next (§9, CPU-side).

## 9. CPU-side: diagnostic thread-scaling (commit `995b4f1`)

Profiling the per-step sections (64×64×32, Nskip=1) by thread count
exposed one section scaling *backwards*:

| section (ms/step)               | OMP=12 | OMP=48 | scaling |
|---------------------------------|--------|--------|---------|
| `evolve.AdvanceRK` (GPU)        | 2.32   | 1.97   | ✓       |
| `advance.sync_curr_from_gpu`    | 1.14   | 1.35   | ~flat   |
| `check_conservation_law`        | 0.69   | 0.52   | ✓       |
| `output_momentum_anisotropy_vs_tau` | 0.83 | **1.59** | ✗ negative |

The contrast with `check_conservation_law` (a similar OMP reduction that
scales *correctly*) pinned the cause: `check_conservation_law` uses one
`collapse(3)` over the whole grid, whereas `output_momentum_anisotropy_vs_tau`
ran **two reductions inside a serial `for (ieta)` loop** — one parallel
region per in-range η-slice, each a 25-variable reduction over only a
64×64 slice.  At high thread counts the fork/join + reduction-tree
overhead per slice dominates the ~85 cells/thread of real work.

**Fix:** two full-grid passes — a centroid pass, then a single
`collapse(3)` anisotropy/eccentricity reduction — iterating a **dense
list of in-range η-slice indices** (`ir_etas`).  Dense is essential: the
in-range slices are a small contiguous η block, so `collapse(3)` over
*all* slices with a `continue` filter idles most threads under static
scheduling (a first attempt that way *regressed* to 3.98 ms — recorded
here as a caution).

Result: `output_momentum_anisotropy_vs_tau` ~0.82 ms/step at **both**
OMP=12 and 48 (flat); per-step bench 64×64×32 at OMP=48 **6.51 → 4.77
ms/step, matching `main_gpu` (4.81)**.  Validated bit-identical at
OMP=1 across all four output files for 2D (single-slice) and 3D
(multi-slice — the `collapse(3)` preserves the ieta-outer accumulation
order, so the cross-slice sums are byte-for-byte unchanged).

This optimization is backend-agnostic (it's CPU diagnostic code) and
applies equally to `main_gpu` and the CPU build.  Smaller grids
(64×64×16, 128×128×1) still carry a residual gap to `main_gpu` from the
untimed `EvolveIt` evolve-loop overhead (§8.3), which is proportionally
larger when there are fewer cells of GPU work per step.
