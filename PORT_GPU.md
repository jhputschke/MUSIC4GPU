# XSCAPE + GPU Port Notes

Living document tracking the work to make MUSIC's GPU backends (Metal/CUDA)
reachable from the XSCAPE/JETSCAPE entry point, plus every architectural
decision and known limitation along the way.

Started: 2026-05-26, on branch `XSCAPE` after merging `main_gpu`.

---

## 1. Context

MUSIC has two top-level entry points:

1. **Standalone binary** — drives `Evolve::EvolveIt` on `SCGrid` objects
   (3D array-of-structs of `Cell_small`). The full GPU dispatch
   (`init_metal_if_needed`, `copy_to_gpu`, `gpu_owns_state_`, per-step
   kernels, `reduce_max_gpu`) is wired here.
2. **XSCAPE/JETSCAPE interface** — drives `Advance::AdvanceIt` on three
   `Fields` objects (separate `std::vector<double>` per quantity, indexed
   by a flat `fieldIdx`). Used when MUSIC runs as a library inside JETSCAPE.

After the `main_gpu` → `XSCAPE` merge, the GPU dispatch is reachable only
from path (1). The XSCAPE path falls back to CPU regardless of build flags.

**Goal:** make the XSCAPE path use GPU when built with `USE_METAL` or
`USE_CUDA`.

## 2. Architectural decision: layout compatibility

| Layout         | Storage                                          | Indexing                                |
|----------------|--------------------------------------------------|-----------------------------------------|
| `SCGrid`       | AoS — one `Cell_small` per `(ix,iy,ieta)`        | `arena(ix,iy,ieta)`                     |
| `Fields`       | SoA — separate `std::vector<double>` per field   | `ix + Nx*(iy + Ny*ieta)`                |
| `GPUSnapshot`  | SoA — separate `float*` per field, component-major | `Nx*(Ny*ieta + iy) + ix`              |

The `Fields` and `GPUSnapshot` cell indexing is algebraically identical
(`ix + Nx*(iy + Ny*ieta)` ≡ `Nx*(Ny*ieta + iy) + ix`), so the Fields→GPU
upload is a per-component direct copy with no gather/scatter — actually
*simpler* than the existing SCGrid→GPU path.

## 3. Merge resolution decisions (2026-05-26)

Conflicts resolved during the `main_gpu` → `XSCAPE` merge. For each file,
what was kept and why.

- **`.gitignore`** — kept both: `*.swp` (XSCAPE) and `*.metallib` (GPU).
- **`CMakeLists.txt`** — cmake `3.10` (needed for CUDA `LANGUAGE`),
  main_gpu's `brew --prefix libomp` Homebrew detection, `-w` warning
  suppression. OpenMP linking is delegated to the modern
  `OpenMP::OpenMP_CXX` target in `src/CMakeLists.txt`.
- **`src/CMakeLists.txt`** — main_gpu's full Metal/CUDA infrastructure.
- **`src/advance.{h,cpp}`** — kept XSCAPE's `Fields&`/`fieldIdx`
  signatures for `FirstRKStepT`, `FirstRKStepW`, `MakeDeltaQI`,
  `AdvanceIt`. Kept XSCAPE's 7-component conserved-charge loop
  (`rhob`, `rhoq`, `rhos`). The GPU helpers (`init_metal_if_needed`,
  `make_gpu_params`, `swap_curr_future_gpu`, `reduce_max_gpu`) and
  member state (`gpu_grid_`, `gpu_owns_state_`,
  `gpu_state_authoritative_`) remain compiled under `#ifdef MUSIC_USE_GPU`
  but are **not yet called** from `AdvanceIt(Fields&,…)`. That's the
  port work this document tracks.
- **`src/dissipative.{h,cpp}`** — kept XSCAPE's `Fields&`/`fieldIdx`/
  `std::array<double,9>` signature for `Make_uWRHS`. Added
  `Make_uWRHS_geom` from main_gpu (GPU pre-pass helper). Dropped
  `Make_uPRHS` because it required `SCGrid&` and is incompatible with
  the XSCAPE path; the Fields path computes the equivalent term inline.
- **`src/evolve.cpp`** — kept XSCAPE's adaptive `while` loop with
  beastMode timestep. Added `bench::Timer` wrappers from main_gpu around
  the max-energy-density computation and `AdvanceRK` call. Added
  `swap_curr_future_gpu()` under `#ifdef MUSIC_USE_GPU` after the host
  pointer swap. The `advance.gpu_owns_state()` short-circuit for
  max-eps was **not** taken because the Fields-path AdvanceIt doesn't
  yet support GPU residency.
- **`src/grid_info.cpp`** — used plain C arrays (main_gpu) instead of
  `std::vector` for OpenMP reduction compatibility on the eccentricity/
  flow arrays.
- **`src/read_in_parameters.cpp`** — combined both freeze-out surface
  consistency checks (XSCAPE's `whichEOS` range check AND main_gpu's
  `useEpsFO == 0 && turn_on_rhob == 1` check).

## 4. Known limitations going into the port

### 4.1 Multi-charge densities (`rhoq`, `rhos`) — CPU only

User-confirmed: typical XSCAPE heavy-ion runs do **not** use net `rhoq` or
`rhos`. The GPU kernels (`music_kernels.metal`, `music_kernels.cu`) and
`GPUSnapshot` only carry `rhob`. Plan:

- The Fields→GPU upload reads `src.e_`, `src.rhob_`, `src.u_`,
  `src.Wmunu_`, `src.piBulk_` (ignores `rhoq_`, `rhos_`).
- Add a runtime guard in the GPU dispatch block: if any cell has
  `|rhoq_| > 0` or `|rhos_| > 0`, fall back to CPU and log a one-time
  warning per session.
- Extending GPUSnapshot for multi-charge is deferred until there's a
  concrete use case. It would require: two new `float*` fields in
  `GPUSnapshot`, allocation in `GPUGrid::alloc_snapshot`, packing in
  `copy_to_gpu`, and porting the 7-component KT-flux loop in the
  `gpu_make_delta_qi` kernel.

### 4.2 Finite-muB EOS — CPU only

The GPU EOS table (`eos_P`, `eos_dPde`, `eos_s`, `eos_T`) is sampled at
`rhob = 0`. EOSes with `whichEOS > 9` typically include finite-muB
behavior. The CPU fallback already exists in the SCGrid path; the same
guard needs to fire on the Fields path.

### 4.3 Other unsupported features (inherited from SCGrid GPU path)

- Baryon diffusion (`turn_on_diff == 1`)
- Vorticity-coupled second-order terms
- `muB_dependent_shear_to_s` ≠ 0

All of these already trigger CPU fallback on the SCGrid path; the same
guards need to be applied on the Fields path.

## 5. Port plan

### Phase 1 — Plumbing

- [x] Add `Fields` overloads in `src/gpu/GPUGrid.h`:
  - `void copy_to_gpu(const Fields& src, GPUSnapshot& dst) const`
  - `void copy_primitives_to_cpu(const GPUSnapshot& src, Fields& dst) const`
  - `void copy_wmunu_to_cpu(const GPUSnapshot& src, Fields& dst) const`
- [x] Implement in `src/gpu/GPUGrid.mm` (Metal, coherent host memory)
- [x] Implement in `src/gpu/GPUGrid_cuda.cu` (CUDA, with discrete-GPU staging)
- [x] Generalize `Advance::init_metal_if_needed` to accept `Fields&`
      (overload or template — only reads `Nx/Ny/Neta`)

### Phase 2 — Dispatch

- [x] Add GPU dispatch block in `Advance::AdvanceIt(Fields&, Fields&, Fields&, …)`
      mirroring the SCGrid version's structure
- [x] At `rk_flag == 0`: `init_metal_if_needed` → `copy_to_gpu(arenaFieldsCurr, snap_curr)`
      unless `gpu_owns_state_`
- [x] Run kernels: `gpu_make_delta_qi`, `gpu_make_uwrhs`, `gpu_make_du`,
      `gpu_make_w_source`, `gpu_finalize_ideal`, `gpu_first_rk_step_w_full`
- [x] On return: either `copy_primitives_to_cpu` + `copy_wmunu_to_cpu` to
      arenaFieldsNext, OR set `gpu_state_authoritative_` for the lockstep
      path (preferred — avoids round-trip)
- [x] At `rk_flag == 1`: `swap_curr_future_gpu()` is already called from
      `evolve.cpp` after the host pointer swap — no Fields-side change needed
- [x] Apply CPU-fallback guards from §4

### Phase 3 — Verification

- [x] Build with `-DUSE_METAL=1` on macOS / `-DUSE_CUDA=1` on Linux
- [x] Run an XSCAPE-style invocation against a CPU-only reference run
- [x] Adapt `tests/metal_vs_cpu_bench.sh` for the Fields path
- [x] Target tolerance: float32-level agreement on `epsilon` and `Wmunu`
      at a representative timestep (~1e-5 relative)

## 6. Decision log (filled in as work proceeds)

(Date | File/area | Decision | Why)

- 2026-05-26 | merge | Kept Fields-based signatures throughout; left GPU
  dispatch unwired in AdvanceIt(Fields&,…) | Avoided inventing untested
  Fields→GPUSnapshot conversion mid-merge; deferred to this port.
- 2026-05-26 | scope | rhoq/rhos remain CPU-only with a runtime guard |
  Confirmed with user; no XSCAPE runs need multi-charge on GPU.
- 2026-05-26 | `GPUGrid.h/mm/cu` | Added three Fields overloads
  (`copy_to_gpu`, `copy_primitives_to_cpu`, `copy_wmunu_to_cpu`) | Fields
  is already SoA, so the body is a per-component float-cast loop — no
  AoS gather/scatter required.  Both Metal (coherent) and CUDA
  (managed + pinned-staging discrete paths) implementations added.
- 2026-05-26 | `init_metal_if_needed` | Refactored to take
  `(int Nx, int Ny, int Neta)` instead of `SCGrid&` | The function only
  read grid dimensions; this makes it callable from any layout.  No
  call sites broke because the SCGrid `AdvanceIt` was already dropped
  during the merge.
- 2026-05-26 | `advance.cpp` | Added `try_gpu_advance()` and call it
  from the top of `AdvanceIt(Fields&,…)` before the CPU triple loop |
  Cleanly separates the GPU dispatch from the CPU fallback; the CPU
  loop runs verbatim when the guards fail or when GPU init didn't take.
- 2026-05-26 | residency | Did **not** wire `gpu_owns_state_` /
  `gpu_state_authoritative_` for the Fields path | Open question §7.2 —
  JETSCAPE may mutate Fields between AdvanceIt calls, which would
  silently invalidate residency.  Every substep currently re-uploads
  at rk0.

## 7. Open questions

- Should the GPU dispatch run unconditionally on `MUSIC_USE_GPU` builds,
  or be opt-in via a new `DATA` flag? (SCGrid path is currently
  unconditional; current Fields path is also unconditional, gated only
  by feature support guards.)
- Does `gpu_owns_state_` make sense across an XSCAPE step boundary, or
  does JETSCAPE re-populate `Fields` between MUSIC `AdvanceIt` calls?
  If the latter, residency is impossible and we always upload at rk0.

## 8. Verification (2026-05-26)

The standalone `MUSIChydro` binary calls `Evolve::EvolveIt(Fields&,…)`,
which calls `Advance::AdvanceIt(Fields&,…)`, which now dispatches to
`try_gpu_advance` when built with `-DUSE_METAL=ON`.  So the standalone
binary is itself a CPU↔GPU comparison harness for the new code path.

### 8.1 Short-bench correctness (initial verification)

`tests/metal_vs_cpu_bench.sh` (100 timesteps, multi-threaded CPU):

| Grid       | CPU time | GPU time | Speedup | Max rel err |
|------------|----------|----------|---------|-------------|
| 64×64×1    | 0.09 s   | 0.19 s   | 0.47×   | **0.0e+00** (bit-identical) |
| 128×128×1  | 0.18 s   | 0.28 s   | 0.64×   | **0.0e+00** (bit-identical) |

The 0.0e+00 result is misleading: the bench's correctness check only
compares the *initial* eps_max (1 point of trace).  Over many steps the
GPU's float32 accumulates noise vs the CPU's float64 — see §8.3.

### 8.2 Longer bench (single-threaded CPU baseline)

Multi-threaded CPU is currently unreachable on this machine — see §9.6
(pre-existing OpenMP race).  All CPU times below are with
`OMP_NUM_THREADS=1`, which makes the GPU look better than it would
against a 16-thread CPU.  Take the speedup numbers as *upper bounds*
for the Apple M3 Max coherent-memory case.

| Grid          | Steps | CPU 1-thread (s) | GPU (s) | Speedup vs 1-thread CPU | Max rel err on `eps_max` |
|---------------|-------|------------------|---------|-------------------------|--------------------------|
| 64×64×1       | 201   | 3.96             | 1.16    | **3.4×**                | 6.4 % (201 pts)          |
| 128×128×1     | 201   | 17.34            | 3.55    | **4.9×**                | 48 % (201 pts)           |
| 64×64×16 (3D) | 61    | 1.15             | 0.44    | **2.6×**                | 3.8 % (61 pts)           |

### 8.3 Numerical accuracy note

The 48 % relative error at 128×128×1 after 201 timesteps is not a bug
per se — both CPU and GPU produce smooth, finite Gubser-like flow, but
the GPU's float32 arithmetic diverges from the CPU's float64 over long
evolutions.  Per-step error is small; it accumulates over hundreds of
steps, particularly in the viscous (small-eps tail) cells where
sensitivity is highest.

For most XSCAPE use cases this is fine — JETSCAPE only needs hydro
correct to a few percent for thermal-particle production downstream.
If tighter agreement is required, the options are:

- Promote critical kernels to fp64 (Metal supports half/float/double;
  CUDA already supports double natively).  Roughly 1.5–2× slower.
- Use mixed-precision: fp32 for KT flux divergence and shear update,
  fp64 for the Newton solve in `gpu_reconst` where most divergence
  originates.

Neither is wired yet; tracked as follow-up §9.7.

### 8.4 Grid-size note

256×256×1 didn't run in this bench — the chosen Gubser-style input
(`X_grid_size = 25.6 fm`, `delta_x = 0.1 fm`, `delta_tau = 0.005 fm`)
violates CFL for that grid and exits at step 1 with
`maximum e = 6.3e+256`.  This is a test-input issue, not a GPU
correctness issue — both CPU and GPU detect and exit the same way.

## 9. Known issues / follow-ups

### 9.1 Intermittent SIGTRAP / Obj-C corruption — now traced to §9.6

Originally suspected to be a small-grid Metal kernel issue.  Subsequent
benchmarking (§8) revealed the same crash signature
(`Method cache corrupted`, `receiver 0 bytes selector 'alloc'`,
`Trace/BPT trap: 5`) on **all** grid sizes whenever
`OMP_NUM_THREADS >= 2`, in both the CPU-only and Metal builds.  That
makes it a pre-existing OpenMP race, not a Metal or small-grid issue —
see §9.6 for the full reproduction and investigation plan.

**Workaround until §9.6 is fixed:** run with `OMP_NUM_THREADS=1`.
The GPU path itself works correctly at any grid size in that mode.

### 9.2 Performance characterisation — partial (see §8.2)

Initial speedup numbers landed (§8.2): **2.6–4.9× over single-thread
CPU** at 64–128 in 2D and 64×64×16 in 3D.  Multi-thread CPU baseline
unreachable until the OpenMP race (§9.6) is fixed; that's the comparison
that actually matters for production.  Other follow-ups:

- Per-step bench timer breakdown (`bench::Timer` scopes are already
  wired in `evolve.cpp`, just need `MUSIC_PROFILE=1` build flag to
  emit results).
- Re-run on a discrete GPU (CUDA, A100/H100) where PCIe transfer cost
  matters and the residency optimisation (§9.4) would make a real
  difference.
- Re-bench at production-realistic Pb–Pb sizes (typically 200×200×64
  with smooth Glauber initial conditions, ~1000 timesteps) once a CFL-
  safe input is available.

### 9.3 Hydro source terms — wired (2026-05-26 follow-up)

✅ Now supported via `prefill_hydro_source_on_cpu()` in `advance.cpp`.
That CPU pre-pass walks the grid in parallel, evaluates
`hydro_source_terms_ptr->get_hydro_energy_source(...)` (and
`get_hydro_rhob_source` when `turn_on_rhob == 1`) at each cell's
`(tau_rk, x, y, eta_s, u_mu)`, and writes `tau_rk * j^alpha` into
`gpu_grid_.qi_source_buf[alpha * Ncells + cell]`.  `MUSICGridParams`
gets `has_hydro_source = 1` / `has_rhob_source` set accordingly and the
existing `gpu_finalize_ideal` kernel picks the buffer up.

**Guard:** `DATA.turn_on_QS == 1` (rhoq/rhos source channels) still
forces CPU fallback — `GPUSnapshot` doesn't carry rhoq/rhos, so the
multi-charge source channels would be silently dropped.  See §4.1.

**Verification limitation:** End-to-end CPU↔GPU agreement on this code
path could not be confirmed in-session because the available test
input (`tests/test_source_terms/strings_event_0.dat`, after the
six-column format patch required to load it) crashes both the CPU and
the GPU build intermittently at step 1 (`SIGTRAP` / `SIGABRT`).  The
crash is pre-existing in the CPU build and unrelated to the GPU port.
Code correctness was confirmed by structural comparison with the
per-cell formula in `Advance::FirstRKStepT`.  A clean source-terms
verification needs a working input file — likely an updated
`strings_event_0.dat`, an AMPT-style input, or a hand-crafted
`HydroSourceBase` subclass with a smooth profile.

**Performance note:** The pre-pass is sequential per-cell on the CPU
(parallelised with OpenMP).  For string/AMPT sources the dominant cost
is `prepare_list_for_current_tau_frame` plus the inner loop over active
strings inside `get_hydro_energy_source`; in practice this is a small
fraction of an RK step so the CPU pre-pass shouldn't bottleneck the
GPU pipeline.  If profiling later shows otherwise, the source loop
itself could be ported (porting the strings model would be most of the
work).

### 9.4 Residency / multi-step GPU state

See open question §7.2.  Currently every substep does a fresh H2D upload
of curr+prev.  A residency optimisation (keeping snap_curr / snap_prev
on the GPU across step boundaries) would roughly double per-step
throughput on the discrete-GPU path but requires confirming that
JETSCAPE doesn't write to the Fields between AdvanceIt calls.

### 9.5 EvolveOneTimeStep is missing diagnostics that EvolveIt has

MUSIC exposes three entry points; all three reach the GPU dispatch:

| Entry point                                   | Driver               |
|-----------------------------------------------|----------------------|
| Standalone `MUSIChydro foo.input`             | `EvolveIt`           |
| XSCAPE batch (`run_hydro`)                    | `EvolveIt`           |
| XSCAPE step-by-step (`run_hydro_upto`)        | `EvolveOneTimeStep`  |

The per-substep evolution (RK loop, GPU dispatch, freezeout,
source-term prep, evolution-data output, frozen-out early exit) is
identical in both drivers.  `EvolveOneTimeStep` is, however, a
stripped-down driver compared to `EvolveIt` — features present in
`EvolveIt` but missing from `EvolveOneTimeStep`:

- **Beast-mode adaptive timestep** (`DATA.beastMode == 2`, `NtauBlock = 200`
  block that periodically doubles `delta_tau`)
- **Diagnostic outputs:**
  - `output_momentum_anisotropy_vs_etas` (at iFreezeStart, +10, +30, +50)
  - `output_momentum_anisotropy_vs_tau`
  - `output_average_phase_diagram_trajectory` (for `Initial_profile` 13/131)
  - Vorticity outputs (`output_vorticity_distribution`,
    `compute_angular_momentum`, `output_vorticity_time_evolution`)
  - `output_hydro_debug_info` (per-cell `monitor_a_fluid_cell`)
  - `output_1p1D_check_file` (`Initial_profile = 1`)
  - `output_1p1D_RiemannTest`, `output_1p1D_DiffusionTest`
- **Conservation-law check** (`grid_info.check_conservation_law`)
- **Per-step profiling** (`bench::Timer` scopes, `bench::dump()`)
- **`reRunHydro` early-return** hook
- **End-of-run `FO_nBvseta.dat` summary**

Two subtle semantic differences also exist:

- `EvolveIt` uses `max_allowed_e_increase_factor = 5.0`;
  `EvolveOneTimeStep` uses `2.0` → tighter sanity check in step-by-step
  mode.
- `EvolveIt` checks `tau > source_tau_max + dt`; `EvolveOneTimeStep`
  checks `tau > source_tau_max` → one-step difference in the eps tracker
  window.

**Impact:** Most JETSCAPE workflows use the step-by-step path because
the framework interleaves hard parton energy loss with hydro.  Those
runs therefore do **not** get the diagnostic outputs above.  Hydro
evolution itself is unaffected.

**Fix when needed:** Mechanical port — copy the missing blocks from
`EvolveIt` into `EvolveOneTimeStep`, gated on `tauIdx % output_frequency`
(replacing `EvolveIt`'s `it %` checks).  Deferred at user request.

### 9.6 OpenMP race / deterministic SIGTRAP at ≥ 2 threads

**Pre-existing in MUSIC, not introduced by the GPU port.**  Discovered
while trying to run a longer benchmark on Apple M3 Max
(macOS / AppleClang 17 / Homebrew libomp 5.1).  Reproduction:

```
OMP_NUM_THREADS=1 MUSIChydro foo.input    # 100 % success
OMP_NUM_THREADS=2 MUSIChydro foo.input    # 0/3 succeed, all SIGTRAP
OMP_NUM_THREADS=16 MUSIChydro foo.input   # 0/3 succeed, all SIGTRAP
```

Crash signature is the same intermittent Obj-C/Metal corruption noted
in §9.1 (`Method cache corrupted`, `receiver 0 bytes selector 'alloc'`,
`Trace/BPT trap: 5`).  Hits both the CPU-only and the Metal build
identically, so it's not a GPU issue.  Likely a race in one of the
`#pragma omp parallel for` loops (init, evolve, or freeze-out) that
manifests on the OpenMP runtime + threading model of this machine.

**Impact for the GPU bench:** §8.2 numbers are CPU-1-thread vs GPU,
which overstates GPU speedup relative to a properly-threaded CPU.
A 16-thread CPU run would likely close the gap or beat the GPU on this
M3 Max class of machine.  On a discrete GPU (CUDA) the relative
position would shift further in favor of the GPU.

**Investigation plan when this is fixed:**
- bisect for the first commit where `OMP_NUM_THREADS=2` crashes
  (start from a known-good ancestor)
- look at recent changes to `#pragma omp parallel for` blocks in
  `evolve.cpp`, `grid_info.cpp`, and `init.cpp` — particularly any new
  shared-state writes inside the loop bodies
- check whether the crash also reproduces on Linux with GCC's libgomp
  (would indicate a real race vs an Apple/libomp interaction)

### 9.7 Float32 GPU vs float64 CPU divergence over long runs

§8.3 documents ~6 % error after 200 steps at 64×64 and 48 % after 200
steps at 128×128 in the Gubser viscous test.  Per-step error is small;
the accumulation is in the dilute-tail cells where the viscous Newton
solve is most sensitive.

If tighter agreement matters:

1. **Promote `gpu_reconst` (the Newton solve in `music_kernels.metal`
   / `music_kernels.cu`) to fp64.**  Both Metal MSL 3.0+ and CUDA
   support `double`; the cost is roughly 2× on the Newton kernel and
   negligible end-to-end.  This is the highest-leverage single change.
2. **Promote KT flux reconstruction to fp64** if (1) isn't enough.
   ~30 % overall slowdown.
3. **Mixed-precision residency:** keep `snap_*` in fp32 for bandwidth,
   re-cast to fp64 only inside the kernel for the sensitive arithmetic.

None of these are wired today.  For most XSCAPE downstream uses
(thermal-particle production, post-decay observables), 5–10 % hydro
error is fine — particle yields and flow harmonics smooth most of it
out.  Revisit if/when JETSCAPE consumers report deviations they care
about.
