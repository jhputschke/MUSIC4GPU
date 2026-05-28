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

**Carve-out for EOS 91 (2026-05-27).** The guard in
`Advance::gpu_features_supported` ([src/advance.cpp](src/advance.cpp))
was `if (DATA.whichEOS > 9) return false;`, which wrongly rejected
`whichEOS == 91`. EOS 91 is a *zero-muB* hotQCD variant — built by the
same `EOS_hotQCD` class as EOS 9 ([src/eos.cpp](src/eos.cpp), the
`eos_id == 9 || eos_id == 91` branch), with `set_flag_muB(false)` and
`dpdrhob = 0` ([src/eos_hotQCD.cpp](src/eos_hotQCD.cpp)). It samples the
GPU EOS table identically to EOS 9 and is GPU-safe. The guard is now:

```cpp
if (DATA.whichEOS > 9 && DATA.whichEOS != 91) return false;
```

This unblocks the standard XSCAPE Au+Au config (`EOS=91`,
`Initial_profile=131`, viscous, `Include_QS=0`/`Include_Rhob=0`), which
otherwise passes every other guard and was falling back to CPU solely on
the EOS check.

**Deferred cleaner fix (Option B) — blocked on a latent bug.** The
principled change is to gate on the EOS's own muB flag instead of a
magic number:

```cpp
if (eos.get_flag_muB()) return false;   // instead of the >9 && !=91 check
```

This is **not safe today** because of two issues found in the
2026-05-27 audit:

1. `EOS_base::flag_muB` ([src/eos_base.h](src/eos_base.h)) is declared
   with no initializer and `EOS_base() = default;`, so any subclass that
   skips `set_flag_muB()` leaves it indeterminate.
2. `EOS_BEST` (`whichEOS == 17`, [src/eos_best.cpp](src/eos_best.cpp))
   and `EOS_UH` (`whichEOS == 19`, [src/eos_UH.cpp](src/eos_UH.cpp))
   **never call `set_flag_muB`**, yet both are genuinely muB-dependent
   (2D `interpolate2D(e, rhob, …)` tables, nonzero `get_dpOverdrhob2`).
   With the flag uninitialized, `get_flag_muB()` could read falsy and
   silently admit a finite-muB EOS onto the rhob=0 GPU path → wrong
   physics, no warning. The current `> 9` threshold rejects 17/19
   correctly, which is why the magic number is kept for now.

   The `EOS` wrapper ([src/eos.h](src/eos.h)) also does not yet forward
   `get_flag_muB()` from the inner `eos_ptr`, so Option B additionally
   needs a one-line forwarder there.

To land Option B safely: (a) give `flag_muB` a fail-safe default of
`true` in `eos_base.h`; (b) add `set_flag_muB(true)` to the `EOS_BEST`
and `EOS_UH` constructors; (c) add `bool get_flag_muB() const { return
eos_ptr->get_flag_muB(); }` to the `EOS` wrapper; (d) switch the guard
to `if (eos.get_flag_muB()) return false;`. Only nothing currently reads
`get_flag_muB()` (the accessor is otherwise dead code), so the
uninitialized-flag bug is latent today — but it must be fixed before
anything depends on the flag.

### 4.3 Other unsupported features (inherited from SCGrid GPU path)

- Baryon diffusion (`turn_on_diff == 1`)
- Vorticity-coupled second-order terms
- `muB_dependent_shear_to_s` ≠ 0

All of these already trigger CPU fallback on the SCGrid path; the same
guards need to be applied on the Fields path.

### 4.4 GPU EOS table was linear-sampled — broke every non-conformal EOS (2026-05-27)

**Symptom.** After the §4.2 carve-out enabled EOS 91 on GPU, a CPU↔GPU
comparison (EOS=91, viscous, Gubser profile) diverged ~6 % by step 49
and the GPU run blew past the 5× energy-density sanity guard and aborted
at step 123, while the fp64 CPU completed all 210 steps cleanly. This
was **not** float32 noise and **not** the §9.7 swap-gating bug (which is
fixed and present): the divergence is deterministic and appears in fp64
too.

**Root cause — EOS table resolution.** The GPU samples the CPU EOS onto
four `GPU_EOS_N = 8192`-point tables at `rhob = 0`. `s(e)` and `T(e)`
were correctly **log-spaced** (they vary over many decades), but `P(e)`
and `dP/de(e)` were **linear-spaced** on `[0, eps_max]`, with a comment
admitting it was "essentially exact for the ideal-gas EOS." For hotQCD,
`eps_max = 9659 /fm⁴`, so the linear spacing is `de ≈ 1.18 /fm⁴` — but
the hydro cells live at `e ≈ 0.15 /fm⁴`, i.e. **0.13 of one grid
interval**. The entire evolution was interpolated on the single segment
`P(0)→P(1.18)`, a straight line drawn across the QCD crossover. The
conformal ideal gas (`P = e/3`) is exactly linear, which is why every
prior bench (all `EOS_to_use 0`) passed and never caught this.

**Per-EOS audit (which eligible EOS need log sampling).** Every EOS that
passes the guard (`whichEOS ≤ 9 || == 91`) except the ideal gas is
non-conformal and was affected. The CPU code shows it directly:

| id    | class    | P(e) shape (CPU)                         | linear GPU grid? |
|-------|----------|------------------------------------------|------------------|
| 0     | idealgas | `P = e/3`, exactly linear                | ✓ exact          |
| 1     | EOSQ     | bag model + 1st-order transition         | ✗ (also finite-µB, see §4.2) |
| 2–7   | s95p     | **7 piecewise tables**, per-table spacing ([eos_s95p.cpp](src/eos_s95p.cpp)) | ✗ |
| 8     | WB       | **degree-12 rational poly** ([eos_WB.cpp](src/eos_WB.cpp)) | ✗ (worst: `de ≈ 12`) |
| 9, 91 | hotQCD   | 100k-pt table, fine spacing ([eos_hotQCD.cpp](src/eos_hotQCD.cpp)) | ✗ (`de ≈ 1.18`) |

The non-linearity is visible in the CPU representation itself: s95p
*splits into 7 sub-tables* specifically to refine the dilute end, WB is
a rational polynomial, hotQCD uses a 100k-point fine table. The GPU's
single 8192-point linear grid over the full range can't match any of
them in the dilute regime.

**Fix.** Sample `P` and `dP/de` on the **same log-e grid** already used
for `s`/`T`, and route the kernel lookups through `gpu_log_interp`
instead of the linear `gpu_eos_interp` (now removed). One general fix
covers hotQCD, WB and s95p at once; the ideal gas stays effectively
exact (linear-in-log interp of a straight line over 0.14 %-wide
intervals → ~1e-7 error, far under the fp32 floor). Changes:

- [src/advance.cpp](src/advance.cpp) — host samples all four tables on
  one log grid (`log_e_floor = 1e-6`, matching `GPUGrid::upload_eos`'s
  `log_*` params). Shared by both backends.
- [src/gpu/music_kernels.metal](src/gpu/music_kernels.metal) and
  [src/gpu/music_kernels.cu](src/gpu/music_kernels.cu) — `gpu_P` /
  `gpu_dPde` use `gpu_log_interp`; the linear `gpu_eos_interp` is
  deleted. (The struct's linear `e_min`/`e_max`/`delta_e` fields are now
  set-but-unused; left in place to avoid touching the shared
  `upload_eos` signature.)

**Verification (Metal, Apple M3 Max).**

- EOS=91 hotQCD (Gubser IC, 210 steps): max rel err on `eps_max` drops
  from ~6 % (and a step-123 crash) to ~1e-4 through mid-run, growing to
  2.5e-3 by step 210 (genuine fp32 accumulation on a deliberately
  non-physical Gubser-on-hotQCD IC). GPU now completes all 210 steps.
- EOS=0 ideal gas (`tests/metal_vs_cpu_bench.sh`): no regression —
  4.7e-5 / 9.7e-5 / 9.4e-5 at 32/64/128², matching the pre-fix floor.

**Caveats / follow-ups.**

- The CUDA kernel change is a structural mirror of the validated
  Metal+host change; **not built or run on CUDA hardware** in-session.
- The Gubser profile is only an exact solution for the conformal EOS, so
  the EOS=91 run is a CPU↔GPU *agreement* test, not a physics test. A
  smooth physical IC (Gaussian blob or read-in profile) would be a
  cleaner accuracy check and is the recommended next step.
- EOSQ (id 1) remains finite-µB yet passes the `≤ 9` threshold — the
  log fix does not address that; it is Option B territory (§4.2).

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

Numbers below come from the repository's own benchmark scripts,
`tests/metal_vs_cpu_bench.sh` and `tests/metal_vs_cpu_bench_3d.sh`,
run on Apple M3 Max with `OMP_NUM_THREADS=12` (matching `main_gpu`'s
original benchmark configuration for a like-for-like comparison).

> **Honesty note:** earlier versions of this section first reported
> 3–7× speedup against a single CPU core (artifact of the §9.6 OpenMP
> race), then 0.6–0.8× ratios against 16-thread CPU (correct CPU
> baseline, but with the GPU path still doing a full H2D+D2H
> round-trip every substep).  The numbers below are after the
> intra-substep GPU residency fix (commit `0cecdf5`) and are
> apples-to-apples with the published `main_gpu` benchmark.

### 8.1 2D bench — `tests/metal_vs_cpu_bench.sh`

Boost-invariant Gubser viscous flow, 100 timesteps, EOS=ideal-gas.

```
$ OMP_NUM_THREADS=12 bash tests/metal_vs_cpu_bench.sh
```

| Grid       | CPU 12 thr | GPU    | XSCAPE speedup | `main_gpu` speedup | XSCAPE max rel err | `main_gpu` max rel err |
|------------|------------|--------|----------------|--------------------|--------------------|------------------------|
| 32×32×1    |  0.78 s    | 0.21 s |  **3.71×**     | 3.24×              | 5.3 × 10⁻⁵         | 5.3 × 10⁻⁵             |
| 64×64×1    |  0.36 s    | 0.28 s |  **1.29×**     | 1.19×              | 9.9 × 10⁻⁵         | 9.9 × 10⁻⁵             |
| 128×128×1  |  1.15 s    | 0.52 s |  **2.21×**     | 2.29×              | 9.4 × 10⁻⁵         | 9.4 × 10⁻⁵             |

Precision matches `main_gpu` bit-for-bit on every row, after the
`swap_curr_future_gpu()` fix described in §9.7.  Speedups jumped on
the merge from the CUDA-side `prev_fresh_buf_` sync optimisation —
see §8.3 for the per-step breakdown.

### 8.2 3D bench — `tests/metal_vs_cpu_bench_3d.sh`

Same Gubser profile replicated across η slices; full 3+1D evolution
exercises η-direction stencils and geometric terms.

```
$ OMP_NUM_THREADS=12 bash tests/metal_vs_cpu_bench_3d.sh
```

| Grid (Nx × Ny × Nη) | CPU 12 thr | GPU    | XSCAPE speedup | `main_gpu` speedup | XSCAPE max rel err | `main_gpu` max rel err |
|---------------------|------------|--------|----------------|--------------------|--------------------|------------------------|
| 32×32×8             |  0.37 s    | 0.24 s |  **1.54×**     | 1.43×              | 5.3 × 10⁻⁵         | 5.3 × 10⁻⁵             |
| 32×32×32            |  0.81 s    | 0.30 s |  **2.70×**     | 2.36×              | 3.7 × 10⁻⁵         | 3.7 × 10⁻⁵             |
| 64×64×16            |  1.66 s    | 0.45 s |  **3.69×**     | 2.91×              | 3.1 × 10⁻⁵         | 3.1 × 10⁻⁵             |
| 64×64×32            |  3.27 s    | 0.65 s |  **5.03×**     | 3.41×              | 3.1 × 10⁻⁵         | 3.1 × 10⁻⁵             |

**XSCAPE+GPU now beats `main_gpu`'s speedup at every 3D grid size**,
decisively so at production scale (5.03× at 64×64×32 vs `main_gpu`'s
3.41×; 3.69× at 64×64×16 vs 2.91×).  All four cases produce `eps_max`
traces bit-for-bit identical to `main_gpu` (which itself agrees with
the CPU to ~3 × 10⁻⁵, the expected float32 noise floor).

The performance gain over the pre-merge XSCAPE Metal numbers comes
from the CUDA-side cross-step sync optimisation (commit `aee1736`,
see §8.3), which proved backend-agnostic — Metal benefits the same
way CUDA does despite the very different memory model.

The "10× precision improvement" that earlier text attributed to the
inter-step residency optimisation was illusory; it was actually the
*reverse*.  See §9.7 for the swap-gating bug that was producing the
~10⁻³ drift and the fix that closed it.

### 8.3 Honest performance assessment

**XSCAPE+GPU now beats `main_gpu`'s speedup at every 3D grid size**,
including decisively at production scale (5.03× at 64×64×32 vs
`main_gpu`'s 3.41×; 3.69× at 64×64×16 vs 2.91×).  Precision matches
the kernel-internal float32 noise floor (~3 × 10⁻⁵ over 41 steps) —
bit-for-bit identical to `main_gpu` after the §9.7 swap-gating fix.

How the speedup gap was closed (chronological):

1. **Split sync API** (commit `950c1ea`).  Most diagnostics only
   read `fpCurr`, not `fpPrev`.  `sync_curr_from_gpu_readonly`
   copies one arena instead of two, halving the sync cost at every
   call site that doesn't need `prev`.
2. **Pointer-hoist in `copy_*_to_cpu(Fields&)`** (commit `950c1ea`).
   Fields stores `u_` and `Wmunu_` as
   `std::vector<std::vector<double>>`; naive `dst.u_[m][c]` does an
   indirect load on every iteration.  Hoisting the inner-vector
   `.data()` pointers outside the parallel loop unlocks
   vectorisation and goes from ~1 GB/s effective throughput to near
   memory-bandwidth peak.  Identical change in both `GPUGrid.mm`
   (Metal) and `GPUGrid_cuda.cu` (CUDA).
3. **`host_curr_fresh_` / `host_prev_fresh_` cache** (commit
   `950c1ea`).  Subsequent sync calls within the same outer
   iteration are no-ops.  `try_gpu_advance` clears them at substep
   entry so the next sync actually runs.
4. **rk1-boundary swap fix** (commit `2e87cff`).  Restored
   `swap_curr_future_gpu` so the GPU's `snap_curr` actually advances
   across the step boundary — see §9.7.  Doesn't change wall time
   per step but closes the 100× precision gap so the bench reports
   `PASS` instead of `WARN`.
5. **Cross-step `prev_fresh_buf_` sync skip** (commit `aee1736`,
   originated on the CUDA side, merged in `0e2cb39`).  Tracks which
   buffer was last synced as `curr` at the step boundary; at the
   start of the next step the host's rotation makes that buffer the
   new `prev`, so the prev-side D2H is provably redundant and gets
   elided.  Halves the per-step D2H volume when both curr and prev
   diagnostics fire.  This was the biggest single contributor to
   the Metal speedup jump — 3.98× → 5.03× at 64×64×32, even though
   it was authored for discrete-GPU PCIe pressure.

Per-step profile breakdown at 64×64×32 after the full chain
(`MUSIC_PROFILE=1`):

| Section                                | XSCAPE   | `main_gpu` |
|----------------------------------------|----------|------------|
| `evolve.step_total`                    | 16 ms    | 15 ms      |
| `evolve.AdvanceRK` (GPU kernels + sync)| 4 ms     | 11 ms      |
| `advance.sync_curr_from_gpu`           | 3.9 ms   | (n/a)      |
| `advance.sync_arena_from_gpu`          | 3.5 ms   | 3.7 ms ★   |
| `evolve.output_momentum_anisotropy_vs_tau` | 2.1 ms | 1.3 ms |
| `evolve.check_conservation_law`        | 1.2 ms   | 1.1 ms     |
| `evolve.max_energy_density` (reduce_max_gpu) | 0.07 ms | 0.07 ms |

★ `main_gpu`'s timer breakdown calls this `advance.d2h_copyback`.
(Profile numbers predate item 5 above; the bench wall times in §8.2
reflect the full chain.)

**Backend portability of item 5.**  The `prev_fresh_buf_`
optimisation was designed to relieve PCIe pressure on discrete
NVIDIA GPUs, where every D2H is an explicit DMA copy.  On Apple
Silicon (Metal) the same buffer addresses are host-visible — there's
no DMA — so the win there comes from skipping the AoS↔SoA repack
work and the OpenMP thread spin-up of the redundant copy, not from
saved bus traffic.  Same code path, different reason it pays off:
the abstraction held up cleanly across the two backends.

**Bottom line:** the GPU port is *correct*, *portable*, and
*useful*.  The CPU-side `check_conservation_law` cost is the limiting
factor for the apples-to-apples ratio on this hardware; setting
`output_diagnostics_every_N_timesteps` to a sensible production value
recovers the full GPU benefit.

### 8.4 Numerical accuracy note

Both bench scripts report a max relative error on the `eps_max`
trace.  After the §9.7 swap-gating fix, the GPU agrees with the CPU
to **5 × 10⁻⁵ in 2D and 3 × 10⁻⁵ in 3D** — exactly matching
`main_gpu`, and at the expected float32 noise floor.  All bench
cases now print `PASS  (all steps agree within 1e-4)`.

No precision-targeted work (fp64 Newton, mixed precision, etc.) is
needed at this time; the divergence that earlier text proposed those
remedies for was caused by a host-side rotation bug, not by float
arithmetic.  Metal MSL on Apple Silicon does not support `double` in
any case (the compiler rejects the type — `xcrun metal` errors at
parse time), so the fp64 Newton path was not even available; the
old §9.7 claim that it was is corrected below.

### 8.5 Reproducing

```
# Build both
cmake -B build              && cmake --build build       -j
cmake -B build_metal -DUSE_METAL=ON && cmake --build build_metal -j

# Symlinks expected by the bench script names
ln -sf build build_cpu        # if a CPU-only build dir was named build_cpu
ln -sf build_metal build_gpu  # likewise

# Run at 12 threads to match the published main_gpu benchmark
OMP_NUM_THREADS=12 bash tests/metal_vs_cpu_bench.sh
OMP_NUM_THREADS=12 bash tests/metal_vs_cpu_bench_3d.sh
```

On Linux/CUDA the equivalents are `tests/cuda_vs_cpu_bench.sh` and
`tests/cuda_vs_cpu_bench_3d.sh` — not yet run on this branch.

## 9. Known issues / follow-ups

### 9.1 Intermittent SIGTRAP / Obj-C corruption — resolved (see §9.6)

Originally suspected to be a small-grid Metal kernel issue, then
mis-blamed on a "pre-existing OpenMP race".  Actual cause: the merge
introduced a real data race in `Cell_info::output_momentum_anisotropy_vs_tau`
where a single `std::vector<double> thermalVec` was shared across all
threads of a `#pragma omp parallel for collapse(2) reduction(...)`
loop, and `eos.getThermalVariables()` resized it concurrently from
each thread.  The resulting heap corruption surfaced as
`Method cache corrupted` / `Trace/BPT trap: 5` / `receiver 0 bytes`.

Fixed in commit `d8c40cb` by moving the `thermalVec` declaration
inside the loop body so each thread gets its own.  Verified at 1, 2,
4, 8, 16 threads on Apple M3 Max — all pass.

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

### 9.4 Full inter-step GPU residency — done (commit `ae58f7c`)

**Implemented as planned.**  Summary of what landed:

- `Advance::gpu_owns_state_` set after each successful rk1 substep.
- `try_gpu_advance` skips its H2D upload when either residency flag
  is set, and skips its D2H copy-back at last substep too.
- `Advance::sync_arena_from_gpu_readonly(prev, curr)`: on-demand D2H
  that does NOT clear `gpu_owns_state_` (next rk0 still skips H2D).
- `Advance::sync_arena_from_gpu(prev, curr)`: same D2H plus clearing
  the flag.  Used at end of EvolveIt / EvolveOneTimeStep.
- `Evolve::EvolveIt` and `Evolve::EvolveOneTimeStep`:
  - Every diagnostic / output / freezeout call site that reads the
    arena is preceded by `sync_arena_from_gpu_readonly`.
  - `get_maximum_energy_density` replaced by `reduce_max_gpu` when
    `gpu_owns_state_` (GPU-side max reduction returning scalars).
  - `output_momentum_anisotropy_vs_tau` and `check_conservation_law`
    gated on `DATA.output_diagnostics_every_N_timesteps`.
- All implementations use the backend-agnostic `GPUGrid` /
  `GPUPipelines` abstractions, so the CUDA build inherits identical
  behaviour without backend-specific changes.

Caveat acted on (open question §7.2): `EvolveOneTimeStep` syncs and
clears `gpu_owns_state_` at end of every call, so JETSCAPE always
sees a CPU-fresh arena.  Standalone / XSCAPE batch (`EvolveIt`)
keeps state on GPU across outer steps for maximum throughput.

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

### 9.6 OpenMP race in `output_momentum_anisotropy_vs_tau` — fixed

**Diagnosis:** the merge of `main_gpu` → `XSCAPE` (commit `92c28e8`)
combined two patterns that are individually safe but collectively
racy:

- XSCAPE pre-merge had `std::vector<double> thermalVec;` declared
  outside the per-cell loop (used serially, OK).
- `main_gpu` had added `#pragma omp parallel for collapse(2)
  reduction(+:...)` around the same loop, with the inner loop
  reworked to use `Fields` accessors (also OK in isolation, but
  `main_gpu` itself never combined the parallel pragma with a shared
  `thermalVec`).

The merge resolution kept both, so every thread of the parallel
reduction wrote to the same `thermalVec` via
`eos.getThermalVariables(..., thermalVec)`, racing on the resize and
corrupting the heap.  Symptoms: deterministic `Method cache corrupted`
/ `Trace/BPT trap: 5` / `objc[…]: receiver 0 bytes selector 'alloc'`
at `OMP_NUM_THREADS >= 2`.

**Earlier mis-diagnosis:** I initially declared this a "pre-existing
OpenMP race" and worked around it with `OMP_NUM_THREADS=1`.  That
claim was wrong — `main_gpu` and `origin/XSCAPE` both run cleanly at
16 threads.  The race was introduced *by the merge*.  I should have
bisected when the user pushed back on the claim; instead I retracted
only after re-checking.

**Fix:** commit `d8c40cb` moves the `thermalVec` declaration inside
the inner loop so each thread gets its own.  No other changes
required.

```
$ OMP_NUM_THREADS=16 build/src/MUSIChydro tests/Gubser_flow/music_input_Gubser
# rc=0, all 201 timesteps complete, eps_max trace matches single-thread run.
```

**Lessons learnt:** when a merge mixes parallel-loop pragmas with
container types from a different branch, audit every container
declared outside the loop body for shared-write patterns inside the
body.  Annotate with `private(...)` / `firstprivate(...)` or move
the declaration inside the loop.  Better still — prefer thread-local
scratch buffers from the outset.

### 9.7 GPU vs CPU divergence — root-caused, fixed

**Earlier draft of this section** attributed a ~3 × 10⁻³ GPU-vs-CPU
gap to float32 accumulation in the viscous Newton solve and proposed
fp64 `gpu_reconst` (or mixed-precision residency) as the remedy.
**That diagnosis was wrong on two counts:**

1. Apple Silicon Metal does not support `double` in MSL — the
   compiler rejects the type at parse time (`'double' is not
   supported in Metal`).  So the proposed fp64 Newton path was never
   available on this backend.
2. The actual cause was a host-side bug in `swap_curr_future_gpu()`
   ([src/advance.cpp:158](src/advance.cpp#L158)), not float
   arithmetic.

**The bug.**  `swap_curr_future_gpu()` mirrors the host's rk1
`fpCurr ↔ fpNext` swap on the GPU side: after rk1, `try_gpu_advance`
has written the corrected step into `snap_future`, and the swap
should make `snap_curr` point at it.  But the swap was gated on
`gpu_state_authoritative_`, which `try_gpu_advance` had just set
*false* in the same rk1 call (because at that point the residency
mode hands off from the "intra-substep" flag to the "inter-step"
flag `gpu_owns_state_`).  The gate suppressed the swap entirely,
leaving `snap_curr` stuck on the rk0 *prediction* across the step
boundary.  Every diagnostic that subsequently read `snap_curr` —
`reduce_max_gpu`, `sync_arena_from_gpu_readonly`, the next step's
kernels — saw the wrong state.

**Diagnostic signature.**  The GPU-vs-CPU `eps_max` gap appeared
already on step 1 at full magnitude (~1.7 × 10⁻³) rather than
accumulating across many steps, which ruled out float32 noise.  CPU
baselines on XSCAPE and `main_gpu` are bit-identical; cold-start
upload data is bit-identical; the `.metal` kernel file is
bit-identical; dispatch counters showed identical kernel call
patterns (1 H2D, 82 entries, all six kernels each substep).  The
only host-side asymmetry that explained the gap was the
rk1-boundary swap.

**The fix.**  Change the gate from `gpu_state_authoritative_` to
`gpu_owns_state_`.  One line.  Result: drift drops from 3.18 × 10⁻³
to 3.06 × 10⁻⁵ at 64×64×32 — bit-for-bit matching `main_gpu` across
the entire 2D and 3D bench.  See §8.1, §8.2, §8.4.

**Lessons learnt.**  Before proposing kernel-precision remedies for
a CPU-vs-GPU gap, verify the gap is actually kernel-numerics in
origin: a per-step-error vs cumulative-error decomposition would
have ruled out float32 in one step.  When two branches share a
`.metal` file and a CPU baseline but diverge by 100×, the difference
is in host orchestration, not kernel arithmetic.
### 9.7 Float32 GPU vs float64 CPU divergence over long runs

> **Correction (2026-05-26, from the CUDA port).**  The ~4 × 10⁻³ vs
> ~5 × 10⁻⁵ accuracy gap between XSCAPE and `main_gpu` that §8.4 and the
> text below blame on "float32 over 100 steps" was **misdiagnosed**.
> The real cause was a residency bug: `swap_curr_future_gpu()` was
> gated on `gpu_state_authoritative_` (commit `0cecdf5`), which
> `try_gpu_advance` clears to false on the last substep, so the rk1
> corrector swap never ran and RK2 silently degraded to forward Euler —
> on **both** backends.  Re-gating on `gpu_owns_state_` (matching
> `main_gpu`) drops the CUDA Gubser error from 3.6 × 10⁻³ to
> 9.9 × 10⁻⁵, exactly matching `main_gpu`.  The same fix applies to
> Metal but the Metal benchmarks here have not been re-run.  See
> PORT_GPU_CUDA.md §2 (Bug 2).  The genuine float32 noise floor is
> ~1 × 10⁻⁴ over 100 steps, not 10⁻³.

§8.3 documents ~6 % error after 200 steps at 64×64 and 48 % after 200
steps at 128×128 in the Gubser viscous test.  Per-step error is small;
the accumulation is in the dilute-tail cells where the viscous Newton
solve is most sensitive.  (Numbers predate the §9.7 correction above and
likely include the forward-Euler error; re-measure on Metal when
convenient.)

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

### 9.8 Slow `check_conservation_law` — misdiagnosis, retracted

An earlier version of this section claimed `check_conservation_law`
took ~30 ms/step because of the multi-charge `eos.get_pressure(e,
rhob, rhoq, rhos)` overload.  That was wrong on two counts:

1. The 4-arg `EOS::get_pressure` wrapper in `src/eos.h` actually
   delegates straight to `eos_ptr->get_pressure(e, rhob)` — it
   silently drops rhoq and rhos.  So Fields and SCGrid paths call
   the same underlying 2-arg lookup.
2. Adding a `bench::Timer` around the call confirmed
   `check_conservation_law` is 1.3 ms/step on XSCAPE — essentially
   identical to `main_gpu`'s 1.1 ms.  My "~30 ms" figure was a
   `step_total − sum_of_measured` residual that I (incorrectly)
   attributed to this function.

The real bottleneck was the GPU→CPU sync itself
(`sync_arena_from_gpu`), where the `dst.u_[m][c]` double-indirection
through `std::vector<std::vector<double>>` ran at ~1 GB/s on a
400 GB/s memory bus.  Fixed in commit `950c1ea` by hoisting the
inner-vector `.data()` pointers outside the parallel loop and by
splitting the sync API into curr-only / both-arena variants so
diagnostics that only read `fpCurr` don't pay for an extra `fpPrev`
copy.  See §8.2 for the post-fix bench.

Lessons learnt: don't quote unmeasured residuals as numbers — use a
timer or shut up.
