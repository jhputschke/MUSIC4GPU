# Plan — GPU-Resident State Across Timesteps

Status: **proposed, not started.** Deferred until a discrete GPU is available
for performance validation (the win is largest on PCIe-attached parts; on the
coherent GB10 the gain is the host-side pack/unpack cost only).

This is the highest-value remaining optimization identified during the CUDA
port (see `README_CUDA.md` → "the host-bound finding"). It is an **evolve-loop**
restructure, not a kernel change.

---

## 1. Motivation and evidence

Profiling the production grid (64×64×32) on GB10 showed the hydro **kernels are
only ~5.5% of wall time** (≈22% of the evolution phase). The run is host-bound:
every timestep packs AoS→SoA on the host, moves the grid host↔device, and
unpacks. On a discrete GPU this host↔device move is **PCIe traffic** (~14 MB
H2D for `current`+`prev`, ~7 MB D2H for `future`, per step) and is expected to
dominate even more.

Snapshot rotation already removes the *intra*-timestep re-upload. Residency only
breaks at the **timestep boundary**, forced by two things:
1. `get_maximum_energy_density(*ap_current)` runs on the host **every** step.
2. The next `AdvanceRK` re-uploads the evolving state from the host arena.

Key enabling fact (verified by reading `evolve.cpp`): **between `AdvanceRK`
calls, nothing writes the host arenas `ap_*`** — diagnostics, output, and
freeze-out only *read* them (`store_previous_step_for_freezeout` writes the
separate `arena_freezeout`). So the GPU can own the evolving state across
timesteps; the host arenas become a cache, synced on demand.

---

## 2. Current flow (per timestep `it` at `tau`)

```
EvolveIt loop (src/evolve.cpp):
  [top]  get_maximum_energy_density(*ap_current)      ← reads HOST every step
         output / freezeout / conservation checks     ← read HOST (periodic)
  AdvanceRK(tau, prev, current, future):              ← src/evolve.cpp
     rk0: AdvanceIt → upload(cur,prev) H2D, kernels → snap_future,
                      (authoritative ⇒ no copyback)
          host rotate:  prev←cur←fut←temp
          GPU rotate_snapshots()  mirrors the host 3-cycle
     rk1: AdvanceIt → rotate_snapshots, kernels → snap_future,
                      copyback D2H → arena_future
          host swap(cur, fut)
```

The GPU is authoritative *within* a timestep (the `gpu_state_authoritative_`
heuristic in `Advance::AdvanceIt`). The copy-back at rk1 + re-upload at the next
rk0 is the cost to remove.

---

## 3. Proposed flow

```
EvolveIt loop:
  [top]  if (gpu_owns_state):
             {eps_max, rhob_max} ← gpu_reduce_max(snap_current)   ← GPU reduction
             T_max  = eos.get_temperature(eps_max)                ← host scalar (T(e) monotone)
             if (step needs host data: output | freezeout | conservation):
                 sync_host_from_gpu(ap_current [, ap_prev])       ← copyback ONLY then
         else:
             get_maximum_energy_density(*ap_current)              ← unchanged fallback
  AdvanceRK:
     rk0: AdvanceIt (NO upload, NO copyback)
          host rotate  +  GPU rotate_snapshots()
     rk1: AdvanceIt (NO upload, NO copyback)
          host swap     +  GPU swap_curr_future()
```

State is uploaded **once** at init; afterward it lives on the GPU and rotates in
lockstep with the host arena pointers. Copy-back happens only on steps that
consume host data (output cadence, freeze-out, conservation) — usually a small
fraction of steps.

---

## 4. Concrete pieces (file by file)

### 4.1 `GPUGrid` (`src/gpu/GPUGrid.h`, `GPUGrid_cuda.cu`)
- Add `void swap_curr_future();` — swaps the `snap_curr` / `snap_future`
  structs (pointer-alias swap, no data move), mirroring `AdvanceRK`'s rk1
  `std::swap(arena_current, arena_future)`.
- `rotate_snapshots()` already matches the rk0 3-cycle — reuse as is.

### 4.2 GPU reduction kernel (`music_kernels.cu` / `.cuh`, `CUDAPipelines`)
- `__global__ void gpu_reduce_max(const float* eps, const float* rhob,
  int Ncells, float* out_eps, float* out_rhob)` — block reduction +
  `atomicMax` on device scalars (or `cub::DeviceReduce::Max` twice).
- `CUDAPipelines::reduce_max(GPUGrid&, double& eps_max, double& rhob_max)` —
  launches it on the compute stream, copies the two scalars back, syncs.
- `T_max` is **not** reduced on the GPU: `T(e)` is monotone in `e` for the
  rhob=0 EOS, so `T_max = eos.get_temperature(eps_max, 0)` on the host. (If a
  finite-μB EOS is ever ported, revisit — reduce `T` per cell instead.)
- Mirror in `MetalPipelines` (a Metal reduction or a small CPU fallback) so the
  shared `GPUPipelines` alias keeps compiling; or guard the call by back-end.

### 4.3 `Advance::AdvanceIt` (`src/advance.cpp`)
- Replace the per-substep `gpu_state_authoritative_` heuristic with a
  run-level `gpu_owns_state_` flag (member of `Advance`, set once when the
  fully-GPU path is active for the whole run; see §4.5).
- When `gpu_owns_state_` is true:
  - rk0/rk1: do **not** `copy_to_gpu` and do **not** `copy_*_to_cpu`. The GPU
    snapshots already hold the state; the rotations (done in `AdvanceRK`) keep
    `snap_X ≡ ap_X`.
  - Keep the existing within-substep dispatch sequence unchanged.
- Init upload: on the very first `AdvanceIt` (or an explicit
  `upload_initial_state()`), `copy_to_gpu` the initial-condition arenas once.

### 4.4 `Evolve` (`src/evolve.cpp`)
- `AdvanceRK`: after each host pointer move, mirror it on the GPU snapshots —
  `rotate_snapshots()` after the rk0 3-cycle, `swap_curr_future()` after the
  rk1 swap. (Guard by back-end / `gpu_owns_state_`.)
- `EvolveIt` top-of-loop:
  - Replace `get_maximum_energy_density(*ap_current)` with the GPU reduction
    path when `gpu_owns_state_`.
  - Before any block that reads `ap_current`/`ap_prev` on the host (output,
    freeze-out storage, Cornelius, conservation), call
    `sync_host_from_gpu(...)`. Freeze-out needs **two** consecutive frames, so
    sync both `ap_current` and `ap_prev` on freeze-out steps.
- Add `Evolve::sync_host_from_gpu(GridPointer& current[, prev])` wrapping the
  existing `GPUGrid::copy_primitives_to_cpu` + `copy_wmunu_to_cpu`.

### 4.5 Run-level guard `gpu_owns_state_`
- True only when the fully-GPU path is active for the **whole** run:
  `gpu_ready_ && gpu_w_full_active-equivalent config` (a run-constant decision
  derived from `DATA`, same predicate the dispatch already computes). Any
  fallback config keeps today's per-substep copy-back path untouched.
- If any host code path ever writes `ap_*` mid-run (none today — must re-audit
  if new features are added), mark the state dirty and force a re-upload.

---

## 5. Correctness validation

- **Lockstep invariant.** In a debug build, after each `AdvanceRK`, sync the
  GPU snapshots to scratch host arrays and assert they equal `ap_*` for the
  first few steps. This catches any rotation/permutation mismatch immediately.
- **Regression guard.** The existing `tests/cuda_vs_cpu_bench{,_3d}.sh`
  (max rel `eps_max` error < 1e-3) already exercises multi-step evolution end
  to end — it will catch a broken residency/sync.
- **Reduction kernel.** Unit-check `gpu_reduce_max` against
  `get_maximum_energy_density` on the same grid before wiring it in.
- **`MUSIC_CUDA_FORCE_DISCRETE=1`** exercises the discrete memory path on
  coherent hardware (correctness only — see README caveat).

---

## 6. Expected benefit and where to measure

- Eliminates the per-timestep H2D upload (`current`+`prev`) and D2H copy-back
  (`future`) on the majority of steps → removes the dominant PCIe traffic on a
  discrete GPU, and the host AoS↔SoA pack/unpack cost on every platform
  (directly attacks GB10's ~94% host-bound fraction).
- **Measure on a real discrete GPU** (A100/RTX/H100): `nsys` timeline before/
  after to confirm the host↔device gaps shrink; `tests/cuda_perstep_bench.sh`
  for per-step wall time. On GB10, measure the reduction in host pack/unpack
  time (kernels are already a tiny slice, so wall change will be smaller).

---

## 7. Effort / risk

- **Effort:** medium. Most of it is in `evolve.cpp` (loop + sync points) and
  `advance.cpp` (gate the copies), plus a small reduction kernel and one
  `GPUGrid` method. No change to the seven physics kernels.
- **Risk:** the rotation lockstep and sync-point placement are the hazards;
  the debug-build invariant in §5 de-risks them. The change is behind
  `gpu_owns_state_`, so unsupported configs are unaffected.

---

## 8. Suggested sequencing

1. **GPU `eps_max`/`rhob_max` reduction** (§4.2) — standalone, low-risk,
   independently testable, and removes the every-step host read that is the
   precondition for residency. Do this first.
2. **`swap_curr_future` + lockstep mirroring + `gpu_owns_state_` gating +
   on-demand `sync_host_from_gpu`** (§4.1, §4.3, §4.4, §4.5) — the residency
   itself.
3. Validate (§5), then benchmark on discrete hardware (§6).
