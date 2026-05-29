# Add GPU Improvements — `output_evolution` hot path in music4gpu

> **Status:** design / proposal. Companion to the `README_CUDA.md` /
> `PORT_GPU.md` / `PORT_GPU_CUDA.md` port-doc series. Phases are intended to be
> implemented in order; measured numbers are folded back in as each lands.

## Context

On the CUDA-GPU MUSIC path, `output_evolution_every_N_timesteps` is the dominant
wall-clock cost. Root cause: when `it % Nskip_timestep == 0`
(`external_packages/music4gpu/src/evolve.cpp:135`) two things fire that the
Phase 6b/6c OpenMP work never touched (that work only fixed the *diagnostic*
functions, `output_momentum_anisotropy_vs_tau` / `check_conservation_law`):

1. **Full-arena D2H** — `Advance::sync_arena_from_gpu_readonly`
   (`evolve.cpp:140`, defined `advance.cpp:488`) copies primitives **and** all
   10–14 `Wmunu` viscous components back to host.
2. **A serial host loop** — `Cell_info::OutputEvolutionDataXYEta_memory`
   (`grid_info.cpp:372`) walks every cell doing **3 EOS calls**
   (`get_pressure` + `get_temperature` + `get_entropy`, lines 391–396) — note
   `get_entropy` *internally re-calls* `get_pressure` + `get_temperature`
   (`eos_base.cpp:211–212`), so P and T are interpolated twice — and a
   `lattice_ideal.push_back` (`HydroinfoMUSIC.cpp:389`) onto a vector that is
   **never `reserve`d**. No `#pragma omp` in this function — single-threaded.

The XSCAPE path (`store_hydro_info_in_memory=1`, set in `src/hydro/MusicWrapper.cc:79`)
consumes only **ideal** info — `dump_ideal_info_to_memory` stores `e, p, s, T,
ux, uy, ueta` (`fluidCell_ideal`, `data_struct.h:43`) and never reads `Wmunu` —
yet the sync still pays for the full viscous D2H + repack.

Goal: make each output **cheaper** (not rarer), so the stored hydro for the iSS
sampler keeps its τ-resolution.

### Memory-model note (drives the phasing)

Both `copy_primitives_to_cpu` / `copy_wmunu_to_cpu`
(`gpu/GPUGrid_cuda.cu:395–453`, mirrored in `gpu/GPUGrid.mm`) have two parts:
a PCIe `cudaMemcpy` gated behind `if (!g_cuda_coherent)` (**skipped on
GB10 / Apple-Metal unified memory**), and an always-run SoA-float→host-double
**repack loop**. So the *transfer*-elimination value is discrete-CUDA-only; the
host loop and the repack are paid on every backend.

- **Phase 1 (host-side) is portable** — helps identically on discrete CUDA,
  GB10, and Metal, and is *relatively more* impactful on the unified-memory
  machines because there is no D2H to mask the serial loop.
- **Phase 2a/2b** target the transfer + viscous repack; the transfer half is
  discrete-only, the repack-avoidance / EOS-on-device half still helps unified.

---

## Phase 1 — Portable host-side wins (value-preserving, all backends)

Target files: `external_packages/music4gpu/src/HydroinfoMUSIC.{cpp,h}`,
`external_packages/music4gpu/src/grid_info.cpp`.

**1.1 — Reserve `lattice_ideal`.** In `HydroinfoMUSIC::set_grid_infomatioin`
(`HydroinfoMUSIC.cpp:332`) the per-frame cell count `ixmax*iymax*ietamax` is
computed (lines 349–354). The frame count ≈ `DATA.nt /
DATA.output_evolution_every_N_timesteps + 1`. Add
`lattice_ideal.reserve(cellsPerFrame * estFrames)` (+ small margin). This kills
the O(N) realloc/copy of the growing vector across the run. `reserve` is only a
capacity hint, so the `beastMode==2` path that mutates
`output_evolution_every_N_timesteps` mid-run (`evolve.cpp:99–105`) is harmless —
an underestimate just degrades to occasional reallocation.

**1.2 — OpenMP-parallelize the output loop.** This is the main per-output win.
The current `push_back` is order-dependent and blocks parallelism. Restructure
`OutputEvolutionDataXYEta_memory` (`grid_info.cpp:372`):

- Compute `nx_out / ny_out / neta_out` (the `for (i=0; i<n; i+=n_skip)` iteration
  counts) and allocate a per-frame local `std::vector<fluidCell_ideal>
  frame(nx_out*ny_out*neta_out)`.
- Fill it with `#pragma omp parallel for collapse(3)`, each iteration writing its
  **own** slot at `flatIdx = (ix_idx*ny_out + iy_idx)*neta_out + ieta_idx` where
  `ix_idx = ix/n_skip_x` etc. This flat index reproduces the **exact original
  push order** (ix outer, iy middle, ieta inner), so the resulting `lattice_ideal`
  is byte-identical to the serial version regardless of the read-side
  `position[...]` indexing in `getHydroValues` (`HydroinfoMUSIC.cpp:~199`). The
  per-cell EOS getters are `const` and already proven thread-safe in the
  parallelized diagnostic loop at `grid_info.cpp:2631`.
- Add a batched sink on `HydroinfoMUSIC` — `dump_ideal_info_frame_to_memory(double
  tau, const std::vector<fluidCell_ideal>& frame)` — that does the τ-bookkeeping
  **once** (`if (tau > hydroTauMax) { hydroTauMax = tau; ++itaumax; }`, identical
  to the per-cell logic at `HydroinfoMUSIC.cpp:376–378` since all cells in a frame
  share `tau`) and bulk-`insert`s the frame in order. Keeps `lattice_ideal`
  private and the append order exact.

Expected: per-output host cost drops ≈ thread-count× with **no change to stored
values**.

**Measured (Phase 1 — `OO_one_event`, EOS 91, 98.4 M cells over 164 output
frames, 48-core host + CUDA GPU):** `grid.output_evolution_memory` (new
`bench::Timer` hook, `MUSIC_PROFILE=1`) dropped from **8.57 s → 2.63 s**
(52.3 → 16.1 ms/frame) going 1 → 16 threads — **3.3×** — and is no longer the
dominant host cost. The stored-lattice FNV-1a checksum is **identical** at 1 and
16 threads (`4786563510388689633`), confirming the parallel fill is race-free and
order-preserving (and the 1-thread path reproduces the former serial arithmetic
exactly). Sub-linear scaling is Amdahl-limited by the per-cell EOS-table gather
(latency/bandwidth bound) — the deeper win is Phase 2b (EOS on device). As a
side effect of running threaded, the `advance.sync_arena_from_gpu` repack also
fell 4.31 → 0.84 s. Total run wall: 46 s → 32 s. Status: **done**.

**1.3 — (optional, deferred) value-preserving EOS de-dup.** The redundant
double-interpolation of P and T (see Context) is ~2 of the ~7 effective lookups.
⚠️ **Do NOT swap in `eos.getThermalVariables`** as a shortcut: its entropy
(`eos_base.cpp:251`) uses the raw passed `rhoq/rhos` with a **swapped muS/muQ
pairing** and ignores `get_rhoS`/`get_rhoQ`, whereas `get_entropy`
(`eos_base.cpp:209`) recomputes `rhoS=get_rhoS(e,rhob)` / `rhoQ=get_rhoQ(e,rhob)`
(for `whichEOS!=20`; the config uses `EOS_id_MUSIC=91` → `EOS_hotQCD`,
`eos.cpp:25`). The two formulas only coincide when all chemical
potentials/charges vanish — not a safe general assumption, and `sd` *is* consumed
downstream (`HydroinfoMUSIC.cpp:227,308`). A bit-identical de-dup therefore needs
a **new combined accessor** that computes P and T once and reuses them in
`get_entropy`'s exact formula. Since Phase 1.2 already parallelizes these calls,
this is a low-priority follow-up, not part of the main push.

---

## Phase 2a — Ideal-only D2H sync (mainly discrete CUDA; partial on unified)

Target files: `external_packages/music4gpu/src/advance.{h,cpp}`, call site
`evolve.cpp:140`.

- Add `sync_curr_ideal_from_gpu_readonly` (or a flag on the existing readonly
  sync) that calls only `copy_primitives_to_cpu` and **skips**
  `copy_wmunu_to_cpu`, used at `evolve.cpp:140` when
  `DATA.store_hydro_info_in_memory == 1 && DATA.outputEvolutionData == 0`
  (memory-ideal path only).
- Discrete CUDA: removes the `Wmunu` PCIe transfer (~14 of ~19 components).
  GB10 / Metal: removes the `Wmunu` repack loop.
- **Care required:** the freshness flags (`host_curr_fresh_`, `host_prev_fresh_`,
  `advance.cpp:488–527`) currently mean "full arena synced." An ideal-only sync
  must not set them in a way that lets a later freeze-out read
  (`store_previous_step_for_freezeout`, the `iFreezeStart`/movie syncs at
  `evolve.cpp:121,170`) or an `outputEvolutionData>0` writer see stale/absent
  `Wmunu`. Introduce a separate `host_curr_wmunu_fresh_` flag, or keep the
  ideal-only sync strictly scoped to the memory-ideal output path that provably
  never needs `Wmunu`. Verify against both paths.

---

## Phase 2b — GPU-side evolution packing kernel (highest ceiling)

Target files: `gpu/GPUGrid_cuda.cu` + `gpu/music_kernels.cu(.cuh)`
(`gpu/GPUGrid.mm` / `.metal` for Metal parity), `grid_info.cpp`,
`advance.{h,cpp}`.

- Add a device kernel that, over the **downsampled** grid
  (`output_evolution_every_N_x/y/eta`), computes the ideal-info tuple
  `e, p, s, T, ux, uy, ueta` on-device into a compact buffer of
  `ixmax*iymax*ietamax` entries, then transfers (discrete) / exposes (unified)
  only that small buffer; the host bulk-appends it via the Phase 1.2 batched sink.
- Eliminates in one move: the full-arena D2H, the SoA→AoS repack, and the serial
  host EOS loop. On unified memory the win is repack-removal + moving EOS onto
  thousands of GPU lanes (the dominant remaining cost once transfer is free).
- **De-risked on CUDA — the EOS table is already device-resident.**
  `GPUGrid::upload_eos` (`gpu/GPUGrid_cuda.cu:158`) uploads `eos_P, eos_dPde,
  eos_s, eos_T` as managed buffers, and existing kernels already consume them
  (`gpu/music_kernels.cuh:39–104`). So the packing kernel mostly *reuses* those
  buffers — no new EOS upload on CUDA. Metal would need the analogous table in
  the `.metal` path.
- **Note (not bit-identical):** device entropy/temperature come from the EOS
  *table* (`eos_s`, `eos_T`), whereas the current host path computes entropy from
  the thermodynamic identity in `get_entropy`. For the μ_B=0 hotQCD EOS these
  agree to interpolation tolerance, but Phase 2b should be validated as
  *within-tolerance*, not bit-identical.
- This is the proper "Phase C for evolution output," analogous to the deferred
  Phase C in `README_CUDA.md:574`.

---

## Verification

- **Correctness, Phase 1 (must be bit-identical):** run a MUSIC test with
  `store_hydro_info_in_memory=1`, dump `lattice_ideal`, and diff byte-for-byte
  against the pre-change baseline. The reserve + reorder-preserving parallel fill
  guarantee identical output; any diff is a bug.
- **Correctness, Phase 2a/2b (within tolerance):** same comparison, accept the
  documented ≈1e-3 regression threshold (float repack for 2a; table-vs-formula
  entropy for 2b).
- **Performance:** profile with the existing `bench::Timer` hooks
  (`advance.sync_arena_from_gpu`) and add one around
  `OutputEvolutionDataXYEta_memory`. Confirm Phase 1 scales with OMP threads;
  confirm Phase 2a drops D2H volume on discrete and repack time on coherent.
  Reuse the `tests/cuda_perstep_bench.sh` init-corrected methodology.
- **Cross-backend:** validate Phase 1 on a discrete CUDA build *and* a coherent
  (`g_cuda_coherent`) / Metal build to confirm the portable speedup.
- **End-to-end:** run a full XSCAPE event (3DGlauber→MUSIC→iSS) and confirm the
  sampled particle spectra are unchanged.

## Documentation

After implementation, fold measured numbers into this doc and cross-link the
`README_CUDA.md` §"Phase C" / knobs discussion and the existing
`output_evolution_every_N_timesteps` time-sink note.
