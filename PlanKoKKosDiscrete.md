# Discrete-GPU Optimization Plan for the Kokkos Backend

## Context

The Kokkos backend (`PlanKokkosPort.md` Stages 1–5, see `Port_GPU_KoKKos.md`) was
written and validated on an **NVIDIA GB10** — a Grace-Blackwell part with
**coherent unified memory**, where host and device share the same physical RAM.
To match that hardware the port uses **one memory path for everything**:

```cpp
// GPUGrid_kokkos.cpp
using KView = Kokkos::View<float*, Kokkos::SharedSpace>;   // == CudaUVMSpace (managed)
```

`Kokkos::SharedSpace` resolves to `cudaMallocManaged` (CUDA) / `HIPManagedSpace`
(HIP) / `SYCLSharedUSMSpace` (SYCL) / `HostSpace` (CPU).  The host packs AoS→SoA
straight into the managed buffer, kernels read it in place, and
`upload_snapshots_async()` is a **no-op** — zero-copy on a coherent part.

This is **correct on a discrete GPU** (A100 / H100 / RTX / MI250X / PVC): managed
memory is demand-paged, so it still works — but it is **not optimal**.  The native
CUDA backend carries a *second*, discrete-specific memory path that the Kokkos
backend has no equivalent for.  **This document is the pick-up plan to add it.**

> **Scope:** throughput only.  The kernels are unchanged, so device **precision is
> unaffected** (still 6.44e-04 vs CPU, identical to native CUDA — the goal gate is
> not at risk).  Everything here is a memory-traffic optimization.

> **Caveat:** none of this is testable on the development box (GB10 only — no
> discrete GPU).  The steps below are validated *designs*; each must be
> re-benchmarked on actual discrete hardware (see Verification).

---

## TL;DR verdict

| | Status |
|---|---|
| Runs correctly on a discrete GPU today | ✅ yes (managed/UVM is demand-paged but correct) |
| Steady-state GPU compute on discrete | ✅ fine — managed pages migrate to device once and stay resident (cross-step residency keeps the host from touching them) |
| Bulk host↔device transfers on discrete | ❌ fault-driven page migration instead of pinned bulk DMA |
| Copy/compute overlap on discrete | ❌ none (`upload_snapshots_async` is a no-op) |
| Memory-mode selection (coherent vs discrete) | ❌ always managed; no runtime detection |
| Read-only/`__ldg` EOS cache | ❌ dropped in the port (plain loads) — helps all GPUs, more on discrete |

The fix is mechanical and the seam is already in place: `GPUSnapshot` still
carries the `*_stage` pinned-staging pointers (currently left `nullptr`),
`upload_snapshots_async()` already exists as the H2D hook, and the
`copy_*_to_cpu` methods are the D2H hook.  The work is filling them in with the
Kokkos pinned-mirror + `deep_copy` + `partition_space` equivalents.

---

## What already carries over to discrete (do NOT redo)

1. **Cross-step residency** (`advance.cpp`: `gpu_owns_state_`,
   `gpu_state_authoritative_`) is **backend-agnostic and already active**.  It
   skips the H2D on most substeps/steps and keeps evolving state on the device
   across timestep boundaries.  With managed memory the snapshot pages migrate to
   the device on first touch and **stay there** while the host doesn't read them,
   so steady-state *compute* never faults.  This is why the discrete penalty is
   concentrated at (a) the cold-start upload and (b) host-side diagnostic syncs
   (`sync_arena_from_gpu`, freezeout, output writers) — not the hot loop.
2. **Allocate-once + pointer-rotation snapshots** (`rotate_snapshots`,
   `swap_curr_future`) — View handles never reallocate per step; the snapshot
   roles rotate by alias swap, same as native CUDA.
3. **Stage-2 GPU tuning** (fast-math, occupancy MDRange tile in `range3()`) — the
   tile is geometry-only and device-agnostic; re-tune the *values* on discrete
   (see DG3) but the mechanism stands.
4. **The single-source kernels** (`music_kernels_kokkos.hpp`) — unchanged; the
   discrete work touches only `GPUGrid_kokkos.cpp` and `KokkosPipelines.cpp`.

---

## The gap: native-CUDA discrete optimizations → Kokkos equivalents

The native backend (`GPUGrid_cuda.cu`, `CUDAPipelines.cu`) picks its path at
runtime from `g_cuda_coherent`.  The discrete branch and its Kokkos translation:

| Native CUDA (discrete branch) | Kokkos equivalent to add | File |
|---|---|---|
| `cudaMalloc` device snapshot buffers | `Kokkos::View<float*>` in the **default device space** (`CudaSpace`/`HIPSpace`/`SYCLDeviceUSMSpace`) | `GPUGrid_kokkos.cpp` |
| `cudaHostAlloc` **pinned host staging** (`*_stage`) | `Kokkos::View<float*, Kokkos::SharedHostPinnedSpace>` mirrors (page-locked, host-written) | `GPUGrid_kokkos.cpp` |
| host packs into pinned staging | pack into the pinned mirror's `data()` (unchanged loop) | `GPUGrid_kokkos.cpp` |
| `upload_snapshots_async`: `cudaMemcpyAsync` H2D on a copy stream + event, compute stream waits | `Kokkos::deep_copy(copy_inst, dev_view, pinned_mirror)` on a **`partition_space`** instance, gate with `.fence()` | `KokkosPipelines.cpp` |
| D2H `cudaMemcpy` in `copy_wmunu_to_cpu` / `copy_primitives_to_cpu` / `refresh_u_curr_stage` | `Kokkos::deep_copy(pinned_mirror, dev_view)` then unpack from the mirror | `GPUGrid_kokkos.cpp` |
| coherence detect (`cudaDevAttrPageableMemoryAccess`, `prop.integrated`) + `MUSIC_CUDA_FORCE_{DISCRETE,COHERENT}` | compile-time `Kokkos::has_shared_space` **+** a runtime device-coherence query; `MUSIC_KOKKOS_FORCE_{DISCRETE,COHERENT}` env overrides | `GPUGrid_kokkos.cpp` |
| dual `cudaStream_t` (compute + copy) | `Kokkos::partition_space(exec, 1, 1)` → two exec-space instances | `KokkosPipelines.cpp` |
| `cudaMemPrefetchAsync` hints (managed path) | keep only if the managed path is retained for coherent parts; N/A for the device+pinned path | — |
| `__ldg` read-only EOS cache | `View<const float*, MemoryTraits<RandomAccess>>` for the EOS tables | both |

**Construct note (portability win over CUDA):** keying the choice off
`Kokkos::has_shared_space` + a coherence query — not a CUDA-only bool — gives the
*same* discrete optimization on **AMD MI250X/Frontier and Intel PVC/Aurora**
discrete parts, and automatically keeps the zero-copy managed path on **APUs**
(MI300A) and **GB10/Grace** unified-memory parts.  One mechanism, every vendor.

---

## Design decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| **DD1** | Path selection | Compile-time `has_shared_space` gate, then a **runtime** coherence query inside `GPUGrid::allocate()`; cache the result in a TU-static `bool g_kokkos_coherent` (mirrors `g_cuda_coherent`). Env overrides `MUSIC_KOKKOS_FORCE_DISCRETE` / `MUSIC_KOKKOS_FORCE_COHERENT`. | A discrete GPU still *has* UVM (`has_shared_space==true`), so the compile-time trait alone can't decide — the runtime query (NVML/`cudaDeviceGetAttribute` under a CUDA guard, or the Kokkos device-property API) answers "*should* we use managed." |
| **DD2** | Pinned staging type | `Kokkos::View<float*, Kokkos::SharedHostPinnedSpace>` mirrors, allocated once next to each device snapshot View; reuse the existing `GPUSnapshot::*_stage` pointers to hold their `data()`. | Page-locked → real async DMA bandwidth; `SharedHostPinnedSpace` is the portable pinned space (CudaHostPinnedSpace/HIPHostPinnedSpace/SYCLHostUSMSpace). Keeps `host_readable_u_curr()` / `refresh_u_curr_stage()` working unchanged. |
| **DD3** | Transfer primitive | `Kokkos::deep_copy(exec_instance, dst, src)` (the exec-instance overload) — H2D in `upload_snapshots_async`, D2H in the copy-back methods. | Bulk DMA, replaces fault migration; the instance overload is what enables overlap (DD4). |
| **DD4** | Overlap | `Kokkos::partition_space(DefaultExecutionSpace(), 1, 1)` → `{compute_inst, copy_inst}`; run the next H2D `deep_copy` on `copy_inst`, gate the compute dispatches via `copy_inst.fence()` / a shared fence. Skip entirely on the coherent path (DD1). | The portable form of the native dual-stream prefetch; only worth it on discrete (no migrate-on-touch to hide). |
| **DD5** | Allocation model | Still **allocate-once** (device Views + pinned mirrors created in `allocate()`, freed in `release()`); rotation stays pointer-alias swap. No per-step alloc. | Same as today and as native CUDA; the stream-ordered allocator is intentionally N/A. |
| **DD6** | Coherent path | **Keep the current `SharedSpace` path unchanged** for coherent parts (GB10/APU); the discrete path is an added branch, not a replacement. | Don't regress the validated GB10 result; DD1 selects between them. |
| **DD7** | Validation tolerance | Unchanged — device precision is identical (same kernels). Gate on the existing **1e-3** EOS trace + the D9 cross-backend ≤1e-4. | This is a perf change; correctness must stay bit-stable vs the managed path. |

---

## Staged roadmap (discrete GPU)

| Stage | Scope | Difficulty | Outcome |
|---|---|---|---|
| **DG0 — Baseline** | On the target discrete GPU, build the port as-is (managed) and capture `eos_gpu_vs_cpu.sh` (precision — should already PASS) + `cuda_perstep_bench`/`kokkos` per-step + an output-heavy run. Profile with Nsight to confirm the cost is fault migration at the host-touch points. | Low | Numbers to beat; proof the bottleneck is transfers, not compute. |
| **DG1 — Device buffers + pinned mirrors (DD1–DD3)** | Add the discrete branch to `GPUGrid_kokkos.cpp`: device-space snapshot Views + `SharedHostPinnedSpace` mirrors (wire `*_stage`); pack into mirrors; `deep_copy` H2D in `upload_snapshots_async`, D2H in `copy_*_to_cpu` / `refresh_u_curr_stage`; runtime coherence query + env overrides. Re-validate precision + D9. | **Medium (core)** | Bulk DMA replaces fault migration; the main discrete win. Expect the cold-start + diagnostic-sync cost to drop sharply. |
| **DG2 — Copy/compute overlap (DD4)** | `partition_space` copy instance; overlap the next H2D with the prior substep's compute; gate with fences. Coherent path skips it. | Medium | Hides the residual H2D behind compute (matters for upload-heavy / non-resident phases). |
| **DG3 — Discrete tuning & polish** | Re-sweep the `range3()` MDRange tile per-kernel on the discrete arch (its register file/L2 differ from GB10); add `RandomAccess` EOS Views (`__ldg`); optionally per-kernel occupancy (`Kokkos::Tools` autotuning). Re-bench vs native CUDA on the same card. | Medium | Close the remaining gap to native CUDA; possibly exceed it via the portable trait abstraction. |
| **DG4 — Multi-vendor discrete (optional)** | Flip `Kokkos_ENABLE_HIP`/`SYCL` + arch, rebuild, re-run DG0→DG3 on MI250X/PVC. The `has_shared_space` selection (DD1) makes the discrete path apply unchanged. | Per-backend | First non-NVIDIA discrete runs; native CUDA can't do this at all. |

DG1 is the bulk of the win; DG2–DG3 chase parity; DG4 is the portability payoff.

---

## Concrete implementation plan (file-by-file)

### `src/gpu/GPUGrid_kokkos.cpp` (the core change)
1. **Two View types in `KStore`:**
   - coherent: `Kokkos::View<float*, Kokkos::SharedSpace>` (today's path).
   - discrete: `Kokkos::View<float*>` (device default space) **+** a parallel
     `Kokkos::View<float*, Kokkos::SharedHostPinnedSpace>` mirror per buffer.
   Store the device `data()` in the snapshot field (`s.epsilon`, …) and the
   mirror `data()` in the existing `s.epsilon_stage`, … (already in `GPUGrid.h`).
2. **Runtime selection (DD1):** in `allocate()`, set a TU-static
   `g_kokkos_coherent` from a device-coherence query (under `#if
   defined(KOKKOS_ENABLE_CUDA)` use `cudaDeviceGetAttribute(...,
   cudaDevAttrPageableMemoryAccess)` + `prop.integrated`; HIP/SYCL analogues
   guarded similarly) honoring `MUSIC_KOKKOS_FORCE_{DISCRETE,COHERENT}`. Pick the
   View type accordingly.
3. **`copy_to_gpu`:** discrete → pack into the pinned mirror (`*_stage`); coherent
   → pack into the managed buffer (unchanged). (Host DMA happens later in
   `upload_snapshots_async`.)
4. **`copy_wmunu_to_cpu` / `copy_primitives_to_cpu`:** discrete → `deep_copy`
   device→mirror first, then unpack from the mirror; coherent → unpack in place
   (unchanged). Mirrors `GPUGrid_cuda.cu`'s `cudaMemcpy` D2H exactly.
5. **`refresh_u_curr_stage`:** discrete → `deep_copy(snap_curr.u_stage_mirror,
   snap_curr.u_dev)`; coherent → no-op (as now).
6. **`host_readable_u_curr()`** already returns `u_stage ? u_stage : u` — works for
   both once `u_stage` is wired on discrete.
7. **`release()`:** free device Views + mirrors (the KStore owns both vectors).

### `src/gpu/KokkosPipelines.cpp`
8. **`upload_snapshots_async`:** coherent → keep no-op; discrete → `deep_copy`
   each `snap_curr`/`snap_prev` buffer device←mirror on the copy instance, then
   make the compute dispatches wait (DD4). Mirror `CUDAPipelines::upload_snapshots_async`.
9. **`partition_space`** the default exec once at `initialize()`; hold the
   `{compute, copy}` instances; issue `dispatch_*` on the compute instance,
   `deep_copy` on the copy instance, gate with fences. (Keep a compile/runtime
   switch so coherent stays single-instance.)
10. **EOS RandomAccess (DG3):** allocate `eos_*` as
    `View<const float*, …, MemoryTraits<RandomAccess>>` (or add a RandomAccess
    alias at the kernel boundary) to restore the `__ldg` path.

### `advance.cpp` (one line)
11. The `#if defined(USE_CUDA)` guard around `upload_snapshots_async`
    (`advance.cpp:505`) must also fire for `USE_KOKKOS` on the discrete path —
    change to `#if defined(USE_CUDA) || defined(USE_KOKKOS)` (the call is already
    a safe no-op on the coherent Kokkos path).

### Build
12. No new CMake needed for DG1–DG2 (same TUs). DG3 RandomAccess is source-only.

---

## Verification (reuse the existing harness)

Run on the **discrete** target, from the repo root:

- **Precision (must stay PASS, unchanged):**
  `GPU_BIN=$PWD/build_kokkos_cuda/src/MUSIChydro bash tests/eos_gpu_vs_cpu.sh`
  → expect the same **6.44e-04** as the managed path (this is a perf change).
- **Cross-backend (must stay PASS):** `bash tests/kokkos_consistency.sh` (≤1e-4).
- **Force-path A/B:** run each of `MUSIC_KOKKOS_FORCE_COHERENT=1` and
  `MUSIC_KOKKOS_FORCE_DISCRETE=1` on the *same* discrete card and diff the
  per-step + the output-heavy wall time — that isolates the DG1/DG2 win from
  device variance (mirrors the native `MUSIC_CUDA_FORCE_*` methodology).
- **Throughput:** per-step compute via `(T(long)−T(short))/Δsteps`, min of 3
  (see `Port_GPU_KoKKos.md`'s benchmark recipe), **plus** an
  `output_evolution_data 1` run — the diagnostic-sync path is where the discrete
  win shows up, and the compute-only per-step may barely move (residency already
  hides it).
- **Profile:** Nsight Systems to confirm fault migration (DG0) is replaced by
  HtoD/DtoH memcpy bars overlapped with kernels (DG2).

**Targets:** DG1 should match native CUDA's discrete transfer behaviour (bulk
DMA, no faults); DG1+DG2+DG3 should reach ≈ the same fraction of native CUDA that
the GB10 port reached of native CUDA on GB10 (~0.8×), with the EOS precision
unchanged.

---

## Risks & open items

- **Coherence query portability.** The CUDA attribute query is easy; HIP/SYCL
  need their own guarded queries. Until then, default discrete-vs-coherent by
  `has_shared_space` + a conservative env override, and document it.
- **D2H sync points are the real cost, not the hot loop.** Because residency
  already keeps compute resident, DG1's headline benefit lands at cold-start and
  at every `sync_arena_from_gpu` / output write — so **measure with diagnostics
  on**, or the win looks smaller than it is.
- **`SharedHostPinnedSpace` is itself device-accessible** (zero-copy pinned). Do
  **not** let kernels read it directly on discrete (every access crosses PCIe) —
  always `deep_copy` into the device View and run kernels on that. (The pinned
  space is staging only.)
- **Overlap correctness (DD4).** The copy instance must be fenced before the
  consuming kernel; get the gating exactly right or kernels read stale device
  memory (the discrete failure mode `PORT_GPU_CUDA.md` documents for native).
- **Don't regress GB10.** DD6 keeps the managed path intact; the discrete branch
  is additive and selected at runtime. Re-run the GB10 gate after every stage.
- **No discrete hardware here.** Everything above is a design; first real numbers
  require an A100/H100/RTX (or MI250X/PVC) — this is the explicit hand-off point.
