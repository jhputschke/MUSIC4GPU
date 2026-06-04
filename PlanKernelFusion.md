# Kernel Fusion — Current CUDA & Metal Backends

## Purpose

A standalone analysis of kernel-fusion opportunities in the **existing (pre-Kokkos)** CUDA
and Metal GPU backends of `music4gpu`. Fusion is a general GPU technique — available in both
backends today, independent of the Kokkos port. The companion `PlanKokkosPort.md` (Stage 3a,
D11) centralizes fusion *after* the port so it is written once for all backends; this document
captures the **pre-port** option and the analysis common to both backends.

---

## 1. Current implementation outline

### 1.1 Per-substep pipeline (shared by both backends)

The host dispatch (`src/advance.cpp:427–436`) runs one batched sequence per RK substep,
identical for CUDA and Metal via the `GPUPipelines` alias:

```
begin_batch();
  dispatch_make_du                 // theta, a, sigma
  dispatch_uwrhs                   // uwrhs_out[5]
  dispatch_w_source                // dwmn[5]
  if (turn_on_bulk) dispatch_uprhs // uprhs_out[1]
  dispatch_delta_qi                // qi_out[5]
  dispatch_finalize_ideal          // snap_future: e, rhob, u
  dispatch_first_rk_step_w_full    // snap_future: Wmunu, pi_b
end_batch(); wait();
```

Kernel data-flow (from `src/gpu/music_kernels.cuh`; SoA component-major,
`cell = ix + Nx*(iy + Ny*ieta)`):

| # | Kernel | Reads (neighbors of) | Writes (per cell) | Stencil |
|---|---|---|---|---|
| 1 | `make_du` | `u` curr/prev | `theta`, `a[4]`, `sigma[10]` | radius-1 (∂u) |
| 2 | `make_uwrhs` | `Wmunu`, `u` curr | `uwrhs_out[5]` | radius-2 (±2 minmod) |
| 3 | `make_w_source` | `Wmunu/pi_b/u` curr+prev | `dwmn[5]` | radius-1 |
| 4 | `make_uprhs` (bulk only) | `pi_b`, `u` curr | `uprhs_out[1]` | radius-2 |
| 5 | `make_delta_qi` | `e/rhob/u` curr, EOS | `qi_out[5]` | radius-2 + Newton |
| 6 | `finalize_ideal` | `qi_out`◆, `dwmn`◆, snap curr/prev, EOS | `e/rhob/u_future` | same-cell (Newton) |
| 7 | `first_rk_step_w_full` | `uwrhs/theta/a/sigma/uprhs`◆, `*_future`◆, snap curr/prev, EOS | `Wmunu/pi_b_future` | same-cell |

◆ = **same-cell** read of an upstream intermediate (no neighbor access).

### 1.2 CUDA backend (`src/gpu/*.cu`)

- **9 `__global__` kernels:** the 7 above + `gpu_make_w_source_tiled` (Phase-3 shared-memory
  halo variant of #3) + `gpu_reduce_max_eps_rhob` (device max-reduction).
- **`CUDAPipelines.cu`:** each `dispatch_*` computes a `dim3` grid/block (8×8×4, ≤256
  threads/block, 2D-adapted when `Neta==1`) and launches on a single compute stream.
  `begin/end_batch` are **no-ops** (the stream already serializes). A second copy stream +
  event (`upload_snapshots_async`) overlaps H2D on discrete GPUs.
- **`GPUGrid_cuda.cu`:** SoA `float` buffers; `cudaMallocManaged` (coherent / GB10) vs
  `cudaMalloc` device + `cudaHostAlloc` pinned staging (discrete). All buffers allocated
  once and persistent.

### 1.3 Metal backend (`src/gpu/*.metal`, `*.mm`)

- **8 `kernel void`** in `music_kernels.metal`: the 7 above + a superseded
  `gpu_first_rk_step_w` (line 253; the active one is `gpu_first_rk_step_w_full`). **No** tiled
  variant and **no** device-reduction kernel.
- **`MetalPipelines.mm`:** one precompiled `MTLComputePipelineState` per kernel (built at
  init). Each `dispatch_*` = command buffer → `computeCommandEncoder` →
  `setComputePipelineState` → set buffers → `dispatchThreads` → `endEncoding` → commit.
  **`begin/end_batch` share a single `MTLCommandBuffer`** across the whole substep (one commit
  + one wait) — so per-dispatch overhead is just one encoder.
- **`GPUGrid.mm`:** `MTLBuffer` shared (unified) storage on Apple Silicon — host-coherent, so
  `reduce_max` scans `epsilon/rhob` on the **host** (no device reduce kernel needed).

### 1.4 Shared vs different

| Aspect | CUDA | Metal |
|---|---|---|
| Dispatch surface | identical (`GPUPipelines` alias, same 7 `dispatch_*`) | identical |
| Kernel math | CUDA C++ (`music_kernels.cu`) | MSL (`music_kernels.metal`) — line-for-line mirror |
| Launch unit | `kernel<<<grid,block>>>` on a stream | encoder `dispatchThreads` in a command buffer |
| Batching | no-op (single stream) | one command buffer per substep |
| Extra kernels | `_tiled` w_source, device reduce | — (host reduce) |
| Memory | managed *or* device+pinned | unified (shared) |

**Implication for fusion:** the dependency graph and legality are **identical** for both
backends; only the launch mechanics differ. Any fusion must be implemented twice (CUDA C++
and MSL).

---

## 2. Why fusion is legal here

Two kernel classes (from §1.1):

- **5 stencil producers** (`make_du`, `make_uwrhs`, `make_w_source`, `make_uprhs`,
  `make_delta_qi`) read **only the read-only input snapshots** at neighbors and write
  **disjoint** per-cell outputs. None consumes another's output.
- **2 per-cell consumers** (`finalize_ideal`, `first_rk_step_w_full`) read upstream
  intermediates **only at the same cell** (◆).

```
        ┌─ make_du ─────────────► theta/a/sigma ─┐
 input  ├─ make_uwrhs ──────────► uwrhs ─────────┤
 (snap  ├─ make_uprhs ──────────► uprhs ─────────┼─► first_rk_step_w_full ─► W/pi_b_future
 curr/  ├─ make_w_source ─► dwmn ┐               │             ▲
 prev)  └─ make_delta_qi ─► qi ──┴─► finalize_ideal ─► e/rhob/u_future ─────┘
```

**Decisive fact:** no kernel reads another kernel's *output* at neighbor cells (every stencil
is self-contained in its producer; all intermediate consumption is same-cell). Therefore
fusion needs **no grid-wide synchronization** — no cooperative-groups grid sync on CUDA, no
cross-threadgroup sync on Metal. The only data crossing the fused boundary is register-resident
per-cell values.

**Legality rule:** fuse A→B iff *either* B reads A's output only at the same cell, *or* A and B
both read only the fixed input snapshot.

---

## 3. Fusion candidates (ranked)

| Fusion | Legal because | Benefit | Risk |
|---|---|---|---|
| **`uwrhs` + `uprhs`** | both stencils over the same `u` neighborhood (independent) | reuse ±2 `u` loads; drop one launch + `uprhs_out` traffic | **Low** (both light) |
| **`delta_qi` + `finalize_ideal`** | `finalize` reads `qi` same-cell (`dwmn` already produced upstream) | eliminate `qi_out[5·Ncells]` write+read on the heaviest data path | **Med** (adds Newton regs to a register-heavy kernel) |
| **`finalize_ideal` + `first_rk_step`** | `first_rk` reads `*_future` same-cell | eliminate the `*_future` (≈6·Ncells) round-trip | **Med–High** (two heavy bodies) |
| **Stencil cluster** (`make_du`+`uwrhs`+`uprhs`+`w_source`) | all read the fixed input; disjoint outputs | share the snap_curr neighbor loads each currently re-reads | rises with cluster size |
| **Tail** (`delta_qi`+`finalize`+`first_rk`) | chained same-cell | maximal round-trip elimination | **High** (register pressure / occupancy) |

---

## 4. Backend-specific mechanics

**CUDA.** Merge the `__global__` bodies into one; `CUDAPipelines` launches one
`kernel<<<grid,block>>>` instead of two. Re-check occupancy after fusing
(`nvcc --ptxas-options=-v`; consider `__launch_bounds__` / `--maxrregcount`); the 8×8×4 block
may need re-tuning because register use rises.

**Metal.** Merge the `kernel void` bodies; build one `MTLComputePipelineState`; emit one
encoder `dispatchThreads`. Because Metal **already batches** the substep into a single command
buffer, the *launch-overhead* saving is just one fewer encoder (small); the **bandwidth**
saving (eliminated round-trips + shared neighbor loads) is the real win. Metal has **no tiled
`w_source`**, so the shared-load benefit of stencil-cluster fusion is comparatively *more*
valuable there. Watch threadgroup occupancy (`maxTotalThreadsPerThreadgroup` drops as
register / threadgroup-memory use rises).

---

## 5. Expected payoff & the caveat

- **Launch / encoder overhead** — moderate on CUDA discrete for small grids; small on Metal
  (already batched).
- **Global-memory traffic** — the main win: eliminate the `qi_out`, `uprhs_out`, and `*_future`
  round-trips, and share redundant input-neighbor reloads across the stencils.
- **Caveat (decides how far to go):** `make_delta_qi` (~60% of runtime: Newton-Brent + 12
  reconstructions/cell) and `first_rk_step_w_full` are **register-/occupancy-bound**, not
  launch-bound. Fusing *those* can lower occupancy and be **net-negative**. Safe wins are the
  **light** fusions and the **round-trip eliminations**; fusing the two heavyweights is a
  measure-it gamble.

### Primer: registers vs occupancy (why the caveat holds)

**Registers** are the fastest, per-thread, on-chip storage holding a thread's live variables.
An SM has a **fixed register file** shared by all its resident threads (e.g. 65,536 32-bit
registers/SM on Ampere/Hopper). Iterative solvers and large per-cell working sets (Newton-Brent;
the 14-component Israel-Stewart update) keep many values live at once → **high register
pressure** → many registers per thread (`nvcc --ptxas-options=-v` reports the count). Exceed the
cap (255, or a `--maxrregcount` you set) and the compiler **spills** to slow local/global memory.

**Occupancy** = active warps ÷ max warps per SM (max 64 on Ampere/Hopper). The SM hides memory
latency by switching among resident warps, so more warps = more latency hiding. When registers
are the binding resource, the resident-thread ceiling is `register_file / registers_per_thread`:

| Registers/thread | Warps/SM | Occupancy |
|---|---|---|
| 32 | 64 | 100% |
| 64 | 32 | 50% |
| 128 | 16 | 25% |
| 200+ | ~10 | ~16% |

So **register pressure → low occupancy → fewer warps to hide latency → the SM stalls.** That is
the limiter on `make_delta_qi` / `first_rk_step_w_full` — *not* launch overhead. **Launch-bound**
kernels (tiny, fast) are dominated by the ~µs launch cost, so fusion helps them; **occupancy-
bound** kernels run long enough that launch cost is noise and the bottleneck is resident warps.

**Fusion implication:** a fused kernel makes each thread hold **both** kernels' live sets →
**more registers/thread → lower occupancy** (or spilling). For the two heavyweights that removes
the very latency-hiding they are already short on, so the launch / round-trip savings can be
**net-negative**. Hence: fuse the *light* kernels, eliminate round-trips, and **measure** (Nsight
Compute occupancy + stall reasons) before fusing the heavy ones. Occupancy isn't everything —
high instruction-level parallelism can hide latency at lower occupancy (Volta+), and
`__launch_bounds__` / `--maxrregcount` trade register count against spilling — so the call is
empirical, per kernel and per architecture.

---

## 6. Recommendation

1. **Land per-kernel golden buffers first** (the `PlanKokkosPort.md` D6 harness): dump each
   intermediate (`dwmn`, `qi_out`, `uwrhs`, `theta/a/sigma`, `*_future`) so any fused kernel can
   be validated against the unfused reference.
2. **If fusing pre-port,** do only the low-risk **`uwrhs`+`uprhs`** (and optionally
   **`delta_qi`+`finalize_ideal`**), behind a compile-time switch, implemented in **both** CUDA
   and MSL, each validated against its golden reference and the end-to-end `eos_gpu_vs_cpu.sh`
   (EOS 91) trace.
3. **Cost of doing it now:** every fusion is maintained **twice** (CUDA C++ + Metal MSL). The
   Kokkos port (`PlanKokkosPort.md` Stage 3a / D11) re-expresses each fusion **once** for
   CUDA/HIP/SYCL/CPU — so unless the throughput is needed before the port, deferring is cheaper
   overall.

---

*Companion document: `PlanKokkosPort.md` (full Kokkos port strategy, decision log D1–D11,
Stages 0–5). Fusion is Stage 3a / decisions D6 (golden buffers) and D11 (optimize behind a
flag, validate against the unfused golden).*
