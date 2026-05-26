# Plan — Multi-GPU Outlook

Status: **outlook / not started.** This records the design and, more
importantly, the *ordering* of prerequisites for using more than one discrete
GPU. It is deliberately gated behind earlier work (see §1).

The 3+1D structured-grid stencil solver is a natural fit for spatial domain
decomposition, so multi-GPU is technically feasible. Whether it is *worthwhile*
depends entirely on the prerequisites below.

---

## 1. Prerequisites and ordering (read first)

Two things must be true before single-grid multi-GPU is worth implementing:

1. **A single GPU must be compute-bound.** Profiling on GB10 showed the GPU is
   only ~5.5% busy at the production grid (64×64×32) — the run is host-bound
   (per-step AoS↔SoA pack/unpack + diagnostics; see `README_CUDA.md`). Adding
   GPUs while the first is 95% idle gains nothing. **`Plan-GPU-resident-state.md`
   is a hard prerequisite** — and multi-GPU makes the host bottleneck *worse*
   (now scatter/gather the grid across N devices each step), so resident state
   is doubly required.

2. **The grid must be large enough to saturate one GPU.** Strong scaling of the
   current 131k-cell grid is poor: split across 4 GPUs it is ~64×64×8 per
   device, where kernel-launch latency + halo sync dominate. Multi-GPU is a
   **weak-scaling** play (much larger grids, e.g. fine 3+1D or ≥256³), not a way
   to speed up today's production size.

### The usually-better alternative: event-level parallelism

For heavy-ion **event-by-event** production (the common workload), do **not**
decompose one event across GPUs. Run **one collision event per GPU** (or per
CUDA MPS slice): embarrassingly parallel, zero halo exchange, near-linear
scaling, and essentially no code change — just per-process device selection
(`CUDA_VISIBLE_DEVICES` / `cudaSetDevice`). Single-grid decomposition (§2) is
only for the case where one event's grid is itself too big/slow on a single GPU.

---

## 2. Single-grid domain decomposition design

### 2.1 Decomposition axis — η slabs
Memory layout is `cell = Nx*(Ny*ieta + iy) + ix` (ix fastest, ieta slowest).
Split the domain into contiguous **η-slabs**, one per GPU: each device owns an
`ieta` range. Because η is the outermost index, a slab — and its boundary
planes — are **contiguous in the SoA layout**, making halo copies simple and
coalesced. (For 2D boost-invariant runs, `Neta == 1`; split along `y` instead.)

### 2.2 Ghost halo — 2 cells
The stencils have radius ≤ 2 (`gpu_make_delta_qi`, `gpu_make_uwrhs`,
`gpu_make_uprhs` are ±2; `gpu_make_w_source`, `gpu_make_du` are ±1). Each device
allocates a **2-plane ghost layer** on each internal η face. Before the stencil
kernels of a substep, exchange the 2 boundary η-planes with neighbour devices:
- `snap_curr` is read by all stencil kernels;
- `snap_prev` is additionally read by `gpu_make_w_source` (time derivative).

So exchange the boundary planes of both `snap_curr` and `snap_prev` once per
substep, before the kernel batch. Halo volume is small: e.g. 64×64 transverse ×
2 planes × ~21 floats × 4 B ≈ 0.7 MB per neighbour — negligible vs ~1 ms of
per-substep compute.

### 2.3 Boundary handling
`clamped_cell` currently clamps at the **global** grid edge. With decomposition:
- **interior** η faces must read the neighbour's exchanged halo (no clamp);
- **global outer** η faces still clamp.

The ghost layer encodes this: fill interior ghosts from the neighbour, fill
outer ghosts by clamping/replication as today. The kernels then read the ghost
layer uniformly (the local index math accounts for the slab origin + halo
offset).

### 2.4 Inter-GPU transport
- **NVLink / NVSwitch:** enable peer access (`cudaDeviceEnablePeerAccess`) and
  exchange with `cudaMemcpyPeerAsync` on a dedicated copy stream — fast, the
  intended path. Topology matters (direct NVLink vs hops).
- **PCIe only:** P2P over PCIe if available, otherwise stage the halo through
  pinned host memory. Workable but slower; keep halos minimal.
- Overlap the halo exchange with interior compute (compute the slab interior,
  which needs no halo, while the boundary planes are in flight; then compute the
  boundary cells). Classic stencil-halo overlap.

### 2.5 Distributed reductions
`eps_max` / `rhob_max` / conservation become a per-device partial reduction
(the GPU reduction from `Plan-GPU-resident-state.md`) followed by a small
cross-device all-reduce of a few scalars — cheap, no grid traffic.

### 2.6 Host integration
The AoS↔SoA pack now **scatters** η-slabs to devices and the copy-back
**gathers** them. With resident state (prerequisite §1) this happens only at
init and on output/freeze-out steps, not every step — which is exactly why
resident state must come first.

---

## 3. Components / effort

| Component | Where | Notes |
|-----------|-------|-------|
| Per-device context: `GPUGrid` + streams + pipeline per GPU | `CUDAPipelines`, `GPUGrid` | one instance per device; track `ndev` |
| Slab index math (local origin + halo offset) | kernels / launch config | kernels gain a slab-origin/extent; clamp only at global faces |
| Ghost-layer allocation (2 planes/face) | `GPUGrid_cuda.cu` | extend snapshot buffers or separate halo buffers |
| Halo pack + `cudaMemcpyPeerAsync` exchange per substep | `CUDAPipelines` | dedicated copy stream; event-gated to compute |
| Interior/boundary compute split (overlap) | dispatch | optional but important for scaling |
| Cross-device all-reduce of scalars | `CUDAPipelines` / `Evolve` | builds on the GPU reduction |
| Host scatter/gather of η-slabs | `GPUGrid` / `Evolve` | only at init + sync points (needs resident state) |

**Effort:** large, and it compounds with the resident-state restructure. No
change to the seven physics kernels themselves beyond slab-aware indexing and
the interior/boundary split.

---

## 4. Recommended path (summary)

1. **`Plan-GPU-resident-state.md`** — make a single GPU compute-bound (hard
   prerequisite).
2. **Event-per-GPU / MPS** — if the goal is event-by-event throughput, stop
   here; it is simpler and scales better than §2.
3. **Single-grid η-slab decomposition (§2)** — only if one event's grid is
   genuinely too large/slow on one GPU, and ideally on NVLink-connected
   devices. Expect weak-scaling benefit on large grids, not strong-scaling of
   today's production size.
