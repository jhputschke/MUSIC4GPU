# Porting MUSIC's CUDA Hydro Backend to Kokkos

## Context

`music4gpu` is the GPU-accelerated build of MUSIC (3+1D viscous relativistic
hydro) used inside X-SCAPE. It currently carries **three** parallel
implementations of the same per-cell physics:

- **CUDA** (`src/gpu/music_kernels.cu`, ~1900 lines, 9 kernels) — NVIDIA only.
- **Metal** (`src/gpu/music_kernels.metal`, ~2200 lines) — Apple Silicon only.
- **CPU/OpenMP** (`advance.cpp`, `dissipative.cpp`, `u_derivative.cpp`) — the
  per-cell reference path, with at least one kernel (`gpu_first_rk_step_w_full`)
  that has *no* exact CPU twin, so the paths can drift.

The goal is to evaluate and plan a move to **Kokkos** as a single performance-
portable source that (1) runs on NVIDIA, **AMD (Frontier/El Capitan)** and
**Intel (Aurora)** GPUs from one kernel body, (2) collapses the CUDA + CPU
triplication into one implementation, (3) keeps/extends NVIDIA performance
(incl. GB10/Grace coherent memory), and (4) future-proofs against vendor API
churn. **Per decision: the Metal backend is kept separate** — Kokkos has no
Apple Metal backend, so Metal remains the lone vendor-specific path for Apple
GPUs while Kokkos subsumes CUDA and adds AMD/Intel/CPU.

This document is **both** a feasibility + staged roadmap *and* a concrete
implementation plan for the first stage.

---

## Feasibility verdict (TL;DR)

**Possible: yes, and the codebase is unusually well-prepared for it.** The hard
architectural work — isolating GPU work behind a clean, backend-agnostic seam —
is already done.

- The host evolution logic (`advance.cpp`) is already backend-agnostic: it talks
  to a `GPUPipelines` **type alias** (`advance.h:16-27`) that resolves to
  `MetalPipelines` or `CUDAPipelines` at compile time. Both expose an *identical*
  interface (`instance()`, `initialize()`, seven `dispatch_*` methods, `wait()`,
  `reduce_max()`, `begin/end_batch()`). **A `KokkosPipelines` class slots in as a
  third option behind that same alias with zero changes to the host dispatch
  flow.**
- The CUDA kernels are already "Kokkos-shaped": flat one-thread-per-cell 3D
  parallelism, SoA component-major buffers, **no warp shuffles / no warp
  intrinsics / no dynamic parallelism**. Porting is close to mechanical.
- There is a **ready-made correctness oracle**: the CPU path (forced via
  `MUSIC_FORCE_CPU=1`) plus `tests/eos_gpu_vs_cpu.sh` (EOS 91), which already
  compares GPU-vs-CPU `eps_max(τ)` traces within tolerance.
- The device path is **float32 throughout** — no precision decisions to litigate.

**Main caveats** (none are blockers): (a) Kokkos can't target Apple Metal — Metal
stays separate by design; (b) `Kokkos::initialize/finalize` lifecycle must be
owned cleanly inside the JETSCAPE process (esp. with MPI device-per-rank); (c)
the one shared-memory tiled kernel and the dual-stream overlap are the only
non-mechanical ports, and both have simpler fallbacks already in tree.

---

## Why Kokkos here — advantages mapped to your motivations

| Motivation | What Kokkos gives you | Mechanism |
|---|---|---|
| **Hardware portability** | One kernel body runs on NVIDIA / AMD / Intel GPUs + multicore CPU | Execution spaces: `Kokkos::Cuda`, `Kokkos::HIP`, `Kokkos::SYCL`, `Kokkos::OpenMP`, `Kokkos::Serial` selected at build time |
| **Maintainability** | Collapse CUDA + CPU into one source; kill CPU/GPU physics drift | Same `KOKKOS_INLINE_FUNCTION` device helpers + `parallel_for` bodies compiled for *both* GPU and CPU backends |
| **NVIDIA performance** | Match/beat current CUDA; keep GB10 coherent-memory path | `Kokkos::SharedSpace` (UVM) for coherent parts; mirror views + `deep_copy` for discrete; per-backend fast-math |
| **Future-proofing** | Insulated from vendor API churn; new accelerators ≈ rebuild | Backend is a CMake/template choice, not a rewrite |

**The "layout for free" win (serves portability *and* maintainability at once).**
A `Kokkos::View` parameterized on the execution space's default layout is
`LayoutLeft` (column-major → **coalesced**) on GPU and `LayoutRight` (row-major →
**cache-friendly**) on CPU. The *same* kernel source therefore gets optimal
memory access on every backend — which is exactly what lets the Kokkos kernels
**replace the hand-tuned SoA packing** in `GPUGrid_cuda.cu` *and* **replace the
CPU OpenMP per-cell loops** without writing the physics twice.

---

## Where Kokkos plugs in (the seam)

Add `USE_KOKKOS` as a third branch alongside `USE_METAL` / `USE_CUDA`:

```cpp
// advance.h — extend the existing alias block (lines 16-31)
#elif defined(USE_KOKKOS)
    #include "gpu/GPUGrid.h"            // existing class; Kokkos-backed impl
    #include "gpu/KokkosPipelines.h"
    #include "gpu/gpu_types.h"
    using GPUPipelines = KokkosPipelines;   // host dispatch unchanged
```

`USE_KOKKOS` and `USE_CUDA` are **mutually exclusive** (both target NVIDIA — pick
one at configure time); Metal is independent and untouched. New classes mirror
the existing ones 1:1:

- **`KokkosPipelines`** ⟷ `CUDAPipelines` — same `instance()`/`dispatch_*`/`wait()`
  surface (`CUDAPipelines.h`). Implemented with a **PIMPL/opaque handle** so the
  header stays Kokkos-free and `advance.cpp` keeps compiling as a plain host TU —
  exactly how `CUDAPipelines.h` already hides `cudaStream_t` behind `void*`
  (`CUDAPipelines.h:74-76`).
- **`GPUGrid` (Kokkos-backed)** — the Kokkos backend implements the *existing*
  `GPUGrid` class (`GPUGrid.h`) in `GPUGrid_kokkos.cpp`, exactly as the CUDA
  backend uses `GPUGrid_cuda.cu` and Metal uses `GPUGrid.mm` — **not** a separate
  class (so `advance.h`'s `GPUGrid gpu_grid_` member is untouched). Same public
  API and snapshot/buffer layout: three rotating `GPUSnapshot`s, `dwmn/qi_out/
  uwrhs_out/uprhs_out/theta_buf/a_buf/sigma_buf`, EOS tables, reduce scalars.
  Internally each `float*` becomes a `Kokkos::View<float*>` (Stage 1); the
  raw-pointer getters (`host_readable_u_curr()`, `qi_source_buf`) return a host
  mirror's `data()` so the host hydro-source pre-pass in `advance.cpp` is
  **unchanged**.

### Build contexts (X-SCAPE primary, stand-alone secondary)

A **single integration point** — `music4gpu/CMakeLists.txt` — covers both builds,
because X-SCAPE compiles the GPU MUSIC by `add_subdirectory(./${MUSIC_BACKEND_DIR})`
(`X-SCAPE/CMakeLists.txt:557`, with `MUSIC_BACKEND_DIR=external_packages/music4gpu`).
So music4gpu's own CMake drives Kokkos discovery in *both* modes:

- **PRIMARY — X-SCAPE embedded:** the parent only adds the `USE_KOKKOS` option, the
  `MUSIC_BACKEND_DIR` arm, and the mutual-exclusion check; the `add_subdirectory`
  at line 557 then triggers music4gpu's Kokkos discovery automatically. Lifecycle:
  `KokkosRuntimeGuard` in the framework driver `main()` (D2). Fetch:
  `bash external_packages/music4gpu/get_kokkos.sh`.
- **SECONDARY — stand-alone `MUSIChydro`:** music4gpu is self-contained (own
  `CMakeLists.txt`, `src/main.cpp`). `get_kokkos.sh` + the same discovery build it
  independently; lifecycle is the `KokkosRuntimeGuard` in `src/main.cpp`.

### Construct mapping

| Current CUDA | Kokkos equivalent |
|---|---|
| `__global__ void k(...)` + `<<<grid,block>>>` | `parallel_for(MDRangePolicy<Rank<3>>({0,0,0},{Neta,Nx,Ny}), KOKKOS_LAMBDA(int ieta,int ix,int iy){...})` |
| `__device__ __forceinline__` (`DFI`) | `KOKKOS_FORCEINLINE_FUNCTION` (helpers: `gpu_reconst` Newton-Brent, minmod, EOS interp — near-verbatim) |
| SoA `float*` (component-major) | `Kokkos::View<float**, exec::array_layout, MemSpace>` `(comp, cell)`, or flat `View<float*>` |
| `cudaMalloc` / `cudaMallocManaged` | `View` in `CudaSpace` / `SharedSpace` (portable UVM) |
| pinned staging + `cudaMemcpyAsync` | `create_mirror_view` (pinned space) + `deep_copy` |
| `gpu_reduce_max_eps_rhob` (atomicMax) | `parallel_reduce` with `Kokkos::Max<float>` (drops the hand-rolled bit-trick) |
| `gpu_make_w_source_tiled` (shared mem) | `TeamPolicy` + `team_scratch` (Stage 3; non-tiled maps to plain `MDRangePolicy`) |
| `__ldg` read-only EOS cache | `View<const float*, …, MemoryTraits<RandomAccess>>` (→ `__ldg`/texture path) |
| `__constant__ WMUNU_IDX/GMUNU_DIAG` | small `constexpr` arrays captured by value |
| dual `cudaStream_t` overlap | `partition_space(exec,…)` → independent execution-space instances |
| `cudaDeviceSynchronize` / stream sync | `Kokkos::fence()` / `exec.fence()` |
| `--use_fast_math` | per-backend compile options via Kokkos CMake |

---

## Complexity assessment

**Mechanical / low-risk (the bulk):** porting the 9 kernels and the device
helpers. The arithmetic is identical; the work is swapping `DFI`→
`KOKKOS_FORCEINLINE_FUNCTION` and kernel-launch math → `MDRangePolicy`. The
existing CUDA file is the line-by-line spec, and the CPU path + EOS test verify
each kernel as it lands.

**Moderate:** the memory layer (`GPUGrid_cuda.cu` → `KokkosGrid`: Views + mirrors
replace hand-rolled pack/copy; the coherent-vs-discrete dual path becomes
`SharedSpace` vs `mirror+deep_copy`), and the build integration (Kokkos as a
submodule, `nvcc_wrapper`/Kokkos-CMake, bump `music4gpu` host C++ standard to
C++17 — already required by parent JETSCAPE and the CUDA TUs).

**Highest-touch (but optional / deferrable):**
1. **`Kokkos::initialize/finalize` lifecycle inside JETSCAPE** — must run once per
   process, before `MPI_Finalize`, with device-per-rank selection. This is the
   trickiest *non-physics* piece (see Risks).
2. **Tiled stencil kernel** (`gpu_make_w_source_tiled`) → `TeamPolicy`+scratch —
   only needed for peak perf; the plain kernel is a correct fallback.
3. **Dual-stream overlap** (CUDA "Phase 4") → execution-space instances — a perf
   optimization, not a correctness requirement.

**Rough sizing** (fluent-with-Kokkos developer; the physics is already validated):
Stage 0 ≈ S, Stage 1 ≈ L (the core), Stage 2 ≈ M, Stage 3 ≈ M–L, Stage 4 ≈ M
(mostly deletion). Total on the order of a few focused weeks to parity, more for
the Stage 3 optimizations and the AMD/Intel bring-up.

---

## Resolved design decisions

| # | Decision | Choice | Why / where it bites |
|---|---|---|---|
| **D1** | Kokkos acquisition | In-repo `music4gpu/get_kokkos.sh` — **latest release by default**, pin via arg (`get_kokkos.sh 5.1.1`) — clones into `external/kokkos`. CMake discovery: `-DKOKKOS_SOURCE_DIR` override → in-repo `external/kokkos` → `../kokkos` sibling → `find_package(Kokkos)` (HPC/Spack/module). | One script + one discovery path serve **both** the X-SCAPE build (which `add_subdirectory`s music4gpu) and stand-alone. In-repo so stand-alone works; pin the version for reproducible CI/HPC builds. Repo has no submodules. |
| **D2** | Init/finalize lifecycle | One `KokkosRuntimeGuard` (host-safe RAII over `Kokkos::initialize/finalize`, defined in `kokkos_runtime.cpp`) in the driver `main()`, before the run. **PRIMARY:** the X-SCAPE framework `main()`, before the `JetScape` object — never a `JetScape` member. **SECONDARY:** stand-alone `music4gpu/src/main.cpp`. | `JetScapeTask::tasks` (base) destroyed last → a member guard would `finalize()` before module Views are freed. The shim keeps both mains plain host TUs (D7) and is collision-safe (no double init/finalize if another module brought Kokkos up). |
| **D3** | Data layout | **Phase**: flat 1D `View<Real*>` (`comp*Ncells+cell`) for Stage-1 parity → layout-templated multi-D Views (`LayoutLeft` GPU / `LayoutRight` CPU) in Stage 2 | Fast parity first; the layout switch is what makes the CPU/OpenMP backend fast, which Stage-4 unification needs |
| **D4** | Precision | Kernels templated on `Real`: `float` on GPU, `double` on the CPU/OpenMP backend | Matches today's float device path *and* the legacy double CPU reference, from one source |
| **D5** | Host-data depth | **Shallow now** (pack/`deep_copy` into `KokkosGrid` Views; `Fields` untouched) → **View-back `Fields` in Stage 4** | De-risks Stages 0–2; defers the wide `Fields` refactor (evolve/grid_info/…) to the unification stage |
| **D6** | Validation | Per-kernel golden-buffer diff vs native CUDA **+** end-to-end `eos_gpu_vs_cpu.sh` (EOS 91) | Localizes divergence to the offending kernel; keeps the existing trace oracle as the integration gate |
| **D7** | Compile surface | PIMPL boundary through Stages 0–2 (only `KokkosPipelines`/`KokkosGrid` TUs compile under the Kokkos compiler; rest of MUSIC stays host TUs, Kokkos behind opaque handles); **widen in Stage 4** | Mirrors how `CUDAPipelines.h` hides `cudaStream_t` behind `void*`; smallest blast radius early, full Views where Stage-4 unification needs them |
| **D8** | GPU feature coverage | **Progressively extend** Kokkos to the currently CPU-only configs (baryon diffusion, finite-µB EOS, multi-charge `rhoq/rhos`, hydro sources); keep the legacy CPU fallback until each lands | More configs accelerated over time; legacy CPU path retires per-feature → new **Stage 5** |
| **D9** | CI / backend matrix | Build + EOS-test `Serial`, `OpenMP`, `Cuda` from Stage 1 with a Serial/OpenMP/Cuda trace-**consistency gate**; add HIP/SYCL when runners exist | Continuous portability + single-source correctness guarantee |
| **D10** | Parity tolerance | **Relative, not bitwise**: same-backend per-kernel ≤ ~1e-6, cross-backend per-kernel ≤ ~1e-4, end-to-end / CI trace ≤ ~1e-3; tune empirically | Fast-math + `float` + cross-backend FP reordering preclude exact match |
| **D11** | Optimization posture | Unfused 1:1 kernels = default **and** golden reference; fusion / scratch behind compile-time switches, enabled per backend only after passing the D9 gate | Conservative correctness, progressive speed; keeps D6 golden diffs meaningful |

---

## Staged roadmap (difficulty ↗, speedup ↗)

| Stage | Scope | Difficulty | Speed/portability outcome |
|---|---|---|---|
| **0 — Infrastructure** | Kokkos in build (a `get_kokkos.sh` clone script + `USE_KOKKOS` option, mutually exclusive w/ CUDA); init/finalize lifecycle; `KokkosPipelines`/`KokkosGrid` skeletons behind the `GPUPipelines` alias; build & run on **Serial/OpenMP host backend** end-to-end | Low | None yet (CPU); green build + correct fall-through |
| **1 — Functional GPU parity** | Port all 9 kernels → `parallel_for`/`parallel_reduce`; device helpers → `KOKKOS_FORCEINLINE_FUNCTION`; Views + mirrors for snapshots/intermediates; build `Kokkos::Cuda`; validate vs CPU & native CUDA via `MUSIC_FORCE_CPU` + `tests/eos_gpu_vs_cpu.sh` (EOS 91) | **High (core)** | ~0.8–1.0× native CUDA (correct, untuned) |
| **2 — Perf parity + new HW** | **Migrate flat→layout-templated multi-D Views** (D3): default layout (`LayoutLeft` GPU / `LayoutRight` CPU); tune `MDRangePolicy` tiles vs the 8×8×4 blocks; `RandomAccess` EOS Views; per-backend fast-math; **bring up AMD (HIP) + Intel (SYCL)** by recompiling and re-running the same verification | Medium | ≈1.0× CUDA on NVIDIA **+ first-ever AMD/Intel GPU runs** |
| **3 — Kokkos-native optimizations** | **Kernel fusion** of the 8-kernel chain (cut launches + global round-trips on `theta/a/sigma/uwrhs/dwmn/qi`); `TeamPolicy`+scratch stencil tiling (portable, beats the CUDA-only tiled kernel); register-pressure relief on the `delta_qi` bottleneck (~60% of runtime: Newton-Brent + 12 reconstructions/cell); `partition_space` copy/compute overlap; persistent resident Views across timesteps (pointer rotation → cheap View-handle swaps) | Medium–High | **> current CUDA**; biggest wins on the 3 heaviest kernels |
| **4 — Single-source unification** | **View-back `Fields`** (D5), **widen the Kokkos-compiled surface** (D7), and replace the **CPU OpenMP per-cell path** with the *same* Kokkos kernels (OpenMP/Serial backend) → removes CPU/GPU physics drift; deprecate native CUDA once Kokkos/CUDA ≥ parity (**Metal retained** per decision) | Medium (mostly deletion) | Maintainability/correctness; new physics lands once for all backends |
| **5 — Extend GPU coverage** (ongoing, parallel to 2–4) | Progressively add Kokkos kernels for the currently CPU-only configs (D8): baryon diffusion (`turn_on_diff`), finite-µB EOS (e.g. EOS 20), multi-charge `rhoq/rhos`, hydro sources — retiring the legacy CPU fallback per-feature as each lands | Per-feature | More configurations run accelerated; legacy CPU path shrinks toward removal |

Stages 0→2 are the path to "Kokkos at CUDA parity, plus AMD/Intel." Stages 3–4
are where Kokkos pays back beyond what the hand-written backends could.

---

## Stage 0–1 concrete implementation plan (the actionable first step)

### Stage 0 — Infrastructure

> **Status:** the Stage-0 infrastructure below is **implemented on the
> `KoKKos-Port` branch** — `get_kokkos.sh`, the `USE_KOKKOS` CMake option +
> discovery (music4gpu + X-SCAPE parent), the lifecycle shim, and the
> skeleton TUs. The skeleton's `KokkosPipelines::initialize()` returns `false`,
> so a `USE_KOKKOS` build runs the CPU reference; Stage 1 ports the kernels.

1. **Vendor Kokkos.** `music4gpu/get_kokkos.sh` clones Kokkos into the in-repo
   `external/kokkos` (git-ignored), **defaulting to the latest release** and
   accepting an optional pinned tag (`bash get_kokkos.sh 5.1.1`). It lives *in
   the music4gpu repo* (not `external_packages/`) so the stand-alone build is
   self-contained; under X-SCAPE it is reached at
   `external_packages/music4gpu/get_kokkos.sh`. Latest tag is resolved with
   `git ls-remote --sort=-v:refname` (no GitHub API / `jq`). Pin a version for
   reproducible CI/HPC builds. (Alternative on HPC: skip the script and use a
   system/Spack/module Kokkos via `find_package` — see Step 2. No `.gitmodules`,
   matching the repo workflow.)
2. **Build options.** In `music4gpu/CMakeLists.txt` add `option(USE_KOKKOS …)`
   beside the existing `USE_CUDA`/`USE_METAL` block, mutually exclusive with
   both. Under `USE_KOKKOS`: re-assert `cmake_minimum_required(VERSION 3.16)`
   (Kokkos floor; the CUDA/Metal/CPU builds keep the 3.10 floor), set
   `CMAKE_CXX_STANDARD 17` **and** `string(REPLACE "-std=c++11" "-std=c++17" …)`
   (the compiler blocks hard-set `-std=c++11`), then run the discovery chain
   (override → `external/kokkos` → `../kokkos` → `find_package(Kokkos)`) with a
   `if(NOT TARGET Kokkos::kokkos)` guard. `Kokkos_ENABLE_SERIAL/OPENMP` (+ CUDA
   later) come from the configure line. In `src/CMakeLists.txt` add the
   `USE_KOKKOS` source branch and link `Kokkos::kokkos` into `libmusic`. Mirror
   the `USE_KOKKOS` option + `MUSIC_BACKEND_DIR` arm + mutual-exclusion in the
   parent `X-SCAPE/CMakeLists.txt`; its existing `add_subdirectory` (line 557)
   then builds music4gpu with Kokkos automatically.
3. **Lifecycle (resolved — see Risks).** A host-safe `KokkosRuntimeGuard`
   (RAII over `Kokkos::initialize/finalize`, declared in `gpu/kokkos_runtime.h`,
   defined in `kokkos_runtime.cpp` — the only TU that includes Kokkos headers)
   brackets the run. **PRIMARY (X-SCAPE):** construct it in the framework `main()`
   *before* the `JetScape` object, so one init/finalize spans the whole
   `Init→Exec→Clear→Finish` run. **Not** a `JetScape` member — the child modules
   live in the `JetScapeTask` base (`tasks`, `JetScapeTask.h:314`) and are
   destroyed *after* any derived member, so a member guard would `finalize()`
   while MUSIC's Views are still alive (use-after-finalize). **SECONDARY
   (stand-alone):** the guard sits at the top of `music4gpu/src/main.cpp`. The
   guard is collision-safe (skips init/finalize if Kokkos is already up), and
   MUSIC's `GPUGrid` stays lifecycle-agnostic. Device selection
   (`--kokkos-map-device-id-by=mpi_rank`) only matters if a given executable
   initializes MPI — the core X-SCAPE framework does not.
4. **Skeletons.** Create `src/gpu/KokkosPipelines.{h,cpp}` (mirrors
   `CUDAPipelines.h` 1:1, PIMPL — header stays Kokkos-free, D7) and
   `src/gpu/GPUGrid_kokkos.cpp` (Kokkos impl of the *existing* `GPUGrid` class,
   like `GPUGrid_cuda.cu`), plus the `kokkos_runtime.{h,cpp}` shim. Extend the
   `advance.h` alias block + `MUSIC_USE_GPU` guard for `USE_KOKKOS`. Wire the
   `.cpp` sources into `src/CMakeLists.txt` under a `USE_KOKKOS` branch (parallel
   to the `USE_CUDA` block). **Stage-0 behaviour:** `KokkosPipelines::initialize()`
   returns `false` (and `GPUGrid::allocate()` as a second net), so `advance.cpp`
   keeps `gpu_ready_ == false` and runs the CPU path — every dispatch/grid method
   exists only for linkage. Stage 1 fills them with real Views + kernels.
5. **Bring-up on host backend first.** Build with `Kokkos::Serial`/`OpenMP`, route
   `dispatch_*` to correct (even if trivial) implementations, and confirm an
   end-to-end MUSIC run reproduces the CPU reference. This validates *all* the
   plumbing (alias, grid, lifecycle, build) before any GPU concern.

### Stage 1 — Functional GPU parity

1. **Device helpers** (`music_kernels.cu` statics) → a shared
   `KOKKOS_FORCEINLINE_FUNCTION` header **templated on `Real`** (D4): `gpu_reconst`
   (Newton-Brent u⁰/primitive recovery), minmod slope limiter, EOS log-table
   interp, index helpers. Reused verbatim by every kernel and both backends.
2. **Kernels** → `parallel_for` over `MDRangePolicy<Rank<3>>({0,0,0},{Neta,Nx,Ny})`
   in pipeline order (`make_du` → `uwrhs` → `w_source` → `uprhs` → `delta_qi` →
   `finalize_ideal` → `first_rk_step_w_full`); reduction → `parallel_reduce` with
   `Kokkos::Max<Real>`. **Stage 1 keeps flat 1D `View<Real*>` with the existing
   `comp*Ncells+cell` indexing** (D3) — the multi-D/layout switch is Stage 2.
3. **Memory (shallow, D5)** — `Fields` (`std::vector<double>`) is left untouched;
   `KokkosGrid` holds the only Views. Coherent path → `SharedSpace`; discrete path
   → device Views + pinned mirror + `deep_copy` (replacing the manual staging in
   `GPUGrid_cuda.cu`). Snapshot rotation/swap → cheap View-handle swaps; host-
   readable getters return host-mirror `data()` so the `advance.cpp` hydro-source
   pre-pass is unchanged. **Allocation model:** all Views are allocated once at
   init and persist for the whole run (rotated by handle-swap, never per-step) —
   the same allocate-once model as today's CUDA backend, so the stream-ordered
   allocator (`cudaMallocAsync` / memory pools) is intentionally N/A and Kokkos's
   default `cudaMalloc`-backed Views are sufficient.
4. **Per-kernel golden buffers (D6)** — add a debug hook dumping each intermediate
   device buffer (`dwmn`, `qi_out`, `theta/a/sigma`, `uwrhs`, …) from both the
   native-CUDA and Kokkos backends; diff offline and gate each kernel as it lands.
   Then validate end-to-end with `Kokkos::Cuda` against `MUSIC_FORCE_CPU` and
   `tests/eos_gpu_vs_cpu.sh` (EOS 91).

---

## Stage 2–3 deep-dive

### Stage 2 — performance parity + portability

**2a · Layout migration (D3 realized).** Replace the flat `View<Real*>` indexed
`flat[comp*Ncells+cell]` with a rank-2 `View<Real**, ExecSpace::array_layout>(Ncells,
Ncomp)` indexed `V(cell, comp)`, keeping the existing `cell = ix + Nx*(iy + Ny*ieta)`
math. The execution-space default layout then does the work for free:
- **GPU** (`LayoutLeft`): `V(cell,comp)` → address `cell + Ncells*comp` = today's
  component-major layout → adjacent cells/threads contiguous → **coalesced** (the exact
  access pattern the CUDA kernels were hand-tuned for).
- **CPU** (`LayoutRight`): `V(cell,comp)` → address `cell*Ncomp + comp` → a cell's
  components are contiguous → **cache-friendly** and auto-vectorizable for per-cell compute.

One kernel body (`V(cell,comp)`) compiles to the right stride on each backend — the single
change that makes the OpenMP backend fast enough to later replace the legacy CPU loops
(Stage 4). Do it once; re-run the D6 golden buffers to prove the GPU numbers are unchanged.

**2b · MDRangePolicy iteration + tiling.** `MDRangePolicy<Rank<3>>({0,0,0},{Neta,Ny,Nx})`
with **ix innermost (last index)** so adjacent work items hit adjacent cells — coalescing
consistent with 2a. Match the hand-tuned CUDA 8×8×4 block via an explicit tile `{tX,tY,tEta}`
(product ≤ device max threads/block); expose it as a tunable and sweep vs the native-CUDA
baseline. For the boost-invariant 2D case (`Neta==1`) drop to `Rank<2>` (or tile `…×1`) so no
work-item dimension is wasted, mirroring the existing 2D adaptation. On CPU the tile is a
cache-blocking factor (auto-tile first, tune if needed).

**2c · Memory traits & math.** EOS tables → `View<const Real*, MemoryTraits<RandomAccess>>`
(the `__ldg`/texture path the CUDA code uses today). Reductions stay `parallel_reduce` with
`Kokkos::Max<Real>` (no atomics). Enable fast-math per backend (CUDA `--use_fast_math`
already; HIP/SYCL `-ffast-math`) — this is exactly what makes cross-backend **bitwise** match
impossible and motivates the relative tolerance (D10).

**2d · AMD (HIP) + Intel (SYCL) bring-up.** Once Stage 1 is correct on `Cuda` and the layout
is execution-space-templated (2a), this is mostly a build-matrix step:
- Flip `Kokkos_ENABLE_HIP` / `Kokkos_ENABLE_SYCL`, set the arch (`Kokkos_ARCH_AMD_GFX90A` for
  MI250X/Frontier, `Kokkos_ARCH_INTEL_PVC` for Aurora), compile with `hipcc` / `icpx`.
- **Coherent-memory generalization:** the GB10 coherent-vs-discrete split becomes
  `Kokkos::SharedSpace` where the device offers unified memory (also MI300A APU, Intel USM) vs
  device Views + mirror `deep_copy` otherwise — one path keyed off a space trait, not `#ifdef`.
- **Gotchas to budget:** AMD wavefront = 64 (vs warp 32) shifts block-size/scratch assumptions
  (MDRangePolicy hides most; Stage-3 scratch tiles need re-sizing); SYCL USM vs HIP managed
  semantics differ; device `printf`/`assert` support varies. Each backend gated by the D9
  consistency check.

### Stage 3 — Kokkos-native optimizations

**3a · Kernel fusion (highest ROI).** The substep is 8 kernels exchanging seven global
intermediates (`theta/a/sigma`, `uwrhs`, `dwmn`, `uprhs`, `qi`). Two fusions stand out:
- **`uwrhs` + `uprhs`** → one radius-2 pass: both are ±2-cell minmod reconstructions over the
  *same* `u`-neighborhood, so fusing reuses those neighbor loads and drops a launch + the
  `uprhs_out` traffic.
- **`delta_qi` + `finalize_ideal`** → consume the reconstructed `qi` in-register and run the
  Newton solve immediately, eliminating the `qi_out[5*Ncells]` global round-trip on the
  pipeline's heaviest data path.

Fusion trades launch/bandwidth for register pressure, so keep it **behind a compile-time
switch** (D11) with the unfused kernels retained as the D6 golden reference.

These two fusions apply equally to the **current CUDA *and* Metal backends** today — fusion
is a general GPU technique, not Kokkos-specific. The pre-port analysis (dependency graph,
legality, backend-specific mechanics, register-pressure caveat) lives in
**`PlanKernelFusion.md`**. Doing it pre-port means maintaining each fusion *twice* (CUDA C++
+ Metal MSL); the Kokkos port re-expresses each one *once* for all backends — so unless the
throughput is needed sooner, deferring fusion to here is cheaper.

**3b · Portable stencil scratch (TeamPolicy).** Re-express the CUDA-only
`gpu_make_w_source_tiled` shared-memory halo as `TeamPolicy` + `team_scratch`: each team loads
a `(tile + 2·radius)` halo into scratch, barriers, then computes — now portable to HIP / SYCL /
CPU (on CPU scratch is just a buffer). Prioritize the **bandwidth-bound** stencils (`w_source`
radius-1, `uwrhs/uprhs` radius-2); it helps the compute-bound `delta_qi` far less.

**3c · `delta_qi` bottleneck (~60% of runtime).** Newton-Brent u⁰ recovery + 12
reconstructions/cell ⇒ heavy register pressure. Levers in ROI order: (1) `TeamPolicy` +
`ThreadVectorRange` across the 12 reconstructions / 3 directions so solves spread over vector
lanes and per-work-item registers drop (raises occupancy); (2) split reconstruct vs
Riemann/Newton into two kernels (trades a buffer for occupancy — measure); (3) seed the Newton
solve from the previous timestep's u⁰ to cut iterations (physics-level).

**3d · Async overlap & residency.** Replace the dual-stream Phase-4 prefetch with
`partition_space(exec, …)` → independent execution-space instances (= streams): run the next
H2D `deep_copy` on a copy instance concurrently with the prior substep's compute, gate via
`fence`. On coherent parts (GB10 / MI300A / PVC) skip the explicit overlap (migrate-on-touch),
keyed off the same space trait as 2d. Keep evolving state in **persistent device Views** rotated
by cheap handle-swap (the existing snapshot rotation), so per-step H2D is eliminated and overlap
only matters for diagnostic syncs.

### Deferred perf decisions — now settled

- **D10 — parity tolerance (relative, not bitwise).** Fast-math + `float` + cross-backend FP
  reordering rule out exact match. Starting policy, tightened empirically: per-kernel golden diff
  *same backend* (CUDA vs Kokkos/Cuda, matched fast-math) ≤ ~1e-6 rel; *cross-backend* per-kernel
  ≤ ~1e-4 rel; end-to-end `eps_max(τ)` and the D9 gate ≤ ~1e-3 rel (the tolerance
  `eos_gpu_vs_cpu.sh` already uses for hotQCD).
- **D11 — optimization posture: "optimize behind a flag, validate against the unfused golden."**
  The unfused, un-scratched 1:1 kernels stay the **default and the D6 golden reference**. Fusion
  (3a) and scratch tiling (3b) live behind compile-time switches (`MUSIC_KOKKOS_FUSE`, …) and are
  enabled per backend only after they pass the D9 consistency gate against that golden path.

---

## Key risks & decisions

- **Kokkos init/finalize ownership (resolved).** Conceptual owner = the
  `JetScape` process-singleton driver; mechanical home = a `Kokkos::ScopeGuard`
  in the framework `main()`, declared *before* the `JetScape` object (one
  init/finalize per process, outliving every event, hydro reuse, and module
  View). **Never a `JetScape` member:** the module list `tasks` lives in the
  `JetScapeTask` base and is destroyed last — after all derived members — so a
  member guard would finalize before MUSIC's Views are freed → use-after-finalize.
  In-class fallback (only if the driver `main` is off-limits): `initialize()` in
  `JetScape::Init()` + `finalize()` in `~JetScape()` *after* an explicit
  `tasks.clear()`. Get this right early — it gates everything and must not collide
  if another module (e.g. a future Kokkos `SurfaceFinder`) also uses Kokkos.
- **NVIDIA backend choice during transition.** Keep native CUDA as the default and
  bring up `USE_KOKKOS` in parallel; only deprecate CUDA after Stage 2 parity.
  De-risks by always having a reference.
- **Metal stays separate** (decided). Document that Apple-GPU acceleration is the
  one path Kokkos does not cover.
- **Tiled stencil + dual-stream** are perf-only; ship Stage 1 with the plain
  kernel + single stream, optimize in Stage 3.
- **Reduction reproducibility** — `Kokkos::Max` float reduction may differ in the
  last bit from the atomicMax bit-trick; bounded by the same float tolerance the
  EOS test already uses.

---

## Verification

**Standard test scripts (`tests/`)** — the same harness that validated the CUDA
and Metal ports; reuse it for Kokkos. All run from the repo root with Gubser ICs
(no external data; the EOS test needs the hotQCD table). Point at the Kokkos build
via `GPU_BIN=` (the scripts auto-detect `build_metal/build_cuda/build_gpu`):

- **Correctness — `tests/eos_gpu_vs_cpu.sh`** (EOS 91 hotQCD default): runs one
  input on `build/src/MUSIChydro` (CPU) and the GPU binary; compares `eps_max(τ)`
  within `TOL` (1e-3). The Stage-0 invocation:
  ```
  cmake -S . -B build        -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  bash get_kokkos.sh                                     # latest Kokkos (or pin a tag)
  cmake -S . -B build_kokkos -DUSE_KOKKOS=ON             && cmake --build build_kokkos -j
  GPU_BIN=$PWD/build_kokkos/src/MUSIChydro bash tests/eos_gpu_vs_cpu.sh
  ```
- **Throughput — `tests/cuda_vs_cpu_bench.sh`**: total wall-time GPU vs CPU across
  a grid sweep (needs `build/` + the GPU build).
- **Per-step cost — `tests/cuda_perstep_bench.sh`**: isolates per-timestep compute
  (`(T(long) − T(short))/100`, min of 3 runs) to factor out fixed init overhead.

> **Stage-0 result (this branch):** `eos_gpu_vs_cpu.sh` (GPU_BIN = the Kokkos
> build) reports **[3/3] PASS, max rel error `0.00e+00`** — the `USE_KOKKOS` build
> reproduces the CPU reference bit-for-bit over 101 steps — and **[2/3] PASS**.
> **[1/3] (GPU-dispatch) FAILs by design**: the skeleton's
> `KokkosPipelines::initialize()` returns false, so MUSIC runs the CPU path; [1/3]
> flips to PASS at Stage 1 when the kernels land.

> **Future (Stage 1+):** add backend-explicit variants
> (`kokkos_cuda_vs_cuda.sh`, `kokkos_vs_cpu_bench.sh`, `kokkos_perstep_bench.sh`, …)
> that compare the Kokkos/Cuda build against the native-CUDA build, and the Kokkos
> host backend against the CPU reference — keeping the CUDA/Metal scripts for the
> legacy paths.

- **CPU reference:** `MUSIC_FORCE_CPU=1` forces the CPU path for an explicit diff.
- **Native-CUDA cross-check:** identical input on the native-CUDA and Kokkos/Cuda
  builds; `eps_max(τ)` traces must agree within float tolerance.
- **Cross-backend (Kokkos) consistency:** same input on Kokkos `Serial`, `OpenMP`,
  `Cuda` (later `HIP`/`SYCL`); diff traces — a single-source consistency gate from
  Stage 1 (D9).
- **Conservation:** existing `check_conservation_law` in `grid_info.cpp`.
- **Performance:** per-step `EvolveIt` wall-time of Kokkos/Cuda vs native CUDA on
  the same NVIDIA GPU (this GB10/Grace box); track per-stage against the parity
  target.

---

## Open items to confirm during execution

- ~~Kokkos acquisition method~~ — **resolved (D1):** in-repo `get_kokkos.sh`
  (latest-default, pin via arg) + `find_package` fallback for HPC.
- ~~Exact pinned Kokkos version~~ — **resolved:** latest release by default
  (5.1.1 at time of writing); pin via `get_kokkos.sh <tag>` for reproducible builds.
- The AMD/Intel toolchains available for the Stage 2 bring-up (ROCm/HIP,
  oneAPI/SYCL) and the exact `Kokkos_ARCH_*` flags for the target machines.
- Kokkos's exact `cmake_minimum_required` for the pinned tag (5.x ≳ 3.25) —
  confirm the host CMake satisfies it; the build raises the floor to 3.16 under
  `USE_KOKKOS` and Kokkos itself enforces the rest.
