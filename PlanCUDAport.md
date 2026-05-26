# CUDA Port Plan for MUSIC4GPU

## Context

MUSIC4GPU implements a 3+1D relativistic second-order viscous hydrodynamics solver for heavy-ion physics, accelerated with Metal 3 GPU on Apple Silicon (M3 Max). The Metal implementation achieves 1.50× speedup over 12-thread OpenMP CPU at the production-relevant 64×64×32 grid scale (131k cells). This plan ports the entire GPU acceleration layer to CUDA for NVIDIA GPUs, preserving the same public API while leveraging CUDA-specific architectural advantages (HBM bandwidth, constant/texture memory, warp primitives, explicit stream control) to improve on the Metal baseline.

**README benchmark baseline (Metal, M3 Max, 64×64×32, 40 steps):**
- GPU: 1.62s, CPU 12T: 2.43s → **1.50× speedup**
- README roadmap documents +18–30% additional speedup via shared memory tiling

---

## Codebase Overview

### Current GPU File Structure
```
src/gpu/
├── gpu_types.h           # Shared C++/MSL types (MUSICGridParams, GPUEosParams)
├── GPUGrid.h             # GPU memory manager: 3 SoA snapshots, AoS↔SoA, EOS upload
├── GPUGrid.mm            # Objective-C++: MTLBuffer alloc, unified memory
├── MetalPipelines.h      # Metal singleton: 7 dispatch_*() methods
├── MetalPipelines.mm     # Metal API: device, command queue, batch command buffer
└── music_kernels.metal   # 2156-line MSL: 7 kernels + device helpers
```

### 7 Compute Kernels (all threads-per-cell, `dim3(8,8,4)` threadgroup)
| Kernel | Source CPU Function | Role |
|--------|---------------------|------|
| `gpu_make_w_source` | `Diss::MakeWSource()` | Divergence of viscous stress tensor |
| `gpu_make_delta_qi` | `Advance::MakeDeltaQI()` | Ideal KT flux (Newton-Brent solver per cell) |
| `gpu_finalize_ideal` | `Reconst::ReconstIt_shell()` | Conserved→primitive Newton inversion |
| `gpu_make_uwrhs` | `Diss::Make_uWRHS()` | KT flux divergence for 5 shear components |
| `gpu_make_uprhs` | `Diss::Make_uPRHS()` | KT flux divergence for bulk pressure |
| `gpu_make_du` | `U_derivative::MakedU()` | θ, a^μ, σ^μν velocity gradient tensors |
| `gpu_first_rk_step_w_full` | `Diss::Make_uWSource()` + RK algebra | Full viscous RK update (shear + bulk) |

### 3 Completed CPU Optimizations (Steps 1–3, commit `5a45a67`)
- **Step 1**: OpenMP-parallelized AoS↔SoA layout conversion
- **Step 2**: Batched Metal command buffers (7 kernels in one commit/wait)
- **Step 3**: Snapshot pointer rotation (avoids round-trip copy when GPU state is authoritative)

### Key Architectural Differences: Metal vs CUDA
| Feature | Metal (M3 Max) | CUDA (A100) |
|---------|----------------|-------------|
| Memory BW | ~400 GB/s unified | ~2000 GB/s HBM2e |
| FP32 TFLOPS | ~7 | ~77.6 |
| Shared memory/SM | 32 KB | 164 KB configurable |
| Constant memory | None | 64 KB `__constant__` |
| Memory model | Unified (zero-copy) | Discrete (explicit PCIe transfer) |
| Stream model | Command buffers | `cudaStream_t` |
| Warp primitives | Limited | `__ballot_sync`, `__shfl_sync` |

---

## Implementation Plan

### Phase 1: Direct Port — Correctness First

**New files to create:**
- `src/gpu/CUDAPipelines.h` — mirrors `MetalPipelines.h` API exactly
- `src/gpu/CUDAPipelines.cu` — CUDA singleton replacing `MetalPipelines.mm`
- `src/gpu/GPUGrid_cuda.cu` — CUDA memory management replacing `GPUGrid.mm`
- `src/gpu/music_kernels.cu` — all 7 kernels translated from MSL to CUDA C++

**Files to modify:**
- `src/gpu/gpu_types.h` — add `#elif defined(__CUDACC__)` branch for `WMUNU_IDX`/`GMUNU_DIAG` arrays (replace Metal `constant` qualifier with `__device__ __constant__`)
- `src/advance.h` — add `#ifdef USE_CUDA` include block alongside `#ifdef USE_METAL`
- `src/advance.cpp` — add `#elif defined(USE_CUDA)` dispatch blocks; logic mirrors Metal
- `CMakeLists.txt` — add `option(USE_CUDA ...)`, `enable_language(CUDA)`, `find_package(CUDAToolkit)`
- `src/CMakeLists.txt` — add CUDA source files, `LANGUAGE CUDA` property, `CUDA::cudart` link

#### Metal → CUDA Translation Map
| Metal | CUDA |
|-------|------|
| `kernel void foo(device float* b [[buffer(0)]], ...)` | `__global__ void foo(float* __restrict__ b, ...)` |
| `[[thread_position_in_grid]]` (uint3 gid) | `uint3 gid = {blockIdx.x*blockDim.x+threadIdx.x, ...}` |
| `threadgroup float s[]` | `__shared__ float s[]` |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads()` |
| `device const float*` | `const float* __restrict__` |
| `constant struct params [[buffer(N)]]` | by-value kernel argument (fits in 4096B limit) |
| `clamp(x, lo, hi)` | `fminf(fmaxf(x, lo), hi)` |
| `MTLResourceStorageModeShared` | `cudaMallocManaged()` + `cudaMemPrefetchAsync()` |
| `[cmdBuf commit]` + `[cmdBuf waitUntilCompleted]` | `cudaStreamSynchronize(stream)` |
| `inline` on device helper | `__device__ __forceinline__` |

#### `CUDAPipelines.cu` Key Decisions
- `initialize()`: `cudaGetDeviceCount`, `cudaSetDevice(0)`, `cudaStreamCreate`
- No library loading (kernels compiled directly into binary by `nvcc`)
- All 7 `dispatch_*()` methods: compute `dim3 grid`, call `kernel<<<grid, block_dim_, 0, stream>>>(...)`
- `MUSICGridParams` (≈424 bytes) passed by-value to kernels — within 4096B argument limit
- `begin_batch()` / `end_batch()`: no-ops in Phase 1 (CUDA streams serialize automatically)
- `wait()`: `cudaStreamSynchronize(compute_stream_)`

#### `GPUGrid_cuda.cu` Key Decisions
- Replace `[device newBufferWithLength:options:MTLResourceStorageModeShared]` with `cudaMallocManaged(&ptr, bytes)`
- Call `cudaMemPrefetchAsync(ptr, bytes, device_id, 0)` immediately after allocation to avoid page-fault jitter
- `release()`: `cudaFree` on each handle
- AoS↔SoA converters (`copy_to_gpu`, `copy_wmunu_to_cpu`, `copy_primitives_to_cpu`, `rotate_snapshots`) are pure C++ with OpenMP — reuse verbatim, no CUDA API in these functions
- EOS upload: allocate via `cudaMallocManaged` + prefetch to device

#### Build Config
```cmake
# CMakeLists.txt (top-level)
option(USE_CUDA "Enable CUDA GPU acceleration (NVIDIA)" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")  # V100, A100, RTX30xx, RTX40xx, H100
endif()

# src/CMakeLists.txt
if(USE_CUDA)
    target_sources(${libname} PRIVATE
        gpu/CUDAPipelines.cu gpu/GPUGrid_cuda.cu gpu/music_kernels.cu)
    set_source_files_properties(
        gpu/CUDAPipelines.cu gpu/GPUGrid_cuda.cu gpu/music_kernels.cu
        PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${libname} PRIVATE USE_CUDA)
    target_link_libraries(${libname} CUDA::cudart)
endif()
```

Build command:
```bash
cmake -S . -B build_cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build_cuda -j$(nproc)
```

#### Implementation Order Within Phase 1
1. Update `gpu_types.h` guard (30 min)
2. `CMakeLists.txt` + `src/CMakeLists.txt` CUDA config (1h)
3. `GPUGrid_cuda.cu` — allocation only, stub dispatch methods (2h)
4. `CUDAPipelines.h/.cu` skeleton with `initialize()` + `wait()` (1h)
5. Port `gpu_make_w_source` to `music_kernels.cu` — simplest stencil (3h)
6. Get first build + launch on 2×2×1 grid without crash
7. Port remaining 6 kernels in dependency order:
   `gpu_make_delta_qi` → `gpu_finalize_ideal` → `gpu_make_du` → `gpu_make_uwrhs` → `gpu_make_uprhs` → `gpu_first_rk_step_w_full` (2 days)
8. Update `advance.h` + `advance.cpp` with `USE_CUDA` guards (3h)
9. Correctness test at 32×32×1: `eps_max` traces within `1e-3`

---

### Phase 2: CUDA Memory & Cache Optimizations

**EOS tables → `__constant__` memory**

`eos_P` and `eos_dPde` (8192 floats each = 32 KB each, 64 KB total) fit exactly in CUDA's `__constant__` cache. These are broadcast reads (all threads in a block read the same EOS index) — the ideal `__constant__` use case:

```cpp
// In music_kernels.cu (file scope):
__constant__ float d_eos_P[GPU_EOS_N];
__constant__ float d_eos_dPde[GPU_EOS_N];

// Upload once in CUDAPipelines::initialize():
cudaMemcpyToSymbol(d_eos_P,    P_data,    n_pts*sizeof(float));
cudaMemcpyToSymbol(d_eos_dPde, dPde_data, n_pts*sizeof(float));
```

`eos_s` and `eos_T` (64 KB total, exceeds constant cache limit) → CUDA texture objects via `tex1Dfetch<float>()`.

**`__ldg()` for read-only stencil inputs**

All neighbor stencil reads in helper functions `get_Wmunu()`, `get_u()`, `get_pi_b()`:
```cpp
__device__ __forceinline__ float get_Wmunu(
    const float* __restrict__ W, int comp, int c, int Ncells) {
    return __ldg(&W[comp * Ncells + c]);  // routes through read-only L1 cache
}
```

**Block dimension tuning**

```cpp
// In CUDAPipelines::initialize() — tune per heavy kernel:
int block_size_opt, min_grid;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size_opt, gpu_make_delta_qi, 0, 0);
// Map 1D optimal size back to 3D: keep z=4 (eta), balance x/y
```

---

### Phase 3: Shared Memory Tiling (+18–30% from README roadmap)

CUDA's 164 KB configurable shared memory (vs Metal's 32 KB threadgroup limit) makes the tiling strategy from the README roadmap fully viable.

**Tiling for `gpu_make_delta_qi` (±2 stencil, ~60% of runtime)**

For block shape `(Bx, By, Bz)` = `(8, 8, 4)`, the shared tile is `(12)(12)(8)` cells. Loading `epsilon + rhob + u[4]` = 6 floats per cell → 27 KB. Fits in 164 KB Ampere shared memory.

```cuda
__shared__ float s_eps[12*12*8];
__shared__ float s_u[4][12*12*8];
// Cooperative halo loading; all stencil reads hit shared memory
// L1 latency: ~4 cycles vs ~200 cycles HBM global
```

Configure shared memory preference:
```cpp
cudaFuncSetAttribute(gpu_make_delta_qi,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
```

**Priority order for tiling:**
1. `gpu_make_delta_qi` — ±2 stencil, 5 reconstructions per direction, ~60% of runtime
2. `gpu_make_uwrhs` — ±2 stencil, 5 Wmunu components, ~10% of runtime
3. `gpu_make_uprhs` — ±2 stencil, scalar bulk pressure
4. `gpu_make_du` — ±1 stencil, velocity gradients
5. `gpu_make_w_source` — ±1 stencil, 14 Wmunu + 5 u components

---

### Phase 4: CUDA Stream Parallelism

**Dual-stream design:**
- `compute_stream_`: all 7 kernel dispatches
- `copy_stream_`: AoS→SoA `cudaMemcpyAsync` host→device transfers

**Pinned host staging buffers:**
```cpp
cudaHostAlloc(&host_stage.epsilon, Ncells*sizeof(float), cudaHostAllocDefault);
```

Enables DMA at PCIe 4.0 peak (~32 GB/s) vs pageable transfer (~8 GB/s). For 131k cells, full SoA transfer is ~21 MB → ~0.65 ms at 32 GB/s.

**Overlap pattern:** OpenMP AoS→SoA pack on CPU → `cudaMemcpyAsync` to device on `copy_stream_` → `cudaStreamWaitEvent` gates compute stream. Copy of `prev` snapshot overlaps with compute kernels for `curr` on next substep.

---

### Phase 5: Advanced CUDA Features

**Warp-level Newton convergence (`gpu_make_delta_qi`, `gpu_finalize_ideal`)**

```cuda
for (int it = 0; it < 60; it++) {
    // ... Newton-Brent step ...
    unsigned mask = __ballot_sync(0xFFFFFFFF,
        fabsf(dv_curr) < ABS_ERR);
    if (mask == 0xFFFFFFFF) break;  // all 32 threads converged
}
```

Breaks warp divergence — exits as soon as all 32 threads converge, rather than spinning until the last thread satisfies its individual condition.

**Half-precision (FP16) — exploratory**

For bandwidth-bound kernels (`gpu_make_w_source`, `gpu_make_du`): store Wmunu in FP16, upcast to FP32 for computation, store back in FP16. Halves global memory bandwidth. Validate accumulation error carefully against CPU reference.

---

### Benchmark Scripts

**`tests/cuda_vs_cpu_bench.sh`** — 2D boost-invariant
- Grid sizes: 32×32×1, 64×64×1, 128×128×1 (matching Metal benchmarks)
- 100 timesteps, Delta_Tau=0.005
- Compare `build/src/MUSIChydro` (CPU) vs `build_cuda/src/MUSIChydro` (GPU)
- Report timing, speedup, max relative error in eps_max trace
- Add `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` header

**`tests/cuda_vs_cpu_bench_3d.sh`** — 3+1D production
- Grid sizes: 32×32×8, 32×32×32, 64×64×16, 64×64×32 (matching Metal benchmarks)
- 40 timesteps each
- Correctness threshold: max relative eps_max error < 1e-3

---

## Performance Expectations

| Grid | Metal baseline | Phase 1 (cudaManaged) | Phase 2+3 (tuned+tiled) | Phase 4 (async) |
|------|---------------|-----------------------|--------------------------|-----------------|
| 64×64×32 (131k) | 1.62s | ~0.3–0.6s est. | ~0.1–0.2s est. | ~0.08–0.15s est. |

On A100 vs M3 Max:
- Memory BW: 5× advantage → stencil kernels ~5× faster
- FP32 TFLOPS: 11× advantage → Newton solver ~11× faster
- Shared memory: 5× larger → full ±2 halo tiling viable (not possible in Metal 32KB)
- Net expected speedup vs Metal: **5–10× on A100** at production scale

---

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `src/gpu/music_kernels.metal` | Source to translate | Primary kernel reference (2156 lines) |
| `src/gpu/music_kernels.cu` | **Create** | CUDA kernel translation |
| `src/gpu/CUDAPipelines.h` | **Create** | CUDA singleton API header |
| `src/gpu/CUDAPipelines.cu` | **Create** | CUDA dispatch implementation |
| `src/gpu/GPUGrid_cuda.cu` | **Create** | cudaMallocManaged + prefetch |
| `src/gpu/gpu_types.h` | **Modify** | Add `__CUDACC__` branch for device arrays |
| `src/advance.h` | **Modify** | Add `#ifdef USE_CUDA` include block |
| `src/advance.cpp` | **Modify** | Add `#elif defined(USE_CUDA)` dispatch blocks |
| `CMakeLists.txt` | **Modify** | `USE_CUDA` option, `enable_language(CUDA)` |
| `src/CMakeLists.txt` | **Modify** | CUDA source files, `LANGUAGE CUDA` property |
| `tests/cuda_vs_cpu_bench.sh` | **Create** | 2D benchmark script |
| `tests/cuda_vs_cpu_bench_3d.sh` | **Create** | 3D production benchmark script |

---

## Verification Strategy

**Phase 1 correctness:**
```bash
cmake -S . -B build_cuda -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda -j$(nproc)
# Quick smoke test (tiny grid, 2 steps):
./build_cuda/src/MUSIChydro music_input_tiny
# Full 2D correctness (same pass/fail threshold as Metal):
OMP_NUM_THREADS=4 bash tests/cuda_vs_cpu_bench.sh
# Accept: max relative eps_max error < 1e-3
```

**Phase 2–4 performance:**
```bash
OMP_NUM_THREADS=$(nproc) bash tests/cuda_vs_cpu_bench_3d.sh
# Observe timing at each phase; compare to Metal README numbers
```

**Regression guard:** Re-run 2D correctness test after each phase. The `1e-3` relative error threshold applies throughout.
