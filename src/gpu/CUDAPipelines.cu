// CUDA implementation of CUDAPipelines.
// Selects the GPU, creates a compute stream, and launches the seven hydro
// kernels.  Mirrors MetalPipelines.mm one-to-one so the host dispatch logic in
// advance.cpp can be shared between the two back-ends.

#include <cuda_runtime.h>
#include <cstdio>
#include "CUDAPipelines.h"
#include "GPUGrid.h"
#include "gpu_types.h"
#include "music_kernels.cuh"

// Selected device id, shared with GPUGrid_cuda.cu for prefetch hints.
int g_cuda_device_id = 0;

// Factor a target thread count into a 3-D block adapted to the grid's eta
// extent.  The eta dimension is capped at 4 (matching the original Metal
// threadgroup) but collapses to Neta for thin grids, so a 2-D boost-invariant
// run (Neta==1) keeps all threads in the x/y plane instead of idling 3 of
// every 4 eta lanes.  x is sized to 32 for coalesced SoA loads where possible.
static void compute_launch(const GPUGrid& gpu, int max_threads,
                           dim3& block, dim3& grid) {
    int neta = gpu.Neta();
    int bz = (neta >= 4) ? 4 : neta;
    if (bz < 1) bz = 1;
    int plane = max_threads / bz;          // budget for bx*by
    int bx = 32;
    while (bx > 1 && bx > plane) bx >>= 1;  // shrink x if the budget is small
    if (bx > gpu.Nx()) {                     // don't over-provision tiny grids
        while (bx > 1 && bx > gpu.Nx()) bx >>= 1;
    }
    int by = plane / bx;
    if (by < 1) by = 1;
    block = dim3((unsigned)bx, (unsigned)by, (unsigned)bz);
    grid  = dim3((gpu.Nx()   + bx - 1) / bx,
                 (gpu.Ny()   + by - 1) / by,
                 (gpu.Neta() + bz - 1) / bz);
}

static inline void check_launch(const char* name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[MUSIC-GPU] kernel launch '%s' failed: %s\n",
                name, cudaGetErrorString(err));
    }
}

// ── singleton ─────────────────────────────────────────────────────────────────

CUDAPipelines& CUDAPipelines::instance() {
    static CUDAPipelines inst;
    return inst;
}

CUDAPipelines::~CUDAPipelines() {
    if (copy_event_) {
        cudaEventDestroy(static_cast<cudaEvent_t>(copy_event_));
        copy_event_ = nullptr;
    }
    if (copy_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(copy_stream_));
        copy_stream_ = nullptr;
    }
    if (compute_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(compute_stream_));
        compute_stream_ = nullptr;
    }
}

// ── initialization ────────────────────────────────────────────────────────────

bool CUDAPipelines::initialize(const char* /*unused*/) {
    if (ready_) return true;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[MUSIC-GPU] No CUDA device found: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    device_id_ = 0;
    g_cuda_device_id = device_id_;
    err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[MUSIC-GPU] cudaSetDevice(0) failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id_) == cudaSuccess) {
        fprintf(stderr, "[MUSIC-GPU] CUDA device: %s (cc %d.%d, %.1f GB)\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // Detect coherent host-memory access (integrated GPU or NVLink-C2C parts).
    // On these the unified buffers are read in place, so the Phase-4 dual-stream
    // prefetch is unnecessary (and the gate would only add latency).
    int pageable_access = 0;
    cudaDeviceGetAttribute(&pageable_access,
                           cudaDevAttrPageableMemoryAccess, device_id_);
    coherent_memory_ = (pageable_access != 0) || (prop.integrated != 0);
    fprintf(stderr, "[MUSIC-GPU] coherent host memory: %s\n",
            coherent_memory_ ? "yes (dual-stream prefetch skipped)"
                             : "no (dual-stream prefetch active)");

    cudaStream_t stream = nullptr;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[MUSIC-GPU] cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    compute_stream_ = static_cast<void*>(stream);

    // Phase 4: dedicated copy stream + completion event for the dual-stream
    // data-movement path.  Non-fatal if creation fails — upload_snapshots_async
    // simply becomes a no-op then.
    cudaStream_t cstream = nullptr;
    if (cudaStreamCreate(&cstream) == cudaSuccess) {
        copy_stream_ = static_cast<void*>(cstream);
        cudaEvent_t ev = nullptr;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess)
            copy_event_ = static_cast<void*>(ev);
    }

    // Occupancy-tuned block size for the heaviest kernel (gpu_make_delta_qi,
    // ~60% of runtime: Newton-Brent solve + 12 reconstructions per cell).  Its
    // register footprint usually caps the useful block at <=256 threads, which
    // we then factor into 3-D per dispatch (see compute_launch).
    int min_grid = 0, opt_block = 0;
    if (cudaOccupancyMaxPotentialBlockSize(&min_grid, &opt_block,
                                           gpu_make_delta_qi, 0, 0) == cudaSuccess
        && opt_block > 0) {
        max_block_threads_ = opt_block;
        if (max_block_threads_ > 256) max_block_threads_ = 256;  // keep 3-D factoring clean
        if (max_block_threads_ < 64)  max_block_threads_ = 64;
    }
    fprintf(stderr, "[MUSIC-GPU] max threads/block = %d\n", max_block_threads_);

    ready_ = true;
    return true;
}

// ── synchronization ─────────────────────────────────────────────────────────

void CUDAPipelines::wait() {
    if (!ready_) return;
    cudaError_t err =
        cudaStreamSynchronize(static_cast<cudaStream_t>(compute_stream_));
    if (err != cudaSuccess) {
        fprintf(stderr, "[MUSIC-GPU] cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(err));
    }
}

// Batch mode is implicit for a single stream — these are no-ops.
void CUDAPipelines::begin_batch() {}
void CUDAPipelines::end_batch()   {}

// ── dual-stream snapshot upload ──────────────────────────────────────────────

void CUDAPipelines::upload_snapshots_async(GPUGrid& gpu) {
    if (!ready_ || !copy_stream_) return;
    // Coherent unified memory (GB10): the kernels read the packed buffers in
    // place, so prefetching + gating only adds latency.  Skip it.
    if (coherent_memory_) return;
    auto cs = static_cast<cudaStream_t>(copy_stream_);
    const size_t nc = static_cast<size_t>(gpu.Ncells());

    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = device_id_;
    auto pf = [&](void* p, size_t bytes) {
        if (p) cudaMemPrefetchAsync(p, bytes, loc, 0, cs);
    };
    // Prefetch the two snapshots the kernels will read this substep onto the
    // copy stream.  On a discrete GPU this is the host→device transfer; on the
    // coherent GB10 it is a residency hint.
    const GPUSnapshot* snaps[2] = {&gpu.snap_curr, &gpu.snap_prev};
    for (const GPUSnapshot* s : snaps) {
        pf(s->epsilon,      nc * sizeof(float));
        pf(s->rhob,         nc * sizeof(float));
        pf(s->u,        4 * nc * sizeof(float));
        pf(s->Wmunu,   14 * nc * sizeof(float));
        pf(s->pi_b,         nc * sizeof(float));
    }
    // Gate the compute stream on the prefetch so the first kernel does not read
    // before the data is resident (correctness-neutral on coherent memory, but
    // the textbook-correct dual-stream handshake on a discrete GPU).
    if (copy_event_) {
        cudaEventRecord(static_cast<cudaEvent_t>(copy_event_), cs);
        cudaStreamWaitEvent(static_cast<cudaStream_t>(compute_stream_),
                            static_cast<cudaEvent_t>(copy_event_), 0);
    }
}

// ── dispatch_w_source ─────────────────────────────────────────────────────────

void CUDAPipelines::dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    // Phase 3: shared-memory tiled launch.  Use a tiling-friendly 8x8xbz block
    // (small, balanced halo) rather than the coalescing block other kernels
    // use.  bz collapses to Neta for thin grids.  Dynamic shared holds the
    // (bx+2)(by+2)(bz+2) halo tile of Wmunu[14]+u[4]+pi_b — all configs here
    // stay under the 48 KB default carveout.
    int bz = (gpu.Neta() >= 4) ? 4 : (gpu.Neta() < 1 ? 1 : gpu.Neta());
    int bx = 8, by = 8;
    dim3 block(bx, by, bz);
    dim3 grid((gpu.Nx()   + bx - 1) / bx,
              (gpu.Ny()   + by - 1) / by,
              (gpu.Neta() + bz - 1) / bz);
    size_t tile_cells = (size_t)(bx + 2) * (by + 2) * (bz + 2);
    size_t shmem = (14 + 4 + 1) * tile_cells * sizeof(float);
    gpu_make_w_source_tiled<<<grid, block, shmem, stream>>>(
        gpu.snap_curr.Wmunu, gpu.snap_curr.pi_b, gpu.snap_curr.u,
        gpu.snap_prev.Wmunu, gpu.snap_prev.pi_b, gpu.snap_prev.u,
        gpu.dwmn, params);
    check_launch("gpu_make_w_source_tiled");
}

// ── dispatch_delta_qi ─────────────────────────────────────────────────────────

void CUDAPipelines::dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    dim3 block, grid;
    compute_launch(gpu, max_block_threads_, block, grid);
    gpu_make_delta_qi<<<grid, block, 0, stream>>>(
        gpu.snap_curr.epsilon, gpu.snap_curr.rhob, gpu.snap_curr.u,
        gpu.eos_P, gpu.eos_dPde,
        gpu.qi_out, params, gpu.eos_params);
    check_launch("gpu_make_delta_qi");
}

// ── dispatch_finalize_ideal ──────────────────────────────────────────────────

void CUDAPipelines::dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    dim3 block, grid;
    compute_launch(gpu, max_block_threads_, block, grid);
    gpu_finalize_ideal<<<grid, block, 0, stream>>>(
        gpu.qi_out, gpu.dwmn,
        gpu.snap_curr.epsilon, gpu.snap_curr.u,
        gpu.snap_prev.epsilon, gpu.snap_prev.rhob, gpu.snap_prev.u,
        gpu.snap_future.epsilon, gpu.snap_future.rhob, gpu.snap_future.u,
        gpu.eos_P, gpu.eos_dPde,
        params, gpu.eos_params, gpu.qi_source_buf);
    check_launch("gpu_finalize_ideal");
}

// ── dispatch_uwrhs ───────────────────────────────────────────────────────────

void CUDAPipelines::dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    dim3 block, grid;
    compute_launch(gpu, max_block_threads_, block, grid);
    gpu_make_uwrhs<<<grid, block, 0, stream>>>(
        gpu.snap_curr.Wmunu, gpu.snap_curr.u, gpu.uwrhs_out, params);
    check_launch("gpu_make_uwrhs");
}

// ── dispatch_make_du ─────────────────────────────────────────────────────────

void CUDAPipelines::dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    dim3 block, grid;
    compute_launch(gpu, max_block_threads_, block, grid);
    gpu_make_du<<<grid, block, 0, stream>>>(
        gpu.snap_curr.u, gpu.snap_prev.u,
        gpu.theta_buf, gpu.a_buf, gpu.sigma_buf, params);
    check_launch("gpu_make_du");
}

// ── dispatch_uprhs ───────────────────────────────────────────────────────────

void CUDAPipelines::dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    dim3 block, grid;
    compute_launch(gpu, max_block_threads_, block, grid);
    gpu_make_uprhs<<<grid, block, 0, stream>>>(
        gpu.snap_curr.pi_b, gpu.snap_curr.u, gpu.uprhs_out, params);
    check_launch("gpu_make_uprhs");
}

// ── dispatch_first_rk_step_w_full ────────────────────────────────────────────

void CUDAPipelines::dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                                  const MUSICGridParams& params) {
    if (!ready_) return;
    auto stream = static_cast<cudaStream_t>(compute_stream_);
    dim3 block, grid;
    compute_launch(gpu, max_block_threads_, block, grid);
    gpu_first_rk_step_w_full<<<grid, block, 0, stream>>>(
        gpu.snap_curr.Wmunu, gpu.snap_curr.pi_b, gpu.snap_curr.u,
        gpu.snap_prev.Wmunu, gpu.snap_prev.pi_b, gpu.snap_prev.u,
        gpu.snap_curr.epsilon, gpu.snap_prev.epsilon,
        gpu.snap_future.u,
        gpu.uwrhs_out, gpu.theta_buf, gpu.a_buf, gpu.sigma_buf,
        gpu.snap_future.Wmunu, gpu.snap_future.pi_b,
        gpu.eos_P, gpu.eos_s, gpu.eos_T, gpu.eos_dPde,
        gpu.uprhs_out, params, gpu.eos_params,
        gpu.snap_future.epsilon, gpu.snap_future.rhob);
    check_launch("gpu_first_rk_step_w_full");
}
