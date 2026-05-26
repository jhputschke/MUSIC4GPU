// C++ interface to Metal compute pipelines.
// Implementation lives in MetalPipelines.mm (Objective-C++).
//
// Usage pattern:
//   MetalPipelines& mp = MetalPipelines::instance();
//   if (!mp.initialize()) { /* fall back to CPU */ }
//   mp.dispatch_w_source(gpu_grid, params);
//   mp.wait();   // block until GPU finishes

#pragma once
#include "GPUGrid.h"
#include "gpu_types.h"

class MetalPipelines {
public:
    static MetalPipelines& instance();

    // Load the precompiled Metal library and build all pipeline states.
    // Returns false if Metal is unavailable or shader loading fails.
    bool initialize(const char* metallib_path = nullptr);

    bool ready() const { return ready_; }

    // Dispatch gpu_make_w_source over the full grid.
    // Reads from gpu.snap_curr and gpu.snap_prev.
    // Writes dwmn[5 * Ncells] into gpu.dwmn.
    // Non-blocking: call wait() to ensure completion.
    void dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_make_delta_qi over the full grid.
    // Reads from gpu.snap_curr (epsilon, rhob, u) and gpu.eos_P / gpu.eos_dPde.
    // Writes qi_out[5 * Ncells] into gpu.qi_out.
    // gpu.upload_eos() must have been called before the first dispatch.
    // Non-blocking: call wait() to ensure completion.
    void dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_first_rk_step_w_full over the full grid.
    // Consumes the outputs of the four earlier viscous kernels (snap_prev,
    // snap_curr, snap_future.u, uwrhs_out, theta_buf, a_buf, sigma_buf) and
    // writes snap_future.Wmunu + snap_future.pi_b directly.  Used only when
    // the simple-config support matrix is satisfied (see kernel docs);
    // otherwise the host runs the CPU FirstRKStepW loop instead.
    void dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                       const MUSICGridParams& params);

    // Dispatch gpu_make_du over the full grid.
    // Reads gpu.snap_curr.u and gpu.snap_prev.u; writes theta_buf,
    // a_buf, sigma_buf (per-cell viscous geometry).  v1 assumes
    // include_vorticity_terms == 0 and zero net baryon (no dUoverTsup /
    // dUTsup / ∂(µ_B/T) terms).  Non-blocking; sync with wait().
    void dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_make_uwrhs over the full grid.
    // Reads gpu.snap_curr.Wmunu and gpu.snap_curr.u; writes
    // uwrhs_out[5 * Ncells] into gpu.uwrhs_out (one entry per shear index
    // 4..8 per cell).  Non-blocking; sync with wait().
    void dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_finalize_ideal over the full grid.
    // Reads qi_out + dwmn (already produced by the two kernels above), the
    // current and previous snapshots, and the EOS tables; writes the
    // post-Newton primitives (epsilon, rhob, u) into gpu.snap_future.
    // Must be called after dispatch_w_source AND dispatch_delta_qi for the
    // same RK step; non-blocking, sync with wait().
    void dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params);

    // Block until all pending GPU work is done.
    void wait();

private:
    MetalPipelines() = default;
    ~MetalPipelines();

    bool   ready_ = false;

    // Opaque pointers to Metal objects (avoid ObjC in header).
    void*  device_        = nullptr;  // id<MTLDevice>
    void*  cmd_queue_     = nullptr;  // id<MTLCommandQueue>
    void*  library_       = nullptr;  // id<MTLLibrary>
    void*  pso_w_source_  = nullptr;  // id<MTLComputePipelineState>
    void*  pso_delta_qi_  = nullptr;  // id<MTLComputePipelineState>
    void*  pso_finalize_  = nullptr;  // id<MTLComputePipelineState>
    void*  pso_uwrhs_     = nullptr;  // id<MTLComputePipelineState>
    void*  pso_make_du_   = nullptr;  // id<MTLComputePipelineState>
    void*  pso_w_full_    = nullptr;  // id<MTLComputePipelineState>
    void*  cmd_buf_       = nullptr;  // last id<MTLCommandBuffer>
};
