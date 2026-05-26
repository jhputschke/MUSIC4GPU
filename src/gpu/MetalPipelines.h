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
    void*  cmd_buf_       = nullptr;  // last id<MTLCommandBuffer>
};
