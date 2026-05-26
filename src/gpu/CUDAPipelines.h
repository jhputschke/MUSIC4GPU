// C++ interface to CUDA compute pipelines (NVIDIA GPUs).
// Implementation lives in CUDAPipelines.cu (compiled by nvcc).
//
// This mirrors MetalPipelines exactly so the host dispatch logic in
// advance.cpp can be shared between the Metal and CUDA back-ends (see the
// `GPUPipelines` alias in advance.h).
//
// Usage pattern:
//   CUDAPipelines& cp = CUDAPipelines::instance();
//   if (!cp.initialize()) { /* fall back to CPU */ }
//   cp.begin_batch();
//   cp.dispatch_w_source(gpu_grid, params);
//   ...
//   cp.end_batch();
//   cp.wait();   // block until the GPU finishes

#pragma once
#include "GPUGrid.h"
#include "gpu_types.h"

class CUDAPipelines {
public:
    static CUDAPipelines& instance();

    // Select device 0 and create the compute stream.  Kernels are compiled
    // directly into the binary by nvcc, so there is no library to load.
    // The argument is accepted (and ignored) to match the MetalPipelines
    // signature used at the shared call site.
    bool initialize(const char* unused = nullptr);

    bool ready() const { return ready_; }

    // All seven dispatches mirror MetalPipelines::dispatch_*.  Each computes a
    // dim3 launch grid and enqueues its kernel on the compute stream.  They are
    // non-blocking; call wait() to synchronize.
    void dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                       const MUSICGridParams& params);
    void dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params);

    // Block until all pending GPU work on the compute stream is done.
    void wait();

    // Batch mode is a no-op for CUDA: a single stream already serializes the
    // kernel launches, so begin/end_batch only exist to match the Metal API.
    void begin_batch();
    void end_batch();

private:
    CUDAPipelines() = default;
    ~CUDAPipelines();

    bool  ready_     = false;
    int   device_id_ = 0;
    // cudaStream_t kept as void* so this header stays free of CUDA headers
    // and can be included by the host C++ compiler (advance.cpp).
    void* compute_stream_ = nullptr;
    // Occupancy-tuned upper bound on threads per block (Phase 2).  The 3-D
    // block is factored from this at dispatch time, adapting to Neta so 2-D
    // (Neta==1) grids don't waste the eta thread dimension.
    int   max_block_threads_ = 256;
};
