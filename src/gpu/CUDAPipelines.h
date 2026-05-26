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

    // Phase 4 (dual-stream): prefetch the freshly-packed snap_curr / snap_prev
    // SoA buffers to the device on a dedicated copy stream, then gate the
    // compute stream on completion via an event.  On a discrete GPU this moves
    // the host→device transfer onto its own stream so it can overlap prior
    // compute; on a coherent unified-memory part (GB10) it is a cheap hint.
    // Safe no-op for the Metal back-end (this method only exists here and is
    // called under a USE_CUDA guard in advance.cpp).
    void upload_snapshots_async(GPUGrid& gpu);

private:
    CUDAPipelines() = default;
    ~CUDAPipelines();

    bool  ready_     = false;
    int   device_id_ = 0;
    // cudaStream_t kept as void* so this header stays free of CUDA headers
    // and can be included by the host C++ compiler (advance.cpp).
    void* compute_stream_ = nullptr;
    // Phase 4 dual-stream: a dedicated copy stream + completion event used to
    // move/prefetch SoA snapshot data independently of the compute stream.
    void* copy_stream_ = nullptr;
    void* copy_event_  = nullptr;
    // True when the GPU can access host/managed memory coherently (integrated
    // or NVLink-C2C parts such as GB10).  On such platforms there is no
    // discrete transfer to overlap, so the dual-stream prefetch+gate is skipped
    // (it would only add migration latency).  Kept active on discrete GPUs.
    bool  coherent_memory_ = false;
    // Occupancy-tuned upper bound on threads per block (Phase 2).  The 3-D
    // block is factored from this at dispatch time, adapting to Neta so 2-D
    // (Neta==1) grids don't waste the eta thread dimension.
    int   max_block_threads_ = 256;
};
