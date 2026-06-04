// C++ interface to the Kokkos compute pipelines (NVIDIA / AMD / Intel GPU +
// multicore CPU).  Implementation lives in KokkosPipelines.cpp.
//
// This mirrors CUDAPipelines / MetalPipelines exactly so the host dispatch
// logic in advance.cpp is back-end agnostic (see the `GPUPipelines` alias in
// advance.h).  The header is kept free of Kokkos headers (PIMPL boundary,
// PlanKokkosPort.md D7) so advance.cpp keeps compiling as a plain host TU.
//
// Stage 0 status: this is a skeleton.  initialize() returns false, so
// advance.cpp keeps gpu_ready_ == false and runs the CPU reference path; the
// dispatch_* / reduce / pack methods are defined for linkage but are not
// reached at runtime until Stage 1 ports the kernels.
//
// Usage pattern (identical to CUDAPipelines):
//   KokkosPipelines& kp = KokkosPipelines::instance();
//   if (!kp.initialize()) { /* fall back to CPU */ }
//   kp.begin_batch();
//   kp.dispatch_w_source(gpu_grid, params);
//   ...
//   kp.end_batch();
//   kp.wait();

#pragma once
#include "GPUGrid.h"
#include "gpu_types.h"

class KokkosPipelines {
public:
    static KokkosPipelines& instance();

    // Bring up the Kokkos execution space for MUSIC.  The argument is accepted
    // (and ignored) to match the Metal/CUDA signature used at the shared call
    // site.  Returns false in the Stage-0 skeleton -> CPU fallback.
    bool initialize(const char* unused = nullptr);

    bool ready() const { return ready_; }

    // The seven dispatches mirror CUDAPipelines::dispatch_*.  Each will enqueue
    // a parallel_for over MDRangePolicy<Rank<3>> in Stage 1.
    void dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                       const MUSICGridParams& params);
    void dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params);

    // Block until pending Kokkos work is done (Kokkos::fence in Stage 1).
    void wait();

    // Batch markers exist only to match the Metal API; no-ops here.
    void begin_batch();
    void end_batch();

    // GPU max-reduction over snap_curr (parallel_reduce with Kokkos::Max in
    // Stage 1).  Writes max(epsilon) / max(rhob) into the out-params.
    void reduce_max(GPUGrid& gpu, double& eps_max, double& rhob_max);

    // Pack the ideal-hydro evolution output on device into host_out (8 floats
    // per down-sampled cell).  Returns false in the skeleton so the caller uses
    // the host output path.
    bool pack_evolution_ideal(GPUGrid& gpu, const GPUPackParams& pp,
                              float* host_out);

    // Async H2D prefetch of the freshly-packed snapshots (partition_space copy
    // instance in Stage 1).  No-op in the skeleton; called under a GPU-residency
    // guard in advance.cpp.
    void upload_snapshots_async(GPUGrid& gpu);

private:
    KokkosPipelines()  = default;
    ~KokkosPipelines() = default;

    bool ready_ = false;
};
