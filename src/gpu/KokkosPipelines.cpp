// Stage-0 skeleton implementation of the Kokkos compute pipelines.
//
// initialize() returns false, which keeps gpu_ready_ == false in advance.cpp
// (advance.cpp:156) so MUSIC runs the pure CPU reference path.  Every method
// below is therefore defined only for linkage and is not reached at runtime
// until Stage 1 wires the real parallel_for / parallel_reduce kernels.  Bodies
// are kept safe-if-called regardless.

#include "KokkosPipelines.h"

KokkosPipelines& KokkosPipelines::instance() {
    static KokkosPipelines inst;
    return inst;
}

bool KokkosPipelines::initialize(const char* /*unused*/) {
    // Not yet functional (Stage 1 ports the 9 kernels).  Returning false makes
    // advance.cpp fall back to CPU -> the run reproduces the CPU reference, the
    // Stage-0 acceptance criterion.
    ready_ = false;
    return false;
}

// ── Kernel dispatches (Stage 1: parallel_for over MDRangePolicy<Rank<3>>) ─────
void KokkosPipelines::dispatch_w_source(GPUGrid&, const MUSICGridParams&) {}
void KokkosPipelines::dispatch_delta_qi(GPUGrid&, const MUSICGridParams&) {}
void KokkosPipelines::dispatch_first_rk_step_w_full(GPUGrid&,
                                                    const MUSICGridParams&) {}
void KokkosPipelines::dispatch_make_du(GPUGrid&, const MUSICGridParams&) {}
void KokkosPipelines::dispatch_uwrhs(GPUGrid&, const MUSICGridParams&) {}
void KokkosPipelines::dispatch_uprhs(GPUGrid&, const MUSICGridParams&) {}
void KokkosPipelines::dispatch_finalize_ideal(GPUGrid&, const MUSICGridParams&) {}

void KokkosPipelines::wait() {}
void KokkosPipelines::begin_batch() {}
void KokkosPipelines::end_batch() {}

void KokkosPipelines::reduce_max(GPUGrid&, double& eps_max, double& rhob_max) {
    // Safe defaults; never reached while ready_ == false.
    eps_max  = 0.0;
    rhob_max = 0.0;
}

bool KokkosPipelines::pack_evolution_ideal(GPUGrid&, const GPUPackParams&,
                                           float* /*host_out*/) {
    return false;   // caller uses the host output path
}

void KokkosPipelines::upload_snapshots_async(GPUGrid&) {}
