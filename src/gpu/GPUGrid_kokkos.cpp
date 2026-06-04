// Kokkos backing for the GPUGrid SoA storage class (declared in GPUGrid.h,
// shared with the CUDA/Metal backends).  This is the Kokkos counterpart of
// GPUGrid_cuda.cu / GPUGrid.mm and is compiled only when USE_KOKKOS is set.
//
// Stage-0 status: a skeleton.  KokkosPipelines::initialize() returns false, so
// advance.cpp keeps gpu_ready_ == false and never reaches the GPU path; the
// methods here exist for linkage and run the safe fall-through.  allocate()
// returns false (the second safety net behind initialize()), so the snapshots
// stay empty and the destructor's release() is a guarded no-op.  Stage 1
// replaces these bodies with real Kokkos::View allocation, mirror/deep_copy
// staging, and pointer-rotation residency.

#include "GPUGrid.h"
#include <utility>   // std::swap

// ── Allocation / teardown ────────────────────────────────────────────────────
bool GPUGrid::alloc_snapshot(GPUSnapshot& /*s*/, int /*Ncells*/) {
    // Stage 1: allocate per-field Kokkos Views (+ pinned host mirrors on the
    // discrete path).  Skeleton allocates nothing.
    return false;
}

bool GPUGrid::allocate(int /*Nx*/, int /*Ny*/, int /*Neta*/) {
    // Returning false keeps the GPU path disabled in advance.cpp -> CPU
    // reference.  (initialize() already returns false; this is a second net.)
    return false;
}

void GPUGrid::release() {
    // Nothing is allocated in the skeleton; guard so the destructor is safe on
    // a default-constructed grid.  Stage 1 frees the Views here.
    if (!allocated_) return;
    allocated_ = false;
}

// ── Host <-> device transfers (Stage 1: deep_copy into/out of Views) ─────────
void GPUGrid::copy_to_gpu(const SCGrid& /*src*/, GPUSnapshot& /*dst*/) const {}
void GPUGrid::copy_to_gpu(const Fields& /*src*/, GPUSnapshot& /*dst*/) const {}

void GPUGrid::copy_wmunu_to_cpu(const GPUSnapshot& /*src*/, SCGrid& /*dst*/) const {}
void GPUGrid::copy_wmunu_to_cpu(const GPUSnapshot& /*src*/, Fields& /*dst*/) const {}

void GPUGrid::copy_primitives_to_cpu(const GPUSnapshot& /*src*/, SCGrid& /*dst*/) const {}
void GPUGrid::copy_primitives_to_cpu(const GPUSnapshot& /*src*/, Fields& /*dst*/) const {}

void GPUGrid::refresh_u_curr_stage() {}

// ── Snapshot pointer bookkeeping (backend-agnostic; matches GPUGrid.h spec) ──
void GPUGrid::rotate_snapshots() {
    // snap_prev <- snap_curr <- snap_future <- (old snap_prev as scratch).
    GPUSnapshot scratch = snap_prev;
    snap_prev   = snap_curr;
    snap_curr   = snap_future;
    snap_future = scratch;
}

void GPUGrid::swap_curr_future() {
    std::swap(snap_curr, snap_future);
}

// ── EOS upload (Stage 1: deep_copy the four tables into RandomAccess Views) ──
bool GPUGrid::upload_eos(const float* /*P_data*/,  const float* /*dPde_data*/,
                         const float* /*s_data*/,  const float* /*T_data*/,
                         int /*n_pts*/, float /*e_min*/, float /*e_max*/) {
    return false;
}
