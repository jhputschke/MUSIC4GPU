// SoA (Struct-of-Arrays) grid layout for GPU-accelerated MUSIC.
//
// Each field is stored in a separate MTLBuffer (shared CPU/GPU memory on
// Apple Silicon).  Component-major ordering gives coalesced GPU reads:
//   field[comp * Ncells + cell]   where  cell = Nx*(Ny*ieta + iy) + ix
//
// Three snapshots are kept (matching arena_prev / arena_current / arena_future).

#pragma once

#include "../grid.h"   // SCGrid = GridT<Cell_small>
#include "gpu_types.h" // GPUEosParams, GPU_EOS_N

// Number of components per field
static const int GPU_WMUNU_COMPS = 14;
static const int GPU_U_COMPS     =  4;

// One snapshot of the grid in SoA layout.
// All float* pointers are backed by Metal shared buffers and are accessible
// from both the CPU and the GPU without any explicit copies.
struct GPUSnapshot {
    float* epsilon  = nullptr;   // [Ncells]
    float* rhob     = nullptr;   // [Ncells]
    float* u        = nullptr;   // [GPU_U_COMPS * Ncells]
    float* Wmunu    = nullptr;   // [GPU_WMUNU_COMPS * Ncells]
    float* pi_b     = nullptr;   // [Ncells]
};

// Manages three grid snapshots (prev / current / future) and an output
// buffer for dwmn (the viscous-source result per cell).
class GPUGrid {
public:
    GPUGrid() = default;
    ~GPUGrid() { release(); }

    // Allocate Metal shared buffers for three snapshots.
    // Returns false if Metal is not available.
    bool allocate(int Nx, int Ny, int Neta);

    // Free all Metal buffers.
    void release();

    bool ready() const { return allocated_; }

    int Nx()     const { return Nx_; }
    int Ny()     const { return Ny_; }
    int Neta()   const { return Neta_; }
    int Ncells() const { return Ncells_; }

    // Metal-internal: expose the buffer handle array so MetalPipelines.mm
    // can reverse-lookup an MTLBuffer from a float* pointer.
    void** buf_handle_ptr()  { return buf_handles_; }
    int    buf_handle_count() const { return n_handles_; }

    // Copy an SCGrid (AoS double) into a GPUSnapshot (SoA float).
    void copy_to_gpu(const SCGrid& src, GPUSnapshot& dst) const;

    // Copy a GPUSnapshot (SoA float) back into an SCGrid (AoS double).
    // Only Wmunu and pi_b are written back; epsilon/rhob/u stay on CPU.
    void copy_wmunu_to_cpu(const GPUSnapshot& src, SCGrid& dst) const;

    // Copy the primitive variables (epsilon, rhob, u) from a GPUSnapshot into
    // an SCGrid (AoS double).  Used after gpu_finalize_ideal to propagate the
    // GPU-reconstructed primitives into arena_future before the CPU viscous
    // pass reads them.
    void copy_primitives_to_cpu(const GPUSnapshot& src, SCGrid& dst) const;

    // The three snapshots (prev, current, future).
    GPUSnapshot snap_prev;
    GPUSnapshot snap_curr;
    GPUSnapshot snap_future;

    // Output buffer: dwmn[5 * Ncells] produced by gpu_make_w_source.
    float* dwmn = nullptr;

    // EOS tables (sampled at rhob=0 on a uniform grid; see GPUEosParams).
    // Uploaded once at init time via upload_eos().
    float*       eos_P    = nullptr;   // pressure table     [GPU_EOS_N]
    float*       eos_dPde = nullptr;   // dP/de table        [GPU_EOS_N]
    float*       eos_s    = nullptr;   // entropy table s(e) [GPU_EOS_N]
    float*       eos_T    = nullptr;   // temperature T(e)   [GPU_EOS_N]
    GPUEosParams eos_params = {};

    // Output buffer: qi_out[5 * Ncells] produced by gpu_make_delta_qi.
    float* qi_out = nullptr;

    // Output buffer: uwrhs_out[5 * Ncells] produced by gpu_make_uwrhs.
    // Layout: [out_idx * Ncells + cell] where out_idx 0..4 maps to the 5
    // shear-stress indices idx_1d = {4, 5, 6, 7, 8}.
    float* uwrhs_out = nullptr;

    // Output buffers produced by gpu_make_du (per-cell viscous geometry):
    //   theta_buf [Ncells]      — expansion rate θ
    //   a_buf     [4 * Ncells]  — a^μ = u^ν ∂_ν u^μ
    //   sigma_buf [10 * Ncells] — velocity shear σ^{μν}
    float* theta_buf = nullptr;
    float* a_buf     = nullptr;
    float* sigma_buf = nullptr;

    // Upload pre-sampled EOS tables (P, dP/de, s, T) to GPU shared memory.
    // P / dP/de are sampled at linearly-spaced e ∈ [e_min, e_max]; s and T
    // are sampled at log-spaced e ∈ [1e-6, e_max] (both are strongly
    // non-linear in e and linear-spacing wastes resolution at the dilute
    // end of the hydro evolution).
    // Must be called after allocate() and before the first dispatch_delta_qi.
    // n_pts must be <= GPU_EOS_N.
    bool upload_eos(const float* P_data, const float* dPde_data,
                    const float* s_data, const float* T_data,
                    int n_pts, float e_min, float e_max);

private:
    // Allocate one snapshot's worth of Metal shared buffers.
    bool alloc_snapshot(GPUSnapshot& s, int Ncells);

    int   Nx_      = 0;
    int   Ny_      = 0;
    int   Neta_    = 0;
    int   Ncells_  = 0;
    bool  allocated_ = false;

    // Opaque MTLBuffer handles kept alive via CF-bridged __bridge_retained.
    // Stored as void* to keep this header free of ObjC.
    // 3 snapshots × 5 fields = 15
    //   + dwmn + qi_out + uwrhs_out + theta_buf + a_buf + sigma_buf = 21
    //   + eos_P + eos_dPde = 23
    // Leave headroom for upcoming Phase-2 buffers (eos_T, eos_s, ...).
    void* buf_handles_[32];
    int   n_handles_ = 0;
};
