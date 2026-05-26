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

    // EOS table (sampled at rhob=0 on a uniform grid; see GPUEosParams).
    // Uploaded once at init time via upload_eos().
    float*       eos_P    = nullptr;   // pressure table    [GPU_EOS_N]
    float*       eos_dPde = nullptr;   // dP/de table       [GPU_EOS_N]
    GPUEosParams eos_params = {};

    // Output buffer: qi_out[5 * Ncells] produced by gpu_make_delta_qi.
    float* qi_out = nullptr;

    // Output buffer: uwrhs_out[5 * Ncells] produced by gpu_make_uwrhs.
    // Layout: [out_idx * Ncells + cell] where out_idx 0..4 maps to the 5
    // shear-stress indices idx_1d = {4, 5, 6, 7, 8}.
    float* uwrhs_out = nullptr;

    // Upload a pre-sampled EOS table (both P and dP/de) to GPU shared memory.
    // Must be called after allocate() and before the first dispatch_delta_qi.
    // n_pts must be <= GPU_EOS_N.
    bool upload_eos(const float* P_data, const float* dPde_data,
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
    // 3 snapshots × 5 fields + dwmn + eos_P + eos_dPde + qi_out = 19 max
    void* buf_handles_[24];
    int   n_handles_ = 0;
};
