// SoA (Struct-of-Arrays) grid layout for GPU-accelerated MUSIC.
//
// Each field is stored in a separate MTLBuffer (shared CPU/GPU memory on
// Apple Silicon).  Component-major ordering gives coalesced GPU reads:
//   field[comp * Ncells + cell]   where  cell = Nx*(Ny*ieta + iy) + ix
//
// Three snapshots are kept (matching arena_prev / arena_current / arena_future).

#pragma once

#include "../grid.h"   // SCGrid = GridT<Cell_small>

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

    // The three snapshots (prev, current, future).
    GPUSnapshot snap_prev;
    GPUSnapshot snap_curr;
    GPUSnapshot snap_future;

    // Output buffer: dwmn[5 * Ncells] produced by gpu_make_w_source.
    float* dwmn = nullptr;

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
    void* buf_handles_[3 * 5 + 1];  // 3 snapshots × 5 fields + dwmn
    int   n_handles_ = 0;
};
