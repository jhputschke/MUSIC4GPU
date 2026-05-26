// Objective-C++ implementation of GPUGrid.
// Handles Metal buffer allocation and AoS<->SoA conversion.

#import <Metal/Metal.h>
#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "GPUGrid.h"
#include "../grid.h"   // SCGrid, Cell_small

// Singleton Metal device (shared with MetalPipelines).
extern id<MTLDevice> g_metal_device;

// ── helpers ──────────────────────────────────────────────────────────────────

// Flat cell index: cell = Nx*(Ny*ieta + iy) + ix
static inline int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

// Allocate a Metal shared buffer, store its handle, and return the CPU pointer.
static float* alloc_metal_buf(id<MTLDevice> device, size_t bytes,
                               void** handles, int& n_handles) {
    id<MTLBuffer> buf = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    if (!buf) return nullptr;
    // Bridge-retain: keeps ARC object alive until we CFRelease it manually.
    handles[n_handles++] = (__bridge_retained void*)buf;
    return static_cast<float*>([buf contents]);
}

// ── GPUGrid ──────────────────────────────────────────────────────────────────

bool GPUGrid::alloc_snapshot(GPUSnapshot& s, int Ncells) {
    auto dev = g_metal_device;
    auto nc  = static_cast<size_t>(Ncells);

    s.epsilon = alloc_metal_buf(dev, nc * sizeof(float),
                                buf_handles_, n_handles_);
    s.rhob    = alloc_metal_buf(dev, nc * sizeof(float),
                                buf_handles_, n_handles_);
    s.u       = alloc_metal_buf(dev, GPU_U_COMPS     * nc * sizeof(float),
                                buf_handles_, n_handles_);
    s.Wmunu   = alloc_metal_buf(dev, GPU_WMUNU_COMPS * nc * sizeof(float),
                                buf_handles_, n_handles_);
    s.pi_b    = alloc_metal_buf(dev, nc * sizeof(float),
                                buf_handles_, n_handles_);
    return s.epsilon && s.rhob && s.u && s.Wmunu && s.pi_b;
}

bool GPUGrid::allocate(int Nx, int Ny, int Neta) {
    if (!g_metal_device) return false;
    Nx_     = Nx;
    Ny_     = Ny;
    Neta_   = Neta;
    Ncells_ = Nx * Ny * Neta;
    n_handles_ = 0;

    bool ok = alloc_snapshot(snap_prev,   Ncells_)
           && alloc_snapshot(snap_curr,   Ncells_)
           && alloc_snapshot(snap_future, Ncells_);

    dwmn = alloc_metal_buf(g_metal_device,
                           5 * static_cast<size_t>(Ncells_) * sizeof(float),
                           buf_handles_, n_handles_);
    ok = ok && (dwmn != nullptr);

    qi_out = alloc_metal_buf(g_metal_device,
                             5 * static_cast<size_t>(Ncells_) * sizeof(float),
                             buf_handles_, n_handles_);
    ok = ok && (qi_out != nullptr);

    uwrhs_out = alloc_metal_buf(g_metal_device,
                                5 * static_cast<size_t>(Ncells_) * sizeof(float),
                                buf_handles_, n_handles_);
    ok = ok && (uwrhs_out != nullptr);

    uprhs_out = alloc_metal_buf(g_metal_device,
                                static_cast<size_t>(Ncells_) * sizeof(float),
                                buf_handles_, n_handles_);
    ok = ok && (uprhs_out != nullptr);

    qi_source_buf = alloc_metal_buf(g_metal_device,
                                    5 * static_cast<size_t>(Ncells_)
                                        * sizeof(float),
                                    buf_handles_, n_handles_);
    ok = ok && (qi_source_buf != nullptr);

    theta_buf = alloc_metal_buf(g_metal_device,
                                static_cast<size_t>(Ncells_) * sizeof(float),
                                buf_handles_, n_handles_);
    a_buf     = alloc_metal_buf(g_metal_device,
                                4 * static_cast<size_t>(Ncells_) * sizeof(float),
                                buf_handles_, n_handles_);
    sigma_buf = alloc_metal_buf(g_metal_device,
                                10 * static_cast<size_t>(Ncells_) * sizeof(float),
                                buf_handles_, n_handles_);
    ok = ok && theta_buf && a_buf && sigma_buf;

    allocated_ = ok;
    return ok;
}

bool GPUGrid::upload_eos(const float* P_data, const float* dPde_data,
                         const float* s_data, const float* T_data,
                         int n_pts, float e_min, float e_max) {
    if (!g_metal_device || !allocated_) return false;

    auto nc = static_cast<size_t>(n_pts);
    eos_P    = alloc_metal_buf(g_metal_device, nc * sizeof(float),
                                buf_handles_, n_handles_);
    eos_dPde = alloc_metal_buf(g_metal_device, nc * sizeof(float),
                                buf_handles_, n_handles_);
    eos_s    = alloc_metal_buf(g_metal_device, nc * sizeof(float),
                                buf_handles_, n_handles_);
    eos_T    = alloc_metal_buf(g_metal_device, nc * sizeof(float),
                                buf_handles_, n_handles_);
    if (!eos_P || !eos_dPde || !eos_s || !eos_T) return false;

    std::memcpy(eos_P,    P_data,    nc * sizeof(float));
    std::memcpy(eos_dPde, dPde_data, nc * sizeof(float));
    std::memcpy(eos_s,    s_data,    nc * sizeof(float));
    std::memcpy(eos_T,    T_data,    nc * sizeof(float));

    eos_params.e_min   = e_min;
    eos_params.e_max   = e_max;
    eos_params.n_pts   = n_pts;
    eos_params.delta_e = (n_pts > 1)
                         ? (e_max - e_min) / static_cast<float>(n_pts - 1)
                         : 1.f;

    // Log-spaced entropy table parameters.  The host sampled s_data at
    // log-spaced e values in [s_log_e_floor, e_max]; here we record the
    // matching grid so the kernel can compute the bin index.
    constexpr float s_log_e_floor = 1.e-6f;   // 1/fm^4 — well below physical range
    eos_params.log_e_min   = std::log(s_log_e_floor);
    eos_params.log_e_max   = std::log(std::max(e_max, s_log_e_floor * 1.01f));
    eos_params.log_delta_e = (n_pts > 1)
                             ? (eos_params.log_e_max - eos_params.log_e_min)
                                 / static_cast<float>(n_pts - 1)
                             : 1.f;
    return true;
}

void GPUGrid::release() {
    for (int i = 0; i < n_handles_; ++i) {
        if (buf_handles_[i]) {
            CFRelease(buf_handles_[i]);
            buf_handles_[i] = nullptr;
        }
    }
    n_handles_ = 0;
    allocated_ = false;
    snap_prev = snap_curr = snap_future = GPUSnapshot{};
    dwmn      = nullptr;
    qi_out    = nullptr;
    uwrhs_out = nullptr;
    uprhs_out = nullptr;
    qi_source_buf = nullptr;
    theta_buf = nullptr;
    a_buf     = nullptr;
    sigma_buf = nullptr;
    eos_P     = nullptr;
    eos_dPde  = nullptr;
    eos_s     = nullptr;
    eos_T     = nullptr;
    eos_params = {};
}

// ── AoS → SoA (double → float) ───────────────────────────────────────────────

void GPUGrid::copy_to_gpu(const SCGrid& src, GPUSnapshot& dst) const {
    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        const auto& cell = src(ix, iy, ieta);

        dst.epsilon[c]             = static_cast<float>(cell.epsilon);
        dst.rhob   [c]             = static_cast<float>(cell.rhob);
        dst.pi_b   [c]             = static_cast<float>(cell.pi_b);

        for (int m = 0; m < GPU_U_COMPS; ++m)
            dst.u[m * Ncells_ + c] = static_cast<float>(cell.u[m]);

        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            dst.Wmunu[m * Ncells_ + c] = static_cast<float>(cell.Wmunu[m]);
    }
}

// ── SoA → AoS (float → double), Wmunu + pi_b only ───────────────────────────

void GPUGrid::copy_wmunu_to_cpu(const GPUSnapshot& src, SCGrid& dst) const {
    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        auto& cell = dst(ix, iy, ieta);

        cell.pi_b = static_cast<double>(src.pi_b[c]);
        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            cell.Wmunu[m] = static_cast<double>(src.Wmunu[m * Ncells_ + c]);
    }
}

void GPUGrid::rotate_snapshots() {
    // GPUSnapshot is just a bundle of float* aliases into permanent Metal
    // buffers — rotating the struct values renames the buffers' roles
    // without touching any GPU memory.  Matches CPU arena rotation in
    // evolve.cpp:310-317 after rk_flag = 0.
    GPUSnapshot temp = snap_prev;
    snap_prev   = snap_curr;
    snap_curr   = snap_future;
    snap_future = temp;
}

void GPUGrid::copy_primitives_to_cpu(const GPUSnapshot& src, SCGrid& dst) const {
    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        auto& cell = dst(ix, iy, ieta);

        cell.epsilon = static_cast<double>(src.epsilon[c]);
        cell.rhob    = static_cast<double>(src.rhob[c]);
        for (int m = 0; m < GPU_U_COMPS; ++m)
            cell.u[m] = static_cast<double>(src.u[m * Ncells_ + c]);
    }
}
