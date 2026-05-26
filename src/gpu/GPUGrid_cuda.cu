// CUDA implementation of GPUGrid.
// Replaces the Metal GPUGrid.mm: buffers are allocated with cudaMallocManaged
// (unified memory, page-migrated on demand) and prefetched to the device to
// avoid first-touch page-fault jitter.  The AoS<->SoA converters are pure
// C++/OpenMP and are shared verbatim with the Metal back-end.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "GPUGrid.h"
#include "../grid.h"   // SCGrid, Cell_small

// The compute stream and selected device id are owned by CUDAPipelines; we
// only need the device id here for prefetch hints.
extern int g_cuda_device_id;

// ── helpers ──────────────────────────────────────────────────────────────────

static inline int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

// Prefetch a managed range to the current device.  CUDA 13 replaced the
// (ptr, count, int device, stream) overload with a cudaMemLocation-based one;
// use that.  Best-effort: ignore failures (e.g. on platforms that report no
// prefetch support — page migration still happens on first touch).
static inline void cuda_prefetch(void* ptr, size_t bytes) {
    if (!ptr) return;
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = g_cuda_device_id;
    cudaMemPrefetchAsync(ptr, bytes, loc, 0, (cudaStream_t)0);
}

// Allocate a unified-memory buffer, record its handle for release(), prefetch
// to the device, and return the (host- and device-accessible) pointer.
static float* alloc_cuda_buf(size_t bytes, void** handles, int& n_handles) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess || !ptr) {
        fprintf(stderr, "[MUSIC-GPU] cudaMallocManaged(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return nullptr;
    }
    handles[n_handles++] = ptr;
    cuda_prefetch(ptr, bytes);
    return static_cast<float*>(ptr);
}

// ── GPUGrid ──────────────────────────────────────────────────────────────────

bool GPUGrid::alloc_snapshot(GPUSnapshot& s, int Ncells) {
    auto nc = static_cast<size_t>(Ncells);
    s.epsilon = alloc_cuda_buf(nc * sizeof(float),                  buf_handles_, n_handles_);
    s.rhob    = alloc_cuda_buf(nc * sizeof(float),                  buf_handles_, n_handles_);
    s.u       = alloc_cuda_buf(GPU_U_COMPS     * nc * sizeof(float), buf_handles_, n_handles_);
    s.Wmunu   = alloc_cuda_buf(GPU_WMUNU_COMPS * nc * sizeof(float), buf_handles_, n_handles_);
    s.pi_b    = alloc_cuda_buf(nc * sizeof(float),                  buf_handles_, n_handles_);
    return s.epsilon && s.rhob && s.u && s.Wmunu && s.pi_b;
}

bool GPUGrid::allocate(int Nx, int Ny, int Neta) {
    Nx_     = Nx;
    Ny_     = Ny;
    Neta_   = Neta;
    Ncells_ = Nx * Ny * Neta;
    n_handles_ = 0;

    const size_t nc = static_cast<size_t>(Ncells_);

    bool ok = alloc_snapshot(snap_prev,   Ncells_)
           && alloc_snapshot(snap_curr,   Ncells_)
           && alloc_snapshot(snap_future, Ncells_);

    dwmn      = alloc_cuda_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    qi_out    = alloc_cuda_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    uwrhs_out = alloc_cuda_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    uprhs_out = alloc_cuda_buf(    nc * sizeof(float), buf_handles_, n_handles_);
    qi_source_buf = alloc_cuda_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    theta_buf = alloc_cuda_buf(     nc * sizeof(float), buf_handles_, n_handles_);
    a_buf     = alloc_cuda_buf( 4 * nc * sizeof(float), buf_handles_, n_handles_);
    sigma_buf = alloc_cuda_buf(10 * nc * sizeof(float), buf_handles_, n_handles_);

    ok = ok && dwmn && qi_out && uwrhs_out && uprhs_out && qi_source_buf
            && theta_buf && a_buf && sigma_buf;

    allocated_ = ok;
    return ok;
}

bool GPUGrid::upload_eos(const float* P_data, const float* dPde_data,
                         const float* s_data, const float* T_data,
                         int n_pts, float e_min, float e_max) {
    if (!allocated_) return false;

    auto nc = static_cast<size_t>(n_pts);
    eos_P    = alloc_cuda_buf(nc * sizeof(float), buf_handles_, n_handles_);
    eos_dPde = alloc_cuda_buf(nc * sizeof(float), buf_handles_, n_handles_);
    eos_s    = alloc_cuda_buf(nc * sizeof(float), buf_handles_, n_handles_);
    eos_T    = alloc_cuda_buf(nc * sizeof(float), buf_handles_, n_handles_);
    if (!eos_P || !eos_dPde || !eos_s || !eos_T) return false;

    std::memcpy(eos_P,    P_data,    nc * sizeof(float));
    std::memcpy(eos_dPde, dPde_data, nc * sizeof(float));
    std::memcpy(eos_s,    s_data,    nc * sizeof(float));
    std::memcpy(eos_T,    T_data,    nc * sizeof(float));

    // Prefetch the freshly-written tables to the device.
    cuda_prefetch(eos_P,    nc * sizeof(float));
    cuda_prefetch(eos_dPde, nc * sizeof(float));
    cuda_prefetch(eos_s,    nc * sizeof(float));
    cuda_prefetch(eos_T,    nc * sizeof(float));

    eos_params.e_min   = e_min;
    eos_params.e_max   = e_max;
    eos_params.n_pts   = n_pts;
    eos_params.delta_e = (n_pts > 1)
                         ? (e_max - e_min) / static_cast<float>(n_pts - 1)
                         : 1.f;

    constexpr float s_log_e_floor = 1.e-6f;
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
            cudaFree(buf_handles_[i]);
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
    // GPUSnapshot is just a bundle of float* aliases into permanent unified
    // buffers — rotating the struct values renames the buffers' roles without
    // touching any GPU memory.
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
