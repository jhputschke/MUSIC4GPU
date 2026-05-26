// CUDA implementation of GPUGrid, with two memory back-ends chosen at runtime
// from the device's coherence capability (set by CUDAPipelines::initialize and
// shared via g_cuda_coherent):
//
//   Coherent host memory (integrated / NVLink-C2C, e.g. GB10):
//     snapshot buffers are cudaMallocManaged and the CPU packs AoS->SoA
//     directly into them; kernels read in place — zero copy.
//
//   Discrete GPU (A100/RTX/H100):
//     snapshot buffers are device-resident cudaMalloc; the CPU packs into
//     pinned host staging buffers and the data is moved with explicit
//     cudaMemcpyAsync (H2D in CUDAPipelines::upload_snapshots_async, D2H here
//     in the copy-back).  This avoids per-fault managed-memory migration over
//     PCIe.
//
// The non-snapshot scratch/output buffers (dwmn, qi_out, uwrhs_out, uprhs_out,
// qi_source_buf, theta/a/sigma, EOS tables) stay cudaMallocManaged in both
// modes: they are device-resident in the fully-GPU path (never host-touched, so
// no migration) but remain host-accessible for the partial-GPU fallback
// configurations.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "GPUGrid.h"
#include "../grid.h"   // SCGrid, Cell_small

// Set by CUDAPipelines::initialize() before GPUGrid::allocate() runs.
extern int  g_cuda_device_id;
extern bool g_cuda_coherent;

// ── helpers ──────────────────────────────────────────────────────────────────

static inline int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

static inline void cuda_prefetch(void* ptr, size_t bytes) {
    if (!ptr) return;
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = g_cuda_device_id;
    cudaMemPrefetchAsync(ptr, bytes, loc, 0, (cudaStream_t)0);
}

// Managed (unified) buffer, recorded in handles for cudaFree, prefetched to GPU.
static float* alloc_managed_buf(size_t bytes, void** handles, int& n_handles) {
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

// Device-resident buffer (discrete path), recorded in handles for cudaFree.
static float* alloc_device_buf(size_t bytes, void** handles, int& n_handles) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess || !ptr) {
        fprintf(stderr, "[MUSIC-GPU] cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return nullptr;
    }
    handles[n_handles++] = ptr;
    return static_cast<float*>(ptr);
}

// Pinned host staging buffer (discrete path).  Freed in release() via the
// snapshot *_stage fields (cudaFreeHost), not the cudaFree handle list.
static float* alloc_pinned_buf(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || !ptr) {
        fprintf(stderr, "[MUSIC-GPU] cudaHostAlloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return nullptr;
    }
    return static_cast<float*>(ptr);
}

// ── GPUGrid ──────────────────────────────────────────────────────────────────

bool GPUGrid::alloc_snapshot(GPUSnapshot& s, int Ncells) {
    const size_t nc  = static_cast<size_t>(Ncells);
    const size_t b1  =                  nc * sizeof(float);
    const size_t b4  = GPU_U_COMPS     * nc * sizeof(float);
    const size_t b14 = GPU_WMUNU_COMPS * nc * sizeof(float);

    if (g_cuda_coherent) {
        s.epsilon = alloc_managed_buf(b1,  buf_handles_, n_handles_);
        s.rhob    = alloc_managed_buf(b1,  buf_handles_, n_handles_);
        s.u       = alloc_managed_buf(b4,  buf_handles_, n_handles_);
        s.Wmunu   = alloc_managed_buf(b14, buf_handles_, n_handles_);
        s.pi_b    = alloc_managed_buf(b1,  buf_handles_, n_handles_);
        return s.epsilon && s.rhob && s.u && s.Wmunu && s.pi_b;
    }

    // Discrete: device buffers + pinned host staging.
    s.epsilon = alloc_device_buf(b1,  buf_handles_, n_handles_);
    s.rhob    = alloc_device_buf(b1,  buf_handles_, n_handles_);
    s.u       = alloc_device_buf(b4,  buf_handles_, n_handles_);
    s.Wmunu   = alloc_device_buf(b14, buf_handles_, n_handles_);
    s.pi_b    = alloc_device_buf(b1,  buf_handles_, n_handles_);
    s.epsilon_stage = alloc_pinned_buf(b1);
    s.rhob_stage    = alloc_pinned_buf(b1);
    s.u_stage       = alloc_pinned_buf(b4);
    s.Wmunu_stage   = alloc_pinned_buf(b14);
    s.pi_b_stage    = alloc_pinned_buf(b1);
    return s.epsilon && s.rhob && s.u && s.Wmunu && s.pi_b
        && s.epsilon_stage && s.rhob_stage && s.u_stage
        && s.Wmunu_stage && s.pi_b_stage;
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

    // Scratch / output buffers stay managed in both modes (see file header).
    dwmn      = alloc_managed_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    qi_out    = alloc_managed_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    uwrhs_out = alloc_managed_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    uprhs_out = alloc_managed_buf(    nc * sizeof(float), buf_handles_, n_handles_);
    qi_source_buf = alloc_managed_buf(5 * nc * sizeof(float), buf_handles_, n_handles_);
    theta_buf = alloc_managed_buf(     nc * sizeof(float), buf_handles_, n_handles_);
    a_buf     = alloc_managed_buf( 4 * nc * sizeof(float), buf_handles_, n_handles_);
    sigma_buf = alloc_managed_buf(10 * nc * sizeof(float), buf_handles_, n_handles_);

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
    // EOS tables are read-mostly and host-written once; managed in both modes.
    eos_P    = alloc_managed_buf(nc * sizeof(float), buf_handles_, n_handles_);
    eos_dPde = alloc_managed_buf(nc * sizeof(float), buf_handles_, n_handles_);
    eos_s    = alloc_managed_buf(nc * sizeof(float), buf_handles_, n_handles_);
    eos_T    = alloc_managed_buf(nc * sizeof(float), buf_handles_, n_handles_);
    if (!eos_P || !eos_dPde || !eos_s || !eos_T) return false;

    std::memcpy(eos_P,    P_data,    nc * sizeof(float));
    std::memcpy(eos_dPde, dPde_data, nc * sizeof(float));
    std::memcpy(eos_s,    s_data,    nc * sizeof(float));
    std::memcpy(eos_T,    T_data,    nc * sizeof(float));

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
    // Pinned host staging (discrete path) frees with cudaFreeHost — handle it
    // before the cudaFree handle sweep and the snapshot reset.  All staging
    // pointers live across the three snapshot structs (rotation only permutes
    // them), so freeing all three covers every staging buffer.
    GPUSnapshot* snaps[3] = {&snap_prev, &snap_curr, &snap_future};
    for (GPUSnapshot* s : snaps) {
        if (s->epsilon_stage) cudaFreeHost(s->epsilon_stage);
        if (s->rhob_stage)    cudaFreeHost(s->rhob_stage);
        if (s->u_stage)       cudaFreeHost(s->u_stage);
        if (s->Wmunu_stage)   cudaFreeHost(s->Wmunu_stage);
        if (s->pi_b_stage)    cudaFreeHost(s->pi_b_stage);
    }

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
//
// Packs into the host-accessible target: the managed buffer (coherent) or the
// pinned staging buffer (discrete, then DMA'd H2D by upload_snapshots_async).

void GPUGrid::copy_to_gpu(const SCGrid& src, GPUSnapshot& dst) const {
    const bool disc = !g_cuda_coherent;
    float* d_eps  = disc ? dst.epsilon_stage : dst.epsilon;
    float* d_rhob = disc ? dst.rhob_stage    : dst.rhob;
    float* d_u    = disc ? dst.u_stage       : dst.u;
    float* d_W    = disc ? dst.Wmunu_stage   : dst.Wmunu;
    float* d_pib  = disc ? dst.pi_b_stage    : dst.pi_b;

    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        const auto& cell = src(ix, iy, ieta);

        d_eps [c] = static_cast<float>(cell.epsilon);
        d_rhob[c] = static_cast<float>(cell.rhob);
        d_pib [c] = static_cast<float>(cell.pi_b);

        for (int m = 0; m < GPU_U_COMPS; ++m)
            d_u[m * Ncells_ + c] = static_cast<float>(cell.u[m]);

        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            d_W[m * Ncells_ + c] = static_cast<float>(cell.Wmunu[m]);
    }
}

// ── SoA → AoS (float → double), Wmunu + pi_b only ───────────────────────────

void GPUGrid::copy_wmunu_to_cpu(const GPUSnapshot& src, SCGrid& dst) const {
    const float* s_W   = src.Wmunu;
    const float* s_pib = src.pi_b;
    if (!g_cuda_coherent) {
        // Bring the GPU-written results back to the pinned staging first.
        // The compute stream was already synchronized by CUDAPipelines::wait().
        cudaMemcpy(src.Wmunu_stage, src.Wmunu,
                   GPU_WMUNU_COMPS * static_cast<size_t>(Ncells_) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(src.pi_b_stage, src.pi_b,
                   static_cast<size_t>(Ncells_) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        s_W   = src.Wmunu_stage;
        s_pib = src.pi_b_stage;
    }

    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        auto& cell = dst(ix, iy, ieta);

        cell.pi_b = static_cast<double>(s_pib[c]);
        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            cell.Wmunu[m] = static_cast<double>(s_W[m * Ncells_ + c]);
    }
}

void GPUGrid::rotate_snapshots() {
    // GPUSnapshot is a bundle of pointer aliases (device + staging); rotating
    // the structs renames the buffers' roles without moving any data.
    GPUSnapshot temp = snap_prev;
    snap_prev   = snap_curr;
    snap_curr   = snap_future;
    snap_future = temp;
}

void GPUGrid::copy_primitives_to_cpu(const GPUSnapshot& src, SCGrid& dst) const {
    const float* s_eps  = src.epsilon;
    const float* s_rhob = src.rhob;
    const float* s_u    = src.u;
    if (!g_cuda_coherent) {
        const size_t nc = static_cast<size_t>(Ncells_);
        cudaMemcpy(src.epsilon_stage, src.epsilon, nc * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(src.rhob_stage,    src.rhob,    nc * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(src.u_stage,       src.u,   GPU_U_COMPS * nc * sizeof(float),
                   cudaMemcpyDeviceToHost);
        s_eps  = src.epsilon_stage;
        s_rhob = src.rhob_stage;
        s_u    = src.u_stage;
    }

    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        auto& cell = dst(ix, iy, ieta);

        cell.epsilon = static_cast<double>(s_eps[c]);
        cell.rhob    = static_cast<double>(s_rhob[c]);
        for (int m = 0; m < GPU_U_COMPS; ++m)
            cell.u[m] = static_cast<double>(s_u[m * Ncells_ + c]);
    }
}

void GPUGrid::refresh_u_curr_stage() {
    // Discrete path only: bring snap_curr.u back to its pinned staging so the
    // host-side hydro-source pre-pass can read it after a snapshot rotation
    // (which left the staging stale).  Safe to call synchronously — the caller
    // invokes it only on the rotation path, where the producing compute stream
    // was already synchronized by the previous substep's wait().
    if (g_cuda_coherent || !snap_curr.u || !snap_curr.u_stage) return;
    cudaMemcpy(snap_curr.u_stage, snap_curr.u,
               GPU_U_COMPS * static_cast<size_t>(Ncells_) * sizeof(float),
               cudaMemcpyDeviceToHost);
}
