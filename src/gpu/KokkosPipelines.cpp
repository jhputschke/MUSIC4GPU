// Kokkos implementation of KokkosPipelines.
//
// Each dispatch enqueues a parallel_for over MDRangePolicy<Rank<3>> on the
// default execution-space instance; like the CUDA single compute stream, all
// launches on that instance serialise in issue order, so the pipeline's
// producer->consumer dependencies (theta/a/sigma -> w_full, dwmn -> finalize,
// ...) are respected without explicit fences between kernels.  wait() is a
// single Kokkos::fence().  Mirrors CUDAPipelines.cu 1:1 so the host dispatch
// logic in advance.cpp is backend-agnostic.
//
// The header stays Kokkos-free (PIMPL, D7); all Kokkos types are confined here
// and in music_kernels_kokkos.hpp (the ported device helpers + per-cell bodies).

#include "KokkosPipelines.h"
#include "GPUGrid.h"
#include "gpu_types.h"
#include "music_kernels_kokkos.hpp"

#include <Kokkos_Core.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cstring>

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Range3    = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
using Range1    = Kokkos::RangePolicy<ExecSpace>;

// Defined in GPUGrid_kokkos.cpp: (re)size + fetch the device pack scratch View.
float* kokkos_grid_evo_pack(GPUGrid& gpu, size_t floats);

// 3-D iteration space.  ix is the innermost (last) index so adjacent work-items
// hit adjacent cells (cell = ix + Nx*(iy + Ny*ieta)) — coalesced on the GPU,
// cache-friendly on the CPU.  Bounds {Neta, Ny, Nx} -> lambda (ieta, iy, ix).
static inline Range3 range3(const GPUGrid& gpu) {
    return Range3({0, 0, 0}, {gpu.Neta(), gpu.Ny(), gpu.Nx()});
}

// ── singleton ─────────────────────────────────────────────────────────────────

KokkosPipelines& KokkosPipelines::instance() {
    static KokkosPipelines inst;
    return inst;
}

// ── initialization ────────────────────────────────────────────────────────────

bool KokkosPipelines::initialize(const char* /*unused*/) {
    if (ready_) return true;
    // The KokkosRuntimeGuard in main() (D2) owns initialize/finalize; we only
    // confirm the runtime is up and report the execution space.
    if (!Kokkos::is_initialized()) {
        fprintf(stderr, "[MUSIC-GPU] Kokkos runtime not initialized "
                        "(missing KokkosRuntimeGuard in main?) — CPU fallback.\n");
        return false;
    }
    fprintf(stderr, "[MUSIC-GPU] Kokkos execution space: %s\n",
            ExecSpace::name());
#if defined(KOKKOS_ENABLE_CUDA)
    {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            fprintf(stderr, "[MUSIC-GPU] Kokkos CUDA device: %s (cc %d.%d, %.1f GB)\n",
                    prop.name, prop.major, prop.minor,
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }
    }
#endif
    ready_ = true;
    return true;
}

// ── synchronization ─────────────────────────────────────────────────────────

void KokkosPipelines::wait() {
    if (!ready_) return;
    Kokkos::fence("KokkosPipelines::wait");
}

// Single-instance ordering makes batch markers no-ops (match the CUDA/Metal API).
void KokkosPipelines::begin_batch() {}
void KokkosPipelines::end_batch()   {}

// Coherent/UVM memory: copy_to_gpu packed straight into the managed Views the
// kernels read, so there is nothing to move (no-op, like the CUDA coherent path).
void KokkosPipelines::upload_snapshots_async(GPUGrid&) {}

// ── reduce_max (parallel_reduce with Kokkos::Max — drops the atomicMax trick) ─

void KokkosPipelines::reduce_max(GPUGrid& gpu, double& eps_max, double& rhob_max) {
    if (!ready_ || !gpu.snap_curr.epsilon || !gpu.snap_curr.rhob) {
        eps_max = rhob_max = 0.0;
        return;
    }
    const float* eps = gpu.snap_curr.epsilon;
    const float* rho = gpu.snap_curr.rhob;
    const int N = gpu.Ncells();

    float me = 0.f, mr = 0.f;
    Kokkos::parallel_reduce("reduce_max_eps_rhob", Range1(0, N),
        KOKKOS_LAMBDA(int i, float& le, float& lr) {
            le = fmaxf(le, eps[i]);
            lr = fmaxf(lr, rho[i]);
        },
        Kokkos::Max<float>(me), Kokkos::Max<float>(mr));
    Kokkos::fence("KokkosPipelines::reduce_max");
    eps_max  = static_cast<double>(me);
    rhob_max = static_cast<double>(mr);
}

// ── pack_evolution_ideal ──────────────────────────────────────────────────────

bool KokkosPipelines::pack_evolution_ideal(GPUGrid& gpu, const GPUPackParams& pp,
                                           float* host_out) {
    if (!ready_ || !host_out) return false;
    const int n_out = pp.nx_out * pp.ny_out * pp.neta_out;
    if (n_out <= 0) return false;
    const size_t need = static_cast<size_t>(n_out) * 8;   // 8 floats per cell

    float* pack = kokkos_grid_evo_pack(gpu, need);
    if (!pack) return false;

    const float* eps = gpu.snap_curr.epsilon;
    const float* u   = gpu.snap_curr.u;
    const float* eP  = gpu.eos_P;
    const float* eS  = gpu.eos_s;
    const float* eT  = gpu.eos_T;
    const GPUEosParams  ep = gpu.eos_params;
    const GPUPackParams p  = pp;

    Kokkos::parallel_for("pack_evolution_ideal", Range1(0, n_out),
        KOKKOS_LAMBDA(int o) {
            mkok::apply_pack_evolution_ideal(o, eps, u, eP, eS, eT, pack, ep, p);
        });
    Kokkos::fence("KokkosPipelines::pack_evolution_ideal");
    std::memcpy(host_out, pack, need * sizeof(float));
    return true;
}

// ── kernel dispatches ─────────────────────────────────────────────────────────

void KokkosPipelines::dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    const float* u_curr = gpu.snap_curr.u;
    const float* u_prev = gpu.snap_prev.u;
    float* theta = gpu.theta_buf;
    float* a     = gpu.a_buf;
    float* sigma = gpu.sigma_buf;
    const MUSICGridParams p = params;
    Kokkos::parallel_for("make_du", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_make_du(ix, iy, ieta, u_curr, u_prev, theta, a, sigma, p);
        });
}

void KokkosPipelines::dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    const float* W = gpu.snap_curr.Wmunu;
    const float* u = gpu.snap_curr.u;
    float* out = gpu.uwrhs_out;
    const MUSICGridParams p = params;
    Kokkos::parallel_for("make_uwrhs", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_make_uwrhs(ix, iy, ieta, W, u, out, p);
        });
}

void KokkosPipelines::dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    const float* Wc = gpu.snap_curr.Wmunu;
    const float* pc = gpu.snap_curr.pi_b;
    const float* uc = gpu.snap_curr.u;
    const float* Wp = gpu.snap_prev.Wmunu;
    const float* pp_ = gpu.snap_prev.pi_b;
    const float* upv = gpu.snap_prev.u;
    float* out = gpu.dwmn;
    const MUSICGridParams p = params;
    Kokkos::parallel_for("make_w_source", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_make_w_source(ix, iy, ieta, Wc, pc, uc, Wp, pp_, upv, out, p);
        });
}

void KokkosPipelines::dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    const float* pi = gpu.snap_curr.pi_b;
    const float* u  = gpu.snap_curr.u;
    float* out = gpu.uprhs_out;
    const MUSICGridParams p = params;
    Kokkos::parallel_for("make_uprhs", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_make_uprhs(ix, iy, ieta, pi, u, out, p);
        });
}

void KokkosPipelines::dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    const float* eps = gpu.snap_curr.epsilon;
    const float* rho = gpu.snap_curr.rhob;
    const float* u   = gpu.snap_curr.u;
    const float* eP  = gpu.eos_P;
    const float* eD  = gpu.eos_dPde;
    float* out = gpu.qi_out;
    const MUSICGridParams p = params;
    const GPUEosParams ep = gpu.eos_params;
    Kokkos::parallel_for("make_delta_qi", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_make_delta_qi(ix, iy, ieta, eps, rho, u, eP, eD, out, p, ep);
        });
}

void KokkosPipelines::dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;
    const float* qi   = gpu.qi_out;
    const float* dwmn = gpu.dwmn;
    const float* ec   = gpu.snap_curr.epsilon;
    const float* uc   = gpu.snap_curr.u;
    const float* ep_  = gpu.snap_prev.epsilon;
    const float* rp   = gpu.snap_prev.rhob;
    const float* up   = gpu.snap_prev.u;
    float* ef = gpu.snap_future.epsilon;
    float* rf = gpu.snap_future.rhob;
    float* uf = gpu.snap_future.u;
    const float* eP = gpu.eos_P;
    const float* eD = gpu.eos_dPde;
    const float* qsrc = gpu.qi_source_buf;
    const MUSICGridParams p = params;
    const GPUEosParams ep = gpu.eos_params;
    Kokkos::parallel_for("finalize_ideal", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_finalize_ideal(ix, iy, ieta, qi, dwmn, ec, uc, ep_, rp, up,
                                       ef, rf, uf, eP, eD, p, ep, qsrc);
        });
}

void KokkosPipelines::dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                                    const MUSICGridParams& params) {
    if (!ready_) return;
    const float* Wc = gpu.snap_curr.Wmunu;
    const float* pc = gpu.snap_curr.pi_b;
    const float* uc = gpu.snap_curr.u;
    const float* Wp = gpu.snap_prev.Wmunu;
    const float* pp_ = gpu.snap_prev.pi_b;
    const float* up = gpu.snap_prev.u;
    const float* ec = gpu.snap_curr.epsilon;
    const float* ep_ = gpu.snap_prev.epsilon;
    const float* uf = gpu.snap_future.u;
    const float* uwrhs = gpu.uwrhs_out;
    const float* theta = gpu.theta_buf;
    const float* a = gpu.a_buf;
    const float* sigma = gpu.sigma_buf;
    float* Wf = gpu.snap_future.Wmunu;
    float* pf = gpu.snap_future.pi_b;
    const float* eP = gpu.eos_P;
    const float* eS = gpu.eos_s;
    const float* eT = gpu.eos_T;
    const float* eD = gpu.eos_dPde;
    const float* uprhs = gpu.uprhs_out;
    const float* ef = gpu.snap_future.epsilon;
    const float* rf = gpu.snap_future.rhob;
    const MUSICGridParams p = params;
    const GPUEosParams ep = gpu.eos_params;
    Kokkos::parallel_for("first_rk_step_w_full", range3(gpu),
        KOKKOS_LAMBDA(int ieta, int iy, int ix) {
            mkok::apply_first_rk_step_w_full(ix, iy, ieta, Wc, pc, uc, Wp, pp_, up,
                                             ec, ep_, uf, uwrhs, theta, a, sigma,
                                             Wf, pf, eP, eS, eT, eD, uprhs, p, ep,
                                             ef, rf);
        });
}
