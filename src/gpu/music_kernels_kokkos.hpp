// Performance-portable Kokkos compute kernels for MUSIC relativistic
// hydrodynamics (NVIDIA / AMD / Intel GPU + multicore CPU).
//
// Direct port of src/gpu/music_kernels.cu (CUDA C++).  The physics, stencils,
// and per-cell algebra are identical, line-for-line — this is what gives the
// Kokkos backend the same precision-vs-CPU as the validated CUDA path.  Only
// the language surface changes:
//   __device__ __forceinline__ f(...)         -> KOKKOS_INLINE_FUNCTION f(...)
//   __global__ void k(...) { i = blockIdx...;  -> per-cell apply_*(ix,iy,ieta,...)
//                            if (i>=N) return; }   (bounds handled by the policy)
//   __ldg(&p[i])                              -> p[i]   (Stage 2 adds RandomAccess)
//   isfinite / isnan                          -> Kokkos::isfinite / Kokkos::isnan
//   __constant__ WMUNU_IDX                    -> local constexpr table (host+device)
//
// Memory layout (SoA, component-major), matching the CUDA/Metal/CPU convention:
//   field[comp * Ncells + cell]   where cell = Nx*(Ny*ieta + iy) + ix
//
// Included only by the Kokkos device TUs (KokkosPipelines.cpp).  The kernel
// launch wrappers live there as parallel_for over MDRangePolicy<Rank<3>>.

#pragma once

#include <Kokkos_Core.hpp>
#include "gpu_types.h"   // MUSICGridParams, GPUEosParams, GPUPackParams

namespace mkok {

// Stage 1 precision: float device path, matching today's CUDA backend (D4).
// Stage 4 will template the helpers on Real to add the double CPU reference.
using Real = float;

// ── math shims (CUDA has no float clamp; clampi mirrors min(max(...))) ─────────
KOKKOS_FORCEINLINE_FUNCTION float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}
KOKKOS_FORCEINLINE_FUNCTION int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ── index helpers ─────────────────────────────────────────────────────────────
KOKKOS_FORCEINLINE_FUNCTION int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

// Boundary-clamped cell index (replicates Cell_small::getHalo).
KOKKOS_FORCEINLINE_FUNCTION int clamped_cell(int ix, int iy, int ieta,
                                             int Nx, int Ny, int Neta) {
    ix   = clampi(ix,   0, Nx   - 1);
    iy   = clampi(iy,   0, Ny   - 1);
    ieta = clampi(ieta, 0, Neta - 1);
    return cell_idx(ix, iy, ieta, Nx, Ny);
}

// Read-only-data load.  On CUDA/HIP device code this routes through the
// read-only / texture cache (`__ldg`) — what `View<const T*, RandomAccess>`
// lowers to, restored here for the EOS-table + stencil-neighbour reads without
// threading View types through every device helper.  Returns the exact same
// float (a cache hint, not a value change), so it is numerically neutral; on
// host / SYCL it is a plain load.
KOKKOS_FORCEINLINE_FUNCTION float ldg_ro(const float* p, int i) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __ldg(&p[i]);
#else
    return p[i];
#endif
}

// Flat 1-D work index -> (ix, iy, ieta).  Inverse of cell_idx, so a
// RangePolicy<Ncells> work item maps to the same cell the MDRange did.
KOKKOS_FORCEINLINE_FUNCTION void deindex(int c, int Nx, int Ny,
                                         int& ix, int& iy, int& ieta) {
    ix = c % Nx;
    int t = c / Nx;
    iy = t % Ny;
    ieta = t / Ny;
}

KOKKOS_FORCEINLINE_FUNCTION float get_Wmunu(const float* W, int comp,
                                            int ix, int iy, int ieta,
                                            int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return ldg_ro(W, comp * Ncells + c);
}
KOKKOS_FORCEINLINE_FUNCTION float get_u(const float* u, int comp,
                                        int ix, int iy, int ieta,
                                        int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return ldg_ro(u, comp * Ncells + c);
}
KOKKOS_FORCEINLINE_FUNCTION float get_pi_b(const float* pi_b,
                                           int ix, int iy, int ieta,
                                           int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return ldg_ro(pi_b, c);
}

// Wmunu 2D->1D index table (= Util::map_2d_idx_to_1d / gpu_types.h WMUNU_IDX).
// A local constexpr keeps it usable from both host and device code (the
// gpu_types.h __constant__ form is device-only under nvcc, so unusable in the
// host half of a KOKKOS_INLINE_FUNCTION).
KOKKOS_FORCEINLINE_FUNCTION int WIDX(int alpha, int direction) {
    constexpr int M[5][4] = {
        { 0,  1,  2,  3},
        { 1,  4,  5,  6},
        { 2,  5,  7,  8},
        { 3,  6,  8,  9},
        {10, 11, 12, 13}};
    return M[alpha][direction];
}

// ── EOS table helpers (log-spaced lookup shared by P, dP/de, s, T) ────────────
KOKKOS_INLINE_FUNCTION float gpu_log_interp(float e, const float* tab,
                                            const GPUEosParams& ep) {
    if (e <= 0.f) return 0.f;
    float le = logf(e);
    le = clampf(le, ep.log_e_min, ep.log_e_max);
    float fe   = (le - ep.log_e_min) / ep.log_delta_e;
    int   idx  = clampi((int)fe, 0, ep.n_pts - 2);
    float frac = fe - (float)idx;
    // EOS tables are read-only for the whole run -> read-only cache (RandomAccess
    // / __ldg). Data-dependent index, so warp lanes hit different entries; the
    // read-only L1 path handles that far better than the default load.
    return fmaxf(0.f, ldg_ro(tab, idx) * (1.f - frac) + ldg_ro(tab, idx + 1) * frac);
}
KOKKOS_INLINE_FUNCTION float gpu_P(float e, const float* P_tab, const GPUEosParams& ep) {
    return fmaxf(1.e-20f, gpu_log_interp(e, P_tab, ep));
}
KOKKOS_INLINE_FUNCTION float gpu_dPde(float e, const float* dPde_tab, const GPUEosParams& ep) {
    return gpu_log_interp(e, dPde_tab, ep);
}
KOKKOS_INLINE_FUNCTION float gpu_cs2(float e, const float* P_tab,
                                     const float* dPde_tab, const GPUEosParams& ep) {
    return clampf(gpu_dPde(e, dPde_tab, ep), 0.01f, 0.333333f);
}
KOKKOS_INLINE_FUNCTION float gpu_s(float e, const float* s_tab, const GPUEosParams& ep) {
    return gpu_log_interp(e, s_tab, ep);
}
KOKKOS_INLINE_FUNCTION float gpu_T_e(float e, const float* T_tab, const GPUEosParams& ep) {
    return gpu_log_interp(e, T_tab, ep);
}

// ── minmod slope limiter ──────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_minmod_dx(float up1, float u, float um1, float theta) {
    float diffup   = (up1 - u)   * theta;
    float diffdown = (u   - um1) * theta;
    float diffmid  = (up1 - um1) * 0.5f;
    if (diffup == 0.f) return 0.f;
    return diffup * fmaxf(0.f, fminf(1.f, fminf(diffdown / diffup, diffmid / diffup)));
}

// ── T^{alpha,0} from SoA cell ─────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_TJb0(int alpha, int c, int Ncells,
                                      const float* eps_buf, const float* rhob_buf,
                                      const float* u_buf, const float* P_tab,
                                      const GPUEosParams& ep) {
    float u0 = ldg_ro(u_buf, 0 * Ncells + c);
    if (alpha == 4) return ldg_ro(rhob_buf, c) * u0;
    float e = ldg_ro(eps_buf, c);
    float P = gpu_P(e, P_tab, ep);
    if (alpha == 0) return (e + P) * u0 * u0 - P;
    return (e + P) * ldg_ro(u_buf, alpha * Ncells + c) * u0;
}

// ── Newton-Brent reconstruction helpers ──────────────────────────────────────
struct ReconstResult {
    float e;
    float rhob;
    float u[4];
};

KOKKOS_INLINE_FUNCTION void gpu_vel_fdf(float v, float T00, float M, float J0,
                                        float& fv, float& dfdv,
                                        const float* P_tab, const float* dPde_tab,
                                        const GPUEosParams& ep) {
    float eps  = T00 - v * M;
    float P    = gpu_P(eps, P_tab, ep);
    float dPde = gpu_dPde(eps, dPde_tab, ep);
    float t1   = T00 + P;
    fv   = v - M / t1;
    dfdv = 1.f - M * M * dPde / (t1 * t1);
    (void)J0;
}

KOKKOS_INLINE_FUNCTION float gpu_solve_v(float v_guess, float T00, float M, float J0,
                                         const float* P_tab, const float* dPde_tab,
                                         const GPUEosParams& ep) {
    const float ABS_ERR = 1.e-7f;
    float fv_l, dfdv_l, fv_h, dfdv_h;
    float v_l = 0.f, v_h = 1.f;
    gpu_vel_fdf(v_l, T00, M, J0, fv_l, dfdv_l, P_tab, dPde_tab, ep);
    gpu_vel_fdf(v_h, T00, M, J0, fv_h, dfdv_h, P_tab, dPde_tab, ep);

    if (fabsf(fv_l) < ABS_ERR) return v_l;
    if (fabsf(fv_h) < ABS_ERR) return v_h;
    if (fv_l * fv_h > 0.f) return -1.f;

    float dv_prev = v_h - v_l;
    float dv_curr = dv_prev;
    float v_root  = (v_h + v_l) * 0.5f;
    float fv, dfdv;
    gpu_vel_fdf(v_root, T00, M, J0, fv, dfdv, P_tab, dPde_tab, ep);

    for (int it = 0; it < 60; it++) {
        if (((v_root - v_h) * dfdv - fv) * ((v_root - v_l) * dfdv - fv) > 0.f
            || fabsf(2.f * fv) > fabsf(dv_prev * dfdv)) {
            dv_prev = dv_curr;
            dv_curr = (v_h - v_l) * 0.5f;
            v_root  = v_l + dv_curr;
        } else {
            dv_prev = dv_curr;
            dv_curr = fv / dfdv;
            v_root  = v_root - dv_curr;
        }
        gpu_vel_fdf(v_root, T00, M, J0, fv, dfdv, P_tab, dPde_tab, ep);
        if (fv * fv_l < 0.f) { v_h = v_root; fv_h = fv; }
        else                  { v_l = v_root; fv_l = fv; }
        if (fabsf(dv_curr) < ABS_ERR) break;
    }
    return v_root;
}

KOKKOS_INLINE_FUNCTION void gpu_u0_fdf(float u0, float T00, float K00, float M, float J0,
                                       float& fu0, float& dfdu0,
                                       const float* P_tab, const float* dPde_tab,
                                       const GPUEosParams& ep) {
    const float ABS_ERR = 1.e-10f;
    float v       = sqrtf(fmaxf(0.f, 1.f - 1.f / (u0 * u0)));
    float epsilon = T00 - v * M;
    float dedu0   = -M / (u0 * u0 * u0 * v + ABS_ERR);
    float P       = gpu_P(epsilon, P_tab, ep);
    float dPde    = gpu_dPde(epsilon, dPde_tab, ep);
    float temp1   = (T00 + P) * (T00 + P) - K00;
    float den1    = sqrtf(fmaxf(0.f, temp1));
    float temp    = (T00 + P) / fmaxf(den1, ABS_ERR);
    fu0    = u0 - temp;
    dfdu0  = 1.f + dedu0 * dPde * K00 / fmaxf(temp1 * den1, ABS_ERR);
    (void)J0;
}

KOKKOS_INLINE_FUNCTION float gpu_solve_u0(float u0_guess, float T00, float K00,
                                          float M, float J0,
                                          const float* P_tab, const float* dPde_tab,
                                          const GPUEosParams& ep) {
    const float ABS_ERR = 1.e-7f;
    float u0_l = fmaxf(1.f, 0.5f * u0_guess);
    float u0_h = fminf(1.e4f, 1.5f * u0_guess);
    if (u0_h < 1.f + ABS_ERR) u0_h = 2.f;

    float fu0_l, dfdu0_l, fu0_h, dfdu0_h;
    gpu_u0_fdf(u0_l, T00, K00, M, J0, fu0_l, dfdu0_l, P_tab, dPde_tab, ep);
    gpu_u0_fdf(u0_h, T00, K00, M, J0, fu0_h, dfdu0_h, P_tab, dPde_tab, ep);

    if (fabsf(fu0_l) < ABS_ERR) return u0_l;
    if (fabsf(fu0_h) < ABS_ERR) return u0_h;
    if (fu0_l * fu0_h > 0.f)  return u0_guess;

    float du0_prev = u0_h - u0_l;
    float du0_curr = du0_prev;
    float u0_root  = (u0_h + u0_l) * 0.5f;
    float fu0, dfdu0;
    gpu_u0_fdf(u0_root, T00, K00, M, J0, fu0, dfdu0, P_tab, dPde_tab, ep);

    for (int it = 0; it < 60; it++) {
        if (((u0_root - u0_h) * dfdu0 - fu0) * ((u0_root - u0_l) * dfdu0 - fu0) > 0.f
            || fabsf(2.f * fu0) > fabsf(du0_prev * dfdu0)) {
            du0_prev = du0_curr;
            du0_curr = (u0_h - u0_l) * 0.5f;
            u0_root  = u0_l + du0_curr;
        } else {
            du0_prev = du0_curr;
            du0_curr = fu0 / dfdu0;
            u0_root  = u0_root - du0_curr;
        }
        gpu_u0_fdf(u0_root, T00, K00, M, J0, fu0, dfdu0, P_tab, dPde_tab, ep);
        if (fu0 * fu0_l < 0.f) { u0_h = u0_root; fu0_h = fu0; }
        else                    { u0_l = u0_root; fu0_l = fu0; }
        if (fabsf(du0_curr) < ABS_ERR) break;
    }
    return u0_root;
}

// ── Main reconstruction (matches Reconst::ReconstIt_shell on CPU) ─────────────
KOKKOS_INLINE_FUNCTION ReconstResult gpu_reconst(float tau, float tauq[5],
                                                 const float prev_u[4], float prev_eps,
                                                 const float* P_tab, const float* dPde_tab,
                                                 const GPUEosParams& ep) {
    const float ABS_ERR = 1.e-8f;

    ReconstResult res;
    res.rhob = 0.f;
    res.u[0] = 1.f;  res.u[1] = 0.f;  res.u[2] = 0.f;  res.u[3] = 0.f;

    float q[5];
    for (int i = 0; i < 5; i++) q[i] = tauq[i] / tau;

    float K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    float M   = sqrtf(K00);
    float T00 = q[0];
    float J0  = q[4];

    if (!Kokkos::isfinite(T00) || !Kokkos::isfinite(M)) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
        return res;
    }

    if (T00 < ABS_ERR) {
        res.e = ABS_ERR;
        return res;
    }
    if (T00 < M) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
        return res;
    }

    float v_guess = sqrtf(fmaxf(0.f, 1.f - 1.f / (prev_u[0] * prev_u[0] + ABS_ERR)));
    float v_sol   = gpu_solve_v(v_guess, T00, M, J0, P_tab, dPde_tab, ep);

    if (v_sol < 0.f) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
        return res;
    }

    float u0      = 1.f / (sqrtf(fmaxf(0.f, 1.f - v_sol * v_sol)) + v_sol * ABS_ERR);
    float epsilon = T00 - v_sol * M;
    float rhob    = J0 / u0;

    if (v_sol > 0.563624f && epsilon > 1e-6f) {
        float u0_sol = gpu_solve_u0(u0, T00, K00, M, J0, P_tab, dPde_tab, ep);
        if (u0_sol >= 1.f) {
            u0      = u0_sol;
            epsilon = T00 - sqrtf(fmaxf(0.f, (1.f - 1.f / (u0 * u0)) * K00));
            rhob    = J0 / u0;
        }
    }

    if (fabsf(u0 - prev_u[0]) / prev_u[0] > 100.f) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
        return res;
    }

    res.e    = epsilon;
    res.rhob = rhob;

    float P       = gpu_P(epsilon, P_tab, ep);
    float vel_inv = u0 / (T00 + P);
    res.u[0]  = u0;
    res.u[1]  = q[1] * vel_inv;
    res.u[2]  = q[2] * vel_inv;
    res.u[3]  = q[3] * vel_inv;

    float u_sp_sq = res.u[1]*res.u[1] + res.u[2]*res.u[2] + res.u[3]*res.u[3];
    if (fabsf(u0*u0 - u_sp_sq - 1.f) > ABS_ERR) {
        float scale = sqrtf(fmaxf(0.f, (u0*u0 - 1.f) / (u_sp_sq + ABS_ERR)));
        res.u[1] *= scale;
        res.u[2] *= scale;
        res.u[3] *= scale;
    }

    if (!Kokkos::isfinite(res.e)) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
    }
    return res;
}

// ── MaxSpeed (matches Advance::MaxSpeed on CPU) ───────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_max_speed(float tau, int direction, ReconstResult r,
                                           const float* P_tab, const float* dPde_tab,
                                           const GPUEosParams& ep) {
    float gfac = (direction == 3) ? 1.f / tau : 1.f;

    float utau    = r.u[0];
    float ux      = fabsf(r.u[direction]);
    float utau2   = utau * utau;
    float ut2mux2 = utau2 - ux * ux;

    float cs2    = gpu_cs2(r.e, P_tab, dPde_tab, ep);
    float num_sq = (ut2mux2 - (ut2mux2 - 1.f) * cs2) * cs2;
    float num;
    if (num_sq >= 0.f) {
        num = utau * ux * (1.f - cs2) + sqrtf(num_sq);
    } else {
        float dPde = gpu_dPde(r.e, dPde_tab, ep);
        float P    = gpu_P(r.e, P_tab, ep);
        float h    = P + r.e;
        num = (dPde < 0.001f)
              ? sqrtf(fmaxf(0.f, -(h*dPde*h*(dPde*(-1.f + ut2mux2) - ut2mux2))))
                - h*(-1.f + dPde)*utau*ux
              : 1.f;
    }
    float den = utau2 * (1.f - cs2) + cs2;
    float f   = num / fmaxf(den, 1.e-20f);
    f = clampf(f, ux / utau, 1.f);
    return f * gfac;
}

// ── T^{alpha,direction} from a ReconstResult ─────────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_get_TJb_reconst(ReconstResult r, int alpha, int direction,
                                                 float tau_fac, const float* P_tab,
                                                 const GPUEosParams& ep) {
    float u_nu = r.u[direction];
    if (alpha == 4) return r.rhob * u_nu * tau_fac;
    float P    = gpu_P(r.e, P_tab, ep);
    float gfac = 0.f;
    float u_mu;
    if (alpha == direction) {
        u_mu = u_nu;
        gfac = (alpha == 0) ? -1.f : 1.f;
    } else {
        u_mu = r.u[alpha];
    }
    return ((r.e + P) * u_mu * u_nu + P * gfac) * tau_fac;
}

// ── η/s profile functions (ports of transport_coeffs.cpp) ────────────────────
constexpr float GPU_HBARC = 0.197326980f;   // GeV · fm

KOKKOS_INLINE_FUNCTION float gpu_eta_over_s_default(float T_in_fm, float shear_to_s_baseline) {
    const float Ttr   = 0.18f / GPU_HBARC;
    float Tfrac = T_in_fm / Ttr;
    if (Tfrac < 1.f) {
        return shear_to_s_baseline + 0.0594f * (1.f - Tfrac)
             + 0.544f * (1.f - Tfrac * Tfrac);
    } else {
        return shear_to_s_baseline + 0.288f * (Tfrac - 1.f)
             + 0.0818f * (Tfrac * Tfrac - 1.f);
    }
}
KOKKOS_INLINE_FUNCTION float gpu_eta_over_s_duke(float T_in_fm, float shear_2_min,
                                                 float shear_2_slope, float shear_2_curv) {
    float T_in_GeV   = T_in_fm * GPU_HBARC;
    float Ttr_in_GeV = 0.154f;
    float Tfrac      = T_in_GeV / Ttr_in_GeV;
    return shear_2_min + shear_2_slope * (T_in_GeV - Ttr_in_GeV) * powf(Tfrac, shear_2_curv);
}
KOKKOS_INLINE_FUNCTION float gpu_eta_over_s_sims(float T_in_fm, float T_kink_GeV,
                                                 float low_slope, float high_slope,
                                                 float at_kink) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float eta_over_s_min = 1.e-6f;
    float eta_over_s;
    if (T_in_GeV < T_kink_GeV) eta_over_s = at_kink + low_slope  * (T_in_GeV - T_kink_GeV);
    else                       eta_over_s = at_kink + high_slope * (T_in_GeV - T_kink_GeV);
    return fmaxf(eta_over_s, eta_over_s_min);
}
KOKKOS_INLINE_FUNCTION float gpu_eta_over_s_profile_mult(float T_in_fm, float shear_to_s_baseline) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float Tc = 0.165f;
    float f_T = 1.f;
    if (T_in_GeV < Tc) {
        const float Tslope = 1.2f;
        const float Tlow   = 0.1f;
        f_T += Tslope * (Tc - T_in_GeV) / (Tc - Tlow);
    } else {
        const float Tslope2 = 0.0f;
        const float Thigh   = 0.4f;
        f_T += Tslope2 * (T_in_GeV - Tc) / (Thigh - Tc);
    }
    return shear_to_s_baseline * f_T;
}

// ── ζ/s profile functions ────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_zeta_over_s_default(float T_in_fm) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float Ttr = 0.18f;
    float dummy = T_in_GeV / Ttr;
    float bulk;
    if (T_in_GeV < 0.995f * Ttr) {
        const float lambda3 = 0.9f, lambda4 = 0.22f, sigma3 = 0.0025f, sigma4 = 0.022f;
        bulk = lambda3 * expf((dummy - 1.f) / sigma3)
             + lambda4 * expf((dummy - 1.f) / sigma4) + 0.03f;
    } else if (T_in_GeV > 1.05f * Ttr) {
        const float lambda1 = 0.9f, lambda2 = 0.25f, sigma1 = 0.025f, sigma2 = 0.13f;
        bulk = lambda1 * expf(-(dummy - 1.f) / sigma1)
             + lambda2 * expf(-(dummy - 1.f) / sigma2) + 0.001f;
    } else {
        const float A1 = -13.77f, A2 = 27.55f, A3 = 13.45f;
        bulk = A1 * dummy * dummy + A2 * dummy - A3;
    }
    return bulk;
}
KOKKOS_INLINE_FUNCTION float gpu_zeta_over_s_duke(float T_in_fm, float norm,
                                                  float width_GeV, float peak_GeV) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float diff_ratio = (T_in_GeV - peak_GeV) / width_GeV;
    return norm / (1.f + diff_ratio * diff_ratio);
}
KOKKOS_INLINE_FUNCTION float gpu_zeta_over_s_sims(float T_in_fm, float max_norm,
                                                  float width_GeV, float T_peak_GeV,
                                                  float lambda) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float diff = T_in_GeV - T_peak_GeV;
    float s    = (diff > 0.f) ? 1.f : ((diff < 0.f) ? -1.f : 0.f);
    float diff_ratio = diff / (width_GeV * (lambda * s + 1.f));
    return max_norm / (1.f + diff_ratio * diff_ratio);
}
KOKKOS_INLINE_FUNCTION float gpu_zeta_over_s_asym_gaussian(float T_in_fm, float B_norm,
                                                           float B_width_low_GeV,
                                                           float B_width_high_GeV,
                                                           float Tpeak_GeV) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float Tdiff    = T_in_GeV - Tpeak_GeV;
    Tdiff = (Tdiff > 0.f) ? (Tdiff / B_width_high_GeV) : (Tdiff / B_width_low_GeV);
    return B_norm * expf(-Tdiff * Tdiff);
}
KOKKOS_INLINE_FUNCTION float gpu_zeta_over_s_bigbroadP(float T_in_fm) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float B_norm = 0.24f, B_width = 1.5f, Tpeak = 0.165f;
    float Ttilde = (T_in_GeV / Tpeak - 1.f) / B_width;
    float bulk   = B_norm / (Ttilde * Ttilde + 1.f);
    if (T_in_GeV < Tpeak) {
        float Tdiff = (T_in_GeV - Tpeak) / 0.01f;
        bulk = B_norm * expf(-Tdiff * Tdiff);
    }
    return bulk;
}

KOKKOS_INLINE_FUNCTION float gpu_zeta_over_s(float T_in_fm, const MUSICGridParams& params) {
    float zoverS = 0.f;
    switch (params.T_dep_bulk_mode) {
        case 0:  zoverS = 0.f; break;
        case 1:  zoverS = gpu_zeta_over_s_default(T_in_fm); break;
        case 2:  zoverS = gpu_zeta_over_s_duke(T_in_fm, params.bulk_duke_norm,
                                               params.bulk_duke_width_GeV,
                                               params.bulk_duke_peak_GeV); break;
        case 3:  zoverS = gpu_zeta_over_s_sims(T_in_fm, params.bulk_sims_max,
                                               params.bulk_sims_width_GeV,
                                               params.bulk_sims_T_peak_GeV,
                                               params.bulk_sims_lambda); break;
        case 7:  zoverS = gpu_zeta_over_s_bigbroadP(T_in_fm); break;
        case 8:  zoverS = gpu_zeta_over_s_asym_gaussian(T_in_fm, 0.13f, 0.01f, 0.12f, 0.160f); break;
        case 9:  zoverS = gpu_zeta_over_s_asym_gaussian(T_in_fm, 0.175f, 0.01f, 0.12f, 0.160f); break;
        case 10: zoverS = gpu_zeta_over_s_asym_gaussian(T_in_fm, params.bulk_asym10_max,
                                                        params.bulk_asym10_width_low,
                                                        params.bulk_asym10_width_high,
                                                        params.bulk_asym10_Tpeak); break;
        default: zoverS = 0.f; break;
    }
    return fmaxf(0.f, zoverS);
}

KOKKOS_INLINE_FUNCTION float gpu_eta_over_s(float T_in_fm, const MUSICGridParams& params) {
    switch (params.T_dep_shear_mode) {
        case 0:  return params.shear_to_s;
        case 1:  return gpu_eta_over_s_default(T_in_fm, params.shear_to_s);
        case 2:  return gpu_eta_over_s_duke(T_in_fm, params.shear_duke_min,
                                            params.shear_duke_slope, params.shear_duke_curv);
        case 3:  return gpu_eta_over_s_sims(T_in_fm, params.shear_sims_T_kink_GeV,
                                            params.shear_sims_low_slope,
                                            params.shear_sims_high_slope,
                                            params.shear_sims_at_kink);
        case 11: return gpu_eta_over_s_profile_mult(T_in_fm, params.shear_to_s);
        default: return params.shear_to_s;
    }
}

// ── Make_uPiSource (port of Diss::Make_uPiSource) ────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_uPi_source(float pi_b, float theta, float eps_src,
                                            float Wsigma_scalar, const float* P_tab,
                                            const float* dPde_tab, const float* T_tab,
                                            const GPUEosParams& ep,
                                            const MUSICGridParams& params, float delta_tau) {
    float P_local  = gpu_P (eps_src, P_tab, ep);
    float T_local  = gpu_T_e(eps_src, T_tab, ep);
    float cs2      = clampf(gpu_dPde(eps_src, dPde_tab, ep), 0.01f, 0.333333f);

    float zeta_s    = gpu_zeta_over_s(T_local, params);
    float epsP      = fmaxf(eps_src + P_local, 1.e-20f);
    float T_safe    = fmaxf(T_local, 1.e-20f);
    float bulk_zeta = zeta_s * epsP / T_safe;

    float csfactor = fmaxf(1.f/3.f - cs2, 1.e-20f);
    float tau_Pi;
    if (params.bulk_relaxation_type == 1) {
        tau_Pi = bulk_zeta / (params.bulk_relax_time_factor * csfactor) / epsP;
    } else {
        tau_Pi = params.bulk_relax_time_factor / (csfactor * csfactor) / epsP * bulk_zeta;
    }
    tau_Pi = fminf(10.f, fmaxf(3.f * delta_tau, tau_Pi));

    const float delta_PiPi = 2.f / 3.f;
    float transport_coeff1 = delta_PiPi * tau_Pi;

    float NS_term  = -bulk_zeta * theta;
    float relax    = -pi_b - transport_coeff1 * theta * pi_b;

    float coupling_to_shear = 0.f;
    if (params.include_second_order_terms == 1) {
        const float lambda_bulkPipi = 8.f / 5.f;
        float transport_coeff1_s = lambda_bulkPipi * (1.f/3.f - cs2) * tau_Pi;
        coupling_to_shear = -Wsigma_scalar * transport_coeff1_s;
    }

    return (NS_term + relax + coupling_to_shear) / tau_Pi;
}

// ── Make_uWSource (port of Diss::Make_uWSource) ──────────────────────────────
KOKKOS_INLINE_FUNCTION float gpu_uW_source(int mu, int nu, const float W4[4][4],
                                           const float s4[4][4], const float u[4],
                                           float pi_b, float theta, float eps_src,
                                           float Wsigma_scalar, float Wsquare,
                                           const float* P_tab, const float* s_tab,
                                           const float* T_tab, const GPUEosParams& ep,
                                           const MUSICGridParams& params, float delta_tau) {
    float P       = gpu_P(eps_src, P_tab, ep);
    float entropy = gpu_s(eps_src, s_tab, ep);
    float T_local = gpu_T_e(eps_src, T_tab, ep);
    float shear_to_s_T = gpu_eta_over_s(T_local, params);
    float shear   = shear_to_s_T * entropy;

    float epsP = fmaxf(eps_src + P, 1.e-20f);
    float tau_pi = params.shear_relax_time_factor * shear / epsP;
    tau_pi = fminf(10.f, fmaxf(3.f * delta_tau, tau_pi));

    const float dpi_pi = 4.f / 3.f;
    float transport_coefficient2 = dpi_pi * tau_pi;

    float W_mn      = W4[mu][nu];
    float sigma_mn  = s4[mu][nu];

    float NS_term = -2.f * shear * sigma_mn;
    float relax   = -(1.f + transport_coefficient2 * theta) * W_mn;

    float Wsigma_term = 0.f;
    float WW_term     = 0.f;
    if (params.include_second_order_terms == 1 && params.init_profile_zero == 0) {
        const float phi7_45      = (9.f / 70.f) * (4.f / 5.f);
        float transport_coeff_W  = phi7_45 * tau_pi / fmaxf(shear, 1.e-20f);
        const float tau_pipi     = 10.f / 7.f;
        float transport_coeff_Ws = tau_pipi * tau_pi;

        float gmunu_uu  = ((mu == nu) ? ((mu == 0) ? -1.f : 1.f) : 0.f) + u[mu] * u[nu];

        float t1_Ws = (
              -W4[mu][0]*s4[nu][0] - W4[nu][0]*s4[mu][0]
              +W4[mu][1]*s4[nu][1] + W4[nu][1]*s4[mu][1]
              +W4[mu][2]*s4[nu][2] + W4[nu][2]*s4[mu][2]
              +W4[mu][3]*s4[nu][3] + W4[nu][3]*s4[mu][3]) * 0.5f;
        float t2_Ws = -(1.f/3.f) * gmunu_uu * Wsigma_scalar;
        Wsigma_term = -transport_coeff_Ws * (t1_Ws + t2_Ws);

        float t1_WW = -W4[mu][0]*W4[nu][0]
                     + W4[mu][1]*W4[nu][1]
                     + W4[mu][2]*W4[nu][2]
                     + W4[mu][3]*W4[nu][3];
        float t2_WW = -(1.f/3.f) * gmunu_uu * Wsquare;
        WW_term = -transport_coeff_W * (t1_WW + t2_WW);
    }

    float Coupling_to_Bulk = 0.f;
    if (params.include_second_order_terms == 1) {
        const float lambda_pibulkPi = 6.f / 5.f;
        float transport_coeff_b = lambda_pibulkPi * tau_pi;
        Coupling_to_Bulk = -pi_b * sigma_mn * transport_coeff_b;
    }

    return (NS_term + relax + Wsigma_term + WW_term + Coupling_to_Bulk) / tau_pi;
}

// ── Make_uWRHS geometric tail (matches Diss::Make_uWRHS_geom) ─────────────────
KOKKOS_INLINE_FUNCTION float gpu_uWRHS_geom(const float W_local[4][4], const float u[4],
                                            const float a_loc[4], int mu, int nu,
                                            float theta, float tau, float delta_tau) {
    float g3m = (mu == 3) ? 1.f : 0.f;
    float g3n = (nu == 3) ? 1.f : 0.f;
    float g0m = (mu == 0) ? -1.f : 0.f;
    float g0n = (nu == 0) ? -1.f : 0.f;

    float tempf = (
         - g3m * W_local[0][nu]
         - g3n * W_local[0][mu]
         + g0m * W_local[3][nu]
         + g0n * W_local[3][mu]
         + W_local[3][nu] * u[mu] * u[0]
         + W_local[3][mu] * u[nu] * u[0]
         - W_local[0][nu] * u[mu] * u[3]
         - W_local[0][mu] * u[nu] * u[3]) * (u[3] / tau);

    for (int ic = 0; ic < 4; ic++) {
        float ic_fac = (ic == 0) ? -1.f : 1.f;
        tempf +=  W_local[ic][nu] * u[mu] * a_loc[ic] * ic_fac
                + W_local[ic][mu] * u[nu] * a_loc[ic] * ic_fac;
    }

    return tempf * delta_tau
           + (-(u[0] * W_local[mu][nu]) / tau + theta * W_local[mu][nu]) * delta_tau;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Per-cell kernel bodies (apply_*).  Each is the CUDA __global__ body minus the
//  blockIdx index math + bounds check — those are provided by the launch policy.
// ═══════════════════════════════════════════════════════════════════════════════

// ── gpu_make_w_source (non-tiled; Stage 3 adds the scratch-tiled variant) ─────
KOKKOS_INLINE_FUNCTION void apply_make_w_source(
    int ix, int iy, int ieta,
    const float* Wmunu_curr, const float* pi_b_curr,
    const float* u_curr,
    const float* Wmunu_prev, const float* pi_b_prev,
    const float* u_prev,
    float* dwmn_out, const MUSICGridParams& params)
{
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta, Ncells = params.Ncells;

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, params.tau, params.tau, 1.f};

    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    float Wc[14], Wp[14];
    for (int m = 0; m < 14; ++m) {
        Wc[m] = Wmunu_curr[m * Ncells + c];
        Wp[m] = Wmunu_prev[m * Ncells + c];
    }
    float uc[4], up[4];
    for (int m = 0; m < 4; ++m) {
        uc[m] = u_curr[m * Ncells + c];
        up[m] = u_prev[m * Ncells + c];
    }
    float pib_c = pi_b_curr[c];
    float pib_p = pi_b_prev[c];

    const int dix[3]   = {1, 0, 0};
    const int diy[3]   = {0, 1, 0};
    const int dieta[3] = {0, 0, 1};

    float W_eta_p[4] = {0.f, 0.f, 0.f, 0.f};
    float W_eta_m[4] = {0.f, 0.f, 0.f, 0.f};
    float dwmn[5]    = {0.f, 0.f, 0.f, 0.f, 0.f};

    for (int alpha = 0; alpha < 5; ++alpha) {
        int idx_alpha0 = WIDX(alpha, 0);

        float dWdtau = (Wc[idx_alpha0] - Wp[idx_alpha0]) / params.delta_tau;

        float dPidtau   = 0.f;
        float Pi_alpha0 = 0.f;
        if (alpha < 4 && params.turn_on_bulk) {
            float gfac  = (alpha == 0) ? -1.f : 0.f;
            Pi_alpha0   = pib_c * (gfac + uc[alpha] * uc[0]);
            float Pi_p0 = pib_p * (gfac + up[alpha] * up[0]);
            dPidtau = (Pi_alpha0 - Pi_p0) / params.delta_tau;
        }

        float dWdx  = 0.f;
        float dPidx = 0.f;

        for (int dir = 0; dir < 3; ++dir) {
            int direction = dir + 1;
            int idx_1d = WIDX(alpha, direction);

            int ixp = ix   + dix[dir];
            int ixm = ix   - dix[dir];
            int iyp = iy   + diy[dir];
            int iym = iy   - diy[dir];
            int iep = ieta + dieta[dir];
            int iem = ieta - dieta[dir];

            float tf = tau_fac[direction];
            float dx = delta[direction];

            float sg   = Wc[idx_1d] * tf;
            float sgp1 = get_Wmunu(Wmunu_curr, idx_1d, ixp, iyp, iep, Nx, Ny, Neta, Ncells) * tf;
            float sgm1 = get_Wmunu(Wmunu_curr, idx_1d, ixm, iym, iem, Nx, Ny, Neta, Ncells) * tf;

            float W_m = (sg + sgm1) * 0.5f;
            float W_p = (sg + sgp1) * 0.5f;

            if (direction == 3 && (alpha == 0 || alpha == 3)) {
                W_eta_p[alpha] += W_p;
                W_eta_m[alpha] += W_m;
            } else {
                dWdx += (W_p - W_m) / dx;
            }

            if (alpha < 4 && params.turn_on_bulk) {
                float gfac1 = (alpha == direction) ? 1.f : 0.f;
                float bgp1  = get_pi_b(pi_b_curr, ixp, iyp, iep, Nx, Ny, Neta, Ncells)
                              * (gfac1 + get_u(u_curr, alpha, ixp, iyp, iep, Nx, Ny, Neta, Ncells)
                                       * get_u(u_curr, direction, ixp, iyp, iep, Nx, Ny, Neta, Ncells))
                              * tf;
                float bg    = pib_c * (gfac1 + uc[alpha] * uc[direction]) * tf;
                float bgm1  = get_pi_b(pi_b_curr, ixm, iym, iem, Nx, Ny, Neta, Ncells)
                              * (gfac1 + get_u(u_curr, alpha, ixm, iym, iem, Nx, Ny, Neta, Ncells)
                                       * get_u(u_curr, direction, ixm, iym, iem, Nx, Ny, Neta, Ncells))
                              * tf;
                float Pi_m = (bg + bgm1) * 0.5f;
                float Pi_p = (bg + bgp1) * 0.5f;

                if (direction == 3 && (alpha == 0 || alpha == 3)) {
                    W_eta_p[alpha] += Pi_p;
                    W_eta_m[alpha] += Pi_m;
                } else {
                    dPidx += (Pi_p - Pi_m) / dx;
                }
            }
        }

        float sf = params.tau * dWdtau + Wc[idx_alpha0] + dWdx;
        float bf = params.tau * dPidtau + Pi_alpha0 + dPidx;
        dwmn[alpha] += sf + bf;
    }

    dwmn[0] += (  (W_eta_p[0] - W_eta_m[0]) * params.cosh_deta
               + (W_eta_p[3] + W_eta_m[3]) * params.sinh_deta);
    dwmn[3] += (  (W_eta_p[3] - W_eta_m[3]) * params.cosh_deta
               + (W_eta_p[0] + W_eta_m[0]) * params.sinh_deta);

    for (int alpha = 0; alpha < 5; ++alpha)
        dwmn_out[alpha * Ncells + c] = dwmn[alpha];
}

// ── gpu_make_delta_qi ─────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void apply_make_delta_qi(
    int ix, int iy, int ieta,
    const float* epsilon_curr, const float* rhob_curr,
    const float* u_curr,
    const float* eos_P, const float* eos_dPde,
    float* qi_out, const MUSICGridParams& params, const GPUEosParams& eos_p)
{
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta, Ncells = params.Ncells;
    const float tau   = params.tau;
    const float theta = params.minmod_theta;

    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    float e_c = epsilon_curr[c];
    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];

    float qi[5];
    for (int alpha = 0; alpha < 5; alpha++)
        qi[alpha] = tau * gpu_TJb0(alpha, c, Ncells,
                                   epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, tau, tau, 1.f};

    float rhs[5]     = {0.f, 0.f, 0.f, 0.f, 0.f};
    float T_eta_m[4] = {0.f, 0.f, 0.f, 0.f};
    float T_eta_p[4] = {0.f, 0.f, 0.f, 0.f};

    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    for (int dir = 0; dir < 3; dir++) {
        int direction = dir + 1;

        int ip1 = clamped_cell(ix+  DX[dir], iy+  DY[dir], ieta+  DETA[dir], Nx, Ny, Neta);
        int ip2 = clamped_cell(ix+2*DX[dir], iy+2*DY[dir], ieta+2*DETA[dir], Nx, Ny, Neta);
        int im1 = clamped_cell(ix-  DX[dir], iy-  DY[dir], ieta-  DETA[dir], Nx, Ny, Neta);
        int im2 = clamped_cell(ix-2*DX[dir], iy-2*DY[dir], ieta-2*DETA[dir], Nx, Ny, Neta);

        float qiphL[5], qiphR[5], qimhL[5], qimhR[5];
        for (int alpha = 0; alpha < 5; alpha++) {
            float gc  = qi[alpha];
            float gp1 = tau * gpu_TJb0(alpha, ip1, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gp2 = tau * gpu_TJb0(alpha, ip2, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gm1 = tau * gpu_TJb0(alpha, im1, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gm2 = tau * gpu_TJb0(alpha, im2, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);

            float fphL =  0.5f * gpu_minmod_dx(gp1, gc,  gm1, theta);
            float fphR = -0.5f * gpu_minmod_dx(gp2, gp1, gc,  theta);
            float fmhL =  0.5f * gpu_minmod_dx(gc,  gm1, gm2, theta);
            float fmhR = -fphL;

            qiphL[alpha] = gc  + fphL;
            qiphR[alpha] = gp1 + fphR;
            qimhL[alpha] = gm1 + fmhL;
            qimhR[alpha] = gc  + fmhR;
        }

        ReconstResult r_phL = gpu_reconst(tau, qiphL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_phR = gpu_reconst(tau, qiphR, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_mhL = gpu_reconst(tau, qimhL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_mhR = gpu_reconst(tau, qimhR, u_c, e_c, eos_P, eos_dPde, eos_p);

        float aiph_L = gpu_max_speed(tau, direction, r_phL, eos_P, eos_dPde, eos_p);
        float aiph_R = gpu_max_speed(tau, direction, r_phR, eos_P, eos_dPde, eos_p);
        float aimh_L = gpu_max_speed(tau, direction, r_mhL, eos_P, eos_dPde, eos_p);
        float aimh_R = gpu_max_speed(tau, direction, r_mhR, eos_P, eos_dPde, eos_p);
        float aiph   = fmaxf(aiph_L, aiph_R);
        float aimh   = fmaxf(aimh_L, aimh_R);

        float tf = tau_fac[direction];
        float dx = delta[direction];

        for (int alpha = 0; alpha < 5; alpha++) {
            float FiphL = gpu_get_TJb_reconst(r_phL, alpha, direction, tf, eos_P, eos_p);
            float FiphR = gpu_get_TJb_reconst(r_phR, alpha, direction, tf, eos_P, eos_p);
            float FimhL = gpu_get_TJb_reconst(r_mhL, alpha, direction, tf, eos_P, eos_p);
            float FimhR = gpu_get_TJb_reconst(r_mhR, alpha, direction, tf, eos_P, eos_p);

            float Fiph = 0.5f * ((FiphL + FiphR) - aiph * (qiphR[alpha] - qiphL[alpha]));
            float Fimh = 0.5f * ((FimhL + FimhR) - aimh * (qimhR[alpha] - qimhL[alpha]));

            if (direction == 3 && (alpha == 0 || alpha == 3)) {
                T_eta_m[alpha] = Fimh;
                T_eta_p[alpha] = Fiph;
            } else {
                rhs[alpha] += (Fimh - Fiph) / dx * params.delta_tau;
            }
        }
    }

    float cd = params.cosh_deta;
    float sd = params.sinh_deta;
    rhs[0] += (  (T_eta_m[0] - T_eta_p[0]) * cd
               - (T_eta_m[3] + T_eta_p[3]) * sd) * params.delta_tau;
    rhs[3] += (  (T_eta_m[3] - T_eta_p[3]) * cd
               - (T_eta_m[0] + T_eta_p[0]) * sd) * params.delta_tau;

    for (int alpha = 0; alpha < 5; alpha++)
        qi_out[alpha * Ncells + c] = qi[alpha] + rhs[alpha];
}

// ── gpu_finalize_ideal ────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void apply_finalize_ideal(
    int ix, int iy, int ieta,
    const float* qi_buf, const float* dwmn_buf,
    const float* epsilon_curr, const float* u_curr,
    const float* epsilon_prev, const float* rhob_prev,
    const float* u_prev,
    float* e_future, float* rhob_future,
    float* u_future,
    const float* eos_P, const float* eos_dPde,
    const MUSICGridParams& params, const GPUEosParams& eos_p,
    const float* qi_source_in)
{
    const int Nx = params.Nx, Ny = params.Ny, Ncells = params.Ncells;
    const int c  = cell_idx(ix, iy, ieta, Nx, Ny);

    const int   rkf      = params.rk_flag;
    const float dt       = params.delta_tau;
    const float tau_org  = params.tau_orig;
    const float tau_next = tau_org + dt;
    const float rk_norm  = 1.f / (1.f + (float)rkf);

    float u0p   = u_prev[c];
    float e_p   = epsilon_prev[c];
    float rho_p = rhob_prev[c];
    float P_p   = (rkf > 0) ? gpu_P(e_p, eos_P, eos_p) : 0.f;

    const int has_src = params.has_hydro_source;

    float qi[5];
    for (int a = 0; a < 5; a++) {
        float qv = qi_buf[a * Ncells + c] - dwmn_buf[a * Ncells + c] * dt;

        if (has_src) qv += qi_source_in[a * Ncells + c] * dt;

        if (rkf > 0) {
            float prev_TJb0;
            if (a == 4) {
                prev_TJb0 = rho_p * u0p;
            } else if (a == 0) {
                prev_TJb0 = (e_p + P_p) * u0p * u0p - P_p;
            } else {
                float ua_p = u_prev[a * Ncells + c];
                prev_TJb0  = (e_p + P_p) * ua_p * u0p;
            }
            qv += (float)rkf * tau_org * prev_TJb0;
        }
        qi[a] = qv * rk_norm;
    }

    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];
    float e_c = epsilon_curr[c];

    ReconstResult r = gpu_reconst(tau_next, qi, u_c, e_c, eos_P, eos_dPde, eos_p);

    e_future[c]    = r.e;
    rhob_future[c] = r.rhob;
    for (int m = 0; m < 4; m++)
        u_future[m * Ncells + c] = r.u[m];
}

// ── FUSED gpu_make_delta_qi + gpu_finalize_ideal (Stage 3, behind a flag) ─────
//
// The two heaviest data-path kernels exchange qi via a 5*Ncells global buffer
// (qi_out).  Fusing them keeps the ideal qi in registers and runs the Newton
// reconstruction immediately, eliminating that global round-trip.  Bit-identical
// to running apply_make_delta_qi (into qi_out) then apply_finalize_ideal (reading
// qi_out): the delta_qi block below reproduces qi[]+rhs[] exactly, and the
// finalize tail consumes it in place.  Default builds keep the two unfused
// kernels as the D6/D11 golden reference; enable with MUSIC_KOKKOS_FUSE.
KOKKOS_INLINE_FUNCTION void apply_delta_qi_finalize(
    int ix, int iy, int ieta,
    const float* epsilon_curr, const float* rhob_curr, const float* u_curr,
    const float* dwmn_buf,
    const float* epsilon_prev, const float* rhob_prev, const float* u_prev,
    float* e_future, float* rhob_future, float* u_future,
    const float* eos_P, const float* eos_dPde,
    const MUSICGridParams& params, const GPUEosParams& eos_p,
    const float* qi_source_in)
{
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta, Ncells = params.Ncells;
    const float tau   = params.tau;
    const float theta = params.minmod_theta;
    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    float e_c = epsilon_curr[c];
    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];

    // ── delta_qi: build the ideal qi update in registers (no qi_out store) ────
    float qi[5];
    for (int alpha = 0; alpha < 5; alpha++)
        qi[alpha] = tau * gpu_TJb0(alpha, c, Ncells,
                                   epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, tau, tau, 1.f};
    float rhs[5]     = {0.f, 0.f, 0.f, 0.f, 0.f};
    float T_eta_m[4] = {0.f, 0.f, 0.f, 0.f};
    float T_eta_p[4] = {0.f, 0.f, 0.f, 0.f};
    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    for (int dir = 0; dir < 3; dir++) {
        int direction = dir + 1;
        int ip1 = clamped_cell(ix+  DX[dir], iy+  DY[dir], ieta+  DETA[dir], Nx, Ny, Neta);
        int ip2 = clamped_cell(ix+2*DX[dir], iy+2*DY[dir], ieta+2*DETA[dir], Nx, Ny, Neta);
        int im1 = clamped_cell(ix-  DX[dir], iy-  DY[dir], ieta-  DETA[dir], Nx, Ny, Neta);
        int im2 = clamped_cell(ix-2*DX[dir], iy-2*DY[dir], ieta-2*DETA[dir], Nx, Ny, Neta);

        float qiphL[5], qiphR[5], qimhL[5], qimhR[5];
        for (int alpha = 0; alpha < 5; alpha++) {
            float gc  = qi[alpha];
            float gp1 = tau * gpu_TJb0(alpha, ip1, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gp2 = tau * gpu_TJb0(alpha, ip2, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gm1 = tau * gpu_TJb0(alpha, im1, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gm2 = tau * gpu_TJb0(alpha, im2, Ncells, epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float fphL =  0.5f * gpu_minmod_dx(gp1, gc,  gm1, theta);
            float fphR = -0.5f * gpu_minmod_dx(gp2, gp1, gc,  theta);
            float fmhL =  0.5f * gpu_minmod_dx(gc,  gm1, gm2, theta);
            float fmhR = -fphL;
            qiphL[alpha] = gc  + fphL;
            qiphR[alpha] = gp1 + fphR;
            qimhL[alpha] = gm1 + fmhL;
            qimhR[alpha] = gc  + fmhR;
        }

        ReconstResult r_phL = gpu_reconst(tau, qiphL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_phR = gpu_reconst(tau, qiphR, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_mhL = gpu_reconst(tau, qimhL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_mhR = gpu_reconst(tau, qimhR, u_c, e_c, eos_P, eos_dPde, eos_p);

        float aiph = fmaxf(gpu_max_speed(tau, direction, r_phL, eos_P, eos_dPde, eos_p),
                           gpu_max_speed(tau, direction, r_phR, eos_P, eos_dPde, eos_p));
        float aimh = fmaxf(gpu_max_speed(tau, direction, r_mhL, eos_P, eos_dPde, eos_p),
                           gpu_max_speed(tau, direction, r_mhR, eos_P, eos_dPde, eos_p));

        float tf = tau_fac[direction];
        float dx = delta[direction];
        for (int alpha = 0; alpha < 5; alpha++) {
            float FiphL = gpu_get_TJb_reconst(r_phL, alpha, direction, tf, eos_P, eos_p);
            float FiphR = gpu_get_TJb_reconst(r_phR, alpha, direction, tf, eos_P, eos_p);
            float FimhL = gpu_get_TJb_reconst(r_mhL, alpha, direction, tf, eos_P, eos_p);
            float FimhR = gpu_get_TJb_reconst(r_mhR, alpha, direction, tf, eos_P, eos_p);
            float Fiph = 0.5f * ((FiphL + FiphR) - aiph * (qiphR[alpha] - qiphL[alpha]));
            float Fimh = 0.5f * ((FimhL + FimhR) - aimh * (qimhR[alpha] - qimhL[alpha]));
            if (direction == 3 && (alpha == 0 || alpha == 3)) {
                T_eta_m[alpha] = Fimh;
                T_eta_p[alpha] = Fiph;
            } else {
                rhs[alpha] += (Fimh - Fiph) / dx * params.delta_tau;
            }
        }
    }
    float cd = params.cosh_deta, sd = params.sinh_deta;
    rhs[0] += ((T_eta_m[0] - T_eta_p[0]) * cd - (T_eta_m[3] + T_eta_p[3]) * sd) * params.delta_tau;
    rhs[3] += ((T_eta_m[3] - T_eta_p[3]) * cd - (T_eta_m[0] + T_eta_p[0]) * sd) * params.delta_tau;

    float qi_full[5];
    for (int alpha = 0; alpha < 5; alpha++) qi_full[alpha] = qi[alpha] + rhs[alpha];

    // ── finalize_ideal: consume qi_full in place, no qi_out round-trip ────────
    const int   rkf      = params.rk_flag;
    const float dt       = params.delta_tau;
    const float tau_org  = params.tau_orig;
    const float tau_next = tau_org + dt;
    const float rk_norm  = 1.f / (1.f + (float)rkf);

    float u0p   = u_prev[c];
    float e_p   = epsilon_prev[c];
    float rho_p = rhob_prev[c];
    float P_p   = (rkf > 0) ? gpu_P(e_p, eos_P, eos_p) : 0.f;
    const int has_src = params.has_hydro_source;

    float qf[5];
    for (int a = 0; a < 5; a++) {
        float qv = qi_full[a] - dwmn_buf[a * Ncells + c] * dt;
        if (has_src) qv += qi_source_in[a * Ncells + c] * dt;
        if (rkf > 0) {
            float prev_TJb0;
            if (a == 4)      prev_TJb0 = rho_p * u0p;
            else if (a == 0) prev_TJb0 = (e_p + P_p) * u0p * u0p - P_p;
            else             prev_TJb0 = (e_p + P_p) * u_prev[a * Ncells + c] * u0p;
            qv += (float)rkf * tau_org * prev_TJb0;
        }
        qf[a] = qv * rk_norm;
    }

    ReconstResult r = gpu_reconst(tau_next, qf, u_c, e_c, eos_P, eos_dPde, eos_p);
    e_future[c]    = r.e;
    rhob_future[c] = r.rhob;
    for (int m = 0; m < 4; m++) u_future[m * Ncells + c] = r.u[m];
}

// ── gpu_make_uwrhs ───────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void apply_make_uwrhs(
    int ix, int iy, int ieta,
    const float* Wmunu_curr, const float* u_curr,
    float* uwrhs_out, const MUSICGridParams& params)
{
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta, Ncells = params.Ncells;
    const int c  = cell_idx(ix, iy, ieta, Nx, Ny);

    const float delta[4] = {0.f, params.delta_x, params.delta_y, params.delta_eta * params.tau};
    const float theta_l   = params.minmod_theta;
    const float delta_tau = params.delta_tau;

    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};
    const int IDX_1D[5] = {4, 5, 6, 7, 8};

    float u_c0 = u_curr[0 * Ncells + c];
    float u_cd[3];
    for (int d = 0; d < 3; d++) u_cd[d] = u_curr[(d + 1) * Ncells + c];

    for (int k = 0; k < 5; k++) {
        int idx_1d = IDX_1D[k];
        float flux = 0.f;

        for (int dir = 0; dir < 3; dir++) {
            int direction = dir + 1;

            int ip1 = clamped_cell(ix +   DX[dir], iy +   DY[dir], ieta +   DETA[dir], Nx, Ny, Neta);
            int ip2 = clamped_cell(ix + 2*DX[dir], iy + 2*DY[dir], ieta + 2*DETA[dir], Nx, Ny, Neta);
            int im1 = clamped_cell(ix -   DX[dir], iy -   DY[dir], ieta -   DETA[dir], Nx, Ny, Neta);
            int im2 = clamped_cell(ix - 2*DX[dir], iy - 2*DY[dir], ieta - 2*DETA[dir], Nx, Ny, Neta);

            float W_c  = Wmunu_curr[idx_1d * Ncells + c];
            float W_p1 = Wmunu_curr[idx_1d * Ncells + ip1];
            float W_p2 = Wmunu_curr[idx_1d * Ncells + ip2];
            float W_m1 = Wmunu_curr[idx_1d * Ncells + im1];
            float W_m2 = Wmunu_curr[idx_1d * Ncells + im2];

            float ud_c  = u_cd[dir];
            float u0_c  = u_c0;
            float ud_p1 = u_curr[direction * Ncells + ip1];
            float u0_p1 = u_curr[0         * Ncells + ip1];
            float ud_p2 = u_curr[direction * Ncells + ip2];
            float u0_p2 = u_curr[0         * Ncells + ip2];
            float ud_m1 = u_curr[direction * Ncells + im1];
            float u0_m1 = u_curr[0         * Ncells + im1];
            float ud_m2 = u_curr[direction * Ncells + im2];
            float u0_m2 = u_curr[0         * Ncells + im2];

            float f_c  = W_c  * ud_c;    float g_c  = W_c  * u0_c;
            float f_p1 = W_p1 * ud_p1;   float g_p1 = W_p1 * u0_p1;
            float f_p2 = W_p2 * ud_p2;   float g_p2 = W_p2 * u0_p2;
            float f_m1 = W_m1 * ud_m1;   float g_m1 = W_m1 * u0_m1;
            float f_m2 = W_m2 * ud_m2;   float g_m2 = W_m2 * u0_m2;

            float uWphR = f_p1 - 0.5f * gpu_minmod_dx(f_p2, f_p1, f_c , theta_l);
            float temp  = 0.5f * gpu_minmod_dx(f_p1, f_c , f_m1, theta_l);
            float uWphL = f_c  + temp;
            float uWmhR = f_c  - temp;
            float uWmhL = f_m1 + 0.5f * gpu_minmod_dx(f_c , f_m1, f_m2, theta_l);

            float WphR = g_p1 - 0.5f * gpu_minmod_dx(g_p2, g_p1, g_c , theta_l);
            float temp2 = 0.5f * gpu_minmod_dx(g_p1, g_c , g_m1, theta_l);
            float WphL = g_c  + temp2;
            float WmhR = g_c  - temp2;
            float WmhL = g_m1 + 0.5f * gpu_minmod_dx(g_c , g_m1, g_m2, theta_l);

            float a   = fabsf(ud_c ) / u0_c;
            float ap1 = fabsf(ud_p1) / u0_p1;
            float am1 = fabsf(ud_m1) / u0_m1;

            float ax;
            ax = fmaxf(a, ap1);
            float HWph = ((uWphR + uWphL) - ax * (WphR - WphL)) * 0.5f;
            ax = fmaxf(a, am1);
            float HWmh = ((uWmhR + uWmhL) - ax * (WmhR - WmhL)) * 0.5f;

            flux += -((HWph - HWmh) / delta[direction]);
        }

        uwrhs_out[k * Ncells + c] = flux * delta_tau;
    }
}

// ── gpu_make_uprhs ───────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void apply_make_uprhs(
    int ix, int iy, int ieta,
    const float* pi_b_curr, const float* u_curr,
    float* uprhs_out, const MUSICGridParams& params)
{
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta, Ncells = params.Ncells;
    const int c  = cell_idx(ix, iy, ieta, Nx, Ny);

    const float delta[4] = {0.f, params.delta_x, params.delta_y, params.delta_eta * params.tau};
    const float theta_l   = params.minmod_theta;
    const float delta_tau = params.delta_tau;

    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    float u_c0 = u_curr[0 * Ncells + c];
    float u_cd[3];
    for (int d = 0; d < 3; d++) u_cd[d] = u_curr[(d + 1) * Ncells + c];

    float flux = 0.f;

    for (int dir = 0; dir < 3; dir++) {
        int direction = dir + 1;

        int ip1 = clamped_cell(ix +   DX[dir], iy +   DY[dir], ieta +   DETA[dir], Nx, Ny, Neta);
        int ip2 = clamped_cell(ix + 2*DX[dir], iy + 2*DY[dir], ieta + 2*DETA[dir], Nx, Ny, Neta);
        int im1 = clamped_cell(ix -   DX[dir], iy -   DY[dir], ieta -   DETA[dir], Nx, Ny, Neta);
        int im2 = clamped_cell(ix - 2*DX[dir], iy - 2*DY[dir], ieta - 2*DETA[dir], Nx, Ny, Neta);

        float pi_c  = pi_b_curr[c];
        float pi_p1 = pi_b_curr[ip1];
        float pi_p2 = pi_b_curr[ip2];
        float pi_m1 = pi_b_curr[im1];
        float pi_m2 = pi_b_curr[im2];

        float ud_c  = u_cd[dir];
        float u0_c  = u_c0;
        float ud_p1 = u_curr[direction * Ncells + ip1];
        float u0_p1 = u_curr[0         * Ncells + ip1];
        float ud_p2 = u_curr[direction * Ncells + ip2];
        float u0_p2 = u_curr[0         * Ncells + ip2];
        float ud_m1 = u_curr[direction * Ncells + im1];
        float u0_m1 = u_curr[0         * Ncells + im1];
        float ud_m2 = u_curr[direction * Ncells + im2];
        float u0_m2 = u_curr[0         * Ncells + im2];

        float f_c  = pi_c  * ud_c;    float g_c  = pi_c  * u0_c;
        float f_p1 = pi_p1 * ud_p1;   float g_p1 = pi_p1 * u0_p1;
        float f_p2 = pi_p2 * ud_p2;   float g_p2 = pi_p2 * u0_p2;
        float f_m1 = pi_m1 * ud_m1;   float g_m1 = pi_m1 * u0_m1;
        float f_m2 = pi_m2 * ud_m2;   float g_m2 = pi_m2 * u0_m2;

        float uPiphR = f_p1 - 0.5f * gpu_minmod_dx(f_p2, f_p1, f_c , theta_l);
        float temp   = 0.5f * gpu_minmod_dx(f_p1, f_c , f_m1, theta_l);
        float uPiphL = f_c  + temp;
        float uPimhR = f_c  - temp;
        float uPimhL = f_m1 + 0.5f * gpu_minmod_dx(f_c , f_m1, f_m2, theta_l);

        float PiphR = g_p1 - 0.5f * gpu_minmod_dx(g_p2, g_p1, g_c , theta_l);
        float temp2 = 0.5f * gpu_minmod_dx(g_p1, g_c , g_m1, theta_l);
        float PiphL = g_c  + temp2;
        float PimhR = g_c  - temp2;
        float PimhL = g_m1 + 0.5f * gpu_minmod_dx(g_c , g_m1, g_m2, theta_l);

        float a   = fabsf(ud_c ) / u0_c;
        float ap1 = fabsf(ud_p1) / u0_p1;
        float am1 = fabsf(ud_m1) / u0_m1;

        float ax;
        ax = fmaxf(a, ap1);
        float HPiph = ((uPiphR + uPiphL) - ax * (PiphR - PiphL)) * 0.5f;
        ax = fmaxf(a, am1);
        float HPimh = ((uPimhR + uPimhL) - ax * (PimhR - PimhL)) * 0.5f;

        flux += -((HPiph - HPimh) / delta[direction]);
    }

    uprhs_out[c] = flux * delta_tau;
}

// ── gpu_make_du ──────────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void apply_make_du(
    int ix, int iy, int ieta,
    const float* u_curr, const float* u_prev,
    float* theta_out, float* a_out,
    float* sigma_out,
    const MUSICGridParams& params)
{
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta, Ncells = params.Ncells;
    const int c  = cell_idx(ix, iy, ieta, Nx, Ny);

    const float tau       = params.tau;
    const float delta_tau = params.delta_tau;
    const float theta_l   = params.minmod_theta;
    const float delta[4]  = {0.f, params.delta_x, params.delta_y, params.delta_eta * tau};

    float u[4];
    for (int m = 0; m < 4; m++) u[m] = u_curr[m * Ncells + c];

    float dUsup_local[4][4];
    for (int m = 0; m < 4; m++)
        for (int n = 0; n < 4; n++)
            dUsup_local[m][n] = 0.f;

    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    for (int dir = 0; dir < 3; dir++) {
        int direction = dir + 1;
        int ip1 = clamped_cell(ix + DX[dir], iy + DY[dir], ieta + DETA[dir], Nx, Ny, Neta);
        int im1 = clamped_cell(ix - DX[dir], iy - DY[dir], ieta - DETA[dir], Nx, Ny, Neta);
        for (int m = 1; m <= 3; m++) {
            float f   = u[m];
            float fp1 = u_curr[m * Ncells + ip1];
            float fm1 = u_curr[m * Ncells + im1];
            dUsup_local[m][direction] = gpu_minmod_dx(fp1, f, fm1, theta_l) / delta[direction];
        }
    }

    for (int n = 1; n <= 3; n++) {
        float f = 0.f;
        for (int m = 1; m <= 3; m++) f += dUsup_local[m][n] * u[m];
        dUsup_local[0][n] = f / u[0];
    }

    for (int m = 0; m < 4; m++) {
        float u_m_c = u[m];
        float u_m_p = u_prev[m * Ncells + c];
        dUsup_local[m][0] = -((u_m_c - u_m_p) / delta_tau);
    }
    {
        float f = 0.f;
        for (int m = 1; m < 4; m++) f += dUsup_local[m][0] * u[m];
        dUsup_local[0][0] = f / u[0];
    }

    float theta_cell = (-dUsup_local[0][0] + dUsup_local[1][1]
                        + dUsup_local[2][2] + dUsup_local[3][3] + u[0] / tau);
    theta_out[c] = theta_cell;

    float a_loc[4];
    for (int m = 0; m < 4; m++) {
        float s = 0.f;
        for (int n = 0; n < 4; n++) {
            float tfac = (n == 0) ? -1.f : 1.f;
            s += tfac * u[n] * dUsup_local[m][n];
        }
        a_loc[m] = s;
        a_out[m * Ncells + c] = s;
    }

    float sigma_local[4][4];
    for (int a = 1; a < 4; a++) {
        for (int b = a; b < 4; b++) {
            float gfac = (a == b) ? 1.f : 0.f;
            float g_a3 = (a == 3) ? 1.f : 0.f;
            float g_b3 = (b == 3) ? 1.f : 0.f;
            sigma_local[a][b] =
                  (dUsup_local[a][b] + dUsup_local[b][a]) * 0.5f
                - (gfac + u[a] * u[b]) * theta_cell / 3.f
                + u[0] / tau * g_a3 * g_b3
                + u[3] * u[0] / tau * 0.5f * (g_a3 * u[b] + g_b3 * u[a])
                + (u[a] * a_loc[b] + u[b] * a_loc[a]) * 0.5f;
            sigma_local[b][a] = sigma_local[a][b];
        }
    }

    sigma_local[3][3] = (
        ( 2.f * (  u[1] * u[2] * sigma_local[1][2]
                 + u[1] * u[3] * sigma_local[1][3]
                 + u[2] * u[3] * sigma_local[2][3])
         - (u[0]*u[0] - u[1]*u[1]) * sigma_local[1][1]
         - (u[0]*u[0] - u[2]*u[2]) * sigma_local[2][2])
        / fmaxf(u[0]*u[0] - u[3]*u[3], 1e-10f));

    for (int a = 1; a < 4; a++) {
        float s = 0.f;
        for (int b = 1; b < 4; b++) s += sigma_local[a][b] * u[b];
        sigma_local[0][a] = s / u[0];
    }
    {
        float s = 0.f;
        for (int a = 1; a < 4; a++) s += sigma_local[0][a] * u[a];
        sigma_local[0][0] = s / u[0];
    }

    sigma_out[0 * Ncells + c] = sigma_local[0][0];
    sigma_out[1 * Ncells + c] = sigma_local[0][1];
    sigma_out[2 * Ncells + c] = sigma_local[0][2];
    sigma_out[3 * Ncells + c] = sigma_local[0][3];
    sigma_out[4 * Ncells + c] = sigma_local[1][1];
    sigma_out[5 * Ncells + c] = sigma_local[1][2];
    sigma_out[6 * Ncells + c] = sigma_local[1][3];
    sigma_out[7 * Ncells + c] = sigma_local[2][2];
    sigma_out[8 * Ncells + c] = sigma_local[2][3];
    sigma_out[9 * Ncells + c] = sigma_local[3][3];
}

// ── gpu_first_rk_step_w_full ─────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void apply_first_rk_step_w_full(
    int ix, int iy, int ieta,
    const float* Wmunu_curr, const float* pi_b_curr,
    const float* u_curr,
    const float* Wmunu_prev, const float* pi_b_prev,
    const float* u_prev,
    const float* epsilon_curr, const float* epsilon_prev,
    const float* u_future,
    const float* uwrhs_in, const float* theta_in,
    const float* a_in, const float* sigma_in,
    float* Wmunu_future, float* pi_b_future,
    const float* eos_P, const float* eos_s,
    const float* eos_T, const float* eos_dPde,
    const float* uprhs_in, const MUSICGridParams& params,
    const GPUEosParams& eos_p,
    const float* epsilon_future, const float* rhob_future)
{
    const int Nx = params.Nx, Ny = params.Ny, Ncells = params.Ncells;
    const int c  = cell_idx(ix, iy, ieta, Nx, Ny);

    const int   rkf     = params.rk_flag;
    const float dt      = params.delta_tau;
    const float tau_now = params.tau;
    const float rk_norm = 1.f / (1.f + (float)rkf);

    if (params.turn_on_shear == 0) {
        for (int m = 0; m < 14; m++) Wmunu_future[m * Ncells + c] = 0.f;
        pi_b_future[c] = 0.f;
        return;
    }

    float u_c[4], u_p[4], u_f[4];
    for (int m = 0; m < 4; m++) {
        u_c[m] = u_curr  [m * Ncells + c];
        u_p[m] = u_prev  [m * Ncells + c];
        u_f[m] = u_future[m * Ncells + c];
    }

    float Wc[14], Wp_cell[14];
    for (int m = 0; m < 14; m++) {
        Wc[m]      = Wmunu_curr[m * Ncells + c];
        Wp_cell[m] = Wmunu_prev[m * Ncells + c];
    }

    float W4[4][4];
    W4[0][0]=Wc[0]; W4[0][1]=Wc[1]; W4[0][2]=Wc[2]; W4[0][3]=Wc[3];
    W4[1][0]=Wc[1]; W4[1][1]=Wc[4]; W4[1][2]=Wc[5]; W4[1][3]=Wc[6];
    W4[2][0]=Wc[2]; W4[2][1]=Wc[5]; W4[2][2]=Wc[7]; W4[2][3]=Wc[8];
    W4[3][0]=Wc[3]; W4[3][1]=Wc[6]; W4[3][2]=Wc[8]; W4[3][3]=Wc[9];

    float theta = theta_in[c];
    float a_loc[4];
    for (int m = 0; m < 4; m++) a_loc[m] = a_in[m * Ncells + c];

    float sigma_vec[10];
    for (int m = 0; m < 10; m++) sigma_vec[m] = sigma_in[m * Ncells + c];

    float s4[4][4];
    s4[0][0]=sigma_vec[0]; s4[0][1]=sigma_vec[1]; s4[0][2]=sigma_vec[2]; s4[0][3]=sigma_vec[3];
    s4[1][0]=sigma_vec[1]; s4[1][1]=sigma_vec[4]; s4[1][2]=sigma_vec[5]; s4[1][3]=sigma_vec[6];
    s4[2][0]=sigma_vec[2]; s4[2][1]=sigma_vec[5]; s4[2][2]=sigma_vec[7]; s4[2][3]=sigma_vec[8];
    s4[3][0]=sigma_vec[3]; s4[3][1]=sigma_vec[6]; s4[3][2]=sigma_vec[8]; s4[3][3]=sigma_vec[9];

    float eps_src = (rkf == 0) ? epsilon_curr[c] : epsilon_prev[c];

    float pi_c_curr = pi_b_curr[c];

    float Wsigma_scalar = 0.f;
    float Wsquare       = 0.f;
    if (params.include_second_order_terms == 1) {
        Wsigma_scalar =
              W4[0][0]*s4[0][0] + W4[1][1]*s4[1][1]
            + W4[2][2]*s4[2][2] + W4[3][3]*s4[3][3]
            - 2.f * (W4[0][1]*s4[0][1] + W4[0][2]*s4[0][2] + W4[0][3]*s4[0][3])
            + 2.f * (W4[1][2]*s4[1][2] + W4[1][3]*s4[1][3] + W4[2][3]*s4[2][3]);
        Wsquare =
              W4[0][0]*W4[0][0] + W4[1][1]*W4[1][1]
            + W4[2][2]*W4[2][2] + W4[3][3]*W4[3][3]
            - 2.f * (W4[0][1]*W4[0][1] + W4[0][2]*W4[0][2] + W4[0][3]*W4[0][3])
            + 2.f * (W4[1][2]*W4[1][2] + W4[1][3]*W4[1][3] + W4[2][3]*W4[2][3]);
    }

    float Wf[14];
    for (int m = 0; m < 14; m++) Wf[m] = 0.f;

    const int MU_LIST[5] = {1, 1, 1, 2, 2};
    const int NU_LIST[5] = {1, 2, 3, 2, 3};

    for (int k = 0; k < 5; k++) {
        int mu = MU_LIST[k];
        int nu = NU_LIST[k];
        int id = WIDX(mu, nu);

        float uwrhs_flux = uwrhs_in[k * Ncells + c];

        float SW = gpu_uW_source(mu, nu, W4, s4, u_c, pi_c_curr, theta, eps_src,
                                 Wsigma_scalar, Wsquare,
                                 eos_P, eos_s, eos_T, eos_p, params, dt);

        float w_rhs_geom = gpu_uWRHS_geom(W4, u_c, a_loc, mu, nu, theta, tau_now, dt);
        float w_rhs = uwrhs_flux + w_rhs_geom;

        float tempf =
              (1.f - (float)rkf) * (Wc[id] * u_c[0])
            +        (float)rkf  * (Wp_cell[id] * u_p[0]);
        tempf += SW * dt;
        tempf += w_rhs;
        tempf += (float)rkf * (Wc[id] * u_c[0]);
        tempf *= rk_norm;

        Wf[id] = tempf / u_f[0];
    }

    Wf[9] = ( 2.f * (  u_f[1]*u_f[2]*Wf[5]
                     + u_f[1]*u_f[3]*Wf[6]
                     + u_f[2]*u_f[3]*Wf[8])
              - (u_f[0]*u_f[0] - u_f[1]*u_f[1]) * Wf[4]
              - (u_f[0]*u_f[0] - u_f[2]*u_f[2]) * Wf[7])
            / fmaxf(u_f[0]*u_f[0] - u_f[3]*u_f[3], 1.e-10f);

    for (int mu = 1; mu < 4; mu++) {
        float s = 0.f;
        for (int nu = 1; nu < 4; nu++) s += Wf[WIDX(mu, nu)] * u_f[nu];
        Wf[mu] = s / u_f[0];
    }
    {
        float s = 0.f;
        for (int nu = 1; nu < 4; nu++) s += Wf[nu] * u_f[nu];
        Wf[0] = s / u_f[0];
    }

    for (int m = 10; m < 14; m++) Wf[m] = 0.f;

    float pi_b_out;
    if (params.turn_on_bulk == 1) {
        float pi_c = pi_b_curr[c];
        float pi_p = pi_b_prev[c];

        float p_rhs = uprhs_in[c] + (-(u_c[0] * pi_c) / tau_now + theta * pi_c) * dt;

        float SPi = gpu_uPi_source(pi_c, theta, eps_src, Wsigma_scalar,
                                   eos_P, eos_dPde, eos_T, eos_p, params, dt);

        float tempf =
              (1.f - (float)rkf) * (pi_c * u_c[0])
            +        (float)rkf  * (pi_p * u_p[0]);
        tempf += SPi * dt;
        tempf += p_rhs;
        tempf += (float)rkf * (pi_c * u_c[0]);
        tempf *= rk_norm;
        pi_b_out = tempf / u_f[0];
    } else {
        pi_b_out = 0.f;
    }

    if (params.do_quest_revert == 1) {
        const float eps_scale     = 0.1f;
        const float xi            = 0.05f;
        const float rho_shear_max = 0.1f;
        const float rho_bulk_max  = 0.1f;

        float e_local = epsilon_future[c];
        float sig_e   = 1.f / (expf(-(e_local - eps_scale) / xi) + 1.f);
        float sig_off = 1.f / (expf(eps_scale / xi)              + 1.f);
        float factor  = 10.f * params.quest_revert_strength * (sig_e - sig_off);

        float pi00 = Wf[0], pi01 = Wf[1], pi02 = Wf[2], pi03 = Wf[3];
        float pi11 = Wf[4], pi12 = Wf[5], pi13 = Wf[6];
        float pi22 = Wf[7], pi23 = Wf[8];
        float pi33 = Wf[9];
        float pisize =
              pi00*pi00 + pi11*pi11 + pi22*pi22 + pi33*pi33
            - 2.f * (pi01*pi01 + pi02*pi02 + pi03*pi03)
            + 2.f * (pi12*pi12 + pi13*pi13 + pi23*pi23);

        float bulksize = 3.f * pi_b_out * pi_b_out;

        float p_local  = gpu_P(e_local, eos_P, eos_p);
        float eq_size  = e_local * e_local + 3.f * p_local * p_local;
        eq_size        = fmaxf(eq_size, 1.e-30f);

        float rho_shear = sqrtf(fmaxf(pisize, 0.f) / eq_size) / factor;
        float rho_bulk  = sqrtf(fmaxf(bulksize, 0.f) / eq_size) / factor;

        if (Kokkos::isnan(rho_shear)) {
            for (int m = 0; m < 10; m++) Wf[m] = 0.f;
        } else if (rho_shear > rho_shear_max) {
            float scale = rho_shear_max / rho_shear;
            for (int m = 0; m < 10; m++) Wf[m] *= scale;
        }
        if (rho_bulk > rho_bulk_max) {
            pi_b_out *= rho_bulk_max / rho_bulk;
        }

        (void)rhob_future;
    }

    for (int m = 0; m < 14; m++) Wmunu_future[m * Ncells + c] = Wf[m];
    pi_b_future[c] = pi_b_out;
}

// ── gpu_pack_evolution_ideal (one work-item per down-sampled output cell) ─────
KOKKOS_INLINE_FUNCTION void apply_pack_evolution_ideal(
    int o,
    const float* epsilon, const float* u,
    const float* eos_P, const float* eos_s,
    const float* eos_T,
    float* out, const GPUEosParams& eos_p, const GPUPackParams& pp)
{
    const int ieta_idx = o % pp.neta_out;
    const int t        = o / pp.neta_out;
    const int iy_idx   = t % pp.ny_out;
    const int ix_idx   = t / pp.ny_out;

    const int ix   = ix_idx   * pp.skip_x;
    const int iy   = iy_idx   * pp.skip_y;
    const int ieta = ieta_idx * pp.skip_eta;

    const int Ncells = pp.Ncells;
    const int c      = ix + pp.Nx * (iy + pp.Ny * ieta);

    const float e = epsilon[c];
    const float p = gpu_log_interp(e, eos_P, eos_p);
    const float s = gpu_log_interp(e, eos_s, eos_p);
    const float T = gpu_log_interp(e, eos_T, eos_p);

    float eta = 0.f;
    if (!pp.boost_invariant)
        eta = (float)ieta * pp.delta_eta - pp.eta_size * 0.5f;

    float* cell = out + (size_t)o * 8;
    cell[0] = eta;
    cell[1] = s;
    cell[2] = e * pp.hbarc;
    cell[3] = p * pp.hbarc;
    cell[4] = T * pp.hbarc;
    cell[5] = u[1 * Ncells + c];
    cell[6] = u[2 * Ncells + c];
    cell[7] = u[3 * Ncells + c];
}

}  // namespace mkok
