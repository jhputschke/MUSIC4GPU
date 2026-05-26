// Metal Shading Language compute kernels for MUSIC relativistic hydrodynamics.
//
// Kernel: gpu_make_w_source
//   Computes the divergence of the viscous stress tensor (W^{mu nu}) and bulk
//   pressure (pi_b) that enters the energy-momentum source term.
//   This is a GPU port of Diss::MakeWSource() in dissipative.cpp.
//
//   For each cell (ix, iy, ieta) independently:
//     dwmn[alpha] = d_m(tau W^{m,alpha}) + geometric terms     for alpha=0..4
//
//   SoA layout (component-major):
//     field[comp * Ncells + cell]    where Ncells = Nx*Ny*Neta
//     cell  = Nx*(Ny*ieta + iy) + ix
//
// Kernel: gpu_first_rk_step_w
//   Updates Wmunu and pi_b for one Runge-Kutta sub-step (viscous sector).
//   GPU port of Advance::FirstRKStepW() in advance.cpp.

#include <metal_stdlib>
using namespace metal;

#include "gpu_types.h"   // MUSICGridParams, WMUNU_IDX, GMUNU_DIAG

// ── index helpers ─────────────────────────────────────────────────────────────

inline int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

// Boundary-clamped cell index (replicates Cell_small::getHalo)
inline int clamped_cell(int ix, int iy, int ieta,
                        int Nx, int Ny, int Neta) {
    ix   = clamp(ix,   0, Nx   - 1);
    iy   = clamp(iy,   0, Ny   - 1);
    ieta = clamp(ieta, 0, Neta - 1);
    return cell_idx(ix, iy, ieta, Nx, Ny);
}

// Read Wmunu[comp] for a cell (with boundary clamping)
inline float get_Wmunu(device const float* Wmunu, int comp,
                       int ix, int iy, int ieta,
                       int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return Wmunu[comp * Ncells + c];
}

inline float get_u(device const float* u, int comp,
                   int ix, int iy, int ieta,
                   int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return u[comp * Ncells + c];
}

inline float get_pi_b(device const float* pi_b,
                      int ix, int iy, int ieta,
                      int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return pi_b[c];
}

// ── gpu_make_w_source ─────────────────────────────────────────────────────────
//
// Matches Diss::MakeWSource() in dissipative.cpp.
// Buffer bindings must match MetalPipelines.mm dispatch_w_source():
//   0  Wmunu_curr [14*Ncells]
//   1  pi_b_curr  [Ncells]
//   2  u_curr     [4*Ncells]
//   3  Wmunu_prev [14*Ncells]
//   4  pi_b_prev  [Ncells]
//   5  u_prev     [4*Ncells]
//   6  dwmn_out   [5*Ncells]
//   7  params

kernel void gpu_make_w_source(
    device const float*       Wmunu_curr [[buffer(0)]],
    device const float*       pi_b_curr  [[buffer(1)]],
    device const float*       u_curr     [[buffer(2)]],
    device const float*       Wmunu_prev [[buffer(3)]],
    device const float*       pi_b_prev  [[buffer(4)]],
    device const float*       u_prev     [[buffer(5)]],
    device       float*       dwmn_out   [[buffer(6)]],
    constant MUSICGridParams& params     [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    int ix   = (int)gid.x;
    int iy   = (int)gid.y;
    int ieta = (int)gid.z;

    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, params.tau, params.tau, 1.f};

    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    // Preload current and previous cell values
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

    // Stencil offsets for 3 spatial directions
    // direction 1=x, 2=y, 3=eta
    const int dix  [3] = { 1, 0, 0};
    const int diy  [3] = { 0, 1, 0};
    const int dieta[3] = { 0, 0, 1};

    float W_eta_p[4] = {0.f, 0.f, 0.f, 0.f};
    float W_eta_m[4] = {0.f, 0.f, 0.f, 0.f};

    // Result accumulator
    float dwmn[5] = {0.f, 0.f, 0.f, 0.f, 0.f};

    for (int alpha = 0; alpha < 5; ++alpha) {
        int idx_alpha0 = WMUNU_IDX[alpha][0];   // W^{alpha, tau}

        // Time derivative: backward difference (first order)
        float dWdtau = (Wc[idx_alpha0] - Wp[idx_alpha0]) / params.delta_tau;

        // Bulk pressure time derivative
        float dPidtau  = 0.f;
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
            int direction = dir + 1;   // 1, 2, 3
            int idx_1d = WMUNU_IDX[alpha][direction];

            // Neighbor indices (clamped at boundary)
            int ixp = ix   + dix  [dir];
            int ixm = ix   - dix  [dir];
            int iyp = iy   + diy  [dir];
            int iym = iy   - diy  [dir];
            int iep = ieta + dieta[dir];
            int iem = ieta - dieta[dir];

            float tf = tau_fac[direction];
            float dx = delta  [direction];

            float sg   = Wc[idx_1d] * tf;
            float sgp1 = get_Wmunu(Wmunu_curr, idx_1d,
                                   ixp, iyp, iep, Nx, Ny, Neta, Ncells) * tf;
            float sgm1 = get_Wmunu(Wmunu_curr, idx_1d,
                                   ixm, iym, iem, Nx, Ny, Neta, Ncells) * tf;

            float W_m = (sg + sgm1) * 0.5f;
            float W_p = (sg + sgp1) * 0.5f;

            if (direction == 3 && (alpha == 0 || alpha == 3)) {
                // Save longitudinal flux for geometric term below
                W_eta_p[alpha] += W_p;
                W_eta_m[alpha] += W_m;
            } else {
                dWdx += (W_p - W_m) / dx;
            }

            if (alpha < 4 && params.turn_on_bulk) {
                float gfac1  = (alpha == direction) ? 1.f : 0.f;
                float bgp1   = get_pi_b(pi_b_curr, ixp, iyp, iep,
                                        Nx, Ny, Neta, Ncells)
                               * (gfac1 + get_u(u_curr, alpha, ixp, iyp, iep,
                                                Nx, Ny, Neta, Ncells)
                                        * get_u(u_curr, direction, ixp, iyp, iep,
                                                Nx, Ny, Neta, Ncells))
                               * tf;
                float bg     = pib_c * (gfac1 + uc[alpha] * uc[direction]) * tf;
                float bgm1   = get_pi_b(pi_b_curr, ixm, iym, iem,
                                        Nx, Ny, Neta, Ncells)
                               * (gfac1 + get_u(u_curr, alpha, ixm, iym, iem,
                                                Nx, Ny, Neta, Ncells)
                                        * get_u(u_curr, direction, ixm, iym, iem,
                                                Nx, Ny, Neta, Ncells))
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
        }  // dir loop

        float sf = params.tau * dWdtau + Wc[idx_alpha0] + dWdx;
        float bf = params.tau * dPidtau + Pi_alpha0 + dPidx;
        dwmn[alpha] += sf + bf;
    }  // alpha loop

    // Longitudinal geometric terms (cosh/sinh pre-computed in params)
    // dwmn[0] += (W_eta_p[0] - W_eta_m[0])*cosh_deta + (W_eta_p[3]+W_eta_m[3])*sinh_deta
    // dwmn[3] += (W_eta_p[3] - W_eta_m[3])*cosh_deta + (W_eta_p[0]+W_eta_m[0])*sinh_deta
    dwmn[0] += (  (W_eta_p[0] - W_eta_m[0]) * params.cosh_deta
               + (W_eta_p[3] + W_eta_m[3]) * params.sinh_deta);
    dwmn[3] += (  (W_eta_p[3] - W_eta_m[3]) * params.cosh_deta
               + (W_eta_p[0] + W_eta_m[0]) * params.sinh_deta);

    // Write output: dwmn[alpha * Ncells + c]
    for (int alpha = 0; alpha < 5; ++alpha)
        dwmn_out[alpha * Ncells + c] = dwmn[alpha];
}

// ── gpu_first_rk_step_w ───────────────────────────────────────────────────────
//
// Updates Wmunu and pi_b of the future grid for the viscous RK sub-step.
// GPU port of Advance::FirstRKStepW() in advance.cpp.
//
// Pre-conditions (computed on CPU and passed as buffers):
//   - arena_future.u is already set by the ideal step (cpu)
//   - theta_buf[c], a_buf[4*Ncells+c], sigma_buf[10*Ncells+c] computed by MakedU
//   - tcoeff_buf[4*Ncells+c] = {shear, tau_pi, bulk_tau_pi, bulk_zeta} per cell
//
// Buffer bindings (must match MetalPipelines dispatch_first_rk_step_w):
//   0  Wmunu_curr  [14*Ncells]
//   1  pi_b_curr   [Ncells]
//   2  u_curr      [4*Ncells]
//   3  Wmunu_prev  [14*Ncells]
//   4  pi_b_prev   [Ncells]
//   5  u_prev      [4*Ncells]
//   6  u_future    [4*Ncells]   (already set by ideal step)
//   7  Wmunu_future [14*Ncells] (output)
//   8  pi_b_future  [Ncells]    (output)
//   9  theta_buf   [Ncells]     expansion rate per cell
//  10  a_buf       [4*Ncells]   Du^mu per cell
//  11  sigma_buf   [10*Ncells]  velocity shear tensor per cell
//  12  tcoeff_buf  [4*Ncells]   transport coefficients per cell
//                               [0]=shear, [1]=tau_pi, [2]=bulk_zeta,
//                               [3]=bulk_tau_Pi
//  13  params

kernel void gpu_first_rk_step_w(
    device const float*       Wmunu_curr   [[buffer(0)]],
    device const float*       pi_b_curr    [[buffer(1)]],
    device const float*       u_curr       [[buffer(2)]],
    device const float*       Wmunu_prev   [[buffer(3)]],
    device const float*       pi_b_prev    [[buffer(4)]],
    device const float*       u_prev       [[buffer(5)]],
    device const float*       u_future     [[buffer(6)]],
    device       float*       Wmunu_future [[buffer(7)]],
    device       float*       pi_b_future  [[buffer(8)]],
    device const float*       theta_buf    [[buffer(9)]],
    device const float*       a_buf        [[buffer(10)]],
    device const float*       sigma_buf    [[buffer(11)]],
    device const float*       tcoeff_buf   [[buffer(12)]],
    constant MUSICGridParams& params       [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int ix   = (int)gid.x;
    int iy   = (int)gid.y;
    int ieta = (int)gid.z;

    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);

    // Load cell-local data
    float Wc[14], Wp_cell[14];
    float uc[4],  up_cell[4], uf[4];
    for (int m = 0; m < 14; ++m) { Wc[m] = Wmunu_curr[m*Ncells+c]; Wp_cell[m] = Wmunu_prev[m*Ncells+c]; }
    for (int m = 0; m < 4;  ++m) { uc[m] = u_curr[m*Ncells+c]; up_cell[m] = u_prev[m*Ncells+c]; uf[m] = u_future[m*Ncells+c]; }

    float pib_c = pi_b_curr[c];
    float pib_p = pi_b_prev[c];

    float theta    = theta_buf[c];
    float shear    = tcoeff_buf[0*Ncells + c];
    float tau_pi   = tcoeff_buf[1*Ncells + c];
    float bulk_zeta= tcoeff_buf[2*Ncells + c];
    float bulk_tau = tcoeff_buf[3*Ncells + c];

    // Load sigma[10] (upper-triangle of velocity shear tensor)
    float sigma[10];
    for (int m = 0; m < 10; ++m) sigma[m] = sigma_buf[m*Ncells + c];

    // Unpack sigma into 4x4 matrix
    float sig4[4][4];
    sig4[0][0]=sigma[0]; sig4[0][1]=sigma[1]; sig4[0][2]=sigma[2]; sig4[0][3]=sigma[3];
    sig4[1][0]=sigma[1]; sig4[1][1]=sigma[4]; sig4[1][2]=sigma[5]; sig4[1][3]=sigma[6];
    sig4[2][0]=sigma[2]; sig4[2][1]=sigma[5]; sig4[2][2]=sigma[7]; sig4[2][3]=sigma[8];
    sig4[3][0]=sigma[3]; sig4[3][1]=sigma[6]; sig4[3][2]=sigma[8]; sig4[3][3]=sigma[9];

    // Unpack Wmunu into 4x4 matrix
    float W4[4][4];
    W4[0][0]=Wc[0]; W4[0][1]=Wc[1]; W4[0][2]=Wc[2]; W4[0][3]=Wc[3];
    W4[1][0]=Wc[1]; W4[1][1]=Wc[4]; W4[1][2]=Wc[5]; W4[1][3]=Wc[6];
    W4[2][0]=Wc[2]; W4[2][1]=Wc[5]; W4[2][2]=Wc[7]; W4[2][3]=Wc[8];
    W4[3][0]=Wc[3]; W4[3][1]=Wc[6]; W4[3][2]=Wc[8]; W4[3][3]=Wc[9];

    const float dt  = params.delta_tau;
    const float tau = params.tau;

    // Make_uWSource for shear indices (idx_1d = 4..8 -> (mu,nu) pairs)
    // (1,1)->4, (1,2)->5, (1,3)->6, (2,2)->7, (2,3)->8
    const int MU_LIST[5] = {1, 1, 1, 2, 2};
    const int NU_LIST[5] = {1, 2, 3, 2, 3};

    // Stencil-based uW RHS: already computed by Make_uWRHS (CPU pre-pass)
    // and stored in a_buf[4*Ncells] as the directional Wmunu fluxes.
    // For now we compute only the source (relaxation) term here; the flux
    // (Make_uWRHS) part is added on CPU and passed via a separate buffer if
    // needed. In this first implementation the RHS buffer is zero.
    // TODO: port Make_uWRHS stencil here.

    float transport2 = 0.f;  // delta_pi_pi * tau_pi (simplification)
    float transport3 = 0.f;  // tau_pi_pi   * tau_pi

    // Update shear stress Wmunu for indices 4..8
    float Wf[14];
    for (int m = 0; m < 14; ++m) Wf[m] = 0.f;

    // Shear indices
    for (int k = 0; k < 5; ++k) {
        int mu     = MU_LIST[k];
        int nu     = NU_LIST[k];
        int idx_1d = WMUNU_IDX[mu][nu];

        // Source term S = -(1 + transport2*theta)*W^{mu nu}
        //               + 2*shear*sigma^{mu nu}
        float src = -(1.f + transport2 * theta) * W4[mu][nu]
                    - 2.f * shear * sig4[mu][nu];

        float tempf = (Wc[idx_1d] * uc[0])
                    + src * dt;
        tempf += (float)params.boost_invariant * 0.f; // placeholder for uWRHS

        Wf[idx_1d] = tempf / uf[0];
    }

    // Bulk pressure (pi_b)
    if (params.turn_on_bulk) {
        // Source: -(pi_b + bulk_zeta*theta) / bulk_tau_Pi
        float bulk_src = -(pib_c + bulk_zeta * theta) / fmax(bulk_tau, 1e-6f);
        float tempf = pib_c * uc[0] + bulk_src * dt;
        pi_b_future[c] = tempf / uf[0];
    } else {
        pi_b_future[c] = 0.f;
    }

    // Re-make Wmunu[3][3] so it is traceless (transversality constraint)
    Wf[9] = ( 2.f*(  uf[1]*uf[2]*Wf[5]
                    + uf[1]*uf[3]*Wf[6]
                    + uf[2]*uf[3]*Wf[8])
              - (uf[0]*uf[0] - uf[1]*uf[1]) * Wf[4]
              - (uf[0]*uf[0] - uf[2]*uf[2]) * Wf[7])
            / fmax(uf[0]*uf[0] - uf[3]*uf[3], 1e-10f);

    // Transversality: Wmunu[mu][0] = sum_nu Wmunu[mu][nu]*u[nu] / u[0]
    for (int mu = 1; mu < 4; ++mu) {
        float sum = 0.f;
        for (int nu = 1; nu < 4; ++nu)
            sum += Wf[WMUNU_IDX[mu][nu]] * uf[nu];
        Wf[mu] = sum / uf[0];
    }
    // Wmunu[0][0]
    float sum00 = 0.f;
    for (int nu = 1; nu < 4; ++nu) sum00 += Wf[nu] * uf[nu];
    Wf[0] = sum00 / uf[0];

    // Zero out diffusion components (handled on CPU)
    for (int m = 10; m < 14; ++m) Wf[m] = 0.f;

    // Write output
    for (int m = 0; m < 14; ++m)
        Wmunu_future[m * Ncells + c] = Wf[m];
}

// ── gpu_make_delta_qi ─────────────────────────────────────────────────────────
//
// Ideal KT flux kernel: GPU port of Advance::MakeDeltaQI() in advance.cpp.
//
// For each cell, computes:
//   qi[alpha] = tau * T^{alpha,0}(c)
//             + sum_dir KT_flux_divergence[alpha,dir] * delta_tau
//             + longitudinal_geometric_terms * delta_tau
//
// EOS requirement (Tier 2): uses pre-sampled P(e) and dP/de(e) tables at
// rhob=0 (stored in eos_P / eos_dPde buffers).  This covers the standard
// zero-net-baryon case.  For finite-muB EOS the CPU path is used as fallback.
//
// Buffer bindings (must match MetalPipelines::dispatch_delta_qi):
//   0  epsilon_curr [Ncells]
//   1  rhob_curr    [Ncells]
//   2  u_curr       [4*Ncells]
//   3  eos_P        [GPU_EOS_N]   pressure table, rhob=0
//   4  eos_dPde     [GPU_EOS_N]   dP/de  table, rhob=0
//   5  qi_out       [5*Ncells]    output
//   6  params
//   7  eos_p

// ── EOS table helpers ─────────────────────────────────────────────────────────

inline float gpu_eos_interp(device const float* table, float e,
                             constant GPUEosParams& ep) {
    e = clamp(e, ep.e_min, ep.e_max);
    float fe  = (e - ep.e_min) / ep.delta_e;
    int   idx = min((int)fe, ep.n_pts - 2);
    idx = max(0, idx);
    float frac = fe - (float)idx;
    return table[idx] * (1.f - frac) + table[idx + 1] * frac;
}

inline float gpu_P(float e, device const float* P_tab,
                   constant GPUEosParams& ep) {
    return max(1.e-20f, gpu_eos_interp(P_tab, e, ep));
}

inline float gpu_dPde(float e, device const float* dPde_tab,
                      constant GPUEosParams& ep) {
    return gpu_eos_interp(dPde_tab, e, ep);
}

// Speed of sound squared, clamped to physical range [0.01, 1/3].
// dP/drhob = 0 assumed (rhob=0 EOS table).
inline float gpu_cs2(float e, device const float* P_tab,
                     device const float* dPde_tab, constant GPUEosParams& ep) {
    return clamp(gpu_dPde(e, dPde_tab, ep), 0.01f, 0.333333f);
}

// ── minmod slope limiter ──────────────────────────────────────────────────────

inline float gpu_minmod_dx(float up1, float u, float um1, float theta) {
    float diffup   = (up1 - u)   * theta;
    float diffdown = (u   - um1) * theta;
    float diffmid  = (up1 - um1) * 0.5f;
    if (diffup == 0.f) return 0.f;
    return diffup * max(0.f, min(1.f, min(diffdown / diffup, diffmid / diffup)));
}

// ── T^{alpha,0} from SoA cell ─────────────────────────────────────────────────

// Returns T^{alpha,0}(cell c) * tau, using stored epsilon/rhob/u.
inline float gpu_TJb0(int alpha, int c, int Ncells,
                      device const float* eps_buf,
                      device const float* rhob_buf,
                      device const float* u_buf,
                      device const float* P_tab,
                      constant GPUEosParams& ep) {
    float u0 = u_buf[0 * Ncells + c];
    if (alpha == 4) return rhob_buf[c] * u0;
    float e = eps_buf[c];
    float P = gpu_P(e, P_tab, ep);
    if (alpha == 0) return (e + P) * u0 * u0 - P;
    return (e + P) * u_buf[alpha * Ncells + c] * u0;
}

// ── Newton-Brent reconstruction helpers ──────────────────────────────────────

struct ReconstResult {
    float e;
    float rhob;
    float u[4];
};

// f(v) and df/dv for the velocity Newton solve.
// Assumes dP/drhob = 0 (rhob=0 EOS table).
inline void gpu_vel_fdf(float v, float T00, float M, float J0,
                        thread float& fv, thread float& dfdv,
                        device const float* P_tab,
                        device const float* dPde_tab,
                        constant GPUEosParams& ep) {
    float eps  = T00 - v * M;
    // For the 1D (rhob=0) EOS, dP/drhob=0 so the J0 / rho terms vanish.
    float P    = gpu_P(eps, P_tab, ep);
    float dPde = gpu_dPde(eps, dPde_tab, ep);
    float t1   = T00 + P;
    fv   = v - M / t1;
    dfdv = 1.f - M * M * dPde / (t1 * t1);
    (void)J0;
}

// Hybrid Newton-Brent root-finder for the velocity.
inline float gpu_solve_v(float v_guess, float T00, float M, float J0,
                          device const float* P_tab,
                          device const float* dPde_tab,
                          constant GPUEosParams& ep) {
    const float ABS_ERR = 1.e-7f;
    float fv_l, dfdv_l, fv_h, dfdv_h;
    float v_l = 0.f, v_h = 1.f;
    gpu_vel_fdf(v_l, T00, M, J0, fv_l, dfdv_l, P_tab, dPde_tab, ep);
    gpu_vel_fdf(v_h, T00, M, J0, fv_h, dfdv_h, P_tab, dPde_tab, ep);

    if (abs(fv_l) < ABS_ERR) return v_l;
    if (abs(fv_h) < ABS_ERR) return v_h;
    if (fv_l * fv_h > 0.f) return 0.f;

    float dv_prev = v_h - v_l;
    float dv_curr = dv_prev;
    float v_root  = (v_h + v_l) * 0.5f;
    float fv, dfdv;
    gpu_vel_fdf(v_root, T00, M, J0, fv, dfdv, P_tab, dPde_tab, ep);

    for (int it = 0; it < 60; it++) {
        if (((v_root - v_h) * dfdv - fv) * ((v_root - v_l) * dfdv - fv) > 0.f
            || abs(2.f * fv) > abs(dv_prev * dfdv)) {
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
        if (abs(dv_curr) < ABS_ERR) break;
    }
    return v_root;
}

// f(u0) and df/du0 for the high-velocity Newton solve.
inline void gpu_u0_fdf(float u0, float T00, float K00, float M, float J0,
                        thread float& fu0, thread float& dfdu0,
                        device const float* P_tab,
                        device const float* dPde_tab,
                        constant GPUEosParams& ep) {
    const float ABS_ERR = 1.e-10f;
    float v       = sqrt(max(0.f, 1.f - 1.f / (u0 * u0)));
    float epsilon = T00 - v * M;
    float dedu0   = -M / (u0 * u0 * u0 * v + ABS_ERR);
    float P       = gpu_P(epsilon, P_tab, ep);
    float dPde    = gpu_dPde(epsilon, dPde_tab, ep);
    float temp1   = (T00 + P) * (T00 + P) - K00;
    float den1    = sqrt(max(0.f, temp1));
    float temp    = (T00 + P) / max(den1, ABS_ERR);
    fu0    = u0 - temp;
    dfdu0  = 1.f + dedu0 * dPde * K00 / max(temp1 * den1, ABS_ERR);
}

inline float gpu_solve_u0(float u0_guess, float T00, float K00, float M, float J0,
                           device const float* P_tab,
                           device const float* dPde_tab,
                           constant GPUEosParams& ep) {
    const float ABS_ERR = 1.e-7f;
    float u0_l = max(1.f, 0.5f * u0_guess);
    float u0_h = min(1.e4f, 1.5f * u0_guess);
    if (u0_h < 1.f + ABS_ERR) u0_h = 2.f;

    float fu0_l, dfdu0_l, fu0_h, dfdu0_h;
    gpu_u0_fdf(u0_l, T00, K00, M, J0, fu0_l, dfdu0_l, P_tab, dPde_tab, ep);
    gpu_u0_fdf(u0_h, T00, K00, M, J0, fu0_h, dfdu0_h, P_tab, dPde_tab, ep);

    if (abs(fu0_l) < ABS_ERR) return u0_l;
    if (abs(fu0_h) < ABS_ERR) return u0_h;
    if (fu0_l * fu0_h > 0.f)  return u0_guess;  // no bracket; return guess

    float du0_prev = u0_h - u0_l;
    float du0_curr = du0_prev;
    float u0_root  = (u0_h + u0_l) * 0.5f;
    float fu0, dfdu0;
    gpu_u0_fdf(u0_root, T00, K00, M, J0, fu0, dfdu0, P_tab, dPde_tab, ep);

    for (int it = 0; it < 60; it++) {
        if (((u0_root - u0_h) * dfdu0 - fu0) * ((u0_root - u0_l) * dfdu0 - fu0) > 0.f
            || abs(2.f * fu0) > abs(du0_prev * dfdu0)) {
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
        if (abs(du0_curr) < ABS_ERR) break;
    }
    return u0_root;
}

// ── Main reconstruction (matches Reconst::ReconstIt_shell on CPU) ─────────────

// tauq[5] = tau * {T^{00}, T^{10}, T^{20}, T^{30}, J^0} at a half-interface.
// prev_u[4], prev_eps: full center-cell 4-velocity and energy used for Newton
// initial guess and for the revert fallback (matches revert_grid on CPU).
ReconstResult gpu_reconst(float tau, float tauq[5],
                           thread const float prev_u[4], float prev_eps,
                           device const float* P_tab,
                           device const float* dPde_tab,
                           constant GPUEosParams& ep) {
    const float ABS_ERR = 1.e-8f;

    ReconstResult res;
    res.rhob   = 0.f;
    res.u[0]   = 1.f;  res.u[1] = 0.f;  res.u[2] = 0.f;  res.u[3] = 0.f;

    // Convert tau*q → q (matching ReconstIt_shell)
    float q[5];
    for (int i = 0; i < 5; i++) q[i] = tauq[i] / tau;

    float K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    float M   = sqrt(K00);
    float T00 = q[0];
    float J0  = q[4];

    // Low energy: regulate (return small e, u=rest frame)
    if (T00 < ABS_ERR) {
        res.e = ABS_ERR;
        return res;
    }
    // Can't invert: revert to previous cell state (full 4-velocity, matches revert_grid)
    if (T00 < M) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];
        res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];
        res.u[3] = prev_u[3];
        return res;
    }

    float v_guess = sqrt(max(0.f, 1.f - 1.f / (prev_u[0] * prev_u[0] + ABS_ERR)));
    float v_sol   = gpu_solve_v(v_guess, T00, M, J0, P_tab, dPde_tab, ep);

    float u0      = 1.f / (sqrt(max(0.f, 1.f - v_sol * v_sol)) + v_sol * ABS_ERR);
    float epsilon = T00 - v_sol * M;
    float rhob    = J0 / u0;

    // High-velocity branch (v > 0.563624)
    if (v_sol > 0.563624f) {
        float u0_sol = gpu_solve_u0(u0, T00, K00, M, J0, P_tab, dPde_tab, ep);
        if (u0_sol >= 1.f) {
            u0      = u0_sol;
            epsilon = T00 - sqrt(max(0.f, (1.f - 1.f / (u0 * u0)) * K00));
            rhob    = J0 / u0;
        }
    }

    res.e    = epsilon;
    res.rhob = rhob;

    float P       = gpu_P(epsilon, P_tab, ep);
    float vel_inv = u0 / (T00 + P);
    res.u[0]  = u0;
    res.u[1]  = q[1] * vel_inv;
    res.u[2]  = q[2] * vel_inv;
    res.u[3]  = q[3] * vel_inv;

    // Enforce unit-norm: u^mu u_mu = -1 (metric -,+,+,+)
    float u_sp_sq = res.u[1]*res.u[1] + res.u[2]*res.u[2] + res.u[3]*res.u[3];
    if (abs(u0*u0 - u_sp_sq - 1.f) > ABS_ERR) {
        float scale = sqrt(max(0.f, (u0*u0 - 1.f) / (u_sp_sq + ABS_ERR)));
        res.u[1] *= scale;
        res.u[2] *= scale;
        res.u[3] *= scale;
    }
    return res;
}

// ── MaxSpeed (matches Advance::MaxSpeed on CPU) ───────────────────────────────

inline float gpu_max_speed(float tau, int direction, ReconstResult r,
                            device const float* P_tab,
                            device const float* dPde_tab,
                            constant GPUEosParams& ep) {
    // g-factor: g[dir-1] = {1, 1, 1/tau} for direction {1,2,3}
    float gfac = (direction == 3) ? 1.f / tau : 1.f;

    float utau    = r.u[0];
    float ux      = abs(r.u[direction]);
    float utau2   = utau * utau;
    float ut2mux2 = utau2 - ux * ux;

    float cs2    = gpu_cs2(r.e, P_tab, dPde_tab, ep);
    float num_sq = (ut2mux2 - (ut2mux2 - 1.f) * cs2) * cs2;
    float num;
    if (num_sq >= 0.f) {
        num = utau * ux * (1.f - cs2) + sqrt(num_sq);
    } else {
        float dPde = gpu_dPde(r.e, dPde_tab, ep);
        float P    = gpu_P(r.e, P_tab, ep);
        float h    = P + r.e;
        num = (dPde < 0.001f)
              ? sqrt(max(0.f, -(h*dPde*h*(dPde*(-1.f + ut2mux2) - ut2mux2))))
                - h*(-1.f + dPde)*utau*ux
              : 1.f;   // fallback for unphysical case
    }
    float den = utau2 * (1.f - cs2) + cs2;
    float f   = num / max(den, 1.e-20f);
    f = clamp(f, ux / utau, 1.f);
    return f * gfac;
}

// ── T^{alpha,direction} from a ReconstResult ─────────────────────────────────

// Returns T^{alpha,nu} * tau_fac where tau_fac = {0,tau,tau,1}[nu].
// Matches Advance::get_TJb(ReconstCell, 0, alpha, nu).
inline float gpu_get_TJb_reconst(ReconstResult r, int alpha, int direction,
                                  float tau_fac,
                                  device const float* P_tab,
                                  constant GPUEosParams& ep) {
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

// ── gpu_make_delta_qi kernel ──────────────────────────────────────────────────

kernel void gpu_make_delta_qi(
    device const float*       epsilon_curr [[buffer(0)]],
    device const float*       rhob_curr    [[buffer(1)]],
    device const float*       u_curr       [[buffer(2)]],
    device const float*       eos_P        [[buffer(3)]],
    device const float*       eos_dPde     [[buffer(4)]],
    device       float*       qi_out       [[buffer(5)]],
    constant MUSICGridParams& params       [[buffer(6)]],
    constant GPUEosParams&    eos_p        [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    int ix   = (int)gid.x;
    int iy   = (int)gid.y;
    int ieta = (int)gid.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;
    const float tau  = params.tau;
    const float theta = params.minmod_theta;

    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    // Center-cell values (used for Newton initial guess and revert fallback)
    float e_c    = epsilon_curr[c];
    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];

    // qi[alpha] = tau * T^{alpha,0}(c)
    float qi[5];
    for (int alpha = 0; alpha < 5; alpha++)
        qi[alpha] = tau * gpu_TJb0(alpha, c, Ncells,
                                   epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, tau, tau, 1.f};

    float rhs[5]     = {0.f, 0.f, 0.f, 0.f, 0.f};
    float T_eta_m[4] = {0.f, 0.f, 0.f, 0.f};
    float T_eta_p[4] = {0.f, 0.f, 0.f, 0.f};

    // Stencil offsets: dir=0→x, dir=1→y, dir=2→eta
    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    for (int dir = 0; dir < 3; dir++) {
        int direction = dir + 1;   // 1, 2, or 3

        int ip1 = clamped_cell(ix+  DX[dir], iy+  DY[dir], ieta+  DETA[dir], Nx, Ny, Neta);
        int ip2 = clamped_cell(ix+2*DX[dir], iy+2*DY[dir], ieta+2*DETA[dir], Nx, Ny, Neta);
        int im1 = clamped_cell(ix-  DX[dir], iy-  DY[dir], ieta-  DETA[dir], Nx, Ny, Neta);
        int im2 = clamped_cell(ix-2*DX[dir], iy-2*DY[dir], ieta-2*DETA[dir], Nx, Ny, Neta);

        // Build minmod-limited half-state conserved vectors
        float qiphL[5], qiphR[5], qimhL[5], qimhR[5];
        for (int alpha = 0; alpha < 5; alpha++) {
            float gc  = qi[alpha];   // tau * T^{alpha,0}(c) — already computed
            float gp1 = tau * gpu_TJb0(alpha, ip1, Ncells,
                                       epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gp2 = tau * gpu_TJb0(alpha, ip2, Ncells,
                                       epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gm1 = tau * gpu_TJb0(alpha, im1, Ncells,
                                       epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);
            float gm2 = tau * gpu_TJb0(alpha, im2, Ncells,
                                       epsilon_curr, rhob_curr, u_curr, eos_P, eos_p);

            float fphL =  0.5f * gpu_minmod_dx(gp1, gc,  gm1, theta);
            float fphR = -0.5f * gpu_minmod_dx(gp2, gp1, gc,  theta);
            float fmhL =  0.5f * gpu_minmod_dx(gc,  gm1, gm2, theta);
            float fmhR = -fphL;   // symmetric: -(1/2)*minmod_dx(gp1, gc, gm1)

            qiphL[alpha] = gc  + fphL;
            qiphR[alpha] = gp1 + fphR;
            qimhL[alpha] = gm1 + fmhL;
            qimhR[alpha] = gc  + fmhR;
        }

        // Reconstruct primitive variables at all four half-interfaces
        ReconstResult r_phL = gpu_reconst(tau, qiphL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_phR = gpu_reconst(tau, qiphR, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_mhL = gpu_reconst(tau, qimhL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_mhR = gpu_reconst(tau, qimhR, u_c, e_c, eos_P, eos_dPde, eos_p);

        // Local propagation speeds (KT upwinding)
        float aiph_L = gpu_max_speed(tau, direction, r_phL, eos_P, eos_dPde, eos_p);
        float aiph_R = gpu_max_speed(tau, direction, r_phR, eos_P, eos_dPde, eos_p);
        float aimh_L = gpu_max_speed(tau, direction, r_mhL, eos_P, eos_dPde, eos_p);
        float aimh_R = gpu_max_speed(tau, direction, r_mhR, eos_P, eos_dPde, eos_p);
        float aiph   = max(aiph_L, aiph_R);
        float aimh   = max(aimh_L, aimh_R);

        float tf = tau_fac[direction];
        float dx = delta[direction];

        // KT numerical flux H_{j+1/2} = (F^+ + F^-)/2 - a*(u^+ - u^-)/2
        for (int alpha = 0; alpha < 5; alpha++) {
            float FiphL = gpu_get_TJb_reconst(r_phL, alpha, direction, tf, eos_P, eos_p);
            float FiphR = gpu_get_TJb_reconst(r_phR, alpha, direction, tf, eos_P, eos_p);
            float FimhL = gpu_get_TJb_reconst(r_mhL, alpha, direction, tf, eos_P, eos_p);
            float FimhR = gpu_get_TJb_reconst(r_mhR, alpha, direction, tf, eos_P, eos_p);

            float Fiph = 0.5f * ((FiphL + FiphR) - aiph * (qiphR[alpha] - qiphL[alpha]));
            float Fimh = 0.5f * ((FimhL + FimhR) - aimh * (qimhR[alpha] - qimhL[alpha]));

            // Longitudinal direction: save for geometric terms below
            if (direction == 3 && (alpha == 0 || alpha == 3)) {
                T_eta_m[alpha] = Fimh;
                T_eta_p[alpha] = Fiph;
            } else {
                rhs[alpha] += (Fimh - Fiph) / dx * params.delta_tau;
            }
        }
    }  // dir loop

    // Longitudinal geometric terms (boost-invariant or full 3+1)
    float cd = params.cosh_deta;
    float sd = params.sinh_deta;
    rhs[0] += (  (T_eta_m[0] - T_eta_p[0]) * cd
               - (T_eta_m[3] + T_eta_p[3]) * sd) * params.delta_tau;
    rhs[3] += (  (T_eta_m[3] - T_eta_p[3]) * cd
               - (T_eta_m[0] + T_eta_p[0]) * sd) * params.delta_tau;

    // Write qi + rhs
    for (int alpha = 0; alpha < 5; alpha++)
        qi_out[alpha * Ncells + c] = qi[alpha] + rhs[alpha];
}

// ── gpu_finalize_ideal ────────────────────────────────────────────────────────
//
// Performs the final per-cell ideal RK update on the GPU, replacing the CPU
// loop body in Advance::FirstRKStepT().  For each cell:
//   1. qi[alpha] = qi_buf[alpha]                              // (from gpu_make_delta_qi)
//                  - dwmn_buf[alpha] * delta_tau              // (from gpu_make_w_source)
//                  + rk_flag * tau_orig * T^{alpha,0}(prev)   // RK mixing term
//   2. qi[alpha] *= 1 / (1 + rk_flag)
//   3. Reconst at tau_next = tau_orig + delta_tau using the same Newton-Brent
//      solve already used inside gpu_make_delta_qi (gpu_reconst()).
//   4. Write results into snap_future.{epsilon, rhob, u}.
//
// Hydro source terms (when flag_add_hydro_source is true on the CPU side)
// are NOT supported here — the CPU code path is used as a fallback in that
// case, since per-cell source evaluation depends on host-only state.
//
// Buffer bindings (must match MetalPipelines::dispatch_finalize_ideal):
//   0  qi_buf        [5*Ncells]   tau_rk * T^{a,0}(c) + KT flux update
//   1  dwmn_buf      [5*Ncells]   viscous-source divergence
//   2  epsilon_curr  [Ncells]     (Newton guess + revert fallback)
//   3  u_curr        [4*Ncells]
//   4  epsilon_prev  [Ncells]     RK mixing: T^{alpha,0}(prev)
//   5  rhob_prev     [Ncells]
//   6  u_prev        [4*Ncells]
//   7  e_future      [Ncells]     output
//   8  rhob_future   [Ncells]     output
//   9  u_future      [4*Ncells]   output
//  10  eos_P         [GPU_EOS_N]
//  11  eos_dPde      [GPU_EOS_N]
//  12  params        (constant struct)
//  13  eos_p         (constant struct)

kernel void gpu_finalize_ideal(
    device const float*       qi_buf        [[buffer(0)]],
    device const float*       dwmn_buf      [[buffer(1)]],
    device const float*       epsilon_curr  [[buffer(2)]],
    device const float*       u_curr        [[buffer(3)]],
    device const float*       epsilon_prev  [[buffer(4)]],
    device const float*       rhob_prev     [[buffer(5)]],
    device const float*       u_prev        [[buffer(6)]],
    device       float*       e_future      [[buffer(7)]],
    device       float*       rhob_future   [[buffer(8)]],
    device       float*       u_future      [[buffer(9)]],
    device const float*       eos_P         [[buffer(10)]],
    device const float*       eos_dPde      [[buffer(11)]],
    constant MUSICGridParams& params        [[buffer(12)]],
    constant GPUEosParams&    eos_p         [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int ix   = (int)gid.x;
    int iy   = (int)gid.y;
    int ieta = (int)gid.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);

    const int   rkf      = params.rk_flag;
    const float dt       = params.delta_tau;
    const float tau_org  = params.tau_orig;
    const float tau_next = tau_org + dt;
    const float rk_norm  = 1.f / (1.f + (float)rkf);

    // Pre-load prev cell for the RK mixing term (only used when rkf > 0).
    float u0p   = u_prev[c];                 // u_prev[0*Ncells + c]
    float e_p   = epsilon_prev[c];
    float rho_p = rhob_prev[c];
    float P_p   = (rkf > 0) ? gpu_P(e_p, eos_P, eos_p) : 0.f;

    float qi[5];
    for (int a = 0; a < 5; a++) {
        float qv = qi_buf  [a * Ncells + c]
                 - dwmn_buf[a * Ncells + c] * dt;

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

    // Current cell primitives — used as Newton initial guess and revert fallback.
    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];
    float e_c = epsilon_curr[c];

    ReconstResult r = gpu_reconst(tau_next, qi, u_c, e_c, eos_P, eos_dPde, eos_p);

    e_future   [c] = r.e;
    rhob_future[c] = r.rhob;
    for (int m = 0; m < 4; m++)
        u_future[m * Ncells + c] = r.u[m];
}

// ── gpu_make_uwrhs ───────────────────────────────────────────────────────────
//
// Computes the Kurganov–Tadmor flux divergence of (u^a W^{mu nu}) used by
// Diss::Make_uWRHS() in dissipative.cpp.  This is the stencil portion only —
// the per-cell algebraic / geometric tail (which depends on theta and Du^mu
// from MakedU) stays on the CPU since those quantities are not yet on the GPU.
//
// For each cell (ix, iy, ieta) and each of the 5 shear indices
//   idx_1d ∈ {4, 5, 6, 7, 8}   (= W^{11}, W^{12}, W^{13}, W^{22}, W^{23})
// the output is:
//   uwrhs_flux[out_idx * Ncells + c] = -delta_tau * Σ_{direction} HW_{div,dir}
// where
//   HW_{div,dir} = (HW_{j+1/2} - HW_{j-1/2}) / delta[direction]
// and HW_{j±1/2} = KT half-cell flux with minmod-limited left/right states.
//
// out_idx layout: 0..4 corresponds to idx_1d 4..8.
//
// Note: delta[eta] in this kernel uses params.delta_eta * params.tau
// (the proper length), matching the CPU Make_uWRHS convention.
//
// Buffer bindings (must match MetalPipelines::dispatch_uwrhs):
//   0  Wmunu_curr   [14*Ncells]
//   1  u_curr       [4*Ncells]
//   2  uwrhs_out    [5*Ncells]   output
//   3  params       (constant struct)

kernel void gpu_make_uwrhs(
    device const float*       Wmunu_curr   [[buffer(0)]],
    device const float*       u_curr       [[buffer(1)]],
    device       float*       uwrhs_out    [[buffer(2)]],
    constant MUSICGridParams& params       [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    int ix   = (int)gid.x;
    int iy   = (int)gid.y;
    int ieta = (int)gid.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);

    const float delta[4] = {0.f,
                            params.delta_x,
                            params.delta_y,
                            params.delta_eta * params.tau};
    const float theta_l   = params.minmod_theta;
    const float delta_tau = params.delta_tau;

    // Stencil offsets in (x, y, eta) for the 3 directions
    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    // 5 shear indices: (1,1)=4, (1,2)=5, (1,3)=6, (2,2)=7, (2,3)=8
    const int IDX_1D[5] = {4, 5, 6, 7, 8};

    // Preload center-cell u
    float u_c0 = u_curr[0 * Ncells + c];
    float u_cd[3];   // u[1], u[2], u[3]
    for (int d = 0; d < 3; d++) u_cd[d] = u_curr[(d + 1) * Ncells + c];

    for (int k = 0; k < 5; k++) {
        int idx_1d = IDX_1D[k];
        float flux = 0.f;

        for (int dir = 0; dir < 3; dir++) {
            int direction = dir + 1;

            int ip1 = clamped_cell(ix +   DX[dir], iy +   DY[dir],
                                   ieta +   DETA[dir], Nx, Ny, Neta);
            int ip2 = clamped_cell(ix + 2*DX[dir], iy + 2*DY[dir],
                                   ieta + 2*DETA[dir], Nx, Ny, Neta);
            int im1 = clamped_cell(ix -   DX[dir], iy -   DY[dir],
                                   ieta -   DETA[dir], Nx, Ny, Neta);
            int im2 = clamped_cell(ix - 2*DX[dir], iy - 2*DY[dir],
                                   ieta - 2*DETA[dir], Nx, Ny, Neta);

            // W^{mu nu}(c) and on neighbors
            float W_c   = Wmunu_curr[idx_1d * Ncells + c];
            float W_p1  = Wmunu_curr[idx_1d * Ncells + ip1];
            float W_p2  = Wmunu_curr[idx_1d * Ncells + ip2];
            float W_m1  = Wmunu_curr[idx_1d * Ncells + im1];
            float W_m2  = Wmunu_curr[idx_1d * Ncells + im2];

            // u^{direction} and u^0 on the same neighbors
            float ud_c   = u_cd[dir];
            float u0_c   = u_c0;
            float ud_p1  = u_curr[direction * Ncells + ip1];
            float u0_p1  = u_curr[0         * Ncells + ip1];
            float ud_p2  = u_curr[direction * Ncells + ip2];
            float u0_p2  = u_curr[0         * Ncells + ip2];
            float ud_m1  = u_curr[direction * Ncells + im1];
            float u0_m1  = u_curr[0         * Ncells + im1];
            float ud_m2  = u_curr[direction * Ncells + im2];
            float u0_m2  = u_curr[0         * Ncells + im2];

            // f = W * u^direction,  g = W * u^0
            float f_c   = W_c  * ud_c;    float g_c   = W_c  * u0_c;
            float f_p1  = W_p1 * ud_p1;   float g_p1  = W_p1 * u0_p1;
            float f_p2  = W_p2 * ud_p2;   float g_p2  = W_p2 * u0_p2;
            float f_m1  = W_m1 * ud_m1;   float g_m1  = W_m1 * u0_m1;
            float f_m2  = W_m2 * ud_m2;   float g_m2  = W_m2 * u0_m2;

            // Half-cell uWmn (minmod-limited)
            float uWphR = f_p1 - 0.5f * gpu_minmod_dx(f_p2, f_p1, f_c , theta_l);
            float temp  = 0.5f * gpu_minmod_dx(f_p1, f_c , f_m1, theta_l);
            float uWphL = f_c  + temp;
            float uWmhR = f_c  - temp;
            float uWmhL = f_m1 + 0.5f * gpu_minmod_dx(f_c , f_m1, f_m2, theta_l);

            // Half-cell Wmn (minmod-limited)
            float WphR = g_p1 - 0.5f * gpu_minmod_dx(g_p2, g_p1, g_c , theta_l);
            float temp2 = 0.5f * gpu_minmod_dx(g_p1, g_c , g_m1, theta_l);
            float WphL = g_c  + temp2;
            float WmhR = g_c  - temp2;
            float WmhL = g_m1 + 0.5f * gpu_minmod_dx(g_c , g_m1, g_m2, theta_l);

            // Local wave speeds
            float a   = fabs(ud_c ) / u0_c;
            float ap1 = fabs(ud_p1) / u0_p1;
            float am1 = fabs(ud_m1) / u0_m1;

            float ax;
            ax = max(a, ap1);
            float HWph = ((uWphR + uWphL) - ax * (WphR - WphL)) * 0.5f;
            ax = max(a, am1);
            float HWmh = ((uWmhR + uWmhL) - ax * (WmhR - WmhL)) * 0.5f;

            flux += -((HWph - HWmh) / delta[direction]);
        }

        uwrhs_out[k * Ncells + c] = flux * delta_tau;
    }
}
