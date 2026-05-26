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
    device const float*       qi_source_in  [[buffer(14)]],
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

    const int has_src = params.has_hydro_source;

    float qi[5];
    for (int a = 0; a < 5; a++) {
        float qv = qi_buf  [a * Ncells + c]
                 - dwmn_buf[a * Ncells + c] * dt;

        // CPU-precomputed hydro source: tau_rk * j^alpha (already tau-scaled)
        if (has_src) {
            qv += qi_source_in[a * Ncells + c] * dt;
        }

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

// ── gpu_make_uprhs ───────────────────────────────────────────────────────────
//
// Bulk-pressure stencil — GPU port of the Neighbourloop portion of
// Diss::Make_uPRHS().  For each cell, writes the KT flux divergence of
// (u^a * pi_b) summed over the three spatial directions, pre-multiplied by
// delta_tau (matching the convention of gpu_make_uwrhs).  The per-cell
// algebraic tail (-u^0/τ + θ)*pi_b * Δτ is added inside
// gpu_first_rk_step_w_full alongside the source term.
//
// Buffer bindings (must match MetalPipelines::dispatch_uprhs):
//   0  pi_b_curr   [Ncells]
//   1  u_curr      [4*Ncells]
//   2  uprhs_out   [Ncells]   output
//   3  params      (constant struct)

kernel void gpu_make_uprhs(
    device const float*       pi_b_curr  [[buffer(0)]],
    device const float*       u_curr     [[buffer(1)]],
    device       float*       uprhs_out  [[buffer(2)]],
    constant MUSICGridParams& params     [[buffer(3)]],
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

    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    float u_c0 = u_curr[0 * Ncells + c];
    float u_cd[3];
    for (int d = 0; d < 3; d++) u_cd[d] = u_curr[(d + 1) * Ncells + c];

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

        // pi_b values
        float pi_c   = pi_b_curr[c];
        float pi_p1  = pi_b_curr[ip1];
        float pi_p2  = pi_b_curr[ip2];
        float pi_m1  = pi_b_curr[im1];
        float pi_m2  = pi_b_curr[im2];

        // u^direction and u^0 on the same neighbors
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

        // f = pi_b * u^direction,  g = pi_b * u^0
        float f_c   = pi_c  * ud_c;    float g_c   = pi_c  * u0_c;
        float f_p1  = pi_p1 * ud_p1;   float g_p1  = pi_p1 * u0_p1;
        float f_p2  = pi_p2 * ud_p2;   float g_p2  = pi_p2 * u0_p2;
        float f_m1  = pi_m1 * ud_m1;   float g_m1  = pi_m1 * u0_m1;
        float f_m2  = pi_m2 * ud_m2;   float g_m2  = pi_m2 * u0_m2;

        // Half-cell u·Pi (minmod-limited)
        float uPiphR = f_p1 - 0.5f * gpu_minmod_dx(f_p2, f_p1, f_c , theta_l);
        float temp   = 0.5f * gpu_minmod_dx(f_p1, f_c , f_m1, theta_l);
        float uPiphL = f_c  + temp;
        float uPimhR = f_c  - temp;
        float uPimhL = f_m1 + 0.5f * gpu_minmod_dx(f_c , f_m1, f_m2, theta_l);

        // Half-cell Pi (minmod-limited)
        float PiphR = g_p1 - 0.5f * gpu_minmod_dx(g_p2, g_p1, g_c , theta_l);
        float temp2 = 0.5f * gpu_minmod_dx(g_p1, g_c , g_m1, theta_l);
        float PiphL = g_c  + temp2;
        float PimhR = g_c  - temp2;
        float PimhL = g_m1 + 0.5f * gpu_minmod_dx(g_c , g_m1, g_m2, theta_l);

        // Wave speeds
        float a   = fabs(ud_c ) / u0_c;
        float ap1 = fabs(ud_p1) / u0_p1;
        float am1 = fabs(ud_m1) / u0_m1;

        float ax;
        ax = max(a, ap1);
        float HPiph = ((uPiphR + uPiphL) - ax * (PiphR - PiphL)) * 0.5f;
        ax = max(a, am1);
        float HPimh = ((uPimhR + uPimhL) - ax * (PimhR - PimhL)) * 0.5f;

        flux += -((HPiph - HPimh) / delta[direction]);
    }

    uprhs_out[c] = flux * delta_tau;
}

// ── gpu_make_du ──────────────────────────────────────────────────────────────
//
// Per-cell port of U_derivative::MakedU + calculate_expansion_rate +
// calculate_Du_supmu + calculate_velocity_shear_tensor.  Writes three
// per-cell output buffers consumed by the CPU viscous loop:
//
//   theta_out[Ncells]      — expansion rate  theta = ∂_μ u^μ + u^0/τ
//   a_out    [4*Ncells]    — a^μ = u^ν ∂_ν u^μ          (DumuVec components 0..3)
//   sigma_out[10*Ncells]   — velocity shear tensor σ^{μν} (VelocityShearVec layout)
//
// Restrictions (v1): assumes vorticity terms OFF (no dUoverTsup / dUTsup)
// and zero net baryon (no ∂^n (µ_B/T) component).  The host code falls back
// to the CPU path when either is enabled.
//
// Spatial stencil uses minmod limiter (matching MakeDSpatial).
// Time derivative is backward first-order (matching MakeDTau).
// delta[3] = params.delta_eta * params.tau  (proper length).
//
// Buffer bindings (must match MetalPipelines::dispatch_make_du):
//   0  u_curr     [4*Ncells]
//   1  u_prev     [4*Ncells]
//   2  theta_out  [Ncells]
//   3  a_out      [4*Ncells]
//   4  sigma_out  [10*Ncells]
//   5  params     (constant struct)

kernel void gpu_make_du(
    device const float*       u_curr     [[buffer(0)]],
    device const float*       u_prev     [[buffer(1)]],
    device       float*       theta_out  [[buffer(2)]],
    device       float*       a_out      [[buffer(3)]],
    device       float*       sigma_out  [[buffer(4)]],
    constant MUSICGridParams& params     [[buffer(5)]],
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

    const float tau       = params.tau;
    const float delta_tau = params.delta_tau;
    const float theta_l   = params.minmod_theta;
    const float delta[4]  = {0.f, params.delta_x, params.delta_y,
                             params.delta_eta * tau};

    // Load center-cell u^μ
    float u[4];
    for (int m = 0; m < 4; m++) u[m] = u_curr[m * Ncells + c];

    // ── dUsup[m][n] = ∂^n u^m ────────────────────────────────────────────────
    //
    // dUsup_local[m][n]; m=0..3 row, n=0..3 column.
    //   n=0   → ∂^τ u^m (backward time difference; with g^00=-1 sign flip)
    //   n=1,2,3 → ∂^n u^m (spatial minmod stencil)
    //   m=0 derived from u·∂u = 0 (transversality) after stencil pass.
    float dUsup_local[4][4];
    for (int m = 0; m < 4; m++)
        for (int n = 0; n < 4; n++)
            dUsup_local[m][n] = 0.f;

    // Spatial stencil: m=1..3, direction=1..3
    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    for (int dir = 0; dir < 3; dir++) {
        int direction = dir + 1;
        int ip1 = clamped_cell(ix + DX[dir], iy + DY[dir],
                               ieta + DETA[dir], Nx, Ny, Neta);
        int im1 = clamped_cell(ix - DX[dir], iy - DY[dir],
                               ieta - DETA[dir], Nx, Ny, Neta);
        for (int m = 1; m <= 3; m++) {
            float f   = u[m];
            float fp1 = u_curr[m * Ncells + ip1];
            float fm1 = u_curr[m * Ncells + im1];
            dUsup_local[m][direction] =
                gpu_minmod_dx(fp1, f, fm1, theta_l) / delta[direction];
        }
    }

    // dUsup[0][n] from u^μ u_μ = -1  →  u_μ ∂^n u^μ = 0
    for (int n = 1; n <= 3; n++) {
        float f = 0.f;
        for (int m = 1; m <= 3; m++)
            f += dUsup_local[m][n] * u[m];
        dUsup_local[0][n] = f / u[0];
    }

    // Time derivative: backward first-order; g^00 = -1 gives the minus sign.
    for (int m = 0; m < 4; m++) {
        float u_m_c = u[m];
        float u_m_p = u_prev[m * Ncells + c];
        dUsup_local[m][0] = -((u_m_c - u_m_p) / delta_tau);
    }
    // dUsup[0][0] from constraint (overwrites the literal time-diff value).
    {
        float f = 0.f;
        for (int m = 1; m < 4; m++)
            f += dUsup_local[m][0] * u[m];
        dUsup_local[0][0] = f / u[0];
    }

    // ── theta = ∂_μ u^μ + u^0/τ ──────────────────────────────────────────────
    // ∂_μ u^μ = -∂^0 u^0 + ∂^1 u^1 + ∂^2 u^2 + ∂^3 u^3
    float theta_cell = (-dUsup_local[0][0] + dUsup_local[1][1]
                        + dUsup_local[2][2] + dUsup_local[3][3]
                        + u[0] / tau);
    theta_out[c] = theta_cell;

    // ── a^μ = u^ν ∂_ν u^μ (DumuVec components 0..3) ─────────────────────────
    // ∂_ν = (-1, +1, +1, +1) * ∂^ν
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

    // ── σ^{μν} (matches calculate_velocity_shear_tensor) ────────────────────
    //
    // Spatial entries (a, b in [1..3], with a ≤ b):
    //   σ[a][b] = ((∂^a u^b + ∂^b u^a)/2
    //             − (gδ_{ab} + u^a u^b) θ/3
    //             + u^0/τ δ_{a3} δ_{b3}
    //             + u^3 u^0/(2τ) (δ_{a3} u^b + δ_{b3} u^a)
    //             + (u^a a^b + u^b a^a)/2)
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

    // σ[3][3] from tracelessness g_{μν} σ^{μν} = 0 (rewritten using
    // transversality u_μ σ^{μν} = 0 to eliminate the σ[0][·] entries).
    sigma_local[3][3] = (
        ( 2.f * (  u[1] * u[2] * sigma_local[1][2]
                 + u[1] * u[3] * sigma_local[1][3]
                 + u[2] * u[3] * sigma_local[2][3])
         - (u[0]*u[0] - u[1]*u[1]) * sigma_local[1][1]
         - (u[0]*u[0] - u[2]*u[2]) * sigma_local[2][2])
        / fmax(u[0]*u[0] - u[3]*u[3], 1e-10f));

    // σ[0][a] from transversality u_μ σ^{μν} = 0
    for (int a = 1; a < 4; a++) {
        float s = 0.f;
        for (int b = 1; b < 4; b++)
            s += sigma_local[a][b] * u[b];
        sigma_local[0][a] = s / u[0];
    }
    // σ[0][0]
    {
        float s = 0.f;
        for (int a = 1; a < 4; a++)
            s += sigma_local[0][a] * u[a];
        sigma_local[0][0] = s / u[0];
    }

    // VelocityShearVec layout:
    //   [0]=σ00 [1]=σ01 [2]=σ02 [3]=σ03
    //   [4]=σ11 [5]=σ12 [6]=σ13
    //   [7]=σ22 [8]=σ23
    //   [9]=σ33
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
//
// GPU port of Advance::FirstRKStepW (shear sector) for the
// constant-shear / no-bulk / no-vorticity / no-diffusion / no-second-order
// configuration.  Consumes the outputs of the four earlier kernels and
// writes the final viscous tensors of arena_future directly into
// snap_future.Wmunu and snap_future.pi_b — no CPU per-cell loop needed.
//
// For each cell:
//   1. Read W^{μν}_curr, u_curr, u_future, theta, a^μ, σ^{μν}, and the
//      precomputed stencil flux uwrhs[5] for shear indices idx_1d ∈ {4..8}.
//   2. For each shear index:
//        w_rhs = uwrhs[idx_1d - 4] + Make_uWRHS_geom(...)
//        SW    = Make_uWSource_constant(...)
//        tempf = (1-rk_flag)*W*u^0 + rk_flag*W_prev*u^0_prev
//                + SW*delta_tau + w_rhs + rk_flag*W*u^0
//        tempf *= 1/(1+rk_flag)
//        W_future[idx_1d] = tempf / u_future^0
//   3. Re-make W^{33} via tracelessness, then W^{0μ} via transversality.
//   4. Zero out the baryon-diffusion components (idx 10..13).
//   5. Set pi_b_future = 0 (turn_on_bulk == 0 in this v1).
//
// Buffer bindings (must match MetalPipelines::dispatch_first_rk_step_w_full):
//   0  Wmunu_curr   [14*Ncells]
//   1  pi_b_curr    [Ncells]
//   2  u_curr       [4*Ncells]
//   3  Wmunu_prev   [14*Ncells]
//   4  pi_b_prev    [Ncells]
//   5  u_prev       [4*Ncells]
//   6  epsilon_curr [Ncells]
//   7  epsilon_prev [Ncells]
//   8  u_future     [4*Ncells]   (from gpu_finalize_ideal)
//   9  uwrhs_in     [5*Ncells]   (from gpu_make_uwrhs)
//  10  theta_in     [Ncells]     (from gpu_make_du)
//  11  a_in         [4*Ncells]
//  12  sigma_in     [10*Ncells]
//  13  Wmunu_future [14*Ncells]  output
//  14  pi_b_future  [Ncells]     output
//  15  eos_P        [GPU_EOS_N]
//  16  eos_s        [GPU_EOS_N]
//  17  params
//  18  eos_p

// Helper: log-spaced table lookup, used for both entropy s(e) ~ e^(3/4)
// and temperature T(e) ~ e^(1/4).  Linear interpolation in log(e) gives
// ~1e-6 relative error across the full physical range.
inline float gpu_log_interp(float e, device const float* tab,
                            constant GPUEosParams& ep) {
    if (e <= 0.f) return 0.f;
    float le = log(e);
    le = clamp(le, ep.log_e_min, ep.log_e_max);
    float fe   = (le - ep.log_e_min) / ep.log_delta_e;
    int   idx  = clamp((int)fe, 0, ep.n_pts - 2);
    float frac = fe - (float)idx;
    return max(0.f, tab[idx] * (1.f - frac) + tab[idx + 1] * frac);
}

inline float gpu_s(float e, device const float* s_tab,
                   constant GPUEosParams& ep) {
    return gpu_log_interp(e, s_tab, ep);
}

inline float gpu_T_e(float e, device const float* T_tab,
                     constant GPUEosParams& ep) {
    return gpu_log_interp(e, T_tab, ep);
}

// ── η/s profile functions (MSL ports of transport_coeffs.cpp) ────────────────
//
// All inputs T in 1/fm.  Outputs the dimensionless η/s.  These mirror
// TransportCoeffs::get_temperature_dependent_eta_over_s_{default,duke,sims}
// and ::get_temperature_dependence_shear_profile.  The muB profile is NOT
// ported here (muB_dependent_shear_to_s == 10 falls back to CPU).

constant float GPU_HBARC = 0.197326980f;   // GeV · fm

inline float gpu_eta_over_s_default(float T_in_fm, float shear_to_s_baseline) {
    // T_in_fm is in 1/fm.  Transition Ttr = 0.18 GeV / hbarc.
    const float Ttr   = 0.18f / GPU_HBARC;
    float Tfrac = T_in_fm / Ttr;
    if (Tfrac < 1.f) {
        return shear_to_s_baseline
             + 0.0594f  * (1.f - Tfrac)
             + 0.544f   * (1.f - Tfrac * Tfrac);
    } else {
        return shear_to_s_baseline
             + 0.288f   * (Tfrac - 1.f)
             + 0.0818f  * (Tfrac * Tfrac - 1.f);
    }
}

inline float gpu_eta_over_s_duke(float T_in_fm,
                                 float shear_2_min, float shear_2_slope,
                                 float shear_2_curv) {
    float T_in_GeV   = T_in_fm * GPU_HBARC;
    float Ttr_in_GeV = 0.154f;
    float Tfrac      = T_in_GeV / Ttr_in_GeV;
    return shear_2_min
         + shear_2_slope * (T_in_GeV - Ttr_in_GeV) * pow(Tfrac, shear_2_curv);
}

inline float gpu_eta_over_s_sims(float T_in_fm,
                                 float T_kink_GeV, float low_slope,
                                 float high_slope, float at_kink) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float eta_over_s_min = 1.e-6f;
    float eta_over_s;
    if (T_in_GeV < T_kink_GeV) {
        eta_over_s = at_kink + low_slope  * (T_in_GeV - T_kink_GeV);
    } else {
        eta_over_s = at_kink + high_slope * (T_in_GeV - T_kink_GeV);
    }
    return max(eta_over_s, eta_over_s_min);
}

// Profile-multiplier (mode 11): returns DATA.shear_to_s * f_T(T).
// Mirrors TransportCoeffs::get_temperature_dependence_shear_profile.
inline float gpu_eta_over_s_profile_mult(float T_in_fm, float shear_to_s_baseline) {
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
//
// MSL ports of TransportCoeffs::get_zeta_over_s and its temperature
// branches.  Supported modes: 0 (off), 1 (default/Gabriel), 2 (Duke,
// Cauchy), 3 (Sims, skewed Cauchy), 8 (AsymGaussian, fixed),
// 9 (AsymGaussian, fixed), 10 (AsymGaussian, DATA-controlled).
// Mode 7 (bigbroadP) and any unsupported value fall back to CPU.

inline float gpu_zeta_over_s_default(float T_in_fm) {
    // T-dependent bulk viscosity from Gabriel (arXiv:1502.01675).
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float Ttr = 0.18f;
    float dummy = T_in_GeV / Ttr;
    float bulk;
    if (T_in_GeV < 0.995f * Ttr) {
        const float lambda3 = 0.9f;
        const float lambda4 = 0.22f;
        const float sigma3  = 0.0025f;
        const float sigma4  = 0.022f;
        bulk = lambda3 * exp((dummy - 1.f) / sigma3)
             + lambda4 * exp((dummy - 1.f) / sigma4) + 0.03f;
    } else if (T_in_GeV > 1.05f * Ttr) {
        const float lambda1 = 0.9f;
        const float lambda2 = 0.25f;
        const float sigma1  = 0.025f;
        const float sigma2  = 0.13f;
        bulk = lambda1 * exp(-(dummy - 1.f) / sigma1)
             + lambda2 * exp(-(dummy - 1.f) / sigma2) + 0.001f;
    } else {
        // 0.995 Ttr <= T <= 1.05 Ttr: polynomial branch
        const float A1 = -13.77f;
        const float A2 =  27.55f;
        const float A3 =  13.45f;
        bulk = A1 * dummy * dummy + A2 * dummy - A3;
    }
    return bulk;
}

inline float gpu_zeta_over_s_duke(float T_in_fm,
                                  float norm, float width_GeV, float peak_GeV) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float diff_ratio = (T_in_GeV - peak_GeV) / width_GeV;
    return norm / (1.f + diff_ratio * diff_ratio);
}

inline float gpu_zeta_over_s_sims(float T_in_fm,
                                  float max_norm, float width_GeV,
                                  float T_peak_GeV, float lambda) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float diff = T_in_GeV - T_peak_GeV;
    float s    = (diff > 0.f) ? 1.f : ((diff < 0.f) ? -1.f : 0.f);
    float diff_ratio = diff / (width_GeV * (lambda * s + 1.f));
    return max_norm / (1.f + diff_ratio * diff_ratio);
}

inline float gpu_zeta_over_s_asym_gaussian(
        float T_in_fm,
        float B_norm, float B_width_low_GeV, float B_width_high_GeV,
        float Tpeak_GeV) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float Tdiff    = T_in_GeV - Tpeak_GeV;
    Tdiff = (Tdiff > 0.f) ? (Tdiff / B_width_high_GeV)
                          : (Tdiff / B_width_low_GeV);
    return B_norm * exp(-Tdiff * Tdiff);
}

inline float gpu_zeta_over_s(float T_in_fm, constant MUSICGridParams& params) {
    float zoverS = 0.f;
    switch (params.T_dep_bulk_mode) {
        case 0:
            zoverS = 0.f;
            break;
        case 1:
            zoverS = gpu_zeta_over_s_default(T_in_fm);
            break;
        case 2:
            zoverS = gpu_zeta_over_s_duke(T_in_fm,
                                          params.bulk_duke_norm,
                                          params.bulk_duke_width_GeV,
                                          params.bulk_duke_peak_GeV);
            break;
        case 3:
            zoverS = gpu_zeta_over_s_sims(T_in_fm,
                                          params.bulk_sims_max,
                                          params.bulk_sims_width_GeV,
                                          params.bulk_sims_T_peak_GeV,
                                          params.bulk_sims_lambda);
            break;
        case 8:
            // IPGlasma+MUSIC+UrQMD fixed params
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm, 0.13f, 0.01f, 0.12f, 0.160f);
            break;
        case 9:
            // IPGlasma+KoMPoST+MUSIC+UrQMD fixed params
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm, 0.175f, 0.01f, 0.12f, 0.160f);
            break;
        case 10:
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm,
                        params.bulk_asym10_max,
                        params.bulk_asym10_width_low,
                        params.bulk_asym10_width_high,
                        params.bulk_asym10_Tpeak);
            break;
        default:
            zoverS = 0.f;
            break;
    }
    return max(0.f, zoverS);
}

// ── Make_uPiSource (port of Diss::Make_uPiSource) ────────────────────────────
//
// Returns the relaxation source SΠ for the bulk pressure pi_b.  Mirrors
// the CPU formula with include_second_order_terms == 0 (so BB_term and
// Coupling_to_Shear are zero) and zero net baryon (T(e), P(e), cs2(e)
// taken from the rhob=0 tables).
inline float gpu_uPi_source(float pi_b, float theta,
                            float eps_src,
                            device const float* P_tab,
                            device const float* dPde_tab,
                            device const float* T_tab,
                            constant GPUEosParams& ep,
                            constant MUSICGridParams& params,
                            float delta_tau)
{
    float P_local  = gpu_P (eps_src, P_tab, ep);
    float T_local  = gpu_T_e(eps_src, T_tab, ep);
    float cs2      = clamp(gpu_dPde(eps_src, dPde_tab, ep), 0.01f, 0.333333f);

    float zeta_s   = gpu_zeta_over_s(T_local, params);
    float epsP     = max(eps_src + P_local, 1.e-20f);
    float T_safe   = max(T_local, 1.e-20f);
    float bulk_zeta = zeta_s * epsP / T_safe;   // η_b ≡ ζ in CPU code

    float csfactor = max(1.f/3.f - cs2, 1.e-20f);
    float tau_Pi;
    if (params.bulk_relaxation_type == 1) {
        tau_Pi = bulk_zeta
               / (params.bulk_relax_time_factor * csfactor)
               / epsP;
    } else {
        tau_Pi = params.bulk_relax_time_factor
               / (csfactor * csfactor)
               / epsP * bulk_zeta;
    }
    tau_Pi = min(10.f, max(3.f * delta_tau, tau_Pi));

    // delta_PiPi = 2/3 — unconditional in CPU Make_uPiSource (BB_term and
    // Coupling_to_Shear are gated on include_second_order_terms == 1).
    const float delta_PiPi = 2.f / 3.f;
    float transport_coeff1 = delta_PiPi * tau_Pi;

    float NS_term  = -bulk_zeta * theta;
    float relax    = -pi_b - transport_coeff1 * theta * pi_b;

    return (NS_term + relax) / tau_Pi;
}

// Dispatch on T_dependent_shear_to_s.  Mirrors get_eta_over_s() in
// transport_coeffs.cpp; the muB-dependence branch (mode 10) is not handled
// here (it falls back to CPU at the host gate).
inline float gpu_eta_over_s(float T_in_fm, constant MUSICGridParams& params) {
    switch (params.T_dep_shear_mode) {
        case 0:
            return params.shear_to_s;
        case 1:
            return gpu_eta_over_s_default(T_in_fm, params.shear_to_s);
        case 2:
            return gpu_eta_over_s_duke(T_in_fm,
                                       params.shear_duke_min,
                                       params.shear_duke_slope,
                                       params.shear_duke_curv);
        case 3:
            return gpu_eta_over_s_sims(T_in_fm,
                                       params.shear_sims_T_kink_GeV,
                                       params.shear_sims_low_slope,
                                       params.shear_sims_high_slope,
                                       params.shear_sims_at_kink);
        case 11:
            return gpu_eta_over_s_profile_mult(T_in_fm, params.shear_to_s);
        default:
            // Unsupported mode: host gate should prevent this.  Fall back
            // to the constant baseline rather than producing garbage.
            return params.shear_to_s;
    }
}

// Algebraic Make_uWSource (port of Diss::Make_uWSource).
// Returns S^{μν} = (NS_term + relaxation_term) / tau_pi.
//
// Supports T-dependent shear viscosity (T_dep_shear_mode ∈ {0,1,2,3,11}).
// Assumes muB_dependent_shear_to_s == 0 (entropy-based shear), no vorticity,
// no second-order terms, no bulk coupling — host gate enforces all of these.
inline float gpu_uW_source(
        float W_mn, float sigma_mn, float theta,
        float eps_src,
        device const float* P_tab,
        device const float* s_tab,
        device const float* T_tab,
        constant GPUEosParams& ep,
        constant MUSICGridParams& params,
        float delta_tau)
{
    float P       = gpu_P(eps_src, P_tab, ep);
    float entropy = gpu_s(eps_src, s_tab, ep);
    float T_local = gpu_T_e(eps_src, T_tab, ep);
    float shear_to_s_T = gpu_eta_over_s(T_local, params);
    float shear   = shear_to_s_T * entropy;

    float epsP = max(eps_src + P, 1.e-20f);
    float tau_pi = params.shear_relax_time_factor * shear / epsP;
    tau_pi = min(10.f, max(3.f * delta_tau, tau_pi));

    // delta_pipi_coeff = 4/3 — unconditional in CPU Make_uWSource (only
    // the WW / Wsigma / Coupling_to_Bulk terms are gated on
    // include_second_order_terms).
    const float dpi_pi = 4.f / 3.f;
    float transport_coefficient2 = dpi_pi * tau_pi;

    float NS_term = -2.f * shear * sigma_mn;
    float relax   = -(1.f + transport_coefficient2 * theta) * W_mn;

    return (NS_term + relax) / tau_pi;
}

// Algebraic Make_uWRHS geometric tail (the per-cell terms that depend on
// W^{μν}, u^μ, a^μ, theta).  Matches Diss::Make_uWRHS_geom on the CPU.
inline float gpu_uWRHS_geom(
        thread const float W_local[4][4], thread const float u[4],
        thread const float a_loc[4],
        int mu, int nu, float theta, float tau, float delta_tau)
{
    // gmunu^{μν} = diag(-1, +1, +1, +1) → gmunu[3][μ] = 1 iff μ==3, else 0
    //                                     gmunu[0][μ] = -1 iff μ==0, else 0
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
         - W_local[0][mu] * u[nu] * u[3])
         * (u[3] / tau);

    for (int ic = 0; ic < 4; ic++) {
        float ic_fac = (ic == 0) ? -1.f : 1.f;
        tempf +=  W_local[ic][nu] * u[mu] * a_loc[ic] * ic_fac
                + W_local[ic][mu] * u[nu] * a_loc[ic] * ic_fac;
    }

    return tempf * delta_tau
           + (-(u[0] * W_local[mu][nu]) / tau
              + theta * W_local[mu][nu]) * delta_tau;
}

kernel void gpu_first_rk_step_w_full(
    device const float*       Wmunu_curr   [[buffer(0)]],
    device const float*       pi_b_curr    [[buffer(1)]],
    device const float*       u_curr       [[buffer(2)]],
    device const float*       Wmunu_prev   [[buffer(3)]],
    device const float*       pi_b_prev    [[buffer(4)]],
    device const float*       u_prev       [[buffer(5)]],
    device const float*       epsilon_curr [[buffer(6)]],
    device const float*       epsilon_prev [[buffer(7)]],
    device const float*       u_future     [[buffer(8)]],
    device const float*       uwrhs_in     [[buffer(9)]],
    device const float*       theta_in     [[buffer(10)]],
    device const float*       a_in         [[buffer(11)]],
    device const float*       sigma_in     [[buffer(12)]],
    device       float*       Wmunu_future [[buffer(13)]],
    device       float*       pi_b_future  [[buffer(14)]],
    device const float*       eos_P        [[buffer(15)]],
    device const float*       eos_s        [[buffer(16)]],
    device const float*       eos_T        [[buffer(17)]],
    device const float*       eos_dPde     [[buffer(18)]],
    device const float*       uprhs_in     [[buffer(19)]],
    constant MUSICGridParams& params       [[buffer(20)]],
    constant GPUEosParams&    eos_p        [[buffer(21)]],
    // For QuestRevert: needs the future-state primitives (post-Newton).
    device const float*       epsilon_future [[buffer(22)]],
    device const float*       rhob_future    [[buffer(23)]],
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

    const int   rkf       = params.rk_flag;
    const float dt        = params.delta_tau;
    const float tau_now   = params.tau;   // tau + rk_flag * delta_tau
    const float rk_norm   = 1.f / (1.f + (float)rkf);

    // Early exit if shear is disabled (host should have skipped the dispatch,
    // but guard anyway): write zeros for all viscous components.
    if (params.turn_on_shear == 0) {
        for (int m = 0; m < 14; m++)
            Wmunu_future[m * Ncells + c] = 0.f;
        pi_b_future[c] = 0.f;
        return;
    }

    // ── Load cell-local state ─────────────────────────────────────────────
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

    // 4×4 matrix views (full Wmunu — only the upper-left 4×4 used here)
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

    float eps_src = (rkf == 0) ? epsilon_curr[c] : epsilon_prev[c];

    // ── Shear update for idx_1d in {4..8} ─────────────────────────────────
    float Wf[14];
    for (int m = 0; m < 14; m++) Wf[m] = 0.f;

    // (μ, ν) for the 5 shear indices.
    const int MU_LIST[5] = {1, 1, 1, 2, 2};
    const int NU_LIST[5] = {1, 2, 3, 2, 3};

    for (int k = 0; k < 5; k++) {
        int mu  = MU_LIST[k];
        int nu  = NU_LIST[k];
        int id  = WMUNU_IDX[mu][nu];     // 4, 5, 6, 7, or 8
        int sid = id;                    // VelocityShearVec uses same idx

        float W_mn       = W4[mu][nu];
        float sigma_mn   = sigma_vec[sid];
        float uwrhs_flux = uwrhs_in[k * Ncells + c];

        float SW = gpu_uW_source(
                       W_mn, sigma_mn, theta, eps_src,
                       eos_P, eos_s, eos_T, eos_p, params, dt);

        float w_rhs_geom = gpu_uWRHS_geom(W4, u_c, a_loc, mu, nu,
                                          theta, tau_now, dt);
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

    // ── Transversality / tracelessness ────────────────────────────────────
    // W^{33}: re-make from traceless condition.
    Wf[9] = ( 2.f * (  u_f[1]*u_f[2]*Wf[5]
                     + u_f[1]*u_f[3]*Wf[6]
                     + u_f[2]*u_f[3]*Wf[8])
              - (u_f[0]*u_f[0] - u_f[1]*u_f[1]) * Wf[4]
              - (u_f[0]*u_f[0] - u_f[2]*u_f[2]) * Wf[7])
            / fmax(u_f[0]*u_f[0] - u_f[3]*u_f[3], 1.e-10f);

    // W^{0μ} for μ = 1..3 via u_ν W^{μν} = 0.
    for (int mu = 1; mu < 4; mu++) {
        float s = 0.f;
        for (int nu = 1; nu < 4; nu++)
            s += Wf[WMUNU_IDX[mu][nu]] * u_f[nu];
        Wf[mu] = s / u_f[0];
    }
    // W^{00} from transversality at μ=0.
    {
        float s = 0.f;
        for (int nu = 1; nu < 4; nu++) s += Wf[nu] * u_f[nu];
        Wf[0] = s / u_f[0];
    }

    // Baryon-diffusion components: zeroed (turn_on_diff == 0 in this v1).
    for (int m = 10; m < 14; m++) Wf[m] = 0.f;

    // ── Bulk pressure update (turn_on_bulk == 1 only) ─────────────────────
    float pi_b_out;
    if (params.turn_on_bulk == 1) {
        float pi_c = pi_b_curr[c];
        float pi_p = pi_b_prev[c];

        // Make_uPRHS pre-pass: stencil flux (already pre-multiplied by Δτ).
        // Add the per-cell algebraic tail.
        float p_rhs = uprhs_in[c]
                    + (-(u_c[0] * pi_c) / tau_now + theta * pi_c) * dt;

        // Source term (relaxation toward Navier-Stokes).
        float SPi = gpu_uPi_source(pi_c, theta, eps_src,
                                   eos_P, eos_dPde, eos_T, eos_p, params, dt);

        // RK mixing — mirrors the FirstRKStepW assembly.
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

    // ── QuestRevert regulator (per-cell, no stencil) ──────────────────────
    //
    // GPU port of Advance::QuestRevert.  Active when the host sets
    // do_quest_revert == 1 (i.e. Initial_profile not in {0, 1}).  Reduces
    // the magnitude of W^{μν} and pi_b when they grow too large relative
    // to the equilibrium scale, matching the CPU regulator that fires
    // after FirstRKStepW.  Uses the future-state primitives (post-Newton)
    // as the equilibrium reference.
    if (params.do_quest_revert == 1) {
        const float eps_scale     = 0.1f;
        const float xi            = 0.05f;
        const float rho_shear_max = 0.1f;
        const float rho_bulk_max  = 0.1f;

        float e_local = epsilon_future[c];
        // Smoothstep factor: matches the CPU `1/(1+exp(-(e-eps)/xi)) - const`
        float sig_e   = 1.f / (exp(-(e_local - eps_scale) / xi) + 1.f);
        float sig_off = 1.f / (exp(eps_scale / xi)              + 1.f);
        float factor  = 10.f * params.quest_revert_strength
                            * (sig_e - sig_off);

        // pisize = Σ g_μ g_ν W^{μν} · W^{μν} (with off-diagonal signs).
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
        eq_size        = max(eq_size, 1.e-30f);

        // Shear regulator: division by `factor` reproduces the CPU sign
        // convention (negative factor at low e ⇒ rho_shear < 0 ⇒ no reduction).
        float rho_shear = sqrt(max(pisize, 0.f) / eq_size) / factor;
        float rho_bulk  = sqrt(max(bulksize, 0.f) / eq_size) / factor;

        if (isnan(rho_shear)) {
            for (int m = 0; m < 10; m++) Wf[m] = 0.f;
        } else if (rho_shear > rho_shear_max) {
            float scale = rho_shear_max / rho_shear;
            for (int m = 0; m < 10; m++) Wf[m] *= scale;
        }
        if (rho_bulk > rho_bulk_max) {
            pi_b_out *= rho_bulk_max / rho_bulk;
        }

        (void)rhob_future;  // EOS at rhob=0 for this v1 — future µB needs Tier 4
    }

    // ── Write outputs ─────────────────────────────────────────────────────
    for (int m = 0; m < 14; m++)
        Wmunu_future[m * Ncells + c] = Wf[m];
    pi_b_future[c] = pi_b_out;
}
