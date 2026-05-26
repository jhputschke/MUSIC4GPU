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
