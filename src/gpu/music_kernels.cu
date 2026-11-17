// CUDA C++ compute kernels for MUSIC relativistic hydrodynamics.
//
// Direct port of src/gpu/music_kernels.metal (Metal Shading Language).  The
// physics, stencils, and per-cell algebra are identical; only the language
// surface differs:
//   kernel void f(device const float* b [[buffer(N)]], ...)  ->  __global__ void f(const float* __restrict__ b, ...)
//   [[thread_position_in_grid]]                              ->  blockIdx*blockDim + threadIdx
//   constant T& params [[buffer(N)]]                         ->  by-value kernel argument
//   clamp(x,lo,hi)                                           ->  clampf / clampi
//   abs/min/max/sqrt/... (float)                             ->  fabsf/fminf/fmaxf/sqrtf/...
//
// Memory layout (SoA, component-major), matching the Metal/CPU convention:
//   field[comp * Ncells + cell]   where cell = Nx*(Ny*ieta + iy) + ix

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_types.h"     // MUSICGridParams, GPUEosParams, WMUNU_IDX, GMUNU_DIAG
#include "music_kernels.cuh"

#define DFI __device__ __forceinline__

#include "eos4d_device.cuh" // NEOS-4D (EOS 20) lookup

// ── math shims (mirror MSL overloaded clamp; CUDA has no float clamp) ─────────
DFI float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
DFI int   clampi(int   x, int   lo, int   hi) { return min(max(x, lo), hi);     }

// ── index helpers ─────────────────────────────────────────────────────────────

DFI int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

// Boundary-clamped cell index (replicates Cell_small::getHalo)
DFI int clamped_cell(int ix, int iy, int ieta, int Nx, int Ny, int Neta) {
    ix   = clampi(ix,   0, Nx   - 1);
    iy   = clampi(iy,   0, Ny   - 1);
    ieta = clampi(ieta, 0, Neta - 1);
    return cell_idx(ix, iy, ieta, Nx, Ny);
}

DFI float get_Wmunu(const float* __restrict__ Wmunu, int comp,
                    int ix, int iy, int ieta,
                    int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return __ldg(&Wmunu[comp * Ncells + c]);
}

DFI float get_u(const float* __restrict__ u, int comp,
                int ix, int iy, int ieta,
                int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return __ldg(&u[comp * Ncells + c]);
}

DFI float get_pi_b(const float* __restrict__ pi_b,
                   int ix, int iy, int ieta,
                   int Nx, int Ny, int Neta, int Ncells) {
    int c = clamped_cell(ix, iy, ieta, Nx, Ny, Neta);
    return __ldg(&pi_b[c]);
}

// ── C3: the active region (opt-in, active_region_eps > 0) ─────────────────────
//
// With the active region on, the seven step kernels run on 8 x 8 x bz blocks
// and get a flag per block (gpu_active_mark); a block whose flag is 0 holds
// only frozen vacuum cells and returns before it reads or writes anything.
// The check is uniform per block, so the shared-memory kernel can return
// before its __syncthreads.  null flags (the default): every block runs.
// The same, per cell, for kernels launched on other blocks (the NEOS-4D and
// diffusion kernels): the flag of the 8 x 8 x act_bz block holding the cell.
DFI bool gpu_cell_inactive(const MUSICGridParams& p, int ix, int iy, int ieta) {
    return p.active_blk != nullptr
        && p.active_blk[(ix >> 3) + p.act_gx * ((iy >> 3)
                        + p.act_gy * (ieta / p.act_bz))] == 0;
}

DFI bool gpu_block_inactive(const unsigned char* __restrict__ active_blk) {
    return active_blk != nullptr
        && active_blk[blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)] == 0;
}

// ── gpu_make_w_source ─────────────────────────────────────────────────────────

__global__ void gpu_make_w_source(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ Wmunu_prev,
    const float* __restrict__ pi_b_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ dwmn_out,
    MUSICGridParams params)
{
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;

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

    const int dix  [3] = { 1, 0, 0};
    const int diy  [3] = { 0, 1, 0};
    const int dieta[3] = { 0, 0, 1};

    float W_eta_p[4] = {0.f, 0.f, 0.f, 0.f};
    float W_eta_m[4] = {0.f, 0.f, 0.f, 0.f};

    float dwmn[5] = {0.f, 0.f, 0.f, 0.f, 0.f};

    for (int alpha = 0; alpha < 5; ++alpha) {
        int idx_alpha0 = WMUNU_IDX[alpha][0];

        float dWdtau = (Wc[idx_alpha0] - Wp[idx_alpha0]) / params.delta_tau;

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
            int direction = dir + 1;
            int idx_1d = WMUNU_IDX[alpha][direction];

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
        if (params.cartesian) {
            // no "+ W", "+ Pi" from partial_tau (tau W^{tau alpha}).  The
            // Cartesian overrides in these kernels use the _rn intrinsics
            // (never contracted, never merged with the Milne arithmetic), so
            // the Milne code compiles exactly as before: bit-identical
            sf = __fadd_rn(dWdtau, dWdx);
            bf = __fadd_rn(dPidtau, dPidx);
        }
        dwmn[alpha] += sf + bf;
    }  // alpha loop

    dwmn[0] += (  (W_eta_p[0] - W_eta_m[0]) * params.cosh_deta
               + (W_eta_p[3] + W_eta_m[3]) * params.sinh_deta);
    dwmn[3] += (  (W_eta_p[3] - W_eta_m[3]) * params.cosh_deta
               + (W_eta_p[0] + W_eta_m[0]) * params.sinh_deta);

    for (int alpha = 0; alpha < 5; ++alpha)
        dwmn_out[alpha * Ncells + c] = dwmn[alpha];
}

// ── gpu_make_w_source_tiled (Phase 3) ─────────────────────────────────────────
//
// Shared-memory tiled port of gpu_make_w_source.  The radius-1 stencil over the
// current snapshot (Wmunu[14], u[4], pi_b) is staged into a (bx+2)(by+2)(bz+2)
// halo tile so every neighbour read is served from shared memory.  The previous
// snapshot is only sampled at the cell centre (the dWdtau / dPidtau time
// derivatives), so it stays in global memory.  Bit-for-bit the same arithmetic
// as the untiled kernel.

__global__ void gpu_make_w_source_tiled(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ Wmunu_prev,
    const float* __restrict__ pi_b_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ dwmn_out,
    MUSICGridParams params,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;

    const int bx = blockDim.x, by = blockDim.y, bz = blockDim.z;
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

    // Tile geometry: 1-cell halo on every face.
    const int TX = bx + 2, TY = by + 2, TZ = bz + 2;
    const int tile_cells = TX * TY * TZ;

    // Dynamic shared layout: Wmunu[14] | u[4] | pi_b[1], component-major.
    extern __shared__ float smem[];
    float* sW = smem;                         // 14 * tile_cells
    float* sU = sW + 14 * tile_cells;         //  4 * tile_cells
    float* sP = sU +  4 * tile_cells;         //  1 * tile_cells

    // Global origin of this tile (halo offset of -1 on each axis).
    const int gx0 = blockIdx.x * bx - 1;
    const int gy0 = blockIdx.y * by - 1;
    const int ge0 = blockIdx.z * bz - 1;

    // Cooperative halo load: grid-stride over the linear tile.
    const int tid      = (tz * by + ty) * bx + tx;
    const int nthreads = bx * by * bz;
    for (int t = tid; t < tile_cells; t += nthreads) {
        int lz  = t / (TX * TY);
        int rem = t - lz * TX * TY;
        int ly  = rem / TX;
        int lx  = rem - ly * TX;
        int gxx = clampi(gx0 + lx, 0, Nx   - 1);
        int gyy = clampi(gy0 + ly, 0, Ny   - 1);
        int gee = clampi(ge0 + lz, 0, Neta - 1);
        int gc  = cell_idx(gxx, gyy, gee, Nx, Ny);
        for (int m = 0; m < 14; ++m) sW[m * tile_cells + t] = __ldg(&Wmunu_curr[m * Ncells + gc]);
        for (int m = 0; m < 4;  ++m) sU[m * tile_cells + t] = __ldg(&u_curr[m * Ncells + gc]);
        sP[t] = __ldg(&pi_b_curr[gc]);
    }
    __syncthreads();

    int ix   = blockIdx.x * bx + tx;
    int iy   = blockIdx.y * by + ty;
    int ieta = blockIdx.z * bz + tz;
    if (ix >= Nx || iy >= Ny || ieta >= Neta) return;

    // This cell's interior tile coordinate (centre at local +1).
    const int lx = tx + 1, ly = ty + 1, lz = tz + 1;
    #define TILE(a,b,cc) (((cc) * TY + (b)) * TX + (a))
    const int self = TILE(lx, ly, lz);

    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, params.tau, params.tau, 1.f};

    // Centre values from shared (current) and global (previous, centre only).
    float Wc[14], Wp[14];
    for (int m = 0; m < 14; ++m) {
        Wc[m] = sW[m * tile_cells + self];
        Wp[m] = Wmunu_prev[m * Ncells + c];
    }
    float uc[4], up[4];
    for (int m = 0; m < 4; ++m) {
        uc[m] = sU[m * tile_cells + self];
        up[m] = u_prev[m * Ncells + c];
    }
    float pib_c = sP[self];
    float pib_p = pi_b_prev[c];

    const int dix  [3] = { 1, 0, 0};
    const int diy  [3] = { 0, 1, 0};
    const int dieta[3] = { 0, 0, 1};

    float W_eta_p[4] = {0.f, 0.f, 0.f, 0.f};
    float W_eta_m[4] = {0.f, 0.f, 0.f, 0.f};
    float dwmn[5]    = {0.f, 0.f, 0.f, 0.f, 0.f};

    for (int alpha = 0; alpha < 5; ++alpha) {
        int idx_alpha0 = WMUNU_IDX[alpha][0];

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
            int idx_1d = WMUNU_IDX[alpha][direction];

            int pL = TILE(lx + dix[dir], ly + diy[dir], lz + dieta[dir]);
            int mL = TILE(lx - dix[dir], ly - diy[dir], lz - dieta[dir]);

            float tf = tau_fac[direction];
            float dx = delta  [direction];

            float sg   = Wc[idx_1d] * tf;
            float sgp1 = sW[idx_1d * tile_cells + pL] * tf;
            float sgm1 = sW[idx_1d * tile_cells + mL] * tf;

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
                float bgp1  = sP[pL]
                              * (gfac1 + sU[alpha * tile_cells + pL]
                                       * sU[direction * tile_cells + pL]) * tf;
                float bg    = pib_c * (gfac1 + uc[alpha] * uc[direction]) * tf;
                float bgm1  = sP[mL]
                              * (gfac1 + sU[alpha * tile_cells + mL]
                                       * sU[direction * tile_cells + mL]) * tf;
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
        if (params.cartesian) {
            // no "+ W", "+ Pi" from partial_tau (tau W^{tau alpha}).  The
            // Cartesian overrides in these kernels use the _rn intrinsics
            // (never contracted, never merged with the Milne arithmetic), so
            // the Milne code compiles exactly as before: bit-identical
            sf = __fadd_rn(dWdtau, dWdx);
            bf = __fadd_rn(dPidtau, dPidx);
        }
        dwmn[alpha] += sf + bf;
    }

    dwmn[0] += (  (W_eta_p[0] - W_eta_m[0]) * params.cosh_deta
               + (W_eta_p[3] + W_eta_m[3]) * params.sinh_deta);
    dwmn[3] += (  (W_eta_p[3] - W_eta_m[3]) * params.cosh_deta
               + (W_eta_p[0] + W_eta_m[0]) * params.sinh_deta);

    for (int alpha = 0; alpha < 5; ++alpha)
        dwmn_out[alpha * Ncells + c] = dwmn[alpha];
    #undef TILE
}

// ── EOS table helpers ─────────────────────────────────────────────────────────

// Log-spaced table lookup, shared by P, dP/de, s and T.  All four are
// non-linear in e for realistic (non-conformal) EOS, and hydro cells live in
// the dilute regime (e ~ 0.1 /fm^4) far below eps_max (~1e4 /fm^4 for
// hotQCD), so log spacing is required to resolve them.  Grid matches
// GPUGrid::upload_eos's log_* params.
// __ldg: route the EOS table through the read-only data cache.  The index
// depends on the cell's local energy density, so warp lanes generally hit
// different entries — the read-only L1 path handles this far better than
// __constant__ memory, which serializes non-broadcast accesses.
DFI float gpu_log_interp(float e, const float* __restrict__ tab,
                         const GPUEosParams& ep) {
    if (e <= 0.f) return 0.f;
    float le = logf(e);
    le = clampf(le, ep.log_e_min, ep.log_e_max);
    float fe   = (le - ep.log_e_min) / ep.log_delta_e;
    int   idx  = clampi((int)fe, 0, ep.n_pts - 2);
    float frac = fe - (float)idx;
    return fmaxf(0.f, __ldg(&tab[idx]) * (1.f - frac) + __ldg(&tab[idx + 1]) * frac);
}

DFI float gpu_P(float e, const float* __restrict__ P_tab,
                const GPUEosParams& ep) {
    return fmaxf(1.e-20f, gpu_log_interp(e, P_tab, ep));
}

DFI float gpu_dPde(float e, const float* __restrict__ dPde_tab,
                   const GPUEosParams& ep) {
    return gpu_log_interp(e, dPde_tab, ep);
}

DFI float gpu_cs2(float e, const float* __restrict__ P_tab,
                  const float* __restrict__ dPde_tab, const GPUEosParams& ep) {
    return clampf(gpu_dPde(e, dPde_tab, ep), 0.01f, 0.333333f);
}

// ── Phase 2b: pack ideal-hydro evolution output on the GPU ────────────────────
// One thread per down-sampled output cell.  Mirror of the host
// Cell_info::OutputEvolutionDataXYEta_memory loop, but the EOS lookups run on
// the GPU through the resident log-spaced tables (sampled at rhob=0 — the same
// approximation the GPU evolution itself uses).  Output is written in
// host fluidCell_ideal field order so the host can bulk-copy it directly.
__global__ void gpu_pack_evolution_ideal(
    const float* __restrict__ epsilon,
    const float* __restrict__ u,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_s,
    const float* __restrict__ eos_T,
    float* __restrict__ out,
    GPUEosParams eos_p,
    GPUPackParams pp,
    const float* __restrict__ rhob,
    const float* __restrict__ rhoq,
    const float* __restrict__ rhos,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p)
{
    const int o     = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_out = pp.nx_out * pp.ny_out * pp.neta_out;
    if (o >= n_out) return;

    // Decode flat output index in the host push order (ix outer, iy middle,
    // ieta inner): o = (ix_idx*ny_out + iy_idx)*neta_out + ieta_idx.
    const int ieta_idx = o % pp.neta_out;
    const int t        = o / pp.neta_out;
    const int iy_idx   = t % pp.ny_out;
    const int ix_idx   = t / pp.ny_out;

    const int ix   = ix_idx   * pp.skip_x;
    const int iy   = iy_idx   * pp.skip_y;
    const int ieta = ieta_idx * pp.skip_eta;

    // Device snapshot cell index: cell = ix + Nx*(iy + Ny*ieta).
    const int Ncells = pp.Ncells;
    const int c      = ix + pp.Nx * (iy + pp.Ny * ieta);

    const float e = epsilon[c];
    float p, s, T;
    if (pp.eos4d) {
        const float rb = rhob[c];
        const float rq = pp.with_qs ? rhoq[c] : 0.f;
        const float rs = pp.with_qs ? rhos[c] : 0.f;
        GPUEos4DState st;
        gpu_eos4d_lookup(e, rb, rq, rs, eos4d_tab, eos4d_p,
                         GPU_EOS4D_P | GPU_EOS4D_THERMO, st);
        p = st.P;
        s = gpu_eos4d_entropy(e, rb, rq, rs, st);
        T = st.T;
    } else {
        p = gpu_log_interp(e, eos_P, eos_p);
        s = gpu_log_interp(e, eos_s, eos_p);
        T = gpu_log_interp(e, eos_T, eos_p);
    }

    float eta = 0.f;
    if (!pp.boost_invariant)
        eta = (float)ieta * pp.delta_eta - pp.eta_size * 0.5f;

    // fluidCell_ideal: { eta; sd, ed, pressure, temperature; ux, uy, ueta }.
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

// ── minmod slope limiter ──────────────────────────────────────────────────────

DFI float gpu_minmod_dx(float up1, float u, float um1, float theta) {
    float diffup   = (up1 - u)   * theta;
    float diffdown = (u   - um1) * theta;
    float diffmid  = (up1 - um1) * 0.5f;
    if (diffup == 0.f) return 0.f;
    return diffup * fmaxf(0.f, fminf(1.f, fminf(diffdown / diffup, diffmid / diffup)));
}

// ── T^{alpha,0} from SoA cell ─────────────────────────────────────────────────

DFI float gpu_TJb0(int alpha, int c, int Ncells,
                   const float* __restrict__ eps_buf,
                   const float* __restrict__ rhob_buf,
                   const float* __restrict__ u_buf,
                   const float* __restrict__ P_tab,
                   const GPUEosParams& ep) {
    float u0 = __ldg(&u_buf[0 * Ncells + c]);
    if (alpha == 4) return __ldg(&rhob_buf[c]) * u0;
    float e = __ldg(&eps_buf[c]);
    float P = gpu_P(e, P_tab, ep);
    if (alpha == 0) return (e + P) * u0 * u0 - P;
    return (e + P) * __ldg(&u_buf[alpha * Ncells + c]) * u0;
}

// ── Newton-Brent reconstruction helpers ──────────────────────────────────────

struct ReconstResult {
    float e;
    float rhob;
    float u[4];
};

DFI void gpu_vel_fdf(float v, float T00, float M, float J0,
                     float& fv, float& dfdv,
                     const float* __restrict__ P_tab,
                     const float* __restrict__ dPde_tab,
                     const GPUEosParams& ep) {
    float eps  = T00 - v * M;
    float P    = gpu_P(eps, P_tab, ep);
    float dPde = gpu_dPde(eps, dPde_tab, ep);
    float t1   = T00 + P;
    fv   = v - M / t1;
    dfdv = 1.f - M * M * dPde / (t1 * t1);
    (void)J0;
}

__device__ float gpu_solve_v(float v_guess, float T00, float M, float J0,
                             const float* __restrict__ P_tab,
                             const float* __restrict__ dPde_tab,
                             const GPUEosParams& ep) {
    const float ABS_ERR = 1.e-7f;
    float fv_l, dfdv_l, fv_h, dfdv_h;
    float v_l = 0.f, v_h = 1.f;
    gpu_vel_fdf(v_l, T00, M, J0, fv_l, dfdv_l, P_tab, dPde_tab, ep);
    gpu_vel_fdf(v_h, T00, M, J0, fv_h, dfdv_h, P_tab, dPde_tab, ep);

    if (fabsf(fv_l) < ABS_ERR) return v_l;
    if (fabsf(fv_h) < ABS_ERR) return v_h;
    // No sign change in [0,1]: no root to bracket.  Return a negative sentinel
    // so the caller can revert to the previous cell (matches CPU
    // solve_v_Hybrid returning status 0 -> revert_grid).  A real velocity is
    // always in [0,1], so <0 is unambiguous.
    if (fv_l * fv_h > 0.f) return -1.f;

    float dv_prev = v_h - v_l;
    float dv_curr = dv_prev;
    float v_root  = (v_h + v_l) * 0.5f;
    float fv, dfdv;
    gpu_vel_fdf(v_root, T00, M, J0, fv, dfdv, P_tab, dPde_tab, ep);

    for (int it = 0; it < 60; it++) {
        const float v_old = v_root;
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
        // Converged: the step is below the tolerance, or the iterate can no
        // longer change in single precision (the root to float resolution; a
        // step of 1e-7 is below the ulp of v near 1, and such a lane would
        // otherwise run all 60 iterations and hold its warp with it).
        if (fabsf(dv_curr) < ABS_ERR || v_root == v_old) break;
    }
    return v_root;
}

DFI void gpu_u0_fdf(float u0, float T00, float K00, float M, float J0,
                    float& fu0, float& dfdu0,
                    const float* __restrict__ P_tab,
                    const float* __restrict__ dPde_tab,
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

__device__ float gpu_solve_u0(float u0_guess, float T00, float K00, float M, float J0,
                              const float* __restrict__ P_tab,
                              const float* __restrict__ dPde_tab,
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
        const float u0_old = u0_root;
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
        // Converged, or stalled at float resolution (an absolute 1e-7 is below
        // the ulp of any u0 above ~1.7; see gpu_solve_v).
        if (fabsf(du0_curr) < ABS_ERR || u0_root == u0_old) break;
    }
    return u0_root;
}

// ── Main reconstruction (matches Reconst::ReconstIt_shell on CPU) ─────────────

__device__ ReconstResult gpu_reconst_raw(float tau, float tauq[5],
                                     const float prev_u[4], float prev_eps,
                                     const float* __restrict__ P_tab,
                                     const float* __restrict__ dPde_tab,
                                     const GPUEosParams& ep) {
    const float ABS_ERR = 1.e-8f;

    ReconstResult res;
    res.rhob   = 0.f;
    res.u[0]   = 1.f;  res.u[1] = 0.f;  res.u[2] = 0.f;  res.u[3] = 0.f;

    float q[5];
    for (int i = 0; i < 5; i++) q[i] = tauq[i] / tau;

    float K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    float M   = sqrtf(K00);
    float T00 = q[0];
    float J0  = q[4];

    // Non-finite input (NaN/Inf propagated from an upstream blow-up, e.g. the
    // viscous W^{mu nu} update at the dilute edge): revert to the previous cell
    // rather than writing NaN downstream, where it would reach the freeze-out
    // finder and abort Cornelius.  Every comparison below is false for NaN, so
    // without this guard a NaN T00 sails through to res.e.
    if (!isfinite(T00) || !isfinite(M)) {
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
        res.u[0] = prev_u[0];
        res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];
        res.u[3] = prev_u[3];
        return res;
    }

    float v_guess = sqrtf(fmaxf(0.f, 1.f - 1.f / (prev_u[0] * prev_u[0] + ABS_ERR)));
    float v_sol   = gpu_solve_v(v_guess, T00, M, J0, P_tab, dPde_tab, ep);

    // Solver could not bracket a root (sentinel < 0): revert to the previous
    // cell instead of fabricating e = T00 from v = 0 (matches CPU revert_grid).
    if (v_sol < 0.f) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
        return res;
    }

    float u0      = 1.f / (sqrtf(fmaxf(0.f, 1.f - v_sol * v_sol)) + v_sol * ABS_ERR);
    float epsilon = T00 - v_sol * M;
    float rhob    = J0 / u0;

    // High-velocity branch (v > v_critical).  The epsilon > 1e-6 guard matches
    // CPU ReconstIt_velocity_Newton — the u0 solve is meaningless at ~zero e.
    if (v_sol > 0.563624f && epsilon > 1e-6f) {
        float u0_sol = gpu_solve_u0(u0, T00, K00, M, J0, P_tab, dPde_tab, ep);
        if (u0_sol >= 1.f) {
            u0      = u0_sol;
            epsilon = T00 - sqrtf(fmaxf(0.f, (1.f - 1.f / (u0 * u0)) * K00));
            rhob    = J0 / u0;
        }
    }

    // Runaway reconstruction: u0 jumped >100x vs the previous step.  The CPU
    // path discards this via revert_grid (check_u0_var).  prev_u[0] >= 1.
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

    // Catch-all: if the solver still produced a non-finite e (e.g. a 0/0 Newton
    // step), revert rather than poison e_future and crash the freeze-out.
    if (!isfinite(res.e)) {
        res.e    = prev_eps;
        res.u[0] = prev_u[0];  res.u[1] = prev_u[1];
        res.u[2] = prev_u[2];  res.u[3] = prev_u[3];
    }
    return res;
}

// Vacuum at rest.  In single precision the energy density of vacuum cells has a
// floor of ~1e-7 1/fm^4 (on the CPU, in double, ~1e-14), and such a cell can
// still come out of the reconstruction with a huge Lorentz factor (u^0 in the
// thousands at the dilute edge of the medium).  T^{tau mu} grows as e u0^2, so
// on the GPU that cell carries an energy-momentum comparable to the dilute
// fluid next to it: when the fluid reaches it, it is dragged to u^0 ~ 100s in
// one step, and the viscous terms then overflow (a whole-grid freeze before the
// non-finite guard, a blow-up after it).  So a vacuum-level cell (e below
// GPU_VACUUM_E, 1e-5 1/fm^4 ~ 2e-6 GeV/fm^3, 1e5 below freeze-out) moving
// faster than u^0 = GPU_VACUUM_U0_MAX is put at rest.  Its e is kept; only the
// spurious kinetic energy of a vacuum cell is dropped.  Real fluid is never
// that dilute, so the evolution of the medium is unaffected.
#define GPU_VACUUM_E       1.e-5f
#define GPU_VACUUM_U0_MAX  10.f

__device__ ReconstResult gpu_reconst(float tau, float tauq[5],
                                     const float prev_u[4], float prev_eps,
                                     const float* __restrict__ P_tab,
                                     const float* __restrict__ dPde_tab,
                                     const GPUEosParams& ep) {
    ReconstResult res = gpu_reconst_raw(tau, tauq, prev_u, prev_eps, P_tab, dPde_tab, ep);
    if (res.e < GPU_VACUUM_E && res.u[0] > GPU_VACUUM_U0_MAX) {
        res.u[0] = 1.f;  res.u[1] = 0.f;  res.u[2] = 0.f;  res.u[3] = 0.f;
    }
    return res;
}

// ── MaxSpeed (matches Advance::MaxSpeed on CPU) ───────────────────────────────

DFI float gpu_max_speed(float tau, int direction, ReconstResult r,
                        const float* __restrict__ P_tab,
                        const float* __restrict__ dPde_tab,
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

DFI float gpu_get_TJb_reconst(ReconstResult r, int alpha, int direction,
                              float tau_fac,
                              const float* __restrict__ P_tab,
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

// ══ Ideal step with NEOS-4D and the charges B, Q, S (params.eos4d == 1) ══════
//
// The 1D-EOS code above is left as it is (its results stay bit-identical).
// With the 4D EOS the conserved vector has params.n_alpha components: 5 with
// baryon number, 7 with B, Q, S. The pressure depends on all three densities,
// and the reconstruction solves for v with rho_i = J_i sqrt(1 - v^2) inside the
// residual, as Reconst::ReconstIt_velocity_Newton does on the CPU; a revert
// keeps the previous cell's charges (Reconst::revert_grid), a regulated cell
// has none (Reconst::regulate_grid).

#define GPU_NA_MAX 7

struct Reconst4D {
    float e;
    float rho[3];   // B, Q, S [1/fm^3]
    float u[4];
    float P;        // P(e, rho) [1/fm^4], for the fluxes and the signal speed
};

DFI float gpu_eos4d_P(float e, const float rho[3],
                      const float* __restrict__ tab, const GPUEos4DParams& p) {
    GPUEos4DState st;
    gpu_eos4d_lookup(e, rho[0], rho[1], rho[2], tab, p, GPU_EOS4D_P, st);
    return st.P;
}

// f(v) = v - M/(T00 + P(e, rho)), e = T00 - v M, rho_i = J_i sqrt(1 - v^2)
DFI void gpu_vel_fdf_4d(float v, float T00, float M, const float J[3],
                        float& fv, float& dfdv,
                        const float* __restrict__ tab, const GPUEos4DParams& p) {
    const float eps = T00 - v * M;
    const float sq  = sqrtf(fmaxf(0.f, 1.f - v * v));
    GPUEos4DState st;
    gpu_eos4d_lookup(eps, J[0]*sq, J[1]*sq, J[2]*sq, tab, p,
                     GPU_EOS4D_P | GPU_EOS4D_DERIV, st);
    const float t1 = T00 + st.P;
    const float t2 = v / fmaxf(sq, 1.e-8f);
    fv   = v - M / t1;
    dfdv = 1.f - M / (t1 * t1) * (M * st.dPde
                                  + t2 * (J[0]*st.dPdrhob + J[1]*st.dPdrhoq
                                          + J[2]*st.dPdrhos));
}

// the bracketed Newton/bisection of gpu_solve_v, with the 4D residual
__device__ float gpu_solve_v_4d(float T00, float M, const float J[3],
                                const float* __restrict__ tab,
                                const GPUEos4DParams& p) {
    const float ABS_ERR = 1.e-7f;
    float fv_l, dfdv_l, fv_h, dfdv_h;
    float v_l = 0.f, v_h = 1.f;
    gpu_vel_fdf_4d(v_l, T00, M, J, fv_l, dfdv_l, tab, p);
    gpu_vel_fdf_4d(v_h, T00, M, J, fv_h, dfdv_h, tab, p);
    if (fabsf(fv_l) < ABS_ERR) return v_l;
    if (fabsf(fv_h) < ABS_ERR) return v_h;
    if (fv_l * fv_h > 0.f) return -1.f;

    float dv_prev = v_h - v_l;
    float dv_curr = dv_prev;
    float v_root  = (v_h + v_l) * 0.5f;
    float fv, dfdv;
    gpu_vel_fdf_4d(v_root, T00, M, J, fv, dfdv, tab, p);
    for (int it = 0; it < 60; it++) {
        const float v_old = v_root;
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
        gpu_vel_fdf_4d(v_root, T00, M, J, fv, dfdv, tab, p);
        if (fv * fv_l < 0.f) { v_h = v_root; fv_h = fv; }
        else                  { v_l = v_root; fv_l = fv; }
        if (fabsf(dv_curr) < ABS_ERR || v_root == v_old) break;
    }
    return v_root;
}

// f(u0) = u0 - (T00 + P)/sqrt((T00 + P)^2 - K00), rho_i = J_i/u0
DFI void gpu_u0_fdf_4d(float u0, float T00, float K00, float M,
                       const float J[3], float& fu0, float& dfdu0,
                       const float* __restrict__ tab,
                       const GPUEos4DParams& p) {
    const float ABS_ERR = 1.e-10f;
    const float v       = sqrtf(fmaxf(0.f, 1.f - 1.f / (u0 * u0)));
    const float epsilon = T00 - v * M;
    const float dedu0   = -M / (u0 * u0 * u0 * v + ABS_ERR);
    const float iu0     = 1.f / u0;
    GPUEos4DState st;
    gpu_eos4d_lookup(epsilon, J[0]*iu0, J[1]*iu0, J[2]*iu0, tab, p,
                     GPU_EOS4D_P | GPU_EOS4D_DERIV, st);
    const float drho = -iu0 * iu0;   // d rho_i/d u0 = -J_i/u0^2
    const float temp1 = (T00 + st.P) * (T00 + st.P) - K00;
    const float den1  = sqrtf(fmaxf(0.f, temp1));
    const float temp  = (T00 + st.P) / fmaxf(den1, ABS_ERR);
    fu0   = u0 - temp;
    dfdu0 = 1.f + (dedu0 * st.dPde
                   + drho * (J[0]*st.dPdrhob + J[1]*st.dPdrhoq
                             + J[2]*st.dPdrhos))
                  * K00 / fmaxf(temp1 * den1, ABS_ERR);
}

__device__ float gpu_solve_u0_4d(float u0_guess, float T00, float K00, float M,
                                 const float J[3],
                                 const float* __restrict__ tab,
                                 const GPUEos4DParams& p) {
    const float ABS_ERR = 1.e-7f;
    float u0_l = fmaxf(1.f, 0.5f * u0_guess);
    float u0_h = fminf(1.e4f, 1.5f * u0_guess);
    if (u0_h < 1.f + ABS_ERR) u0_h = 2.f;
    float fu0_l, dfdu0_l, fu0_h, dfdu0_h;
    gpu_u0_fdf_4d(u0_l, T00, K00, M, J, fu0_l, dfdu0_l, tab, p);
    gpu_u0_fdf_4d(u0_h, T00, K00, M, J, fu0_h, dfdu0_h, tab, p);
    if (fabsf(fu0_l) < ABS_ERR) return u0_l;
    if (fabsf(fu0_h) < ABS_ERR) return u0_h;
    if (fu0_l * fu0_h > 0.f)  return u0_guess;

    float du0_prev = u0_h - u0_l;
    float du0_curr = du0_prev;
    float u0_root  = (u0_h + u0_l) * 0.5f;
    float fu0, dfdu0;
    gpu_u0_fdf_4d(u0_root, T00, K00, M, J, fu0, dfdu0, tab, p);
    for (int it = 0; it < 60; it++) {
        const float u0_old = u0_root;
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
        gpu_u0_fdf_4d(u0_root, T00, K00, M, J, fu0, dfdu0, tab, p);
        if (fu0 * fu0_l < 0.f) { u0_h = u0_root; fu0_h = fu0; }
        else                    { u0_l = u0_root; fu0_l = fu0; }
        if (fabsf(du0_curr) < ABS_ERR || u0_root == u0_old) break;
    }
    return u0_root;
}

DFI void gpu_reconst4d_revert(Reconst4D& r, const float prev_u[4],
                              float prev_e, const float prev_rho[3]) {
    r.e = prev_e;
    for (int i = 0; i < 3; i++) r.rho[i] = prev_rho[i];
    for (int m = 0; m < 4; m++) r.u[m] = prev_u[m];
    r.P = -1.f;   // recomputed by gpu_reconst4d
}

__device__ Reconst4D gpu_reconst4d_raw(float tau, const float tauq[GPU_NA_MAX],
                                       int na, const float prev_u[4],
                                       float prev_e, const float prev_rho[3],
                                       const float* __restrict__ tab,
                                       const GPUEos4DParams& p) {
    const float ABS_ERR = 1.e-8f;
    Reconst4D res;
    float q[GPU_NA_MAX];
    for (int i = 0; i < GPU_NA_MAX; i++) q[i] = (i < na) ? tauq[i] / tau : 0.f;

    const float K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    const float M   = sqrtf(K00);
    const float T00 = q[0];
    const float J[3] = {q[4], q[5], q[6]};

    if (!isfinite(T00) || !isfinite(M)) {
        gpu_reconst4d_revert(res, prev_u, prev_e, prev_rho);
        return res;
    }
    if (T00 < ABS_ERR) {   // Reconst::regulate_grid
        res.e = ABS_ERR;
        res.rho[0] = res.rho[1] = res.rho[2] = 0.f;
        res.u[0] = 1.f; res.u[1] = res.u[2] = res.u[3] = 0.f;
        res.P = -1.f;
        return res;
    }
    if (T00 < M) {
        gpu_reconst4d_revert(res, prev_u, prev_e, prev_rho);
        return res;
    }

    const float v_sol = gpu_solve_v_4d(T00, M, J, tab, p);
    if (v_sol < 0.f) {
        gpu_reconst4d_revert(res, prev_u, prev_e, prev_rho);
        return res;
    }

    float u0      = 1.f / (sqrtf(fmaxf(0.f, 1.f - v_sol * v_sol)) + v_sol * ABS_ERR);
    float epsilon = T00 - v_sol * M;
    if (v_sol > 0.563624f && epsilon > 1e-6f) {
        const float u0_sol = gpu_solve_u0_4d(u0, T00, K00, M, J, tab, p);
        if (u0_sol >= 1.f) {
            u0      = u0_sol;
            epsilon = T00 - sqrtf(fmaxf(0.f, (1.f - 1.f / (u0 * u0)) * K00));
        }
    }
    if (fabsf(u0 - prev_u[0]) / prev_u[0] > 100.f) {
        gpu_reconst4d_revert(res, prev_u, prev_e, prev_rho);
        return res;
    }

    res.e = epsilon;
    for (int i = 0; i < 3; i++) res.rho[i] = J[i] / u0;
    const float P = gpu_eos4d_P(epsilon, res.rho, tab, p);
    const float vel_inv = u0 / (T00 + P);
    res.u[0] = u0;
    res.u[1] = q[1] * vel_inv;
    res.u[2] = q[2] * vel_inv;
    res.u[3] = q[3] * vel_inv;
    const float u_sp_sq = res.u[1]*res.u[1] + res.u[2]*res.u[2] + res.u[3]*res.u[3];
    if (fabsf(u0*u0 - u_sp_sq - 1.f) > ABS_ERR) {
        const float scale = sqrtf(fmaxf(0.f, (u0*u0 - 1.f) / (u_sp_sq + ABS_ERR)));
        res.u[1] *= scale;
        res.u[2] *= scale;
        res.u[3] *= scale;
    }
    res.P = P;
    bool finite = isfinite(res.e);
    for (int i = 0; i < 3; i++) finite = finite && isfinite(res.rho[i]);
    if (!finite) gpu_reconst4d_revert(res, prev_u, prev_e, prev_rho);
    return res;
}

__device__ Reconst4D gpu_reconst4d(float tau, const float tauq[GPU_NA_MAX],
                                   int na, const float prev_u[4], float prev_e,
                                   const float prev_rho[3],
                                   const float* __restrict__ tab,
                                   const GPUEos4DParams& p) {
    Reconst4D r = gpu_reconst4d_raw(tau, tauq, na, prev_u, prev_e, prev_rho,
                                    tab, p);
    if (r.e < GPU_VACUUM_E && r.u[0] > GPU_VACUUM_U0_MAX) {   // see gpu_reconst
        r.u[0] = 1.f;  r.u[1] = 0.f;  r.u[2] = 0.f;  r.u[3] = 0.f;
    }
    if (r.P < 0.f) r.P = gpu_eos4d_P(r.e, r.rho, tab, p);
    return r;
}

// Advance::MaxSpeed with the 4D EOS (c_s^2 from its table)
DFI float gpu_max_speed_4d(float tau, int direction, const Reconst4D& r,
                           const float* __restrict__ tab,
                           const GPUEos4DParams& p) {
    const float gfac = (direction == 3) ? 1.f / tau : 1.f;
    const float utau    = r.u[0];
    const float ux      = fabsf(r.u[direction]);
    const float utau2   = utau * utau;
    const float ut2mux2 = utau2 - ux * ux;
    GPUEos4DState st;
    gpu_eos4d_lookup(r.e, r.rho[0], r.rho[1], r.rho[2], tab, p,
                     GPU_EOS4D_P | GPU_EOS4D_DERIV | GPU_EOS4D_THERMO, st);
    const float cs2    = st.cs2;
    const float num_sq = (ut2mux2 - (ut2mux2 - 1.f) * cs2) * cs2;
    float num;
    if (num_sq >= 0.f) {
        num = utau * ux * (1.f - cs2) + sqrtf(num_sq);
    } else {
        const float dPde = st.dPde;
        const float h    = st.P + r.e;
        num = (dPde < 0.001f)
              ? sqrtf(fmaxf(0.f, -(h*dPde*h*(dPde*(-1.f + ut2mux2) - ut2mux2))))
                - h*(-1.f + dPde)*utau*ux
              : 1.f;
    }
    const float den = utau2 * (1.f - cs2) + cs2;
    float f = num / fmaxf(den, 1.e-20f);
    f = clampf(f, ux / utau, 1.f);
    return f * gfac;
}

// T^{alpha direction} (alpha < 4) or N_i^{direction} (alpha = 4 + i), times
// tau_fac, of a reconstructed state
DFI float gpu_flux_4d(const Reconst4D& r, int alpha, int direction,
                      float tau_fac) {
    const float u_nu = r.u[direction];
    if (alpha >= 4) return r.rho[alpha - 4] * u_nu * tau_fac;
    float gfac = 0.f;
    float u_mu;
    if (alpha == direction) {
        u_mu = u_nu;
        gfac = (alpha == 0) ? -1.f : 1.f;
    } else {
        u_mu = r.u[alpha];
    }
    return ((r.e + r.P) * u_mu * u_nu + r.P * gfac) * tau_fac;
}

// tau T^{tau alpha} (alpha < 4) and tau N_i^tau of cell c, one EOS lookup
DFI void gpu_cell_qi_4d(int c, int Ncells, int na, float tau,
                        const float* __restrict__ eps,
                        const float* __restrict__ rhob,
                        const float* __restrict__ rhoq,
                        const float* __restrict__ rhos,
                        const float* __restrict__ u,
                        const float* __restrict__ tab,
                        const GPUEos4DParams& p, float out[GPU_NA_MAX]) {
    const float e = __ldg(&eps[c]);
    const float rho[3] = {__ldg(&rhob[c]),
                          (na > 5) ? __ldg(&rhoq[c]) : 0.f,
                          (na > 5) ? __ldg(&rhos[c]) : 0.f};
    const float u0 = __ldg(&u[c]);
    const float P = gpu_eos4d_P(e, rho, tab, p);
    out[0] = tau * ((e + P) * u0 * u0 - P);
    for (int a = 1; a < 4; a++) out[a] = tau * ((e + P) * __ldg(&u[a * Ncells + c]) * u0);
    for (int i = 0; i < 3; i++) out[4 + i] = tau * rho[i] * u0;
}

// gpu_make_delta_qi for the 4D EOS
template <bool SYM>
__device__ void gpu_make_delta_qi_4d(
    int ix, int iy, int ieta,
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ qi_out,
    const MUSICGridParams& params,
    const float* __restrict__ tab,
    const GPUEos4DParams& p4)
{
    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;
    const int na     = params.n_alpha;
    const float tau  = params.tau;
    const float theta = params.minmod_theta;
    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    const float e_c = epsilon_curr[c];
    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];
    const float rho_c[3] = {rhob_curr[c],
                            (na > 5) ? rhoq_curr[c] : 0.f,
                            (na > 5) ? rhos_curr[c] : 0.f};

    float qi[GPU_NA_MAX];
    gpu_cell_qi_4d(c, Ncells, na, tau, epsilon_curr, rhob_curr, rhoq_curr,
                   rhos_curr, u_curr, tab, p4, qi);

    const float delta[4]   = {0.f, params.delta_x, params.delta_y, params.delta_eta};
    const float tau_fac[4] = {0.f, tau, tau, 1.f};
    float rhs[GPU_NA_MAX] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float T_eta_m[4] = {0.f, 0.f, 0.f, 0.f};
    float T_eta_p[4] = {0.f, 0.f, 0.f, 0.f};
    const int DX[3]   = {1, 0, 0};
    const int DY[3]   = {0, 1, 0};
    const int DETA[3] = {0, 0, 1};

    for (int dir = 0; dir < 3; dir++) {
        const int direction = dir + 1;
        const int ip1 = clamped_cell(ix+  DX[dir], iy+  DY[dir], ieta+  DETA[dir], Nx, Ny, Neta);
        const int ip2 = clamped_cell(ix+2*DX[dir], iy+2*DY[dir], ieta+2*DETA[dir], Nx, Ny, Neta);
        const int im1 = clamped_cell(ix-  DX[dir], iy-  DY[dir], ieta-  DETA[dir], Nx, Ny, Neta);
        const int im2 = clamped_cell(ix-2*DX[dir], iy-2*DY[dir], ieta-2*DETA[dir], Nx, Ny, Neta);
        float gp1[GPU_NA_MAX], gp2[GPU_NA_MAX], gm1[GPU_NA_MAX], gm2[GPU_NA_MAX];
        gpu_cell_qi_4d(ip1, Ncells, na, tau, epsilon_curr, rhob_curr, rhoq_curr, rhos_curr, u_curr, tab, p4, gp1);
        gpu_cell_qi_4d(ip2, Ncells, na, tau, epsilon_curr, rhob_curr, rhoq_curr, rhos_curr, u_curr, tab, p4, gp2);
        gpu_cell_qi_4d(im1, Ncells, na, tau, epsilon_curr, rhob_curr, rhoq_curr, rhos_curr, u_curr, tab, p4, gm1);
        gpu_cell_qi_4d(im2, Ncells, na, tau, epsilon_curr, rhob_curr, rhoq_curr, rhos_curr, u_curr, tab, p4, gm2);

        float qiphL[GPU_NA_MAX], qiphR[GPU_NA_MAX], qimhL[GPU_NA_MAX], qimhR[GPU_NA_MAX];
        for (int a = 0; a < GPU_NA_MAX; a++) {
            if (a >= na) { qiphL[a] = qiphR[a] = qimhL[a] = qimhR[a] = 0.f; continue; }
            const float gc = qi[a];
            const float fphL =  0.5f * gpu_minmod_dx(gp1[a], gc,     gm1[a], theta);
            const float fphR = -0.5f * gpu_minmod_dx(gp2[a], gp1[a], gc,     theta);
            const float fmhL =  0.5f * gpu_minmod_dx(gc,     gm1[a], gm2[a], theta);
            const float fmhR = -fphL;
            qiphL[a] = gc     + fphL;
            qiphR[a] = gp1[a] + fphR;
            qimhL[a] = gm1[a] + fmhL;
            qimhR[a] = gc     + fmhR;
        }

        // A half-cell state that cannot be reconstructed falls back to a
        // cell's state.  SYM = false (Milne, and Cartesian with
        // symmetric_flux_fallback 0): to this cell's, for all four; cell i
        // and cell i+1 then use different states for the same interface, and
        // its flux out of the one is not its flux into the other.  In the
        // SMASH-fed Cartesian runs (fast, dilute matter) that lost ~10 % of E
        // and B.  SYM = true (gpu_make_delta_qi_eos4d_sym): the left state to
        // the left cell, the right state to the right cell, the same from
        // both sides (Advance::MakeDeltaQI on the CPU alike).  A template, so
        // that the SYM = false code is the old one, bit for bit.
        const Reconst4D r_phL = gpu_reconst4d(tau, qiphL, na, u_c, e_c, rho_c, tab, p4);
        Reconst4D r_phR, r_mhL;
        if constexpr (SYM) {
            float u_p1[4], u_m1[4];
            for (int m = 0; m < 4; m++) {
                u_p1[m] = u_curr[m * Ncells + ip1];
                u_m1[m] = u_curr[m * Ncells + im1];
            }
            const float rho_p1[3] = {rhob_curr[ip1],
                                     (na > 5) ? rhoq_curr[ip1] : 0.f,
                                     (na > 5) ? rhos_curr[ip1] : 0.f};
            const float rho_m1[3] = {rhob_curr[im1],
                                     (na > 5) ? rhoq_curr[im1] : 0.f,
                                     (na > 5) ? rhos_curr[im1] : 0.f};
            r_phR = gpu_reconst4d(tau, qiphR, na, u_p1, epsilon_curr[ip1],
                                  rho_p1, tab, p4);
            r_mhL = gpu_reconst4d(tau, qimhL, na, u_m1, epsilon_curr[im1],
                                  rho_m1, tab, p4);
        } else {
            r_phR = gpu_reconst4d(tau, qiphR, na, u_c, e_c, rho_c, tab, p4);
            r_mhL = gpu_reconst4d(tau, qimhL, na, u_c, e_c, rho_c, tab, p4);
        }
        const Reconst4D r_mhR = gpu_reconst4d(tau, qimhR, na, u_c, e_c, rho_c, tab, p4);

        const float aiph = fmaxf(gpu_max_speed_4d(tau, direction, r_phL, tab, p4),
                                 gpu_max_speed_4d(tau, direction, r_phR, tab, p4));
        const float aimh = fmaxf(gpu_max_speed_4d(tau, direction, r_mhL, tab, p4),
                                 gpu_max_speed_4d(tau, direction, r_mhR, tab, p4));
        const float tf = tau_fac[direction];
        const float dx = delta[direction];
        for (int a = 0; a < na; a++) {
            const float Fiph = 0.5f * ((gpu_flux_4d(r_phL, a, direction, tf)
                                        + gpu_flux_4d(r_phR, a, direction, tf))
                                       - aiph * (qiphR[a] - qiphL[a]));
            const float Fimh = 0.5f * ((gpu_flux_4d(r_mhL, a, direction, tf)
                                        + gpu_flux_4d(r_mhR, a, direction, tf))
                                       - aimh * (qimhR[a] - qimhL[a]));
            if (direction == 3 && (a == 0 || a == 3)) {
                T_eta_m[a] = Fimh;
                T_eta_p[a] = Fiph;
            } else {
                rhs[a] += (Fimh - Fiph) / dx * params.delta_tau;
            }
        }
    }

    const float cd = params.cosh_deta;
    const float sd = params.sinh_deta;
    rhs[0] += (  (T_eta_m[0] - T_eta_p[0]) * cd
               - (T_eta_m[3] + T_eta_p[3]) * sd) * params.delta_tau;
    rhs[3] += (  (T_eta_m[3] - T_eta_p[3]) * cd
               - (T_eta_m[0] + T_eta_p[0]) * sd) * params.delta_tau;
    for (int a = 0; a < na; a++)
        qi_out[a * Ncells + c] = qi[a] + rhs[a];
}

// gpu_finalize_ideal for the 4D EOS
__device__ void gpu_finalize_ideal_4d(
    int c,
    const float* __restrict__ qi_buf,
    const float* __restrict__ dwmn_buf,
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ epsilon_prev,
    const float* __restrict__ rhob_prev,
    const float* __restrict__ rhoq_prev,
    const float* __restrict__ rhos_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ e_future,
    float* __restrict__ rhob_future,
    float* __restrict__ rhoq_future,
    float* __restrict__ rhos_future,
    float* __restrict__ u_future,
    const MUSICGridParams& params,
    const float* __restrict__ qi_source_in,
    const float* __restrict__ tab,
    const GPUEos4DParams& p4)
{
    const int Ncells = params.Ncells;
    const int na     = params.n_alpha;
    const int   rkf      = params.rk_flag;
    const float dt       = params.delta_tau;
    const float tau_org  = params.tau_orig;
    // Cartesian: tau_orig = 1 from the host, tauNextFactor = 1
    const float tau_next = params.cartesian ? 1.f : tau_org + dt;
    const float rk_norm  = 1.f / (1.f + (float)rkf);

    const float u0p = u_prev[c];
    const float e_p = epsilon_prev[c];
    const float rho_p[3] = {rhob_prev[c],
                            (na > 5) ? rhoq_prev[c] : 0.f,
                            (na > 5) ? rhos_prev[c] : 0.f};
    const float P_p = (rkf > 0) ? gpu_eos4d_P(e_p, rho_p, tab, p4) : 0.f;

    float qi[GPU_NA_MAX];
    for (int a = 0; a < GPU_NA_MAX; a++) {
        if (a >= na) { qi[a] = 0.f; continue; }
        // the viscous divergence has five components (q^mu is the baryon
        // current's); Q and S have no diffusion current
        float qv = qi_buf[a * Ncells + c];
        if (a < 5) qv -= dwmn_buf[a * Ncells + c] * dt;
        if (params.has_hydro_source && (a < 5 || params.has_qs_source))
            qv += qi_source_in[a * Ncells + c] * dt;
        if (rkf > 0) {
            float prev_TJb0;
            if (a >= 4) {
                prev_TJb0 = rho_p[a - 4] * u0p;
            } else if (a == 0) {
                prev_TJb0 = (e_p + P_p) * u0p * u0p - P_p;
            } else {
                prev_TJb0 = (e_p + P_p) * u_prev[a * Ncells + c] * u0p;
            }
            qv += (float)rkf * tau_org * prev_TJb0;
        }
        qi[a] = qv * rk_norm;
    }

    float u_c[4];
    for (int m = 0; m < 4; m++) u_c[m] = u_curr[m * Ncells + c];
    const float rho_c[3] = {rhob_curr[c],
                            (na > 5) ? rhoq_curr[c] : 0.f,
                            (na > 5) ? rhos_curr[c] : 0.f};
    const Reconst4D r = gpu_reconst4d(tau_next, qi, na, u_c, epsilon_curr[c],
                                      rho_c, tab, p4);
    e_future[c]    = r.e;
    rhob_future[c] = r.rho[0];
    if (na > 5) {
        rhoq_future[c] = r.rho[1];
        rhos_future[c] = r.rho[2];
    }
    for (int m = 0; m < 4; m++) u_future[m * Ncells + c] = r.u[m];
}

// The two ideal-step kernels of the 4D EOS. Separate from gpu_make_delta_qi
// and gpu_finalize_ideal: with the 4D code inlined into those, the compiled
// 1D path returned non-finite fluxes (seen on the GB10 with CUDA 13), and the
// 1D kernels stay exactly as they were this way.
__global__ void gpu_make_delta_qi_eos4d(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p)
{
    const int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    const int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;
    if (gpu_cell_inactive(params, ix, iy, ieta)) return;   // C3
    gpu_make_delta_qi_4d<false>(ix, iy, ieta, epsilon_curr, rhob_curr, rhoq_curr,
                         rhos_curr, u_curr, qi_out, params, eos4d_tab, eos4d_p);
}

// as gpu_make_delta_qi_eos4d, the failed half-cell states falling back to
// the cell on their side of the interface (Cartesian, symmetric_flux_fallback)
__global__ void gpu_make_delta_qi_eos4d_sym(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p)
{
    const int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    const int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;
    if (gpu_cell_inactive(params, ix, iy, ieta)) return;   // C3
    gpu_make_delta_qi_4d<true>(ix, iy, ieta, epsilon_curr, rhob_curr, rhoq_curr,
                         rhos_curr, u_curr, qi_out, params, eos4d_tab, eos4d_p);
}

__global__ void gpu_finalize_ideal_eos4d(
    const float* __restrict__ qi_buf,
    const float* __restrict__ dwmn_buf,
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ epsilon_prev,
    const float* __restrict__ rhob_prev,
    const float* __restrict__ rhoq_prev,
    const float* __restrict__ rhos_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ e_future,
    float* __restrict__ rhob_future,
    float* __restrict__ rhoq_future,
    float* __restrict__ rhos_future,
    float* __restrict__ u_future,
    MUSICGridParams params,
    const float* __restrict__ qi_source_in,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p)
{
    const int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    const int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;
    if (gpu_cell_inactive(params, ix, iy, ieta)) return;   // C3
    const int c = cell_idx(ix, iy, ieta, params.Nx, params.Ny);
    gpu_finalize_ideal_4d(c, qi_buf, dwmn_buf, epsilon_curr, rhob_curr,
                          rhoq_curr, rhos_curr, u_curr, epsilon_prev,
                          rhob_prev, rhoq_prev, rhos_prev, u_prev,
                          e_future, rhob_future, rhoq_future, rhos_future,
                          u_future, params, qi_source_in, eos4d_tab, eos4d_p);
}

// ── gpu_make_delta_qi kernel ──────────────────────────────────────────────────

template <bool SYM>
__device__ __forceinline__ void gpu_make_delta_qi_body(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_dPde,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    GPUEosParams eos_p,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;
    const float tau  = params.tau;
    const float theta = params.minmod_theta;

    const int c = cell_idx(ix, iy, ieta, Nx, Ny);

    float e_c    = epsilon_curr[c];
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
            float fmhR = -fphL;

            qiphL[alpha] = gc  + fphL;
            qiphR[alpha] = gp1 + fphR;
            qimhL[alpha] = gm1 + fmhL;
            qimhR[alpha] = gc  + fmhR;
        }

        // the fallback cell of a failed reconstruction: see
        // gpu_make_delta_qi_4d (SYM: gpu_make_delta_qi_sym)
        ReconstResult r_phL = gpu_reconst(tau, qiphL, u_c, e_c, eos_P, eos_dPde, eos_p);
        ReconstResult r_phR, r_mhL;
        if constexpr (SYM) {
            float u_p1[4], u_m1[4];
            for (int m = 0; m < 4; m++) {
                u_p1[m] = u_curr[m * Ncells + ip1];
                u_m1[m] = u_curr[m * Ncells + im1];
            }
            r_phR = gpu_reconst(tau, qiphR, u_p1, epsilon_curr[ip1],
                                eos_P, eos_dPde, eos_p);
            r_mhL = gpu_reconst(tau, qimhL, u_m1, epsilon_curr[im1],
                                eos_P, eos_dPde, eos_p);
        } else {
            r_phR = gpu_reconst(tau, qiphR, u_c, e_c, eos_P, eos_dPde, eos_p);
            r_mhL = gpu_reconst(tau, qimhL, u_c, e_c, eos_P, eos_dPde, eos_p);
        }
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
    }  // dir loop

    float cd = params.cosh_deta;
    float sd = params.sinh_deta;
    rhs[0] += (  (T_eta_m[0] - T_eta_p[0]) * cd
               - (T_eta_m[3] + T_eta_p[3]) * sd) * params.delta_tau;
    rhs[3] += (  (T_eta_m[3] - T_eta_p[3]) * cd
               - (T_eta_m[0] + T_eta_p[0]) * sd) * params.delta_tau;

    for (int alpha = 0; alpha < 5; alpha++)
        qi_out[alpha * Ncells + c] = qi[alpha] + rhs[alpha];
}

__global__ void gpu_make_delta_qi(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_dPde,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    GPUEosParams eos_p,
    const unsigned char* __restrict__ active_blk)
{
    gpu_make_delta_qi_body<false>(epsilon_curr, rhob_curr, u_curr, eos_P, eos_dPde, qi_out, params, eos_p, active_blk);
}

__global__ void gpu_make_delta_qi_sym(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_dPde,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    GPUEosParams eos_p,
    const unsigned char* __restrict__ active_blk)
{
    gpu_make_delta_qi_body<true>(epsilon_curr, rhob_curr, u_curr, eos_P, eos_dPde, qi_out, params, eos_p, active_blk);
}

// ── gpu_finalize_ideal ────────────────────────────────────────────────────────

__global__ void gpu_finalize_ideal(
    const float* __restrict__ qi_buf,
    const float* __restrict__ dwmn_buf,
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ epsilon_prev,
    const float* __restrict__ rhob_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ e_future,
    float* __restrict__ rhob_future,
    float* __restrict__ u_future,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_dPde,
    MUSICGridParams params,
    GPUEosParams eos_p,
    const float* __restrict__ qi_source_in,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);

    const int   rkf      = params.rk_flag;
    const float dt       = params.delta_tau;
    const float tau_org  = params.tau_orig;
    // Cartesian: tau_orig = 1 from the host, tauNextFactor = 1
    const float tau_next = params.cartesian ? 1.f : tau_org + dt;
    const float rk_norm  = 1.f / (1.f + (float)rkf);

    float u0p   = u_prev[c];
    float e_p   = epsilon_prev[c];
    float rho_p = rhob_prev[c];
    float P_p   = (rkf > 0) ? gpu_P(e_p, eos_P, eos_p) : 0.f;

    const int has_src = params.has_hydro_source;

    float qi[5];
    for (int a = 0; a < 5; a++) {
        float qv = qi_buf  [a * Ncells + c]
                 - dwmn_buf[a * Ncells + c] * dt;

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

__global__ void gpu_make_uwrhs(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ uwrhs_out,
    MUSICGridParams params,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
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

    const int IDX_1D[5] = {4, 5, 6, 7, 8};

    float u_c0 = u_curr[0 * Ncells + c];
    float u_cd[3];
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

            float W_c   = Wmunu_curr[idx_1d * Ncells + c];
            float W_p1  = Wmunu_curr[idx_1d * Ncells + ip1];
            float W_p2  = Wmunu_curr[idx_1d * Ncells + ip2];
            float W_m1  = Wmunu_curr[idx_1d * Ncells + im1];
            float W_m2  = Wmunu_curr[idx_1d * Ncells + im2];

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

            float f_c   = W_c  * ud_c;    float g_c   = W_c  * u0_c;
            float f_p1  = W_p1 * ud_p1;   float g_p1  = W_p1 * u0_p1;
            float f_p2  = W_p2 * ud_p2;   float g_p2  = W_p2 * u0_p2;
            float f_m1  = W_m1 * ud_m1;   float g_m1  = W_m1 * u0_m1;
            float f_m2  = W_m2 * ud_m2;   float g_m2  = W_m2 * u0_m2;

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

__global__ void gpu_make_uprhs(
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ uprhs_out,
    MUSICGridParams params,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
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

        float pi_c   = pi_b_curr[c];
        float pi_p1  = pi_b_curr[ip1];
        float pi_p2  = pi_b_curr[ip2];
        float pi_m1  = pi_b_curr[im1];
        float pi_m2  = pi_b_curr[im2];

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

        float f_c   = pi_c  * ud_c;    float g_c   = pi_c  * u0_c;
        float f_p1  = pi_p1 * ud_p1;   float g_p1  = pi_p1 * u0_p1;
        float f_p2  = pi_p2 * ud_p2;   float g_p2  = pi_p2 * u0_p2;
        float f_m1  = pi_m1 * ud_m1;   float g_m1  = pi_m1 * u0_m1;
        float f_m2  = pi_m2 * ud_m2;   float g_m2  = pi_m2 * u0_m2;

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

__global__ void gpu_make_du(
    const float* __restrict__ u_curr,
    const float* __restrict__ u_prev,
    float* __restrict__ theta_out,
    float* __restrict__ a_out,
    float* __restrict__ sigma_out,
    MUSICGridParams params,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
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

    for (int n = 1; n <= 3; n++) {
        float f = 0.f;
        for (int m = 1; m <= 3; m++)
            f += dUsup_local[m][n] * u[m];
        dUsup_local[0][n] = f / u[0];
    }

    for (int m = 0; m < 4; m++) {
        float u_m_c = u[m];
        float u_m_p = u_prev[m * Ncells + c];
        dUsup_local[m][0] = -((u_m_c - u_m_p) / delta_tau);
    }
    {
        float f = 0.f;
        for (int m = 1; m < 4; m++)
            f += dUsup_local[m][0] * u[m];
        dUsup_local[0][0] = f / u[0];
    }

    float theta_cell = (-dUsup_local[0][0] + dUsup_local[1][1]
                        + dUsup_local[2][2] + dUsup_local[3][3]
                        + u[0] / tau);
    // Cartesian: no u^tau/tau piece (U_derivative::calculate_expansion_rate)
    if (params.cartesian)
        theta_cell = __fadd_rn(__fadd_rn(__fadd_rn(-dUsup_local[0][0],
                                                   dUsup_local[1][1]),
                                         dUsup_local[2][2]),
                               dUsup_local[3][3]);
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
            if (params.cartesian) {   // no Christoffel terms
                const float sym = __fmul_rn(__fadd_rn(dUsup_local[a][b],
                                                      dUsup_local[b][a]), 0.5f);
                const float proj = __fdiv_rn(
                    __fmul_rn(__fadd_rn(gfac, __fmul_rn(u[a], u[b])),
                              theta_cell), 3.f);
                const float acc = __fmul_rn(
                    __fadd_rn(__fmul_rn(u[a], a_loc[b]),
                              __fmul_rn(u[b], a_loc[a])), 0.5f);
                sigma_local[a][b] = __fadd_rn(__fsub_rn(sym, proj), acc);
            }
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
        for (int b = 1; b < 4; b++)
            s += sigma_local[a][b] * u[b];
        sigma_local[0][a] = s / u[0];
    }
    {
        float s = 0.f;
        for (int a = 1; a < 4; a++)
            s += sigma_local[0][a] * u[a];
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

DFI float gpu_s(float e, const float* __restrict__ s_tab, const GPUEosParams& ep) {
    return gpu_log_interp(e, s_tab, ep);
}

DFI float gpu_T_e(float e, const float* __restrict__ T_tab, const GPUEosParams& ep) {
    return gpu_log_interp(e, T_tab, ep);
}

// ── η/s profile functions (ports of transport_coeffs.cpp) ────────────────────

constexpr float GPU_HBARC = 0.197326980f;   // GeV · fm

DFI float gpu_eta_over_s_default(float T_in_fm, float shear_to_s_baseline) {
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

DFI float gpu_eta_over_s_duke(float T_in_fm,
                              float shear_2_min, float shear_2_slope,
                              float shear_2_curv) {
    float T_in_GeV   = T_in_fm * GPU_HBARC;
    float Ttr_in_GeV = 0.154f;
    float Tfrac      = T_in_GeV / Ttr_in_GeV;
    return shear_2_min
         + shear_2_slope * (T_in_GeV - Ttr_in_GeV) * powf(Tfrac, shear_2_curv);
}

DFI float gpu_eta_over_s_sims(float T_in_fm,
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
    return fmaxf(eta_over_s, eta_over_s_min);
}

DFI float gpu_eta_over_s_profile_mult(float T_in_fm, float shear_to_s_baseline) {
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

DFI float gpu_zeta_over_s_default(float T_in_fm) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float Ttr = 0.18f;
    float dummy = T_in_GeV / Ttr;
    float bulk;
    if (T_in_GeV < 0.995f * Ttr) {
        const float lambda3 = 0.9f;
        const float lambda4 = 0.22f;
        const float sigma3  = 0.0025f;
        const float sigma4  = 0.022f;
        bulk = lambda3 * expf((dummy - 1.f) / sigma3)
             + lambda4 * expf((dummy - 1.f) / sigma4) + 0.03f;
    } else if (T_in_GeV > 1.05f * Ttr) {
        const float lambda1 = 0.9f;
        const float lambda2 = 0.25f;
        const float sigma1  = 0.025f;
        const float sigma2  = 0.13f;
        bulk = lambda1 * expf(-(dummy - 1.f) / sigma1)
             + lambda2 * expf(-(dummy - 1.f) / sigma2) + 0.001f;
    } else {
        const float A1 = -13.77f;
        const float A2 =  27.55f;
        const float A3 =  13.45f;
        bulk = A1 * dummy * dummy + A2 * dummy - A3;
    }
    return bulk;
}

DFI float gpu_zeta_over_s_duke(float T_in_fm,
                               float norm, float width_GeV, float peak_GeV) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float diff_ratio = (T_in_GeV - peak_GeV) / width_GeV;
    return norm / (1.f + diff_ratio * diff_ratio);
}

DFI float gpu_zeta_over_s_sims(float T_in_fm,
                               float max_norm, float width_GeV,
                               float T_peak_GeV, float lambda) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float diff = T_in_GeV - T_peak_GeV;
    float s    = (diff > 0.f) ? 1.f : ((diff < 0.f) ? -1.f : 0.f);
    float diff_ratio = diff / (width_GeV * (lambda * s + 1.f));
    return max_norm / (1.f + diff_ratio * diff_ratio);
}

DFI float gpu_zeta_over_s_asym_gaussian(
        float T_in_fm,
        float B_norm, float B_width_low_GeV, float B_width_high_GeV,
        float Tpeak_GeV) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    float Tdiff    = T_in_GeV - Tpeak_GeV;
    Tdiff = (Tdiff > 0.f) ? (Tdiff / B_width_high_GeV)
                          : (Tdiff / B_width_low_GeV);
    return B_norm * expf(-Tdiff * Tdiff);
}

// Mode 7 — "bigbroadP" (arXiv:1901.04378, 1908.06212): a broad Lorentzian above
// the peak, a narrow Gaussian below.  Fixed parameters (no DATA dependence).
DFI float gpu_zeta_over_s_bigbroadP(float T_in_fm) {
    float T_in_GeV = T_in_fm * GPU_HBARC;
    const float B_norm  = 0.24f;
    const float B_width = 1.5f;
    const float Tpeak   = 0.165f;
    float Ttilde = (T_in_GeV / Tpeak - 1.f) / B_width;
    float bulk   = B_norm / (Ttilde * Ttilde + 1.f);
    if (T_in_GeV < Tpeak) {
        float Tdiff = (T_in_GeV - Tpeak) / 0.01f;
        bulk = B_norm * expf(-Tdiff * Tdiff);
    }
    return bulk;
}

DFI float gpu_zeta_over_s(float T_in_fm, const MUSICGridParams& params) {
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
        case 7:
            zoverS = gpu_zeta_over_s_bigbroadP(T_in_fm);
            break;
        case 8:
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm, 0.13f, 0.01f, 0.12f, 0.160f);
            break;
        case 9:
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
    return fmaxf(0.f, zoverS);
}

// ── Make_uPiSource (port of Diss::Make_uPiSource) ────────────────────────────

DFI float gpu_uPi_source(float pi_b, float theta,
                         float eps_src, float Wsigma_scalar,
                         const float* __restrict__ P_tab,
                         const float* __restrict__ dPde_tab,
                         const float* __restrict__ T_tab,
                         const GPUEosParams& ep,
                         const MUSICGridParams& params,
                         float delta_tau)
{
    float P_local  = gpu_P (eps_src, P_tab, ep);
    float T_local  = gpu_T_e(eps_src, T_tab, ep);
    float cs2      = clampf(gpu_dPde(eps_src, dPde_tab, ep), 0.01f, 0.333333f);

    float zeta_s   = gpu_zeta_over_s(T_local, params);
    float epsP     = fmaxf(eps_src + P_local, 1.e-20f);
    float T_safe   = fmaxf(T_local, 1.e-20f);
    float bulk_zeta = zeta_s * epsP / T_safe;

    float csfactor = fmaxf(1.f/3.f - cs2, 1.e-20f);
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
    tau_Pi = fminf(10.f, fmaxf(3.f * delta_tau, tau_Pi));

    const float delta_PiPi = 2.f / 3.f;
    float transport_coeff1 = delta_PiPi * tau_Pi;

    float NS_term  = -bulk_zeta * theta;
    float relax    = -pi_b - transport_coeff1 * theta * pi_b;

    float coupling_to_shear = 0.f;
    if (params.include_second_order_terms == 1) {
        const float lambda_bulkPipi = 8.f / 5.f;
        float transport_coeff1_s =
                lambda_bulkPipi * (1.f/3.f - cs2) * tau_Pi;
        coupling_to_shear = -Wsigma_scalar * transport_coeff1_s;
    }

    return (NS_term + relax + coupling_to_shear) / tau_Pi;
}

// Dispatch on T_dependent_shear_to_s.  Mirrors get_eta_over_s().
DFI float gpu_eta_over_s(float T_in_fm, const MUSICGridParams& params) {
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
            return params.shear_to_s;
    }
}

// Algebraic Make_uWSource (port of Diss::Make_uWSource).
DFI float gpu_uW_source(
        int mu, int nu,
        const float W4[4][4],
        const float s4[4][4],
        const float u[4],
        float pi_b,
        float theta, float eps_src,
        float Wsigma_scalar, float Wsquare,
        const float* __restrict__ P_tab,
        const float* __restrict__ s_tab,
        const float* __restrict__ T_tab,
        const GPUEosParams& ep,
        const MUSICGridParams& params,
        float delta_tau)
{
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
    if (params.include_second_order_terms == 1
        && params.init_profile_zero == 0) {
        const float phi7_45      = (9.f / 70.f) * (4.f / 5.f);
        float transport_coeff_W  = phi7_45 * tau_pi / fmaxf(shear, 1.e-20f);
        const float tau_pipi     = 10.f / 7.f;
        float transport_coeff_Ws = tau_pipi * tau_pi;

        float gmunu_uu  = ((mu == nu)
                            ? ((mu == 0) ? -1.f : 1.f)
                            : 0.f)
                        + u[mu] * u[nu];

        float t1_Ws = (
              -W4[mu][0]*s4[nu][0] - W4[nu][0]*s4[mu][0]
              +W4[mu][1]*s4[nu][1] + W4[nu][1]*s4[mu][1]
              +W4[mu][2]*s4[nu][2] + W4[nu][2]*s4[mu][2]
              +W4[mu][3]*s4[nu][3] + W4[nu][3]*s4[mu][3]
              ) * 0.5f;
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

    return (NS_term + relax + Wsigma_term + WW_term + Coupling_to_Bulk)
           / tau_pi;
}

// Algebraic Make_uWRHS geometric tail (matches Diss::Make_uWRHS_geom).
DFI float gpu_uWRHS_geom(
        const float W_local[4][4], const float u[4],
        const float a_loc[4],
        int mu, int nu, float theta, float tau, float delta_tau,
        int cartesian)
{
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

    float res = tempf * delta_tau
           + (-(u[0] * W_local[mu][nu]) / tau
              + theta * W_local[mu][nu]) * delta_tau;
    if (cartesian) {
        // Cartesian: no Christoffel terms and no -u^tau W/tau (_rn
        // intrinsics: the Milne arithmetic above compiles as before)
        float tc = 0.f;
        for (int ic = 0; ic < 4; ic++) {
            const float ic_fac = (ic == 0) ? -1.f : 1.f;
            const float aw = __fmul_rn(a_loc[ic], ic_fac);
            tc = __fadd_rn(tc, __fadd_rn(
                     __fmul_rn(__fmul_rn(W_local[ic][nu], u[mu]), aw),
                     __fmul_rn(__fmul_rn(W_local[ic][mu], u[nu]), aw)));
        }
        res = __fadd_rn(__fmul_rn(tc, delta_tau),
                        __fmul_rn(__fmul_rn(theta, W_local[mu][nu]),
                                  delta_tau));
    }
    return res;
}

// ── non-finite counter ───────────────────────────────────────────────────────
// Cell updates in which gpu_first_rk_step_w_full found a non-finite W^{mu nu}
// or Pi and set it to 0.  A device global (no kernel argument, no buffer), so
// the host accessor has to live in this translation unit.
__device__ unsigned int d_nonfinite_count = 0u;

unsigned int gpu_nonfinite_count_read_and_reset(void* stream_v) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    unsigned int count = 0u;
    const unsigned int zero = 0u;
    cudaMemcpyFromSymbolAsync(&count, d_nonfinite_count, sizeof(count), 0,
                              cudaMemcpyDeviceToHost, stream);
    cudaMemcpyToSymbolAsync(d_nonfinite_count, &zero, sizeof(zero), 0,
                            cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return count;
}

// ── gpu_first_rk_step_w_full ─────────────────────────────────────────────────

__global__ void gpu_first_rk_step_w_full(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ Wmunu_prev,
    const float* __restrict__ pi_b_prev,
    const float* __restrict__ u_prev,
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ epsilon_prev,
    const float* __restrict__ u_future,
    const float* __restrict__ uwrhs_in,
    const float* __restrict__ theta_in,
    const float* __restrict__ a_in,
    const float* __restrict__ sigma_in,
    float* __restrict__ Wmunu_future,
    float* __restrict__ pi_b_future,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_s,
    const float* __restrict__ eos_T,
    const float* __restrict__ eos_dPde,
    const float* __restrict__ uprhs_in,
    MUSICGridParams params,
    GPUEosParams eos_p,
    const float* __restrict__ epsilon_future,
    const float* __restrict__ rhob_future,
    const unsigned char* __restrict__ active_blk)
{
    if (gpu_block_inactive(active_blk)) return;
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);

    const int   rkf       = params.rk_flag;
    const float dt        = params.delta_tau;
    const float tau_now   = params.tau;
    const float rk_norm   = 1.f / (1.f + (float)rkf);

    if (params.turn_on_shear == 0) {
        for (int m = 0; m < 14; m++)
            Wmunu_future[m * Ncells + c] = 0.f;
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
        int mu  = MU_LIST[k];
        int nu  = NU_LIST[k];
        int id  = WMUNU_IDX[mu][nu];

        float uwrhs_flux = uwrhs_in[k * Ncells + c];

        float SW = gpu_uW_source(
                       mu, nu, W4, s4, u_c, pi_c_curr,
                       theta, eps_src,
                       Wsigma_scalar, Wsquare,
                       eos_P, eos_s, eos_T, eos_p, params, dt);

        float w_rhs_geom = gpu_uWRHS_geom(W4, u_c, a_loc, mu, nu,
                                          theta, tau_now, dt,
                                          params.cartesian);
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
        for (int nu = 1; nu < 4; nu++)
            s += Wf[WMUNU_IDX[mu][nu]] * u_f[nu];
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

        float p_rhs = uprhs_in[c]
                    + (-(u_c[0] * pi_c) / tau_now + theta * pi_c) * dt;
        if (params.cartesian)   // no -u^tau Pi/tau
            p_rhs = __fadd_rn(uprhs_in[c],
                              __fmul_rn(__fmul_rn(theta, pi_c), dt));

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
        float factor  = 10.f * params.quest_revert_strength
                            * (sig_e - sig_off);

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

        if (isnan(rho_shear)) {
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

    // Non-finite guard.  An overflow in a very dilute cell (e.g. a spurious u^0
    // jump at the edge of the medium) can leave W^{mu nu} or Pi at inf/NaN, and
    // the regulator above cannot remove it (inf * 0 = NaN, and NaN > x is
    // false).  It would then spread through the flux stencil, and
    // gpu_reconstruct reverts every cell it reaches, so the whole grid stops
    // evolving while MUSIC runs on to its maximum time.  Set such a tensor to 0
    // and count the cell update (gpu_nonfinite_count_read_and_reset).  Only
    // non-finite values trigger it, so normal runs are unchanged.
    bool bad_w = false;
    for (int m = 0; m < 14; m++) bad_w |= !isfinite(Wf[m]);
    const bool bad_pi = !isfinite(pi_b_out);
    if (bad_w) for (int m = 0; m < 14; m++) Wf[m] = 0.f;
    if (bad_pi) pi_b_out = 0.f;
    if (bad_w || bad_pi) atomicAdd(&d_nonfinite_count, 1u);

    for (int m = 0; m < 14; m++)
        Wmunu_future[m * Ncells + c] = Wf[m];
    pi_b_future[c] = pi_b_out;
}

DFI float gpu_zeta_over_s_muB(float T_in_fm, float muB_in_fm,
                          const MUSICGridParams& params) {
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
        case 7:
            zoverS = gpu_zeta_over_s_bigbroadP(T_in_fm);
            break;
        case 8:
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm, 0.13f, 0.01f, 0.12f, 0.160f);
            break;
        case 9:
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm, 0.175f, 0.01f, 0.12f, 0.160f);
            break;
        case 10: {
            // the peak height and position move with mu_B
            // (TransportCoeffs::get_zeta_over_s); at mu_B = 0 they are
            // bulk_10_max and bulk_10_Tpeak exactly
            const float muB_GeV = muB_in_fm * GPU_HBARC;
            float peak;
            if (muB_GeV < 0.2f) {
                peak = params.bulk_asym10_max
                       + (params.bulk_asym10_max_muB0p2 - params.bulk_asym10_max)
                         / 0.2f * muB_GeV;
            } else if (muB_GeV < 0.4f) {
                peak = params.bulk_asym10_max_muB0p2
                       + (params.bulk_asym10_max_muB0p4
                          - params.bulk_asym10_max_muB0p2)
                         / 0.2f * (muB_GeV - 0.2f);
            } else {
                peak = params.bulk_asym10_max_muB0p4;
            }
            const float Tpeak = params.bulk_asym10_Tpeak
                + params.bulk_asym10_Tpeak_muBcurv * muB_GeV * muB_GeV;
            zoverS = gpu_zeta_over_s_asym_gaussian(
                        T_in_fm, peak,
                        params.bulk_asym10_width_low,
                        params.bulk_asym10_width_high,
                        Tpeak);
            break;
        }
        default:
            zoverS = 0.f;
            break;
    }
    return fmaxf(0.f, zoverS);
}

// ── Make_uPiSource with precomputed thermodynamics (4D EOS) ─────────────────

// the local thermodynamics the viscous sources need: from the 1D tables
// (e only) or from the NEOS-4D lookup with the charges
struct GPUThermo {
    float P, T, s, cs2, muB;
};

DFI GPUThermo gpu_thermo_1d(float e, const float* __restrict__ P_tab,
                            const float* __restrict__ s_tab,
                            const float* __restrict__ T_tab,
                            const float* __restrict__ dPde_tab,
                            const GPUEosParams& ep) {
    GPUThermo th;
    th.P   = gpu_P(e, P_tab, ep);
    th.T   = gpu_T_e(e, T_tab, ep);
    th.s   = gpu_s(e, s_tab, ep);
    th.cs2 = clampf(gpu_dPde(e, dPde_tab, ep), 0.01f, 0.333333f);
    th.muB = 0.f;
    return th;
}

DFI GPUThermo gpu_thermo_4d(float e, float rhob, float rhoq, float rhos,
                            const float* __restrict__ tab,
                            const GPUEos4DParams& p) {
    GPUEos4DState st;
    gpu_eos4d_lookup(e, rhob, rhoq, rhos, tab, p,
                     GPU_EOS4D_P | GPU_EOS4D_THERMO, st);
    GPUThermo th;
    th.P   = st.P;
    th.T   = st.T;
    th.s   = gpu_eos4d_entropy(e, rhob, rhoq, rhos, st);
    th.cs2 = st.cs2;
    th.muB = st.muB;
    return th;
}

DFI float gpu_uPi_source_th(float pi_b, float theta,
                         float eps_src, float Wsigma_scalar,
                         const GPUThermo& th,
                         const MUSICGridParams& params,
                         float delta_tau)
{
    float P_local  = th.P;
    float T_local  = th.T;
    float cs2      = th.cs2;

    float zeta_s   = gpu_zeta_over_s_muB(T_local, th.muB, params);
    float epsP     = fmaxf(eps_src + P_local, 1.e-20f);
    float T_safe   = fmaxf(T_local, 1.e-20f);
    float bulk_zeta = zeta_s * epsP / T_safe;

    float csfactor = fmaxf(1.f/3.f - cs2, 1.e-20f);
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
    tau_Pi = fminf(10.f, fmaxf(3.f * delta_tau, tau_Pi));

    const float delta_PiPi = 2.f / 3.f;
    float transport_coeff1 = delta_PiPi * tau_Pi;

    float NS_term  = -bulk_zeta * theta;
    float relax    = -pi_b - transport_coeff1 * theta * pi_b;

    float coupling_to_shear = 0.f;
    if (params.include_second_order_terms == 1) {
        const float lambda_bulkPipi = 8.f / 5.f;
        float transport_coeff1_s =
                lambda_bulkPipi * (1.f/3.f - cs2) * tau_Pi;
        coupling_to_shear = -Wsigma_scalar * transport_coeff1_s;
    }

    return (NS_term + relax + coupling_to_shear) / tau_Pi;
}

// eta/s(T, mu_B): gpu_eta_over_s times the mu_B factor of
// TransportCoeffs::get_eta_over_s (modes 7 and 10), clipped at 0 as there;
// used with the 4D EOS only, so the 1D path is unchanged
DFI float gpu_eta_over_s_muB(float T_in_fm, float muB_in_fm,
                             const MUSICGridParams& params) {
    float eta_s = gpu_eta_over_s(T_in_fm, params);
    if (params.muB_dep_shear_mode == 7) {
        const float m = fabsf(muB_in_fm) * GPU_HBARC;
        const float f0 = 1.f, f1 = params.shear_muBf0p2, f2 = params.shear_muBf0p4;
        float f;
        if (m < 0.2f)      f = f0 + (f1 - f0) / 0.2f * m;
        else if (m < 0.4f) f = f1 + (f2 - f1) / 0.2f * (m - 0.2f);
        else               f = f2;
        eta_s *= f;
    } else if (params.muB_dep_shear_mode == 10) {
        // |mu_B|, as TransportCoeffs::get_muB_dependence_shear_profile
        eta_s *= 1.f + params.shear_muBDep_slope
                       * powf(fabsf(muB_in_fm) * GPU_HBARC
                              / params.shear_muBDep_scale,
                              params.shear_muBDep_alpha);
    }
    return fmaxf(0.f, eta_s);
}

// Algebraic Make_uWSource (port of Diss::Make_uWSource).
DFI float gpu_uW_source_th(
        int mu, int nu,
        const float W4[4][4],
        const float s4[4][4],
        const float u[4],
        float pi_b,
        float theta, float eps_src,
        float Wsigma_scalar, float Wsquare,
        const GPUThermo& th,
        const MUSICGridParams& params,
        float delta_tau)
{
    float P       = th.P;
    float entropy = th.s;
    float T_local = th.T;
    float shear_to_s_T = params.eos4d
                         ? gpu_eta_over_s_muB(T_local, th.muB, params)
                         : gpu_eta_over_s(T_local, params);
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
    if (params.include_second_order_terms == 1
        && params.init_profile_zero == 0) {
        const float phi7_45      = (9.f / 70.f) * (4.f / 5.f);
        float transport_coeff_W  = phi7_45 * tau_pi / fmaxf(shear, 1.e-20f);
        const float tau_pipi     = 10.f / 7.f;
        float transport_coeff_Ws = tau_pipi * tau_pi;

        float gmunu_uu  = ((mu == nu)
                            ? ((mu == 0) ? -1.f : 1.f)
                            : 0.f)
                        + u[mu] * u[nu];

        float t1_Ws = (
              -W4[mu][0]*s4[nu][0] - W4[nu][0]*s4[mu][0]
              +W4[mu][1]*s4[nu][1] + W4[nu][1]*s4[mu][1]
              +W4[mu][2]*s4[nu][2] + W4[nu][2]*s4[mu][2]
              +W4[mu][3]*s4[nu][3] + W4[nu][3]*s4[mu][3]
              ) * 0.5f;
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

    return (NS_term + relax + Wsigma_term + WW_term + Coupling_to_Bulk)
           / tau_pi;
}

// ── gpu_first_rk_step_w_full with the 4D EOS ─────────────────────────────────
//
// gpu_first_rk_step_w_full and its source terms with T, s, P, cs2 and mu_B
// from the NEOS-4D lookup at the cell's densities, and eta/s, zeta/s with
// their mu_B dependence. A copy, so that the 1D-EOS kernel and its results
// stay exactly as they were.

__global__ void gpu_first_rk_step_w_full_eos4d(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ Wmunu_prev,
    const float* __restrict__ pi_b_prev,
    const float* __restrict__ u_prev,
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ epsilon_prev,
    const float* __restrict__ u_future,
    const float* __restrict__ uwrhs_in,
    const float* __restrict__ theta_in,
    const float* __restrict__ a_in,
    const float* __restrict__ sigma_in,
    float* __restrict__ Wmunu_future,
    float* __restrict__ pi_b_future,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_s,
    const float* __restrict__ eos_T,
    const float* __restrict__ eos_dPde,
    const float* __restrict__ uprhs_in,
    MUSICGridParams params,
    GPUEosParams eos_p,
    const float* __restrict__ epsilon_future,
    const float* __restrict__ rhob_future,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhob_prev,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhoq_prev,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ rhos_prev,
    const float* __restrict__ rhoq_future,
    const float* __restrict__ rhos_future,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p,
    const float* __restrict__ uwrhs_q_in,
    const float* __restrict__ dmuBT_in)
{
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;
    if (gpu_cell_inactive(params, ix, iy, ieta)) return;   // C3

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);

    const int   rkf       = params.rk_flag;
    const float dt        = params.delta_tau;
    const float tau_now   = params.tau;
    const float rk_norm   = 1.f / (1.f + (float)rkf);

    // nothing to evolve: W, Pi (and q) stay zero. Unlike the 1D kernel, bulk
    // and diffusion are evolved without shear as well (as on the CPU).
    if (params.turn_on_shear == 0 && params.turn_on_bulk == 0
        && params.turn_on_diff == 0) {
        for (int m = 0; m < 14; m++)
            Wmunu_future[m * Ncells + c] = 0.f;
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

    // the thermodynamics of the source cell (Advance::FirstRKStepW: the
    // current cell at rk 0, the previous one at rk 1), once per cell
    GPUThermo th;
    if (params.eos4d) {
        const bool q = (params.n_alpha > 5);
        const float rb = (rkf == 0) ? rhob_curr[c] : rhob_prev[c];
        const float rq = q ? ((rkf == 0) ? rhoq_curr[c] : rhoq_prev[c]) : 0.f;
        const float rs = q ? ((rkf == 0) ? rhos_curr[c] : rhos_prev[c]) : 0.f;
        th = gpu_thermo_4d(eps_src, rb, rq, rs, eos4d_tab, eos4d_p);
    } else {
        th = gpu_thermo_1d(eps_src, eos_P, eos_s, eos_T, eos_dPde, eos_p);
    }

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
        int mu  = MU_LIST[k];
        int nu  = NU_LIST[k];
        int id  = WMUNU_IDX[mu][nu];

        float uwrhs_flux = uwrhs_in[k * Ncells + c];

        float SW = gpu_uW_source_th(
                       mu, nu, W4, s4, u_c, pi_c_curr,
                       theta, eps_src,
                       Wsigma_scalar, Wsquare,
                       th, params, dt);

        float w_rhs_geom = gpu_uWRHS_geom(W4, u_c, a_loc, mu, nu,
                                          theta, tau_now, dt,
                                          params.cartesian);
        float w_rhs = uwrhs_flux + w_rhs_geom;

        float tempf =
              (1.f - (float)rkf) * (Wc[id] * u_c[0])
            +        (float)rkf  * (Wp_cell[id] * u_p[0]);
        tempf += SW * dt;
        tempf += w_rhs;
        tempf += (float)rkf * (Wc[id] * u_c[0]);
        tempf *= rk_norm;

        Wf[id] = (params.turn_on_shear == 1) ? tempf / u_f[0] : 0.f;
    }

    Wf[9] = ( 2.f * (  u_f[1]*u_f[2]*Wf[5]
                     + u_f[1]*u_f[3]*Wf[6]
                     + u_f[2]*u_f[3]*Wf[8])
              - (u_f[0]*u_f[0] - u_f[1]*u_f[1]) * Wf[4]
              - (u_f[0]*u_f[0] - u_f[2]*u_f[2]) * Wf[7])
            / fmaxf(u_f[0]*u_f[0] - u_f[3]*u_f[3], 1.e-10f);

    for (int mu = 1; mu < 4; mu++) {
        float s = 0.f;
        for (int nu = 1; nu < 4; nu++)
            s += Wf[WMUNU_IDX[mu][nu]] * u_f[nu];
        Wf[mu] = s / u_f[0];
    }
    {
        float s = 0.f;
        for (int nu = 1; nu < 4; nu++) s += Wf[nu] * u_f[nu];
        Wf[0] = s / u_f[0];
    }

    for (int m = 10; m < 14; m++) Wf[m] = 0.f;
    if (params.turn_on_diff == 1) {
        // Diss::Make_uqSource and the q part of Advance::FirstRKStepW, with
        // the thermodynamics of the source cell (th) and its rho_B
        const float small_eps = 1.e-16f;
        const float rhob_src = (rkf == 0) ? rhob_curr[c] : rhob_prev[c];
        const float T = th.T;
        float tau_rho = params.kappa_coefficient / fmaxf(T, small_eps);
        tau_rho = fminf(10.f, fmaxf(3.f * dt, tau_rho));
        const float alpha_mu = th.muB / fmaxf(T, small_eps);
        const float den = 3.f * T * tanhf(alpha_mu);
        const float denorm_safe = copysignf(fmaxf(fabsf(den), small_eps), den);
        const float kappa = params.kappa_coefficient
            * (rhob_src / denorm_safe
               - rhob_src * rhob_src / fmaxf(eps_src + th.P, small_eps));
        float qc[4], qp[4];
        for (int i = 0; i < 4; i++) {
            qc[i] = Wc[10 + i];
            qp[i] = Wp_cell[10 + i];
        }
        float dmu[4];
        for (int n = 0; n < 4; n++) dmu[n] = dmuBT_in[n * Ncells + c];
        const float a4 = dmuBT_in[4 * Ncells + c];
        const float tc  = params.delta_qq_coeff  * tau_rho;
        const float tc2 = params.lambda_qq_coeff * tau_rho;
        float q_dot_a = 0.f;   // q^i g_ii a^i
        for (int i = 0; i < 4; i++)
            q_dot_a += qc[i] * ((i == 0) ? -1.f : 1.f) * a_loc[i];
        for (int nu = 1; nu < 4; nu++) {
            const float NS = kappa * (dmu[nu] + u_c[nu] * a4);
            const float NL1 = -tc * qc[nu] * theta;
            float qs = 0.f;   // q^i sigma_i^nu
            for (int i = 0; i < 4; i++)
                qs += qc[i] * s4[i][nu] * ((i == 0) ? -1.f : 1.f);
            const float NL2 = -tc2 * qs;
            float src = (-qc[nu] - NS + NL1 + NL2) / tau_rho;
            const float src_relax = src;
            src += (theta - u_c[0] / tau_now) * qc[nu];
            // the Milne terms: (g^{nu tau} + u^nu u^tau) u^eta q^eta/tau
            //                - (g^{nu eta} + u^nu u^eta) u^eta q^tau/tau
            const float g_nu3 = (nu == 3) ? 1.f : 0.f;
            src += (u_c[nu] * u_c[0]) * u_c[3] * qc[3] / tau_now
                   - (g_nu3 + u_c[nu] * u_c[3]) * u_c[3] * qc[0] / tau_now;
            // Cartesian: theta has no u^tau/tau piece, no Christoffel terms
            if (params.cartesian)
                src = __fadd_rn(src_relax, __fmul_rn(theta, qc[nu]));
            src += u_c[nu] * q_dot_a;

            float t = (1.f - (float)rkf) * (qc[nu] * u_c[0])
                    +        (float)rkf  * (qp[nu] * u_p[0]);
            t += src * dt;
            t += uwrhs_q_in[(nu - 1) * Ncells + c];
            t += (float)rkf * (qc[nu] * u_c[0]);
            t *= rk_norm;
            Wf[10 + nu] = t / u_f[0];
        }
        // q^tau from transversality
        Wf[10] = (Wf[11] * u_f[1] + Wf[12] * u_f[2] + Wf[13] * u_f[3]) / u_f[0];
    }

    float pi_b_out;
    if (params.turn_on_bulk == 1) {
        float pi_c = pi_b_curr[c];
        float pi_p = pi_b_prev[c];

        float p_rhs = uprhs_in[c]
                    + (-(u_c[0] * pi_c) / tau_now + theta * pi_c) * dt;
        if (params.cartesian)   // no -u^tau Pi/tau
            p_rhs = __fadd_rn(uprhs_in[c],
                              __fmul_rn(__fmul_rn(theta, pi_c), dt));

        float SPi = gpu_uPi_source_th(pi_c, theta, eps_src, Wsigma_scalar,
                                   th, params, dt);

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
        float factor  = 10.f * params.quest_revert_strength
                            * (sig_e - sig_off);

        float pi00 = Wf[0], pi01 = Wf[1], pi02 = Wf[2], pi03 = Wf[3];
        float pi11 = Wf[4], pi12 = Wf[5], pi13 = Wf[6];
        float pi22 = Wf[7], pi23 = Wf[8];
        float pi33 = Wf[9];
        float pisize =
              pi00*pi00 + pi11*pi11 + pi22*pi22 + pi33*pi33
            - 2.f * (pi01*pi01 + pi02*pi02 + pi03*pi03)
            + 2.f * (pi12*pi12 + pi13*pi13 + pi23*pi23);

        float bulksize = 3.f * pi_b_out * pi_b_out;

        float p_local;
        if (params.eos4d) {
            const bool q = (params.n_alpha > 5);
            const float rho_f[3] = {rhob_future[c],
                                    q ? rhoq_future[c] : 0.f,
                                    q ? rhos_future[c] : 0.f};
            p_local = gpu_eos4d_P(e_local, rho_f, eos4d_tab, eos4d_p);
        } else {
            p_local = gpu_P(e_local, eos_P, eos_p);
        }
        float eq_size  = e_local * e_local + 3.f * p_local * p_local;
        eq_size        = fmaxf(eq_size, 1.e-30f);

        // As Advance::QuestRevert: a negative pi^{mu nu} pi_{mu nu} gives NaN
        // there and sets W to 0. (The 1D kernel clamps it to 0 instead, which
        // leaves W unregulated: in a fast cell, u^0 ~ 100 and more at the
        // dilute edge of a SMASH initial state, W's components are ~ u0^2 its
        // rest-frame size, the float invariant drowns in their cancellation
        // and came out <= 0, and W grew without bound, to 1e32.)
        float rho_shear = sqrtf(pisize / eq_size) / factor;
        float rho_bulk  = sqrtf(fmaxf(bulksize, 0.f) / eq_size) / factor;

        if (isnan(rho_shear)) {
            for (int m = 0; m < 10; m++) Wf[m] = 0.f;
        } else if (rho_shear > rho_shear_max) {
            float scale = rho_shear_max / rho_shear;
            for (int m = 0; m < 10; m++) Wf[m] *= scale;
        }
        if (rho_bulk > rho_bulk_max) {
            pi_b_out *= rho_bulk_max / rho_bulk;
        }
        if (params.turn_on_diff == 1) {
            // Advance::QuestRevert_qmu (energy and rho_B of the new state)
            float q_size = -Wf[10]*Wf[10] + Wf[11]*Wf[11]
                           + Wf[12]*Wf[12] + Wf[13]*Wf[13];
            if (q_size < 0.f) {
                for (int m = 10; m < 14; m++) Wf[m] = 0.f;
            }
            const float rhob_f = rhob_future[c];
            const float rho_q = sqrtf(q_size / (rhob_f * rhob_f)) / factor;
            const float rho_q_max = 0.1f;
            if (rho_q > rho_q_max) {
                const float scale = rho_q_max / rho_q;
                for (int m = 10; m < 14; m++) Wf[m] *= scale;
            }
        }

        (void)rhob_future;
    }

    // Non-finite guard.  An overflow in a very dilute cell (e.g. a spurious u^0
    // jump at the edge of the medium) can leave W^{mu nu} or Pi at inf/NaN, and
    // the regulator above cannot remove it (inf * 0 = NaN, and NaN > x is
    // false).  It would then spread through the flux stencil, and
    // gpu_reconstruct reverts every cell it reaches, so the whole grid stops
    // evolving while MUSIC runs on to its maximum time.  Set such a tensor to 0
    // and count the cell update (gpu_nonfinite_count_read_and_reset).  Only
    // non-finite values trigger it, so normal runs are unchanged.
    bool bad_w = false;
    for (int m = 0; m < 14; m++) bad_w |= !isfinite(Wf[m]);
    const bool bad_pi = !isfinite(pi_b_out);
    if (bad_w) for (int m = 0; m < 14; m++) Wf[m] = 0.f;
    if (bad_pi) pi_b_out = 0.f;
    if (bad_w || bad_pi) atomicAdd(&d_nonfinite_count, 1u);

    for (int m = 0; m < 14; m++)
        Wmunu_future[m * Ncells + c] = Wf[m];
    pi_b_future[c] = pi_b_out;
}

// ── Net baryon diffusion with the 4D EOS (turn_on_diff) ─────────────────────
//
// mu_B/T of the current and the previous step (one lookup per cell each),
// its gradient partial^mu (mu_B/T) and its derivative along u, as
// U_derivative::MakeDSpatial / MakeDTau / calculate_Du_supmu do for
// dUsup[4][mu]; and the KT flux of u^a q^i (Diss::Make_uWRHS, components 11
// to 13). The q update itself is in gpu_first_rk_step_w_full_eos4d.

__global__ void gpu_muBT_eos4d(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ epsilon_prev,
    const float* __restrict__ rhob_prev,
    const float* __restrict__ rhoq_prev,
    const float* __restrict__ rhos_prev,
    float* __restrict__ muBT_out,
    MUSICGridParams params,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p)
{
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= params.Ncells) return;
    if (gpu_cell_inactive(params, c % params.Nx, (c / params.Nx) % params.Ny,
                          c / (params.Nx * params.Ny))) return;   // C3
    const bool q = (params.n_alpha > 5);
    GPUEos4DState st;
    gpu_eos4d_lookup(epsilon_curr[c], rhob_curr[c],
                     q ? rhoq_curr[c] : 0.f, q ? rhos_curr[c] : 0.f,
                     eos4d_tab, eos4d_p, GPU_EOS4D_P | GPU_EOS4D_THERMO, st);
    muBT_out[c] = st.muB / st.T;
    gpu_eos4d_lookup(epsilon_prev[c], rhob_prev[c],
                     q ? rhoq_prev[c] : 0.f, q ? rhos_prev[c] : 0.f,
                     eos4d_tab, eos4d_p, GPU_EOS4D_P | GPU_EOS4D_THERMO, st);
    muBT_out[params.Ncells + c] = st.muB / st.T;
}

__global__ void gpu_make_dmuBT(
    const float* __restrict__ muBT,
    const float* __restrict__ u_curr,
    float* __restrict__ dmuBT_out,
    MUSICGridParams params)
{
    const int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    const int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;
    if (gpu_cell_inactive(params, ix, iy, ieta)) return;   // C3
    const int Nx = params.Nx, Ny = params.Ny, Neta = params.Neta;
    const int Ncells = params.Ncells;
    const int c = cell_idx(ix, iy, ieta, Nx, Ny);
    const float delta[4] = {0.f, params.delta_x, params.delta_y,
                            params.delta_eta * params.tau};
    const int DX[3] = {1, 0, 0}, DY[3] = {0, 1, 0}, DETA[3] = {0, 0, 1};

    float d[4];
    const float f = muBT[c];
    for (int dir = 0; dir < 3; dir++) {
        const int ip1 = clamped_cell(ix + DX[dir], iy + DY[dir],
                                     ieta + DETA[dir], Nx, Ny, Neta);
        const int im1 = clamped_cell(ix - DX[dir], iy - DY[dir],
                                     ieta - DETA[dir], Nx, Ny, Neta);
        d[dir + 1] = gpu_minmod_dx(muBT[ip1], f, muBT[im1],
                                   params.minmod_theta) / delta[dir + 1];
    }
    // backward difference in tau, g^{00} = -1
    d[0] = -(f - muBT[Ncells + c]) / params.delta_tau;
    float a4 = 0.f;
    for (int n = 0; n < 4; n++) {
        const float tfac = (n == 0) ? -1.f : 1.f;
        a4 += tfac * u_curr[n * Ncells + c] * d[n];
        dmuBT_out[n * Ncells + c] = d[n];
    }
    dmuBT_out[4 * Ncells + c] = a4;
}

// gpu_make_uwrhs for the spatial components of q^mu (Wmunu 11, 12, 13)
__global__ void gpu_make_uwrhs_q(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ uwrhs_q_out,
    MUSICGridParams params)
{
    int ix   = blockIdx.x * blockDim.x + threadIdx.x;
    int iy   = blockIdx.y * blockDim.y + threadIdx.y;
    int ieta = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= params.Nx || iy >= params.Ny || ieta >= params.Neta) return;
    if (gpu_cell_inactive(params, ix, iy, ieta)) return;   // C3

    const int Nx     = params.Nx;
    const int Ny     = params.Ny;
    const int Neta   = params.Neta;
    const int Ncells = params.Ncells;
    const int c      = cell_idx(ix, iy, ieta, Nx, Ny);
    const float delta[4] = {0.f, params.delta_x, params.delta_y,
                            params.delta_eta * params.tau};
    const float theta_l   = params.minmod_theta;
    const int DX[3] = {1, 0, 0}, DY[3] = {0, 1, 0}, DETA[3] = {0, 0, 1};

    for (int k = 0; k < 3; k++) {
        const int idx_1d = 11 + k;
        float flux = 0.f;
        for (int dir = 0; dir < 3; dir++) {
            const int direction = dir + 1;
            const int ip1 = clamped_cell(ix +   DX[dir], iy +   DY[dir], ieta +   DETA[dir], Nx, Ny, Neta);
            const int ip2 = clamped_cell(ix + 2*DX[dir], iy + 2*DY[dir], ieta + 2*DETA[dir], Nx, Ny, Neta);
            const int im1 = clamped_cell(ix -   DX[dir], iy -   DY[dir], ieta -   DETA[dir], Nx, Ny, Neta);
            const int im2 = clamped_cell(ix - 2*DX[dir], iy - 2*DY[dir], ieta - 2*DETA[dir], Nx, Ny, Neta);
            const int cs[5] = {c, ip1, ip2, im1, im2};
            float W[5], ud[5], u0[5];
            for (int j = 0; j < 5; j++) {
                W[j]  = Wmunu_curr[idx_1d * Ncells + cs[j]];
                ud[j] = u_curr[direction * Ncells + cs[j]];
                u0[j] = u_curr[cs[j]];
            }
            const float f_c  = W[0]*ud[0], g_c  = W[0]*u0[0];
            const float f_p1 = W[1]*ud[1], g_p1 = W[1]*u0[1];
            const float f_p2 = W[2]*ud[2], g_p2 = W[2]*u0[2];
            const float f_m1 = W[3]*ud[3], g_m1 = W[3]*u0[3];
            const float f_m2 = W[4]*ud[4], g_m2 = W[4]*u0[4];

            const float uWphR = f_p1 - 0.5f * gpu_minmod_dx(f_p2, f_p1, f_c, theta_l);
            const float temp  = 0.5f * gpu_minmod_dx(f_p1, f_c, f_m1, theta_l);
            const float uWphL = f_c + temp;
            const float uWmhR = f_c - temp;
            const float uWmhL = f_m1 + 0.5f * gpu_minmod_dx(f_c, f_m1, f_m2, theta_l);
            const float WphR  = g_p1 - 0.5f * gpu_minmod_dx(g_p2, g_p1, g_c, theta_l);
            const float temp2 = 0.5f * gpu_minmod_dx(g_p1, g_c, g_m1, theta_l);
            const float WphL  = g_c + temp2;
            const float WmhR  = g_c - temp2;
            const float WmhL  = g_m1 + 0.5f * gpu_minmod_dx(g_c, g_m1, g_m2, theta_l);

            const float a   = fabsf(ud[0]) / u0[0];
            const float ap1 = fabsf(ud[1]) / u0[1];
            const float am1 = fabsf(ud[3]) / u0[3];
            float ax = fmaxf(a, ap1);
            const float HWph = ((uWphR + uWphL) - ax * (WphR - WphL)) * 0.5f;
            ax = fmaxf(a, am1);
            const float HWmh = ((uWmhR + uWmhL) - ax * (WmhR - WmhL)) * 0.5f;
            flux += -((HWph - HWmh) / delta[direction]);
        }
        uwrhs_q_out[k * Ncells + c] = flux * params.delta_tau;
    }
}

// ── Max-reduction over epsilon and rhob ───────────────────────────────────────
//
// Uses dynamic shared memory (2 * blockDim.x floats).
// IEEE-754 trick: for non-negative floats, bit-casting to int preserves order,
// so atomicMax on the integer representation gives the correct float maximum.

__global__ void gpu_reduce_max_eps_edge(
    const float* __restrict__ epsilon,
    int Nx, int Ny, int Ncells, int fx, int fy,
    float* out_eps)
{
    extern __shared__ float sdata[];

    const int tid    = static_cast<int>(threadIdx.x);
    int       idx    = static_cast<int>(blockIdx.x * blockDim.x) + tid;
    const int stride = static_cast<int>(blockDim.x * gridDim.x);

    float le = 0.f;
    for (; idx < Ncells; idx += stride) {
        const int ix = idx % Nx;                  // cell_idx: Nx*(Ny*ieta + iy) + ix
        const int iy = (idx / Nx) % Ny;
        if (ix <= fx || ix >= Nx - 2*fx || iy <= fy || iy >= Ny - 2*fy) {
            le = fmaxf(le, __ldg(&epsilon[idx]));
        }
    }
    sdata[tid] = le;
    __syncthreads();

    for (int s = static_cast<int>(blockDim.x) >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(out_eps), __float_as_int(sdata[0]));
    }
}

// ── Gather the cells the freeze-out search reads ─────────────────────────────

__global__ void gpu_gather_cells(
    const float* __restrict__ epsilon,
    const float* __restrict__ rhob,
    const float* __restrict__ u,
    const float* __restrict__ Wmunu,
    const float* __restrict__ pi_b,
    const float* __restrict__ kept,
    const int* __restrict__ cells, int n, int Ncells,
    float* __restrict__ out,
    const float* __restrict__ rhoq,
    const float* __restrict__ rhos,
    int nf)
{
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n) return;
    const size_t c  = static_cast<size_t>(cells[j]);
    const size_t nc = static_cast<size_t>(Ncells);
    const size_t m  = static_cast<size_t>(n);
    out[j]         = epsilon[c];
    out[m + j]     = rhob[c];
    for (int k = 0; k < 4; ++k)  out[(2 + k)*m + j] = u[k*nc + c];
    for (int k = 0; k < 14; ++k) out[(6 + k)*m + j] = Wmunu[k*nc + c];
    out[20*m + j]  = pi_b[c];
    if (nf > 21) {   // the charges, when they are on the device
        out[21*m + j] = rhoq[c];
        out[22*m + j] = rhos[c];
    }
    if (kept) {
        for (int f = 0; f < nf; ++f) out[(nf + f)*m + j] = kept[f*nc + c];
    }
}

__global__ void gpu_drain_cells(
    float* __restrict__ epsilon,
    float* __restrict__ rhob,
    float* __restrict__ u,
    float* __restrict__ Wmunu,
    float* __restrict__ pi_b,
    float* __restrict__ rhoq,
    float* __restrict__ rhos,
    const int* __restrict__ cells, const float* __restrict__ keep,
    int n, int Ncells, float e_vac)
{
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n) return;
    const size_t c  = static_cast<size_t>(cells[j]);
    const size_t nc = static_cast<size_t>(Ncells);
    const float k = keep[j];
    if (k > 0.f) {
        epsilon[c] *= k;
        rhob[c]    *= k;
        for (int m = 0; m < 14; ++m) Wmunu[m*nc + c] *= k;
        pi_b[c] *= k;
        if (rhoq) rhoq[c] *= k;
        if (rhos) rhos[c] *= k;
        return;
    }
    epsilon[c] = e_vac;
    rhob[c]    = 0.f;
    u[c]       = 1.f;
    for (int m = 1; m < 4; ++m)  u[m*nc + c] = 0.f;
    for (int m = 0; m < 14; ++m) Wmunu[m*nc + c] = 0.f;
    pi_b[c] = 0.f;
    if (rhoq) rhoq[c] = 0.f;
    if (rhos) rhos[c] = 0.f;
}

__global__ void gpu_reduce_max_eps_rhob(
    const float* __restrict__ epsilon,
    const float* __restrict__ rhob,
    int Ncells,
    float* out_eps,
    float* out_rhob)
{
    extern __shared__ float sdata[];
    float* s_eps  = sdata;
    float* s_rhob = sdata + blockDim.x;

    const int tid    = static_cast<int>(threadIdx.x);
    int       idx    = static_cast<int>(blockIdx.x * blockDim.x) + tid;
    const int stride = static_cast<int>(blockDim.x * gridDim.x);

    float le = 0.f, lr = 0.f;
    for (; idx < Ncells; idx += stride) {
        le = fmaxf(le, __ldg(&epsilon[idx]));
        lr = fmaxf(lr, __ldg(&rhob[idx]));
    }
    s_eps[tid]  = le;
    s_rhob[tid] = lr;
    __syncthreads();

    for (int s = static_cast<int>(blockDim.x) >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_eps[tid]  = fmaxf(s_eps[tid],  s_eps[tid + s]);
            s_rhob[tid] = fmaxf(s_rhob[tid], s_rhob[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(out_eps),  __float_as_int(s_eps[0]));
        atomicMax(reinterpret_cast<int*>(out_rhob), __float_as_int(s_rhob[0]));
    }
}

// ── NEOS-4D lookup at given points (self-test, MUSIC_EOS4D_SELFTEST) ────────
// in[4*i + k] = (e, rhob, rhoq, rhos)_i; out[GPU_EOS4D_TEST_NOUT*i + k] =
// (P, T, muB, muQ, muS, cs2, dPde, dPdrhob, dPdrhoq, dPdrhos, s)_i

__global__ void gpu_eos4d_eval(const float* __restrict__ in, int n,
                               const float* __restrict__ tab,
                               GPUEos4DParams p, float* __restrict__ out)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float e = in[4*i], rb = in[4*i + 1], rq = in[4*i + 2], rs = in[4*i + 3];
    GPUEos4DState st;
    gpu_eos4d_lookup(e, rb, rq, rs, tab, p,
                     GPU_EOS4D_P | GPU_EOS4D_DERIV | GPU_EOS4D_THERMO, st);
    float* o = out + static_cast<size_t>(GPU_EOS4D_TEST_NOUT) * i;
    o[0] = st.P;    o[1] = st.T;       o[2] = st.muB;     o[3] = st.muQ;
    o[4] = st.muS;  o[5] = st.cs2;     o[6] = st.dPde;    o[7] = st.dPdrhob;
    o[8] = st.dPdrhoq; o[9] = st.dPdrhos;
    o[10] = gpu_eos4d_entropy(e, rb, rq, rs, st);
}

// ── C3: build the active region ───────────────────────────────────────────────
//
// gpu_active_mark: one thread per cell.  A cell is active if e > eps_act (when
// epsilon is given; NaN counts as active) or if any of its 5 source
// components is non-zero (when qi_source is given).  An active cell sets the
// flag of every block (bx x by x bz cells, gx x gy blocks per eta layer) that
// holds a cell within `halo` cells of it.  The stores race only with stores
// of the same value.
__global__ void gpu_active_mark(
    const float* __restrict__ epsilon,
    const float* __restrict__ qi_source,
    float eps_act, int halo,
    int Nx, int Ny, int Neta,
    int bx, int by, int bz, int gx, int gy,
    unsigned char* __restrict__ active_blk)
{
    const int Ncells = Nx * Ny * Neta;
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= Ncells) return;
    bool act = false;
    if (epsilon) act = !(epsilon[c] <= eps_act);
    if (!act && qi_source) {
        for (int a = 0; a < 5; ++a) act |= (qi_source[a * Ncells + c] != 0.f);
    }
    if (!act) return;
    const int ix   = c % Nx;
    const int iy   = (c / Nx) % Ny;
    const int ieta = c / (Nx * Ny);
    const int x0 = max(ix - halo, 0) / bx,   x1 = min(ix + halo, Nx - 1) / bx;
    const int y0 = max(iy - halo, 0) / by,   y1 = min(iy + halo, Ny - 1) / by;
    const int z0 = max(ieta - halo, 0) / bz, z1 = min(ieta + halo, Neta - 1) / bz;
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                active_blk[x + gx * (y + gy * z)] = 1;
}

// gpu_active_freeze: one thread per cell.  A cell whose block was active
// (old_blk) and is not any more (new_blk) is frozen: snap_curr's fields are
// copied into snap_prev and snap_future, so that the three buffers agree and
// the pointer rotations of the RK steps leave the cell where it is.
__global__ void gpu_active_freeze(
    const unsigned char* __restrict__ old_blk,
    const unsigned char* __restrict__ new_blk,
    int Nx, int Ny, int Neta,
    int bx, int by, int bz, int gx, int gy,
    GPUSnapPtrs curr, GPUSnapPtrs prev, GPUSnapPtrs future)
{
    const int Ncells = Nx * Ny * Neta;
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= Ncells) return;
    const int ix   = c % Nx;
    const int iy   = (c / Nx) % Ny;
    const int ieta = c / (Nx * Ny);
    const int b = ix / bx + gx * (iy / by + gy * (ieta / bz));
    if (!old_blk[b] || new_blk[b]) return;
    const float e = curr.epsilon[c], r = curr.rhob[c], p = curr.pi_b[c];
    prev.epsilon[c] = e;  future.epsilon[c] = e;
    prev.rhob[c]    = r;  future.rhob[c]    = r;
    prev.pi_b[c]    = p;  future.pi_b[c]    = p;
    for (int m = 0; m < 4; ++m) {
        const float v = curr.u[m * Ncells + c];
        prev.u[m * Ncells + c] = v;  future.u[m * Ncells + c] = v;
    }
    for (int m = 0; m < 14; ++m) {
        const float v = curr.Wmunu[m * Ncells + c];
        prev.Wmunu[m * Ncells + c] = v;  future.Wmunu[m * Ncells + c] = v;
    }
    if (curr.rhoq) {      // Q and S on the device (NEOS-4D)
        const float q = curr.rhoq[c], s = curr.rhos[c];
        prev.rhoq[c] = q;  future.rhoq[c] = q;
        prev.rhos[c] = s;  future.rhos[c] = s;
    }
}
