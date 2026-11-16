// __global__ kernel prototypes shared between music_kernels.cu (definitions)
// and CUDAPipelines.cu (launches).  Keeping the signatures in one place avoids
// silent launch/definition mismatches.
//
// Only included from .cu translation units (compiled by nvcc).

#pragma once
#include "gpu_types.h"

__global__ void gpu_make_w_source(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ Wmunu_prev,
    const float* __restrict__ pi_b_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ dwmn_out,
    MUSICGridParams params);

// Phase 3: shared-memory tiled variant of gpu_make_w_source.  Identical result,
// but the current-snapshot Wmunu/u/pi_b are cooperatively staged into a
// (blockDim + 2)^3 halo tile in dynamic shared memory, so the radius-1 stencil
// reads hit shared memory instead of global.  Launch with dynamic shared bytes
// = (14 + 4 + 1) * (bx+2)*(by+2)*(bz+2) * sizeof(float).
__global__ void gpu_make_w_source_tiled(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ Wmunu_prev,
    const float* __restrict__ pi_b_prev,
    const float* __restrict__ u_prev,
    float* __restrict__ dwmn_out,
    MUSICGridParams params);

__global__ void gpu_make_delta_qi(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ u_curr,
    const float* __restrict__ eos_P,
    const float* __restrict__ eos_dPde,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    GPUEosParams eos_p);

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
    const float* __restrict__ qi_source_in);

// the ideal step with the NEOS-4D EOS and the B, Q, S densities
// (params.eos4d == 1, params.n_alpha = 5 or 7)
__global__ void gpu_make_delta_qi_eos4d(
    const float* __restrict__ epsilon_curr,
    const float* __restrict__ rhob_curr,
    const float* __restrict__ rhoq_curr,
    const float* __restrict__ rhos_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ qi_out,
    MUSICGridParams params,
    const float* __restrict__ eos4d_tab,
    GPUEos4DParams eos4d_p);

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
    GPUEos4DParams eos4d_p);

__global__ void gpu_make_uwrhs(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ uwrhs_out,
    MUSICGridParams params);

__global__ void gpu_make_uprhs(
    const float* __restrict__ pi_b_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ uprhs_out,
    MUSICGridParams params);

__global__ void gpu_make_du(
    const float* __restrict__ u_curr,
    const float* __restrict__ u_prev,
    float* __restrict__ theta_out,
    float* __restrict__ a_out,
    float* __restrict__ sigma_out,
    MUSICGridParams params);

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
    const float* __restrict__ rhob_future);

// gpu_first_rk_step_w_full with the NEOS-4D thermodynamics (params.eos4d)
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
    const float* __restrict__ dmuBT_in);

// Block-parallel max-reduction over epsilon and rhob in snap_curr.
// Each block reduces blockDim.x cells via shared memory; blocks collaborate
// via atomicMax (IEEE-754 trick for non-negative floats).
// out_eps and out_rhob must be initialized to 0 before the kernel is launched.
__global__ void gpu_reduce_max_eps_rhob(
    const float* __restrict__ epsilon,
    const float* __restrict__ rhob,
    int Ncells,
    float* out_eps,
    float* out_rhob);

// max(epsilon) over the transverse edge bands of snap_curr only: the cells
// with ix <= fx, ix >= Nx - 2*fx, iy <= fy or iy >= Ny - 2*fy (the bands in
// which Cornelius flags a crossing as a boundary hit), every eta.  Same
// shared-memory + atomicMax scheme as gpu_reduce_max_eps_rhob; out_eps must
// be 0 before the launch, so a band with only e <= 0 reads 0.
__global__ void gpu_reduce_max_eps_edge(
    const float* __restrict__ epsilon,
    int Nx, int Ny, int Ncells, int fx, int fy,
    float* out_eps);

// All 21 fields (epsilon, rhob, u[4], Wmunu[14], pi_b) of the n cells
// cells[0..n), from the snapshot arrays and, unless null, from `kept` (a
// snapshot copied into one array in that order, 21 x Ncells floats).
// Writes field-major: out[f*n + j] from the snapshot, out[(21 + f)*n + j] from
// `kept`.  For the freeze-out search (CUDAPipelines::gather_cells).
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
    int nf);

// Pack ideal-hydro output for the down-sampled evolution grid (Phase 2b).
// One thread per output cell; reads snap_curr (epsilon, u) and the device EOS
// tables, writes 8 floats per cell into `out` in fluidCell_ideal field order
// (eta, sd, ed, pressure, temperature, ux, uy, ueta).  e/p/T are hbarc-scaled.
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
    GPUEos4DParams eos4d_p);

// Cell updates in which gpu_first_rk_step_w_full set a non-finite W^{mu nu} or
// Pi to 0 since the last call; resets the count.  Synchronizes `stream`.
unsigned int gpu_nonfinite_count_read_and_reset(void* stream);

// NEOS-4D lookup at n points (self-test): see music_kernels.cu
__global__ void gpu_eos4d_eval(const float* __restrict__ in, int n,
                               const float* __restrict__ tab,
                               GPUEos4DParams p, float* __restrict__ out);

// net baryon diffusion with the 4D EOS (see music_kernels.cu)
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
    GPUEos4DParams eos4d_p);

__global__ void gpu_make_dmuBT(
    const float* __restrict__ muBT,
    const float* __restrict__ u_curr,
    float* __restrict__ dmuBT_out,
    MUSICGridParams params);

__global__ void gpu_make_uwrhs_q(
    const float* __restrict__ Wmunu_curr,
    const float* __restrict__ u_curr,
    float* __restrict__ uwrhs_q_out,
    MUSICGridParams params);
