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
    GPUPackParams pp);
