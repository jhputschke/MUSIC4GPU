// Shared type definitions usable from both C++ and Metal Shading Language.
// Keep this header free of any C++ STL or Metal-specific includes.
//
// Memory layout convention (SoA, component-major):
//   field[comp * Ncells + cell]  where cell = Nx*(Ny*ieta + iy) + ix
//
// Wmunu index map (mirrors Util::map_2d_idx_to_1d):
//   (alpha,dir)  ->  1D index
//   (0,0)->0  (0,1)->1  (0,2)->2  (0,3)->3
//   (1,1)->4  (1,2)->5  (1,3)->6
//   (2,2)->7  (2,3)->8
//   (3,3)->9
//   (4,0)->10 (4,1)->11 (4,2)->12 (4,3)->13
//   off-diagonal entries are symmetric.

#ifndef MUSIC_GPU_TYPES_H
#define MUSIC_GPU_TYPES_H

struct MUSICGridParams {
    int   Nx;
    int   Ny;
    int   Neta;
    int   Ncells;         // Nx * Ny * Neta
    float delta_x;
    float delta_y;
    float delta_eta;
    float delta_tau;
    float tau;            // current tau (rk-corrected) = tau_orig + rk_flag*delta_tau
    int   boost_invariant;
    int   turn_on_bulk;
    int   turn_on_diff;   // baryon diffusion flag
    float cosh_deta;      // precomputed geometric factor
    float sinh_deta;      // precomputed geometric factor
    float minmod_theta;   // flux limiter parameter (used by gpu_make_delta_qi)
    int   rk_flag;        // RK sub-step index (0 or 1); used by gpu_finalize_ideal
    float tau_orig;       // tau at the start of this RK step (un-shifted)

    // Transport / config inputs for gpu_first_rk_step_w_full (Tier 3c Phase 2).
    // Phase-2 v1 only supports the constant-shear branch
    // (T_dependent_shear_to_s == 0, muB_dependent_shear_to_s == 0,
    //  include_second_order_terms == 0, include_vorticity_terms == 0,
    //  turn_on_diff == 0).  The host falls back to CPU FirstRKStepW for
    //  any other configuration.
    float shear_to_s;             // DATA.shear_to_s   (constant η/s)
    float shear_relax_time_factor; // DATA.shear_relax_time_factor
    int   turn_on_shear;          // DATA.turn_on_shear flag
};

// EOS table sampled on a uniform grid: P(e) and dP/de(e) at rhob=0.
// Covers the majority of use cases (zero net baryon density).
// For finite-muB EOS the CPU fallback is used.
#define GPU_EOS_N 8192

struct GPUEosParams {
    float e_min;    // lower bound (0)
    float e_max;    // upper bound (eos.get_eps_max())
    float delta_e;  // spacing = (e_max - e_min) / (n_pts - 1)
    int   n_pts;    // number of sample points (GPU_EOS_N)

    // Entropy s(e) is sampled in log-e because s ~ e^(3/4) varies over many
    // decades — a linear table at the same e_max would lose most resolution
    // in the dilute regime where the bulk of hydro evolution lives.
    // s_table[i] = s(exp(log_e_min + i * log_delta_e)).
    float log_e_min;     // log(s_e_min)
    float log_e_max;     // log(s_e_max) == log(e_max)
    float log_delta_e;   // (log_e_max - log_e_min) / (n_pts - 1)
};

// Wmunu 2D->1D index table (same as Util::map_2d_idx_to_1d)
// Use as: WMUNU_IDX[alpha][direction]  for alpha in [0,4], direction in [0,3]
#ifdef __METAL_VERSION__
constant int WMUNU_IDX[5][4] = {
#else
static const int WMUNU_IDX[5][4] = {
#endif
    { 0,  1,  2,  3},   // alpha=0 (tau)
    { 1,  4,  5,  6},   // alpha=1 (x)
    { 2,  5,  7,  8},   // alpha=2 (y)
    { 3,  6,  8,  9},   // alpha=3 (eta)
    {10, 11, 12, 13}    // alpha=4 (baryon q^mu)
};

// Metric g^{mu nu} = diag(-1,+1,+1,+1)
#ifdef __METAL_VERSION__
constant float GMUNU_DIAG[4] = {-1.f, 1.f, 1.f, 1.f};
#else
static const float GMUNU_DIAG[4] = {-1.f, 1.f, 1.f, 1.f};
#endif

#endif // MUSIC_GPU_TYPES_H
