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
