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
    float tau;            // current tau (rk-corrected)
    int   boost_invariant;
    int   turn_on_bulk;
    int   turn_on_diff;   // baryon diffusion flag
    float cosh_deta;      // precomputed geometric factor
    float sinh_deta;      // precomputed geometric factor
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
