// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_ADVANCE_H_
#define SRC_ADVANCE_H_

#include <memory>
#include "data.h"
#include "cell.h"
#include "grid.h"
#include "dissipative.h"
#include "minmod.h"
#include "u_derivative.h"
#include "reconst.h"
#include "hydro_source_base.h"
#include "pretty_ostream.h"

#ifdef USE_METAL
#include "gpu/GPUGrid.h"
#include "gpu/MetalPipelines.h"
#include "gpu/gpu_types.h"
// Shared alias so the GPU dispatch logic in advance.cpp is back-end agnostic.
using GPUPipelines = MetalPipelines;
#elif defined(USE_CUDA)
#include "gpu/GPUGrid.h"
#include "gpu/CUDAPipelines.h"
#include "gpu/gpu_types.h"
using GPUPipelines = CUDAPipelines;
#endif

// Single switch covering either GPU back-end.
#if defined(USE_METAL) || defined(USE_CUDA)
#define MUSIC_USE_GPU 1
#endif

class Advance {
 private:
    const InitData &DATA;
    const EOS &eos;
    std::shared_ptr<HydroSourceBase> hydro_source_terms_ptr;

    Diss diss_helper;
    Minmod minmod;
    Reconst reconst_helper;
    pretty_ostream music_message;

    bool flag_add_hydro_source;

#ifdef MUSIC_USE_GPU
    GPUGrid    gpu_grid_;
    bool       gpu_ready_ = false;
    bool       metal_initialized_ = false;
    // True when the previous AdvanceIt substep wrote a complete fresh state
    // into gpu_grid_.snap_future and skipped the CPU copy-back.  The next
    // AdvanceIt entry rotates GPU snapshots and skips the AoS→SoA upload
    // instead of going through the CPU arena.
    bool       gpu_state_authoritative_ = false;
    // Run-level flag: GPU holds the evolving state across timestep boundaries.
    // When true, AdvanceIt skips the H2D upload at the start of rk0 because
    // snap_curr/snap_prev are already current (maintained by swap_curr_future +
    // lockstep rotations).  Set on the first complete full-GPU step.
    bool       gpu_owns_state_ = false;
    void       init_metal_if_needed(SCGrid &arena_current);
    void       make_gpu_params(double tau, int rk_flag,
                               MUSICGridParams &p) const;
#endif

 public:
    Advance(const EOS &eosIn, const InitData &DATA_in,
            std::shared_ptr<HydroSourceBase> hydro_source_ptr_in);

    void AdvanceIt(const double tau_init,
                   SCGrid &arena_prev, SCGrid &arena_current,
                   SCGrid &arena_future, const int rk_flag);

#ifdef MUSIC_USE_GPU
    // Mirror the rk1 host arena swap in GPU snapshot space (snap_curr ↔
    // snap_future).  Called from Evolve::AdvanceRK immediately after the
    // std::swap so the GPU and host pointer roles stay in lockstep.
    void swap_curr_future_gpu();

    // True when the GPU holds the authoritative evolving state across timestep
    // boundaries (H2D upload at rk0 is skipped).
    bool gpu_owns_state() const { return gpu_owns_state_; }

    // GPU max-reduction over snap_curr.epsilon / .rhob.  Returns results in
    // 1/fm^4 units (same as Cell_small::epsilon).  Synchronizes before return.
    // No-op if GPU residency is not active.
    void reduce_max_gpu(double& eps_max, double& rhob_max);
#endif

    // gpu_dwmn_base: pointer to the first alpha-component of the GPU-computed
    // dwmn buffer (component-major, stride = Ncells).  Pass nullptr to run
    // the CPU MakeWSource fallback.
    // gpu_qi_base: pointer to the first alpha-component of the GPU-computed
    // qi_out buffer (component-major, stride = Ncells).  Pass nullptr to run
    // the CPU MakeDeltaQI fallback.
    void FirstRKStepT(const double tau, const double x_local,
                      const double y_local, const double eta_s_local,
                      SCGrid &arena_current, SCGrid &arena_future,
                      SCGrid &arena_prev, const int ix, const int iy,
                      const int ieta, const int rk_flag,
                      const float* gpu_dwmn_base = nullptr,
                      const float* gpu_qi_base   = nullptr,
                      int Ncells = 0);

    // gpu_uwrhs_base (optional): pointer to the first idx-component of the
    // GPU-computed Make_uWRHS stencil flux for the current cell, layout
    // [k * Ncells + cell] with k = idx_1d - 4 (k in [0..4] for the 5 shear
    // indices).  If non-null, FirstRKStepW reads the flux part from this
    // buffer and only computes the per-cell geometric / algebraic tail on
    // the CPU.  Pass nullptr to fall back to Diss::Make_uWRHS().
    void FirstRKStepW(const double tau_it, SCGrid &arena_prev,
                      SCGrid &arena_current, SCGrid &arena_future,
                      const int rk_flag, const double theta_local,
                      const DumuVec &a_local,
                      const VelocityShearVec &sigma_local,
                      const VorticityVec &omega_local,
                      const DmuMuBoverTVec &baryon_diffusion_vector,
                      const int ieta, const int ix, const int iy,
                      const float* gpu_uwrhs_base = nullptr,
                      int Ncells = 0);

    void UpdateTJbRK(const ReconstCell &grid_rk, Cell_small &grid_pt);
    void QuestRevert(const double tau, Cell_small *grid_pt,
                     const int ieta, const int ix, const int iy);
    void QuestRevert_qmu(const double tau, Cell_small *grid_pt,
                         const int ieta, const int ix, const int iy);

    void MakeDeltaQI(const double tau, SCGrid &arena_current,
                     const int ix, const int iy, const int ieta, TJbVec &qi,
                     const int rk_flag);
    double MaxSpeed(const double tau, const int direc,
                    const ReconstCell &grid_p);

    double get_TJb(const ReconstCell &grid_p, const int rk_flag,
                   const int mu, const int nu);
    double get_TJb(const Cell_small &grid_p, const int mu, const int nu);
};

#endif  // SRC_ADVANCE_H_
