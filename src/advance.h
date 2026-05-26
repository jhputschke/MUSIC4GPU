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

#ifdef USE_METAL
    GPUGrid    gpu_grid_;
    bool       gpu_ready_ = false;
    bool       metal_initialized_ = false;
    void       init_metal_if_needed(SCGrid &arena_current);
    void       make_gpu_params(double tau_rk, MUSICGridParams &p) const;
#endif

 public:
    Advance(const EOS &eosIn, const InitData &DATA_in,
            std::shared_ptr<HydroSourceBase> hydro_source_ptr_in);

    void AdvanceIt(const double tau_init,
                   SCGrid &arena_prev, SCGrid &arena_current,
                   SCGrid &arena_future, const int rk_flag);

    // gpu_dwmn_base: pointer to the first alpha-component of the GPU-computed
    // dwmn buffer (component-major, stride = Ncells).  Pass nullptr to run
    // the CPU MakeWSource fallback.
    void FirstRKStepT(const double tau, const double x_local,
                      const double y_local, const double eta_s_local,
                      SCGrid &arena_current, SCGrid &arena_future,
                      SCGrid &arena_prev, const int ix, const int iy,
                      const int ieta, const int rk_flag,
                      const float* gpu_dwmn_base = nullptr,
                      int Ncells = 0);

    void FirstRKStepW(const double tau_it, SCGrid &arena_prev,
                      SCGrid &arena_current, SCGrid &arena_future,
                      const int rk_flag, const double theta_local,
                      const DumuVec &a_local,
                      const VelocityShearVec &sigma_local,
                      const VorticityVec &omega_local,
                      const DmuMuBoverTVec &baryon_diffusion_vector,
                      const int ieta, const int ix, const int iy);

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
