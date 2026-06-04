// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_ADVANCE_H_
#define SRC_ADVANCE_H_

#include <memory>
#include "data.h"
#include "cell.h"
#include "fields.h"
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
#elif defined(USE_KOKKOS)
#include "gpu/GPUGrid.h"
#include "gpu/KokkosPipelines.h"
#include "gpu/gpu_types.h"
// Same alias, third back-end: host dispatch in advance.cpp is unchanged.
using GPUPipelines = KokkosPipelines;
#endif

// Single switch covering any GPU back-end.
#if defined(USE_METAL) || defined(USE_CUDA) || defined(USE_KOKKOS)
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
    // Cross-step prev residency.  Under inter-step residency, MUSIC's RK
    // rotation makes this step's fpPrev the *same host buffer* that was
    // fpCurr last step (new fpPrev == old fpCurr), and its contents still
    // equal GPU snap_prev — so the per-step prev D2H is redundant whenever
    // curr was synced the immediately-preceding step.  These track that
    // safely: gpu_step_count_ bumps once per completed step (in
    // swap_curr_future_gpu), curr_synced_{buf,step}_ record the last curr
    // sync, and prev_fresh_buf_ is the buffer proven fresh-as-prev for the
    // current step (set only when the curr sync was exactly one step ago,
    // else nullptr → sync prev).  Conservative: we never skip unless proven.
    long        gpu_step_count_   = 0;
    const void* curr_synced_buf_  = nullptr;
    long        curr_synced_step_ = -2;
    const void* prev_fresh_buf_   = nullptr;
    // Lazy GPU init.  Takes grid dimensions as plain ints so it can be called
    // from any caller — Fields-based AdvanceIt today, SCGrid-based EvolveIt
    // in the standalone path if/when that is re-introduced.
    void       init_metal_if_needed(int Nx, int Ny, int Neta);
    void       make_gpu_params(double tau, int rk_flag,
                               MUSICGridParams &p) const;

    // Predicate: is the current DATA configuration entirely supported by the
    // GPU kernels?  Currently requires viscosity_flag==1, no baryon diffusion,
    // no hydro source terms, no multi-charge (rhoq/rhos kept zero), and a
    // CPU EOS that the GPU table samples adequately.  See PORT_GPU.md §4 for
    // the full list.
    bool       gpu_features_supported() const;

    // Per-cell guard that scans the Fields arrays for non-zero rhoq/rhos.
    // Returns false on the first non-zero entry.  The result is cached
    // after the first call — once we've confirmed rhoq/rhos are zero at
    // the start of a run they stay zero (gpu_features_supported() gates
    // out the only path that could write them, multi-charge sources).
    // Per-substep scanning would cost O(Ncells) extra serial work and
    // significantly hurt throughput at production grid sizes.
    bool       gpu_charges_ok(const Fields &arena);
    bool       charges_checked_ = false;
    bool       charges_ok_cache_ = false;

    // Run a single AdvanceIt substep entirely on the GPU.  Uploads from
    // arenaFieldsCurr/arenaFieldsPrev, dispatches the kernel pipeline,
    // syncs, and writes results into arenaFieldsNext.  Returns false when
    // the configuration is not GPU-supported or initialization failed; the
    // caller must run the CPU loop in that case.
    bool       try_gpu_advance(double tau, Fields &arenaFieldsPrev,
                               Fields &arenaFieldsCurr,
                               Fields &arenaFieldsNext, int rk_flag);

    // CPU pre-pass: evaluate the hydro source term j^alpha per cell and
    // populate gpu_grid_.qi_source_buf in the layout the GPU kernel expects
    // (qi_source_in[alpha * Ncells + cell] = tau_rk * j^alpha).  Mirrors the
    // per-cell formula in Advance::FirstRKStepT.  Caller must guard
    // turn_on_QS == 1 — rhoq/rhos sources are not GPU-supported.
    void       prefill_hydro_source_on_cpu(double tau, int rk_flag,
                                           Fields &arenaFieldsCurr);
#endif

 public:
    Advance(const EOS &eosIn, const InitData &DATA_in,
            std::shared_ptr<HydroSourceBase> hydro_source_ptr_in);

    void AdvanceIt(const double tau_init, Fields &arenaFieldsPrev,
                   Fields &arenaFieldsCurr, Fields &arenaFieldsNext,
                   const int rk_flag);

#ifdef MUSIC_USE_GPU
    // Mirror the rk1 host arena swap in GPU snapshot space (snap_curr ↔
    // snap_future).  Called from Evolve::AdvanceRK immediately after the
    // std::swap so the GPU and host pointer roles stay in lockstep.
    void swap_curr_future_gpu();

    // Mirror the rk0 host arena 3-way rotation in GPU snapshot space
    // (snap_prev ← snap_curr ← snap_future ← old snap_prev).  Called from
    // Evolve::AdvanceRK immediately after the host pointer rotation when
    // gpu_state_authoritative_ is set, so rk1 can read snap_curr+snap_prev
    // without an H2D upload.
    void rotate_snapshots_gpu();

    // True when the GPU holds the authoritative evolving state across timestep
    // boundaries (H2D upload at rk0 is skipped).
    bool gpu_owns_state() const { return gpu_owns_state_; }

    // GPU max-reduction over snap_curr.epsilon / .rhob.  Returns results in
    // 1/fm^4 units (same as Cell_small::epsilon).  Synchronizes before return.
    // No-op if GPU residency is not active.
    void reduce_max_gpu(double& eps_max, double& rhob_max);

    // Bring snap_curr/snap_prev back into the host arenas if the GPU has
    // been authoritative across the timestep boundary.  Multiple flavours
    // for different consumers' needs:
    //
    //   sync_arena_from_gpu(prev, curr)
    //       Full D2H of both snap_curr and snap_prev into host.  Clears
    //       gpu_owns_state_ so the next AdvanceIt re-uploads.  Use when
    //       caller may mutate the arena.
    //
    //   sync_arena_from_gpu_readonly(prev, curr)
    //       Full D2H, but leaves gpu_owns_state_ set so the next AdvanceIt
    //       rk0 still skips its H2D.  Use for pure-read diagnostics that
    //       read both prev and curr (check_conservation_law, vorticity,
    //       freezeout, ...).
    //
    //   sync_curr_from_gpu_readonly(curr)
    //       Half-D2H: only snap_curr → arenaFieldsCurr.  About 2× faster
    //       than the full sync.  Use for diagnostics that only read fpCurr
    //       (Gubser_flow_check_file, output_momentum_anisotropy_vs_tau,
    //       output_evolution_data...).
    //
    // All three variants are no-ops when gpu_owns_state_ is unset (CPU has
    // current state) OR when host_curr_fresh_ / host_prev_fresh_ indicate
    // a previous sync this outer iteration already brought the data over.
    void sync_arena_from_gpu(Fields &arenaFieldsPrev,
                             Fields &arenaFieldsCurr);
    void sync_arena_from_gpu_readonly(Fields &arenaFieldsPrev,
                                      Fields &arenaFieldsCurr);
    void sync_curr_from_gpu_readonly(Fields &arenaFieldsCurr);

    // Phase 2b: pack the ideal-hydro evolution output for the current step
    // directly on the GPU (EOS lookups + down-sampling on device) into `out`,
    // laid out exactly as host fluidCell_ideal.  Avoids the full-arena D2H and
    // the serial host EOS loop for the memory-output path.  Returns false when
    // the GPU is not holding current state or the back-end has no packing
    // kernel (Metal); the caller must then use the host output path.
    bool pack_evolution_ideal(std::vector<fluidCell_ideal> &out);

   private:
    // Set when host arena matches the corresponding GPU snapshot — i.e. a
    // previous sync this iteration already brought it over.  Cleared at
    // the start of the next AdvanceIt substep (state changed on GPU).
    bool host_curr_fresh_ = false;
    bool host_prev_fresh_ = false;
   public:
#endif

    // gpu_dwmn_base: pointer to the first alpha-component of the GPU-computed
    // dwmn buffer (component-major, stride = Ncells).  Pass nullptr to run
    // the CPU MakeWSource fallback.
    // gpu_qi_base: pointer to the first alpha-component of the GPU-computed
    // qi_out buffer (component-major, stride = Ncells).  Pass nullptr to run
    // the CPU MakeDeltaQI fallback.
    void FirstRKStepT(const double tau, const double x_local,
                      const double y_local, const double eta_s_local,
                      const int ix, const int iy,
                      const int ieta, const int rk_flag,
                      const int fieldIdx, Fields &arenaFieldsCurr,
                      Fields &arenaFieldsNext, Fields &arenaFieldsPrev);

    void FirstRKStepW(const double tau, Fields &arenaFieldsPrev,
                      Fields &arenaFieldsCurr, Fields &arenaFieldsNext,
                      const int rk_flag, const double theta_local,
                      const DumuVec &a_local,
                      const VelocityShearVec &sigma_local,
                      const VorticityVec &omega_local,
                      const DmuMuBoverTVec &baryon_diffusion_vector,
                      const int ieta, const int ix, const int iy,
                      const int fieldIdx);

    void QuestRevert(Cell_small &grid_pt,
                     const int ieta, const int ix, const int iy);
    void QuestRevert_qmu(Cell_small &grid_pt,
                         const int ieta, const int ix, const int iy);

    void MakeDeltaQI(const double tau, Fields &arenaFieldsCurr,
                     const int ix, const int iy, const int ieta, TJbVec &qi,
                     const int rk_flag);

    double MaxSpeed(const double tau, const int direc,
                    const ReconstCell &grid_p, double &pressure);

    double get_TJb(const ReconstCell &grid_p, const int mu, const int nu,
                   const double pressure);
};

#endif  // SRC_ADVANCE_H_
