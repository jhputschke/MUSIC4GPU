// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_EVOLVE_H_
#define SRC_EVOLVE_H_

#include <memory>
#include <vector>
#include "util.h"
#include "data.h"
#include "cell.h"
#include "grid_info.h"
#include "eos.h"
#include "advance.h"
#include "fields.h"
#include "hydro_source_base.h"
#include "pretty_ostream.h"
#include "HydroinfoMUSIC.h"
#include "music_checkpoint.h"
#include "surfaceCell.h"

//! The two steps the freeze-out search compares (evolve.cpp, freeze-out
//! steps): the step being checked goes into slot k, the other slot holds the
//! previously checked one; then the slots swap roles.
struct FreezeoutSteps {
    Fields curr[2];
    Fields prev[2];             //!< the step before each; vorticity output only
    int k = 0;
    //! the previous checked step's fields other than e are only in gpu_state
    bool kept_on_gpu = false;
    void* gpu_state = nullptr;  //!< a GPU copy of the previous checked step
    std::vector<unsigned char> visit;   //!< cubes to visit, by their first cell
    std::vector<int> cells;             //!< cells to gather
    std::vector<unsigned char> listed;  //!< scratch for listing them once
    ~FreezeoutSteps();
};

// this is a control class for the hydrodynamic evolution
class Evolve {
 private:
    const EOS &eos;        // declare EOS object
    InitData &DATA;
    std::shared_ptr<HydroSourceBase> hydro_source_terms_ptr;
    std::shared_ptr<HydroSourceBase> hydro_source_terms_from_jets_ptr_;

    Cell_info grid_info;
    Advance advance;
    pretty_ostream music_message;

    // simulation information
    int rk_order;

    // information about freeze-out surface
    // (only used when freezeout_method == 4)
    std::vector<double> epsFO_list;

    std::vector<SurfaceCell> surfaceCellVec_;
    std::vector<double> FO_nBvsEta_;
    std::vector<double> FO_nQvsEta_;
    std::vector<double> FO_nSvsEta_;

    // non-finite W^{mu nu} / Pi set to 0 by the GPU in this run (see
    // check_nonfinite_gpu)
    unsigned long nonfinite_total_ = 0;
    int nonfinite_first_it_ = -1;
    void check_nonfinite_gpu(int it, double tau);
    // MUSIC_DEBUG_NONFINITE / MUSIC_DEBUG_CELL (evolve.cpp); no-ops unless set
    void debug_hooks(Fields &prev, Fields &curr, int it, double tau);

    // MusicCheckpoint (music_checkpoint.h): take one in this run, or start
    // from one -- or, verifying, run from tau0 and compare at its step
    std::shared_ptr<MusicCheckpoint> checkpoint_out_;
    double checkpoint_tau_ = 0.;
    bool   late_switched_       = false;  //!< late_delta_tau applied
    double tau_last_check_      = -1.;    //!< tau of the last freeze-out check
    //! Cornelius' tau spacing for the first check after the step switch
    //! (-1: to be measured at that check; 0: facTau * delta_tau)
    double dtau_check_override_ = 0.;
    double late_delta_tau_org_ = 0.;  //!< the step parameters before the switch,
    int    late_facTau_org_ = 0;      //!< restored when the run ends (DATA is
    int    late_skip_org_ = 0;        //!< shared by every run of the instance)
    void late_step_restore();
    std::shared_ptr<const MusicCheckpoint> resume_from_;
    bool verify_resume_ = false;
    std::string resume_status_;
    std::string checkpoint_key(Fields &arena_initial) const;
    std::string resume_rejection(const MusicCheckpoint &c,
                                 const std::string &key, int iFreezeStart,
                                 double tau0);
    void take_checkpoint(int it, double tau, int iFreezeStart, int frozen,
                         double eps_max_cur, const std::string &key,
                         Fields &arena_current, HydroinfoMUSIC &hydro_info);
    bool restore_checkpoint(const MusicCheckpoint &c, Fields* fpPrev,
                            Fields* fpCurr, FreezeoutSteps &fo,
                            double &eps_max_cur, HydroinfoMUSIC &hydro_info);

    // the freeze-out search's steps (evolve.cpp, freeze-out steps)
    bool freezeout_gather() const;
    void freezeout_store_step(FreezeoutSteps &fo, Fields &prev, Fields &curr);
    int freezeout_check(FreezeoutSteps &fo, double tau, Fields &prev,
                        Fields &curr);
    void freezeout_cubes_to_visit(FreezeoutSteps &fo, Fields &now,
                                  Fields &last);
    void verify_checkpoint(const MusicCheckpoint &c, int it, double tau,
                           int frozen, double eps_max_cur,
                           Fields &arena_current, HydroinfoMUSIC &hydro_info);

 public:
    Evolve(const EOS &eos, InitData &DATA_in,
           std::shared_ptr<HydroSourceBase> hydro_source_ptr_in,
           std::shared_ptr<HydroSourceBase> hydro_source_from_jets_ptr_in
               = nullptr);

    ~Evolve() {clearSurfaceCellVector();}

    void clearSurfaceCellVector() {surfaceCellVec_.clear();}
    long get_number_of_surface_cells() const {return(surfaceCellVec_.size());}
    void get_surface_cell_with_index(const long i, SurfaceCell &cell_i) {
        cell_i = surfaceCellVec_[i];
    }

    int EvolveIt(Fields &arenaFieldsPrev, Fields &arenaFieldsCurr,
                 Fields &arenaFieldsNext, HydroinfoMUSIC &hydro_info_ptr);

    //! Take a MusicCheckpoint in this run, into `out`: at its last
    //! freeze-out step before tau (iteration `it`, after the freeze-out
    //! stage, before AdvanceRK).  GPU runs only.
    void request_checkpoint(std::shared_ptr<MusicCheckpoint> out, double tau) {
        checkpoint_out_ = std::move(out);
        checkpoint_tau_ = tau;
    }
    //! Start this run from `from` -- if it was taken in a run with the same
    //! parameters, initial arena and initial-state source, and this run's jet
    //! source adds nothing before its step; otherwise from tau0.  verify: run
    //! from tau0 regardless and compare the state with `from` at its step.
    void set_resume(std::shared_ptr<const MusicCheckpoint> from, bool verify) {
        resume_from_ = std::move(from);
        verify_resume_ = verify;
    }
    //! what this run did with set_resume, for the log; "" if not asked
    const std::string &resume_status() const { return resume_status_; }

    int EvolveOneTimeStep(const int itau, Fields &arenaFieldsPrev,
                          Fields &arenaFieldsCurr, Fields &arenaFieldsNext,
                          Fields &freezeoutFieldPrev,
                          Fields &freezeoutFieldCurr,
                          HydroinfoMUSIC &hydro_info_ptr);

    void AdvanceRK(double tau, Fields* &fpPrev, Fields* &fpCurr,
                   Fields* &fpNext);

    int FreezeOut_equal_tau_Surface(double tau, Fields &arena_current);
    void FreezeOut_equal_tau_Surface_XY(double tau,
                                        int ieta, Fields &arena_current,
                                        int thread_id, double epsFO);
    //! visit: only the cubes whose first cell is non-zero there (all if null)
    int FindFreezeOutSurface_Cornelius(double tau,
        Fields &arena_prev, Fields &arena_current,
        Fields &arena_freezeout_prev, Fields &arena_freezeout,
        const unsigned char* visit = nullptr);

    int FindFreezeOutSurface_Cornelius_XY(double tau, int ieta,
                                          Fields &arena_prev,
                                          Fields &arena_current,
                                          Fields &arena_freezeout_prev,
                                          Fields &arena_freezeout,
                                          int thread_id, double epsFO,
                                          std::vector<SurfaceCell> &surface_out,
                                          const unsigned char* visit = nullptr);
    int FindFreezeOutSurface_boostinvariant_Cornelius(
                double tau, Fields &arena_current, Fields &arena_freezeout);

    void store_previous_step_for_freezeout(Fields &arenaCurr,
                                           Fields &arenaFreeze);

    // freeze_out_surface = 0: stop without building the surface
    double emax_last_check_ = -1.;  //!< max e at the last check [1/fm^4]
    double max_energy_density_raw(Fields &arena_current);
    int frozen_without_surface(Fields &arena_current);
    bool transverse_edge_is_hot(Fields &arena_current, double eps_fo);
    void regulate_qmu(const FlowVec u, const double q[],
                      double q_regulated[]) const;
    void regulate_Wmunu(const FlowVec u, const double Wmunu[4][4],
                        double Wmunu_regulated[4][4]) const;

    void initialize_freezeout_surface_info();

    Cell_small four_dimension_linear_interpolation(
        double* lattice_spacing, double fraction[2][4], Cell_small**** cube);
    Cell_small three_dimension_linear_interpolation(
        double* lattice_spacing, double fraction[2][3], Cell_small*** cube);
    Cell_aux four_dimension_linear_interpolation(
        double* lattice_spacing, double fraction[2][4], Cell_aux**** cube);
    Cell_aux three_dimension_linear_interpolation(
        double* lattice_spacing, double fraction[2][3], Cell_aux*** cube);
};

#endif  // SRC_EVOLVE_H_

