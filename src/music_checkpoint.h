// A run's complete state at one step, for a second run of the same initial
// condition to start from instead of recomputing it: the two-stage X-SCAPE
// chain's jet leg (MUSIC_2) is bit-identical to the background (MUSIC_1)
// until its first droplet deposits.  See js-contrib
// docs/Plans/PLAN_music2_restart.md.

#ifndef SRC_MUSIC_CHECKPOINT_H_
#define SRC_MUSIC_CHECKPOINT_H_

#include <cstddef>
#include <string>
#include <vector>

#include "data_struct.h"
#include "surfaceCell.h"

//! Frees a GPU state buffer (GPUPipelines::free_state); defined in evolve.cpp.
void free_checkpoint_gpu_state(void* state);

struct MusicCheckpoint {
    MusicCheckpoint() = default;
    ~MusicCheckpoint() { free_checkpoint_gpu_state(gpu_state); }
    MusicCheckpoint(const MusicCheckpoint&) = delete;
    MusicCheckpoint& operator=(const MusicCheckpoint&) = delete;

    //! the checkpoint holds a complete state; false until it is taken
    bool valid = false;
    //! everything that must agree between the two runs: the parameters that
    //! shape the evolution, the initial arena and the initial-state source
    std::string key;

    // ── where: iteration `it` of EvolveIt, after its freeze-out stage and
    //    before AdvanceRK.  A run resumes there with AdvanceRK.
    int    it = -1;
    double tau = 0.;
    double delta_tau = 0.;
    int    facTau = 1;
    int    iFreezeStart = 0;

    // ── freeze-out bookkeeping
    //! the run built the freeze-out surface: `surface` and the FO_n*vsEta
    //! counters are complete up to `it`
    bool has_surface = false;
    std::vector<SurfaceCell> surface;
    std::vector<double> FO_nBvsEta, FO_nQvsEta, FO_nSvsEta;
    //! max e [1/fm^4] at step `it`: what the no-surface stop test holds
    //! after its check there (Evolve::emax_last_check_)
    double emax_last_check = -1.;
    double eps_max_cur = -1.;
    unsigned long nonfinite_total = 0;
    int nonfinite_first_it = -1;

    //! the freeze-out test's result at step `it` (the run stops after its
    //! AdvanceRK if 1)
    int frozen = 0;

    // ── the evolution store: its first frames (those of steps <= it)
    //! the run kept the store's bookkeeping (store_hydro_info_in_memory)
    bool has_store = false;
    //! ... and its frames (store_hydro_info_frames)
    bool has_frames = false;
    std::vector<fluidCell_ideal> frames;
    int    store_ntau = 0;
    double store_tau_max = 0.;

    // ── the GPU state: snap_prev, then snap_curr (GPUPipelines layout)
    void*  gpu_state = nullptr;
    size_t gpu_state_floats = 0;
};

#endif  // SRC_MUSIC_CHECKPOINT_H_
