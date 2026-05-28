// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "util.h"
#include "data.h"
#include "cell.h"
#include "reconst.h"
#include "eos.h"
#include "evolve.h"
#include "advance.h"
#include "bench_timer.h"

using Util::map_2d_idx_to_1d;
using Util::map_1d_idx_to_2d;
using Util::hbarc;

// ── Metal helpers (compiled only when USE_METAL is defined) ───────────────────
#ifdef MUSIC_USE_GPU
void Advance::init_metal_if_needed(int Nx, int Ny, int Neta) {
    if (metal_initialized_) return;
    metal_initialized_ = true;

    auto& mp = GPUPipelines::instance();
    if (!mp.initialize()) {
        music_message << "[MUSIC-GPU] Metal init failed, falling back to CPU.";
        music_message.flush("warning");
        gpu_ready_ = false;
        return;
    }
    if (!gpu_grid_.allocate(Nx, Ny, Neta)) {
        music_message << "[MUSIC-GPU] GPU buffer allocation failed.";
        music_message.flush("warning");
        gpu_ready_ = false;
        return;
    }

    // Sample P, dP/de, s and T at rhob=0 onto ONE log-spaced e grid for the
    // GPU EOS (standard zero-net-baryon case).  All four are non-linear in e
    // for any realistic EOS (hotQCD/WB/s95p): P(e) curves through the QCD
    // crossover just as s~e^(3/4) and T~e^(1/4) do.  Hydro cells live at
    // e ~ 0.1 /fm^4 while eps_max ~ 1e4 /fm^4, so a LINEAR grid would put the
    // entire evolution inside its first interval — exact only for the
    // conformal ideal gas (P=e/3).  Log spacing keeps resolution in the
    // dilute regime.  The floor/grid must match GPUGrid::upload_eos's log_*.
    {
        const int    N_EOS   = GPU_EOS_N;
        double       eps_max = eos.get_eps_max();
        if (eps_max <= 0.0) eps_max = 1.0e4;
        std::vector<float> P_data(N_EOS), dPde_data(N_EOS);
        std::vector<float> s_data(N_EOS), T_data(N_EOS);
        constexpr double log_e_floor = 1.0e-6;  // 1/fm^4
        const double log_e_min = std::log(log_e_floor);
        const double log_e_max = std::log(std::max(eps_max, log_e_floor*1.01));
        const double dle = (log_e_max - log_e_min) / static_cast<double>(N_EOS - 1);
        for (int i = 0; i < N_EOS; i++) {
            const double e = std::exp(log_e_min + i * dle);
            P_data[i]    = static_cast<float>(eos.get_pressure   (e, 0.0));
            dPde_data[i] = static_cast<float>(eos.get_dpde       (e, 0.0));
            s_data[i]    = static_cast<float>(eos.get_entropy    (e, 0.0));
            T_data[i]    = static_cast<float>(eos.get_temperature(e, 0.0));
        }
        if (!gpu_grid_.upload_eos(P_data.data(), dPde_data.data(),
                                  s_data.data(), T_data.data(),
                                  N_EOS, 0.0f, static_cast<float>(eps_max))) {
            music_message << "[MUSIC-GPU] EOS table upload failed.";
            music_message.flush("warning");
            gpu_ready_ = false;
            return;
        }
    }

    gpu_ready_ = true;
    music_message << "[MUSIC-GPU] GPU grid allocated ("
                  << Nx << "x" << Ny << "x" << Neta << " cells).";
    music_message.flush("info");
}

void Advance::make_gpu_params(double tau, int rk_flag,
                              MUSICGridParams &p) const {
    const double tau_rk = tau + rk_flag * DATA.delta_tau;
    p.Nx     = DATA.nx;
    p.Ny     = DATA.ny;
    p.Neta   = DATA.neta;
    p.Ncells = DATA.nx * DATA.ny * DATA.neta;
    p.delta_x   = static_cast<float>(DATA.delta_x);
    p.delta_y   = static_cast<float>(DATA.delta_y);
    p.delta_eta = static_cast<float>(DATA.delta_eta);
    p.delta_tau = static_cast<float>(DATA.delta_tau);
    p.tau       = static_cast<float>(tau_rk);
    p.boost_invariant = DATA.boost_invariant ? 1 : 0;
    p.turn_on_bulk    = DATA.turn_on_bulk;
    p.turn_on_diff    = DATA.turn_on_diff;
    p.minmod_theta    = static_cast<float>(DATA.minmod_theta);
    p.rk_flag         = rk_flag;
    p.tau_orig        = static_cast<float>(tau);

    // Phase-2 transport / config inputs
    p.shear_to_s              = static_cast<float>(DATA.shear_to_s);
    p.shear_relax_time_factor = static_cast<float>(DATA.shear_relax_time_factor);
    p.turn_on_shear           = DATA.turn_on_shear;
    p.T_dep_shear_mode        = DATA.T_dependent_shear_to_s;
    p.shear_duke_min          = static_cast<float>(DATA.shear_2_min);
    p.shear_duke_slope        = static_cast<float>(DATA.shear_2_slope);
    p.shear_duke_curv         = static_cast<float>(DATA.shear_2_curv);
    p.shear_sims_T_kink_GeV   = static_cast<float>(DATA.shear_3_T_kink_in_GeV);
    p.shear_sims_low_slope    = static_cast<float>(DATA.shear_3_low_T_slope_in_GeV);
    p.shear_sims_high_slope   = static_cast<float>(DATA.shear_3_high_T_slope_in_GeV);
    p.shear_sims_at_kink      = static_cast<float>(DATA.shear_3_at_kink);
    // Bulk-viscosity inputs (Phase 3) — turn_on_bulk already set above.
    p.T_dep_bulk_mode         = DATA.T_dependent_bulk_to_s;
    p.bulk_relaxation_type    = DATA.bulk_relaxation_type;
    p.bulk_relax_time_factor  = static_cast<float>(DATA.bulk_relax_time_factor);
    p.bulk_duke_norm          = static_cast<float>(DATA.bulk_2_normalisation);
    p.bulk_duke_width_GeV     = static_cast<float>(DATA.bulk_2_width_in_GeV);
    p.bulk_duke_peak_GeV      = static_cast<float>(DATA.bulk_2_peak_in_GeV);
    p.bulk_sims_max           = static_cast<float>(DATA.bulk_3_max);
    p.bulk_sims_width_GeV     = static_cast<float>(DATA.bulk_3_width_in_GeV);
    p.bulk_sims_T_peak_GeV    = static_cast<float>(DATA.bulk_3_T_peak_in_GeV);
    p.bulk_sims_lambda        = static_cast<float>(DATA.bulk_3_lambda_asymm);
    p.bulk_asym10_max         = static_cast<float>(DATA.bulk_10_max);
    p.bulk_asym10_width_low   = static_cast<float>(DATA.bulk_10_width_low);
    p.bulk_asym10_width_high  = static_cast<float>(DATA.bulk_10_width_high);
    p.bulk_asym10_Tpeak       = static_cast<float>(DATA.bulk_10_Tpeak);
    // QuestRevert
    p.do_quest_revert         = (DATA.Initial_profile != 0
                                 && DATA.Initial_profile != 1) ? 1 : 0;
    p.quest_revert_strength   = static_cast<float>(DATA.quest_revert_strength);
    // Second-order transport-term flags
    p.include_second_order_terms = DATA.include_second_order_terms;
    p.init_profile_zero          = (DATA.Initial_profile == 0) ? 1 : 0;

    // Precompute geometric factors for the longitudinal flux term
    double de = DATA.delta_eta;
    if (DATA.boost_invariant) {
        p.cosh_deta = 0.f;
        p.sinh_deta = 0.5f;
    } else {
        double cd = (de > 1e-10) ? cosh(de/2.) / de : 0.5;
        double sd = (de > 1e-10) ? sinh(de/2.) / de : 0.5;
        p.cosh_deta = static_cast<float>(cd);
        p.sinh_deta = static_cast<float>(std::max(0.5, sd));
    }
}

void Advance::swap_curr_future_gpu() {
    // Mirror the host rk1 fpCurr↔fpNext swap on the GPU side.  After rk1,
    // try_gpu_advance has written results into snap_future; the host swap
    // makes fpCurr point at what was fpNext, so on GPU snap_curr must now
    // point at what was snap_future.
    //
    // Gate on gpu_owns_state_ (set true by the last substep), NOT
    // gpu_state_authoritative_: try_gpu_advance clears the latter to false on
    // the last substep, so gating on it here silently skipped the swap and
    // stranded the rk1 corrector in snap_future — the next step then evolved
    // from the rk0 predictor, degrading RK2 to forward-Euler (~1e-3 drift
    // instead of the kernels' ~1e-5 float32 floor).  This matches main_gpu's
    // original gate; the gpu_state_authoritative_ gate was a regression
    // introduced in commit 0cecdf5.  See PORT_GPU_CUDA.md.
    if (gpu_owns_state_) gpu_grid_.swap_curr_future();

    // Step boundary (called once per completed RK step, at rk1).  After the
    // host rotation, this step's fpCurr buffer becomes next step's fpPrev
    // (new fpPrev == old fpCurr), and the GPU rotation makes next step's
    // snap_prev == this step's snap_curr — so if we synced curr this step,
    // next step's prev D2H is redundant.  Carry that buffer forward as
    // prev_fresh_buf_, but ONLY when curr was actually synced this step
    // (curr_synced_step_ == gpu_step_count_); otherwise clear it so the next
    // prev read syncs normally.  This makes it correct at any diagnostic
    // frequency: on a step that skipped its curr sync, no skip is offered.
    prev_fresh_buf_   = (curr_synced_step_ == gpu_step_count_) ? curr_synced_buf_
                                                              : nullptr;
    ++gpu_step_count_;
    curr_synced_buf_  = nullptr;
}

void Advance::rotate_snapshots_gpu() {
    // 3-way mirror of the host rk0 rotation
    // (fpPrev ← fpCurr ← fpNext ← old fpPrev).  After rk0, try_gpu_advance
    // has written results into snap_future; the host rotation makes fpCurr
    // point at what was fpNext, so on GPU snap_curr must now point at what
    // was snap_future, snap_prev at what was snap_curr, and snap_future
    // becomes scratch (was prev).  Lets the rk1 substep skip its H2D upload.
    if (gpu_state_authoritative_) gpu_grid_.rotate_snapshots();
}

void Advance::reduce_max_gpu(double& eps_max, double& rhob_max) {
    if (!gpu_owns_state_ || !gpu_ready_) { eps_max = rhob_max = 0.0; return; }
    GPUPipelines::instance().reduce_max(gpu_grid_, eps_max, rhob_max);
}

// ── Feature support predicate ────────────────────────────────────────────────
//
// Gate the GPU dispatch on the configuration falling inside the support matrix
// that the kernels actually implement.  Anything outside this set falls back
// to the CPU loop.  Each guard mirrors a kernel-side assumption documented in
// the .metal / .cu sources and PORT_GPU.md §4.
//
// The check is configuration-only (DATA flags); per-cell state checks (rhoq /
// rhos non-zero, finite-muB EOS effects) are done in gpu_charges_ok().
bool Advance::gpu_features_supported() const {
    if (DATA.viscosity_flag != 1)         return false;
    if (DATA.turn_on_diff)                return false;
    if (DATA.muB_dependent_shear_to_s != 0) return false;
    // Reject finite-muB EOS tables (GPU EOS is sampled at rhob=0).  EOS 91 is
    // a zero-muB hotQCD variant (same EOS_hotQCD class as EOS 9, flag_muB=false)
    // so it is GPU-safe and carved out of the threshold.  See PORT_GPU.md §4.2
    // for why this stays a magic-number check rather than eos.get_flag_muB().
    if (DATA.whichEOS > 9 && DATA.whichEOS != 91) return false;
    // Hydro source terms: GPU snapshot only carries rhob, so QS (rhoq/rhos)
    // source contributions can't be applied — fall back to CPU in that case.
    // Pure energy + baryon sources are supported via prefill_hydro_source_on_cpu.
    if (flag_add_hydro_source && DATA.turn_on_QS == 1) return false;
    return true;
}

// CPU-side pre-pass that fills qi_source_buf with tau_rk * j^alpha at each
// cell.  This mirrors the per-cell logic in FirstRKStepT (alpha = 0..3 for
// energy + momentum, alpha = 4 for baryon when turn_on_rhob).  The result is
// consumed by gpu_finalize_ideal when params.has_hydro_source == 1.
void Advance::prefill_hydro_source_on_cpu(double tau, int rk_flag,
                                          Fields &arenaFieldsCurr) {
    const int Nx    = arenaFieldsCurr.nX();
    const int Ny    = arenaFieldsCurr.nY();
    const int Neta  = arenaFieldsCurr.nEta();
    const int N     = Nx * Ny * Neta;
    const double tau_rk      = tau + rk_flag * DATA.delta_tau;
    const double tauRkFactor = (DATA.CoorType == 1) ? 1.0 : tau_rk;

    // qi_source_buf layout: [alpha * Ncells + cell]; written for alpha 0..4.
    // Zero the buffer first so unused alpha slots stay clean.
    float* qb = gpu_grid_.qi_source_buf;
    std::memset(qb, 0, static_cast<size_t>(5) * N * sizeof(float));

    const bool rhob_src = (DATA.turn_on_rhob == 1);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = arenaFieldsCurr.getFieldIdx(ix, iy, ieta);
        const double x_local     = -DATA.x_size  /2. + ix   * DATA.delta_x;
        const double y_local     = -DATA.y_size  /2. + iy   * DATA.delta_y;
        const double eta_s_local = -DATA.eta_size/2. + ieta * DATA.delta_eta;

        FlowVec u_local;
        for (int ii = 0; ii < 4; ii++)
            u_local[ii] = arenaFieldsCurr.u_[ii][c];

        EnergyFlowVec j_mu = {0};
        hydro_source_terms_ptr->get_hydro_energy_source(
                tau_rk, x_local, y_local, eta_s_local, u_local, j_mu);
        for (int ii = 0; ii < 4; ii++) {
            qb[ii * N + c] = static_cast<float>(tauRkFactor * j_mu[ii]);
        }
        if (rhob_src) {
            const double j_rhob = hydro_source_terms_ptr->get_hydro_rhob_source(
                    tau_rk, x_local, y_local, eta_s_local, u_local);
            qb[4 * N + c] = static_cast<float>(tauRkFactor * j_rhob);
        }
    }
}

bool Advance::gpu_charges_ok(const Fields &arena) {
    // First-call scan only — the result is cached for the rest of the run.
    // rhoq/rhos can only become non-zero through a multi-charge source term
    // (DATA.turn_on_QS == 1), which gpu_features_supported() rejects upstream.
    // Skipping per-substep scans is essential for throughput: at 64x64x32
    // and 200 substeps a per-substep scan would cost ~26M serial cell
    // comparisons, eating most of the GPU speedup.
    if (charges_checked_) return charges_ok_cache_;
    const auto &rhoq = arena.rhoq_;
    const auto &rhos = arena.rhos_;
    const size_t N = rhoq.size();
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (rhoq[i] != 0.0 || rhos[i] != 0.0) { ok = false; break; }
    }
    charges_ok_cache_ = ok;
    charges_checked_  = true;
    return ok;
}

// ── Full-GPU AdvanceIt substep (Fields path) ─────────────────────────────────
//
// Mirrors the kernel sequence used by the SCGrid path before the XSCAPE merge:
//   1. Upload curr+prev to snap_curr / snap_prev
//   2. dispatch_make_du     (viscous geometry: theta, a, sigma)
//   3. dispatch_uwrhs       (KT flux divergence of u^a W^{kl})
//   4. dispatch_w_source    (partial_a W^{a mu})
//   5. dispatch_uprhs       (KT flux of u^a pi_b)         — turn_on_bulk only
//   6. dispatch_delta_qi    (KT flux of T^{tau alpha} + geom)
//   7. dispatch_finalize_ideal       (Newton solve -> snap_future primitives)
//   8. dispatch_first_rk_step_w_full (second-order Wmunu / pi_b -> snap_future)
//   9. wait()
//  10. Copy snap_future primitives + Wmunu + pi_b back into arenaFieldsNext.
//
// Residency optimisation (gpu_owns_state_ / gpu_state_authoritative_) is NOT
// enabled on the Fields path in this first cut — every substep does a fresh
// upload at rk_flag==0.  See PORT_GPU.md §7 (open questions) — JETSCAPE may
// mutate the Fields between AdvanceIt calls, which would invalidate any
// residency assumption.
bool Advance::try_gpu_advance(double tau, Fields &arenaFieldsPrev,
                              Fields &arenaFieldsCurr,
                              Fields &arenaFieldsNext, int rk_flag) {
    static bool charge_warning_logged = false;
    static bool feature_warning_logged = false;

    // Any successful GPU substep will write a new state into snap_future
    // (which the rk0→rk1 rotation will then promote to snap_curr).  Mark
    // host arenas as stale so the next sync call actually copies.
    host_curr_fresh_ = false;
    host_prev_fresh_ = false;

    if (!gpu_features_supported()) {
        if (!feature_warning_logged) {
            music_message << "[MUSIC-GPU] Fields path: configuration outside "
                             "GPU support matrix, using CPU.";
            music_message.flush("info");
            feature_warning_logged = true;
        }
        return false;
    }
    if (!gpu_charges_ok(arenaFieldsCurr)) {
        if (!charge_warning_logged) {
            music_message << "[MUSIC-GPU] Fields path: non-zero rhoq/rhos "
                             "detected, falling back to CPU "
                             "(see PORT_GPU.md §4.1).";
            music_message.flush("warning");
            charge_warning_logged = true;
        }
        return false;
    }

    init_metal_if_needed(arenaFieldsCurr.nX(),
                         arenaFieldsCurr.nY(),
                         arenaFieldsCurr.nEta());
    if (!gpu_ready_) return false;  // init failed; init_metal_if_needed already warned

    // Upload host arenas into the GPU snapshots — UNLESS the snapshots are
    // already authoritative from a previous substep/step:
    //   - gpu_state_authoritative_  : snap_future from the previous substep
    //                                 has been rotated into snap_curr by
    //                                 rotate_snapshots_gpu() (intra-substep
    //                                 residency, within an AdvanceRK call).
    //   - gpu_owns_state_           : snap_curr survived across the
    //                                 timestep boundary because the last rk1
    //                                 substep skipped its D2H — host arenas
    //                                 are stale until sync_arena_from_gpu().
    //
    // Either path lets us skip the H2D, which avoids the float64→float32
    // truncation of every cell every step.  Without it, eps_max drifts
    // ~1e-2 over 100 steps; with it, drift stays at the kernel's
    // intrinsic single-precision noise level (~1e-4).
    if (!gpu_state_authoritative_ && !gpu_owns_state_) {
        gpu_grid_.copy_to_gpu(arenaFieldsCurr, gpu_grid_.snap_curr);
        if (rk_flag > 0) {
            gpu_grid_.copy_to_gpu(arenaFieldsPrev, gpu_grid_.snap_prev);
        } else {
            // At rk_flag == 0 of a cold start, prev == curr.
            gpu_grid_.copy_to_gpu(arenaFieldsCurr, gpu_grid_.snap_prev);
        }
#if defined(USE_CUDA)
        // Discrete GPU: the copy_to_gpu calls above only packed the float-cast
        // primitives into pinned host staging.  Push them to the device
        // snapshot buffers now, or the kernels below read uninitialised device
        // memory (eps collapses to 0 within a step).  No-op on coherent unified
        // memory, where copy_to_gpu wrote the device-visible managed buffers
        // directly.  Mirrors main_gpu's SCGrid dispatch — this H2D call was
        // lost when the SCGrid AdvanceIt was dropped in the XSCAPE merge.
        // See PORT_GPU_CUDA.md.
        GPUPipelines::instance().upload_snapshots_async(gpu_grid_);
#endif
    }

    MUSICGridParams p;
    make_gpu_params(tau, rk_flag, p);

    // Hydro source pre-pass (energy + momentum, plus rhob if turn_on_rhob).
    // turn_on_QS == 1 is rejected upstream in gpu_features_supported(), so we
    // only need to evaluate energy + baryon channels here.
    if (flag_add_hydro_source) {
        // prepare_list_for_current_tau_frame is already called once per
        // timestep by Evolve::AdvanceRK before AdvanceIt, matching the CPU
        // FirstRKStepT contract (no per-substep re-prep).
        prefill_hydro_source_on_cpu(tau, rk_flag, arenaFieldsCurr);
        p.has_hydro_source = 1;
        p.has_rhob_source  = (DATA.turn_on_rhob == 1) ? 1 : 0;
    } else {
        p.has_hydro_source = 0;
        p.has_rhob_source  = 0;
    }

    auto& mp = GPUPipelines::instance();
    mp.begin_batch();
      mp.dispatch_make_du(gpu_grid_, p);
      mp.dispatch_uwrhs  (gpu_grid_, p);
      mp.dispatch_w_source(gpu_grid_, p);
      if (p.turn_on_bulk == 1) mp.dispatch_uprhs(gpu_grid_, p);
      mp.dispatch_delta_qi(gpu_grid_, p);
      mp.dispatch_finalize_ideal(gpu_grid_, p);
      mp.dispatch_first_rk_step_w_full(gpu_grid_, p);
    mp.end_batch();
    mp.wait();

    // Decide how state crosses the substep / step boundary.
    //
    //   Intra-substep (rk0 → rk1): snap_future contains rk0 result.
    //     AdvanceRK's host 3-way rotation will be mirrored on the GPU side
    //     via rotate_snapshots_gpu(), so the rk1 substep just reads from
    //     snap_curr / snap_prev with no H2D.  Set gpu_state_authoritative_.
    //
    //   Inter-step (rk1 → next outer step's rk0): snap_future contains
    //     rk1 result; AdvanceRK's host 2-way swap will be mirrored via
    //     swap_curr_future_gpu(), so snap_curr ends up as the latest
    //     state.  We set gpu_owns_state_ to advertise that the host
    //     Fields objects are now STALE — EvolveIt must call
    //     sync_arena_from_gpu() before any code that reads them, and the
    //     next AdvanceIt's rk0 substep skips its H2D upload.
    //
    // We also skip the D2H copy-back in the inter-step path; CPU side
    // catches up lazily via sync_arena_from_gpu() at the diagnostic
    // call sites in evolve.cpp.  Eliminating the per-step round-trip
    // is what closes both the perf and the precision gap to main_gpu.
    const bool last_substep = (rk_flag == DATA.rk_order - 1);
    if (last_substep) {
        gpu_state_authoritative_ = false;   // snap_curr lives in snap_future
                                            // until swap_curr_future_gpu()
                                            // runs in AdvanceRK
        gpu_owns_state_          = true;
    } else {
        gpu_state_authoritative_ = true;    // intra-substep residency
    }
    return true;
}

// On-demand D2H: bring GPU snapshots back into the host Fields objects so
// CPU code (diagnostics, freezeout, output writers) can read them.  The
// host_curr_fresh_ / host_prev_fresh_ flags act as a per-iteration cache —
// once a sync has run, subsequent calls in the same outer iteration are
// no-ops until the next AdvanceIt substep clears the flags.
//
// sync_arena_from_gpu (the non-readonly variant) additionally clears
// gpu_owns_state_, forcing the next AdvanceIt rk0 to re-upload from host.
// Use only when the caller might mutate the arena.
void Advance::sync_arena_from_gpu(Fields &arenaFieldsPrev,
                                  Fields &arenaFieldsCurr) {
    sync_arena_from_gpu_readonly(arenaFieldsPrev, arenaFieldsCurr);
    gpu_owns_state_ = false;
    // Residency epoch ends here (host becomes authoritative).  Drop the
    // cross-step freshness so a later epoch can't match a stale buffer ptr.
    prev_fresh_buf_  = nullptr;
    curr_synced_buf_ = nullptr;
}

void Advance::sync_arena_from_gpu_readonly(Fields &arenaFieldsPrev,
                                           Fields &arenaFieldsCurr) {
    if (!gpu_owns_state_ || !gpu_ready_) return;
    bench::Timer _bt_sync("advance.sync_arena_from_gpu");
    if (!host_curr_fresh_) {
        gpu_grid_.copy_primitives_to_cpu(gpu_grid_.snap_curr, arenaFieldsCurr);
        gpu_grid_.copy_wmunu_to_cpu     (gpu_grid_.snap_curr, arenaFieldsCurr);
        host_curr_fresh_ = true;
    }
    // Record the curr buffer so next step's prev D2H can be skipped.
    curr_synced_buf_  = static_cast<const void*>(&arenaFieldsCurr);
    curr_synced_step_ = gpu_step_count_;
    if (!host_prev_fresh_) {
        // Skip the prev D2H iff this buffer is the one we synced as curr
        // exactly one step ago: the RK rotation makes new fpPrev == old
        // fpCurr and the GPU rotation makes snap_prev == last step's
        // snap_curr, so its host contents are already current.  Otherwise
        // (different buffer, or a step was skipped) copy normally.
        if (static_cast<const void*>(&arenaFieldsPrev) == prev_fresh_buf_) {
            host_prev_fresh_ = true;            // already current — no copy
        } else {
            gpu_grid_.copy_primitives_to_cpu(gpu_grid_.snap_prev, arenaFieldsPrev);
            gpu_grid_.copy_wmunu_to_cpu     (gpu_grid_.snap_prev, arenaFieldsPrev);
            host_prev_fresh_ = true;
        }
    }
}

void Advance::sync_curr_from_gpu_readonly(Fields &arenaFieldsCurr) {
    if (!gpu_owns_state_ || !gpu_ready_) return;
    // Record the curr buffer/step (even if the copy itself is cached) so the
    // step boundary can carry it forward as next step's fresh prev.
    curr_synced_buf_  = static_cast<const void*>(&arenaFieldsCurr);
    curr_synced_step_ = gpu_step_count_;
    if (host_curr_fresh_) return;
    bench::Timer _bt_sync("advance.sync_curr_from_gpu");
    gpu_grid_.copy_primitives_to_cpu(gpu_grid_.snap_curr, arenaFieldsCurr);
    gpu_grid_.copy_wmunu_to_cpu     (gpu_grid_.snap_curr, arenaFieldsCurr);
    host_curr_fresh_ = true;
}
#endif  // MUSIC_USE_GPU

Advance::Advance(const EOS &eosIn, const InitData &DATA_in,
                 std::shared_ptr<HydroSourceBase> hydro_source_ptr_in) :
    DATA(DATA_in), eos(eosIn),
    diss_helper(eosIn, DATA_in),
    minmod(DATA_in),
    reconst_helper(eos, DATA_in.echo_level, DATA_in.beastMode) {

    hydro_source_terms_ptr = hydro_source_ptr_in;
    flag_add_hydro_source = false;
    if (hydro_source_terms_ptr) {
        if (DATA.Initial_profile == 42) {
            if (hydro_source_terms_ptr->get_number_of_sources() > 0) {
                flag_add_hydro_source = true;
            }
        } else {
            flag_add_hydro_source = true;
        }
    }
}

//! this function evolves one Runge-Kutta step in tau
void Advance::AdvanceIt(const double tau, Fields &arenaFieldsPrev,
                        Fields &arenaFieldsCurr, Fields &arenaFieldsNext,
                        const int rk_flag) {
#ifdef MUSIC_USE_GPU
    // Try the full-GPU path first.  If it returns true the substep is done;
    // otherwise (unsupported config, GPU init failed, etc.) fall through to
    // the CPU triple loop below.  See PORT_GPU.md §5 Phase 2.
    if (try_gpu_advance(tau, arenaFieldsPrev,
                        arenaFieldsCurr, arenaFieldsNext, rk_flag)) {
        return;
    }
#endif

    const int grid_neta = arenaFieldsCurr.nEta();
    const int grid_nx   = arenaFieldsCurr.nX();
    const int grid_ny   = arenaFieldsCurr.nY();

    // ── CPU triple loop: ideal evolution + Newton solve ───────────────────────
    #pragma omp parallel for collapse(3) schedule(guided)
    for (int ieta = 0; ieta < grid_neta; ieta++)
    for (int ix   = 0; ix   < grid_nx;   ix++  )
    for (int iy   = 0; iy   < grid_ny;   iy++  ) {
        int fieldIdx = arenaFieldsCurr.getFieldIdx(ix, iy, ieta);

        double eta_s_local = - DATA.eta_size/2. + ieta*DATA.delta_eta;
        double x_local     = - DATA.x_size  /2. +   ix*DATA.delta_x;
        double y_local     = - DATA.y_size  /2. +   iy*DATA.delta_y;

        FirstRKStepT(tau, x_local, y_local, eta_s_local,
                     ix, iy, ieta, rk_flag,
                     fieldIdx, arenaFieldsCurr, arenaFieldsNext,
                     arenaFieldsPrev);

        if (DATA.viscosity_flag == 1) {
            // No viscosity for 4D EoS -> no need to modify u_derivative class.
            U_derivative u_derivative_helper(DATA, eos);
            u_derivative_helper.MakedU(tau, arenaFieldsPrev,
                                       arenaFieldsCurr,
                                       fieldIdx, ix, iy, ieta);
            double theta_local = u_derivative_helper.calculate_expansion_rate(
                                            tau, arenaFieldsCurr, fieldIdx);
            DumuVec a_local;
            u_derivative_helper.calculate_Du_supmu(arenaFieldsCurr,
                                                   fieldIdx, a_local);

            VelocityShearVec sigma_local;
            u_derivative_helper.calculate_velocity_shear_tensor(
                    tau, arenaFieldsCurr, fieldIdx, theta_local,
                    a_local, sigma_local);

            VorticityVec omega_local;
            u_derivative_helper.calculate_kinetic_vorticity_with_spatial_projector(
                    tau, arenaFieldsCurr, fieldIdx, a_local, omega_local);

            DmuMuBoverTVec baryon_diffusion_vector;
            u_derivative_helper.get_DmuMuBoverTVec(baryon_diffusion_vector);

            FirstRKStepW(tau, arenaFieldsPrev, arenaFieldsCurr,
                         arenaFieldsNext, rk_flag, theta_local, a_local,
                         sigma_local, omega_local, baryon_diffusion_vector,
                         ieta, ix, iy, fieldIdx);
        }
    }
}


/* %%%%%%%%%%%%%%%%%%%%%% First steps begins here %%%%%%%%%%%%%%%%%% */
void Advance::FirstRKStepT(
        const double tau, const double x_local, const double y_local,
        const double eta_s_local,
        const int ix, const int iy, const int ieta, const int rk_flag,
        const int fieldIdx, Fields &arenaFieldsCurr,
        Fields &arenaFieldsNext, Fields &arenaFieldsPrev) {
    // this advances the ideal part
    double tau_rk = tau + rk_flag*(DATA.delta_tau);
    double tauFactor = tau;
    double tauRkFactor = tau_rk;
    double tauNextFactor = tau + DATA.delta_tau;
    if (DATA.CoorType == 1) {
        tauFactor = 1.;
        tauRkFactor = 1.;
        tauNextFactor = 1.;
    }

    auto cellPrev = arenaFieldsPrev.getCellIdeal(fieldIdx);
    auto cellCurr = arenaFieldsCurr.getCellIdeal(fieldIdx);

    // Solve partial_a T^{a mu} = -partial_a W^{a mu}
    // Update T^{mu nu}
    // MakeDelatQI gets
    //   qi = q0 if rk_flag = 0 or
    //   qi = q0 + k1 if rk_flag = 1
    // rhs[alpha] is what MakeDeltaQI outputs.
    // It is the spatial derivative part of partial_a T^{a mu}
    // (including geometric terms)
    TJbVec qi = {0};
    double pressure = eos.get_pressure(cellCurr.e, cellCurr.rhob,
                                       cellCurr.rhoq, cellCurr.rhos);

    for (int alpha = 0; alpha < 7; alpha++) {
        qi[alpha] = get_TJb(cellCurr, alpha, 0, pressure)*tauRkFactor;
    }

    MakeDeltaQI(tauRkFactor, arenaFieldsCurr, ix, iy, ieta, qi, rk_flag);

    TJbVec qi_source = {0.0};

    if (flag_add_hydro_source) {
        EnergyFlowVec j_mu = {0};
        FlowVec u_local;
        for (int ii = 0; ii < 4; ii++)
            u_local[ii] = arenaFieldsCurr.u_[ii][fieldIdx];

        hydro_source_terms_ptr->get_hydro_energy_source(
                    tau_rk, x_local, y_local, eta_s_local, u_local, j_mu);
        for (int ii = 0; ii < 4; ii++) {
            qi_source[ii] = tauRkFactor*j_mu[ii];
            if (isnan(qi_source[ii])) {
                music_message << "qi_source is nan. i = " << ii;
                music_message.flush("error");
                exit(0);
            }
        }

        if (DATA.turn_on_rhob == 1) {
            qi_source[4] = (
                tauRkFactor*hydro_source_terms_ptr->get_hydro_rhob_source(
                            tau_rk, x_local, y_local, eta_s_local, u_local));
        }

        if (DATA.turn_on_QS == 1) {
            qi_source[5] = (
                tauRkFactor*hydro_source_terms_ptr->get_hydro_rhoq_source(
                            tau_rk, x_local, y_local, eta_s_local, u_local));
            qi_source[6] = (
                tauRkFactor*hydro_source_terms_ptr->get_hydro_rhos_source(
                            tau_rk, x_local, y_local, eta_s_local, u_local));
        }
    }

    // now MakeWSource returns partial_a W^{a mu}
    // (including geometric terms)

    TJbVec dwmn ={0.0};
    diss_helper.MakeWSource(tauRkFactor, ix, iy, ieta,
                            dwmn, arenaFieldsCurr, arenaFieldsPrev, fieldIdx);


    double pressurePrev = eos.get_pressure(cellPrev.e, cellPrev.rhob,
                                           cellPrev.rhoq, cellPrev.rhos);

    for (int alpha = 0; alpha < 7; alpha++) {
        /* dwmn is the only one with the minus sign */
        qi[alpha] -= dwmn[alpha]*(DATA.delta_tau);

        // add energy momentum and net baryon density source terms
        qi[alpha] += qi_source[alpha]*DATA.delta_tau;

        // set baryon density back to zero if viscous correction made it
        // non-zero remove/modify if rho_b!=0
        // - this is only to remove the viscous correction that
        // can make rho_b negative which we do not want.
        //if (DATA.turn_on_rhob == 0) {
        //    if (alpha == 4 && std::abs(qi[alpha]) > 1e-12)
        //        qi[alpha] = 0.;
        //}

        /* if rk_flag > 0, we now have q0 + k1 + k2.
         * So add q0 and multiply by 1/2 */
        qi[alpha] += (rk_flag
                      *get_TJb(cellPrev, alpha, 0, pressurePrev)*tauFactor);
        qi[alpha] *= 1./(1. + rk_flag);
    }

    auto grid_rk_t = reconst_helper.ReconstIt_shell(tauNextFactor, qi,
                                                    cellCurr);
    arenaFieldsNext.e_[fieldIdx] = grid_rk_t.e;


    arenaFieldsNext.rhob_[fieldIdx] = grid_rk_t.rhob;
    arenaFieldsNext.rhoq_[fieldIdx] = grid_rk_t.rhoq;
    arenaFieldsNext.rhos_[fieldIdx] = grid_rk_t.rhos;

    for (int ii = 0; ii < 4; ii++) {
        arenaFieldsNext.u_[ii][fieldIdx] = grid_rk_t.u[ii];
    }
}


void Advance::FirstRKStepW(const double tau, Fields &arenaFieldsPrev,
                           Fields &arenaFieldsCurr, Fields &arenaFieldsNext,
                           const int rk_flag, const double theta_local,
                           const DumuVec &a_local,
                           const VelocityShearVec &sigma_local,
                           const VorticityVec &omega_local,
                           const DmuMuBoverTVec &baryon_diffusion_vector,
                           const int ieta, const int ix, const int iy,
                           const int fieldIdx) {
    auto grid_c = arenaFieldsCurr.getCell(fieldIdx);
    auto grid_f = arenaFieldsNext.getCell(fieldIdx);

    const double tau_now  = tau + rk_flag*DATA.delta_tau;

    // Solve partial_a (u^a W^{mu nu}) = 0
    // Update W^{mu nu}
    // mu = 4 is the baryon current qmu

    // calculate delta uWmunu
    // need to use u[0][mu], remember rk_flag = 0 here
    // with the KT flux
    // solve partial_tau (u^0 W^{kl}) = -partial_i (u^i W^{kl}
    /* Advance uWmunu */

    // spatial gradients for all viscous quantities
    std::array<double, 9> w_rhs = {0.};
    diss_helper.Make_uWRHS(tau_now, arenaFieldsCurr, fieldIdx,
                           ix, iy, ieta, w_rhs, theta_local, a_local);

    // Adapt for 4D EoS -- For the moment no viscosities for 4D EoS so ok.
    std::vector<double> thermalVec;
    if (rk_flag == 0) {
        eos.getThermalVariables(grid_c.epsilon, grid_c.rhob, grid_c.rhoq,
                                grid_c.rhos, thermalVec);
    } else {
        eos.getThermalVariables(arenaFieldsPrev.e_[fieldIdx],
                                arenaFieldsPrev.rhob_[fieldIdx],
                                arenaFieldsPrev.rhoq_[fieldIdx],
                                arenaFieldsPrev.rhos_[fieldIdx], thermalVec);
    }

    double tempf;
    double u0Prev = arenaFieldsPrev.u_[0][fieldIdx];
    if (DATA.turn_on_shear == 1) {
        std::array<double, 5> sourceTerms = {0.};
        diss_helper.Make_uWSource(
            tau_now, grid_c, theta_local, a_local, sigma_local, omega_local,
            thermalVec, sourceTerms);
        for (int idx_1d = 4; idx_1d < 9; idx_1d++) {
            double WmunuPrev = arenaFieldsPrev.Wmunu_[idx_1d][fieldIdx];
            tempf = ((1. - rk_flag)*(grid_c.Wmunu[idx_1d]*grid_c.u[0])
                     + rk_flag*(WmunuPrev*u0Prev)
            );
            tempf += sourceTerms[idx_1d-4]*(DATA.delta_tau);
            tempf += w_rhs[idx_1d-4];
            tempf += rk_flag*((grid_c.Wmunu[idx_1d])*(grid_c.u[0]));
            tempf *= 1./(1. + rk_flag);
            grid_f.Wmunu[idx_1d] = tempf/(grid_f.u[0]);
        }
    } else {
        for (int idx_1d = 4; idx_1d < 9; idx_1d++) {
            grid_f.Wmunu[idx_1d] = 0.0;
        }
    }

    if (DATA.turn_on_bulk == 1) {
        double piBulkPrev = arenaFieldsPrev.piBulk_[fieldIdx];
        tempf = ((1. - rk_flag)*(grid_c.pi_b*grid_c.u[0])
                 + rk_flag*piBulkPrev*u0Prev);
        double temps = diss_helper.Make_uPiSource(
                    tau_now, grid_c, theta_local, sigma_local, thermalVec);
        tempf += temps*(DATA.delta_tau);
        tempf += w_rhs[5];
        tempf += rk_flag*((grid_c.pi_b)*(grid_c.u[0]));
        tempf *= 1./(1. + rk_flag);
        grid_f.pi_b = tempf/(grid_f.u[0]);
    } else {
        grid_f.pi_b = 0.0;
    }

    // CShen: add source term for baryon diffusion
    if (DATA.turn_on_diff == 1) {
        std::array<double, 3> sourceTerms = {0.};
        diss_helper.Make_uqSource(tau_now, grid_c,
                                  theta_local, a_local, sigma_local,
                                  omega_local, baryon_diffusion_vector,
                                  thermalVec, sourceTerms);
        for (int idx_1d = 11; idx_1d < 14; idx_1d++) {
            double WmunuPrev = arenaFieldsPrev.Wmunu_[idx_1d][fieldIdx];
            tempf = ((1. - rk_flag)*(grid_c.Wmunu[idx_1d]*grid_c.u[0])
                     + rk_flag*WmunuPrev*u0Prev);
            tempf += sourceTerms[idx_1d-11]*(DATA.delta_tau);
            tempf += w_rhs[idx_1d-5];
            tempf += rk_flag*(grid_c.Wmunu[idx_1d]*grid_c.u[0]);
            tempf *= 1./(1. + rk_flag);

            grid_f.Wmunu[idx_1d] = tempf/(grid_f.u[0]);
        }
    } else {
        for (int idx_1d = 10; idx_1d < 14; idx_1d++) {
            grid_f.Wmunu[idx_1d] = 0.0;
        }
    }

    // re-make Wmunu[3][3] so that Wmunu[mu][nu] is traceless
    double u0sq = grid_f.u[0]*grid_f.u[0];
    grid_f.Wmunu[9] = (
        (2.*(  grid_f.u[1]*grid_f.u[2]*grid_f.Wmunu[5]
             + grid_f.u[1]*grid_f.u[3]*grid_f.Wmunu[6]
             + grid_f.u[2]*grid_f.u[3]*grid_f.Wmunu[8])
         - (u0sq - grid_f.u[1]*grid_f.u[1])*grid_f.Wmunu[4]
         - (u0sq - grid_f.u[2]*grid_f.u[2])*grid_f.Wmunu[7])
        /(u0sq - grid_f.u[3]*grid_f.u[3]));

    // make Wmunu[i][0] using the transversality
    for (int mu = 1; mu < 4; mu++) {
        tempf = 0.0;
        for (int nu = 1; nu < 4; nu++) {
            int idx_1d = map_2d_idx_to_1d(mu, nu);
            tempf += grid_f.Wmunu[idx_1d]*grid_f.u[nu];
        }
        grid_f.Wmunu[mu] = tempf/(grid_f.u[0]);
    }

    // make Wmunu[0][0]
    tempf = 0.0;
    for (int nu = 1; nu < 4; nu++)
        tempf += grid_f.Wmunu[nu]*grid_f.u[nu];
    grid_f.Wmunu[0] = tempf/(grid_f.u[0]);

    // make qmu[0] using transversality
    tempf = 0.0;
    for (int idx_1d = 11; idx_1d < 14; idx_1d++) {
        tempf += grid_f.Wmunu[idx_1d]*grid_f.u[idx_1d-10];
    }
    grid_f.Wmunu[10] = DATA.turn_on_diff*tempf/(grid_f.u[0]);

    // If the energy density of the fluid element is smaller than 0.01GeV
    // reduce Wmunu using the QuestRevert algorithm
    if (DATA.Initial_profile != 0 && DATA.Initial_profile != 1) {
        QuestRevert(grid_f, ieta, ix, iy);
        if (DATA.turn_on_diff == 1) {
            QuestRevert_qmu(grid_f, ieta, ix, iy);
        }
    }
    for (int idx_1d = 0; idx_1d < 14; idx_1d++) {
        arenaFieldsNext.Wmunu_[idx_1d][fieldIdx] = grid_f.Wmunu[idx_1d];
        arenaFieldsNext.piBulk_[fieldIdx] = grid_f.pi_b;
    }
}


//! this function reduce the size of shear stress tensor and bulk pressure
//! in the dilute region to stablize numerical simulations
void Advance::QuestRevert(Cell_small &grid_pt,
                          const int ieta, const int ix, const int iy) {
    double eps_scale = 0.1;   // 1/fm^4
    double e_local   = grid_pt.epsilon;
    double rhob      = grid_pt.rhob;
    double rhoq      = grid_pt.rhoq;
    double rhos      = grid_pt.rhos;

    // regulation factor in the default MUSIC
    // double factor = 300.*tanh(grid_pt.epsilon/eps_scale);
    double xi = 0.05;
    double factor = 10.*DATA.quest_revert_strength*(
                        1./(exp(-(e_local - eps_scale)/xi) + 1.)
                            - 1./(exp(eps_scale/xi) + 1.));
    double factor_bulk = factor;

    double pi_00 = grid_pt.Wmunu[0];
    double pi_01 = grid_pt.Wmunu[1];
    double pi_02 = grid_pt.Wmunu[2];
    double pi_03 = grid_pt.Wmunu[3];
    double pi_11 = grid_pt.Wmunu[4];
    double pi_12 = grid_pt.Wmunu[5];
    double pi_13 = grid_pt.Wmunu[6];
    double pi_22 = grid_pt.Wmunu[7];
    double pi_23 = grid_pt.Wmunu[8];
    double pi_33 = grid_pt.Wmunu[9];

    double pisize = (pi_00*pi_00 + pi_11*pi_11 + pi_22*pi_22 + pi_33*pi_33
         - 2.*(pi_01*pi_01 + pi_02*pi_02 + pi_03*pi_03)
         + 2.*(pi_12*pi_12 + pi_13*pi_13 + pi_23*pi_23));

    double pi_local = grid_pt.pi_b;
    double bulksize = 3.*pi_local*pi_local;

    double p_local = eos.get_pressure(e_local, rhob, rhoq, rhos);
    double eq_size = e_local*e_local + 3.*p_local*p_local;

    // In default MUSIC
    double rho_shear = sqrt(pisize/eq_size)/factor;
    double rho_bulk  = sqrt(bulksize/eq_size)/factor_bulk;

    // Reducing the shear stress tensor
    double rho_shear_max = 0.1;
    if (std::isnan(rho_shear)) {
        for (int mu = 0; mu < 10; mu++) {
            grid_pt.Wmunu[mu] = 0.0;
        }
    } else if (rho_shear > rho_shear_max) {
        if (e_local > eps_scale && DATA.echo_level > 5) {
            music_message << "ieta = " << ieta << ", ix = " << ix
                          << ", iy = " << iy
                          << ", energy density = " << e_local*hbarc
                          << " GeV/fm^3, shear |pi/(epsilon+3*P)| = "
                          << rho_shear;
            music_message.flush("warning");
        }
        for (int mu = 0; mu < 10; mu++) {
            grid_pt.Wmunu[mu] = (rho_shear_max/rho_shear)*grid_pt.Wmunu[mu];
        }
    }

    // Reducing bulk viscous pressure
    double rho_bulk_max = 0.1;
    if (rho_bulk > rho_bulk_max) {
        if (e_local > eps_scale && DATA.echo_level > 5) {
            music_message << "ieta = " << ieta << ", ix = " << ix
                          << ", iy = " << iy
                          << ", energy density = " << e_local*hbarc
                          << " GeV/fm^3, bulk |Pi/(epsilon+3*P)| = "
                          << rho_bulk;
            music_message.flush("warning");
        }
        grid_pt.pi_b = (rho_bulk_max/rho_bulk)*grid_pt.pi_b;
    }
}


//! this function reduce the size of net baryon diffusion current
//! in the dilute region to stablize numerical simulations
void Advance::QuestRevert_qmu(Cell_small &grid_pt,
                              const int ieta, const int ix, const int iy) {
    double eps_scale = 0.1;   // in 1/fm^4

    double xi = 0.05;
    double factor = 10.*DATA.quest_revert_strength*(
                            1./(exp(-(grid_pt.epsilon - eps_scale)/xi) + 1.)
                            - 1./(exp(eps_scale/xi) + 1.));

    double q_mu_local[4];
    for (int i = 0; i < 4; i++) {
        // copy the value from the grid
        q_mu_local[i] = grid_pt.Wmunu[10+i];
    }

    // calculate the size of q^\mu
    double q_size = 0.0;
    for (int i = 0; i < 4; i++) {
        double gfac = (i == 0 ? -1.0 : 1.0);
        q_size += gfac*q_mu_local[i]*q_mu_local[i];
    }

    // first check the positivity of q^mu q_mu
    // (in the conversion of gmn = diag(-+++))
    if (q_size < 0.0) {
        music_message << "Advance::QuestRevert_qmu: q^mu q_mu = " << q_size
                      << " < 0!";
        music_message.flush("warning");
        music_message << "Reset it to zero!!!!";
        music_message.flush("warning");
        for (int i = 0; i < 4; i++) {
            int idx_1d = map_2d_idx_to_1d(4, i);
            grid_pt.Wmunu[idx_1d] = 0.0;
        }
    }

    // reduce the size of q^mu according to rhoB
    double e_local = grid_pt.epsilon;
    double rhob_local = grid_pt.rhob;
    double rho_q = sqrt(q_size/(rhob_local*rhob_local))/factor;
    double rho_q_max = 0.1;
    if (rho_q > rho_q_max) {
        if (e_local > eps_scale && DATA.echo_level > 5) {
            music_message << "ieta = " << ieta << ", ix = " << ix
                          << ", iy = " << iy
                          << ", energy density = " << e_local*hbarc
                          << "GeV/fm^3"
                          << ", rhob = " << rhob_local << "1/fm^3"
                          << "-- diffusion |q/rhob| = " << rho_q;
            music_message.flush("warning");
        }
        for (int i = 0; i < 4; i++) {
            grid_pt.Wmunu[10+i] = (rho_q_max/rho_q)*q_mu_local[i];
        }
    }
}


//! This function computes the rhs array. It computes the spatial
//! derivatives of T^\mu\nu using the KT algorithm
void Advance::MakeDeltaQI(const double tauFactor, Fields &arenaFieldsCurr,
                          const int ix, const int iy, const int ieta,
                          TJbVec &qi, const int rk_flag) {
    const double delta[4]   = {0.0, DATA.delta_x, DATA.delta_y, DATA.delta_eta};
    const double tau_fac[4] = {0.0, tauFactor, tauFactor, 1.0};

    TJbVec qiphL   = {0.};
    TJbVec qiphR   = {0.};
    TJbVec qimhL   = {0.};
    TJbVec qimhR   = {0.};

    TJbVec rhs     = {0.};
    EnergyFlowVec T_eta_m = {0.};
    EnergyFlowVec T_eta_p = {0.};
    FieldNeighbourLoopIdeal2(arenaFieldsCurr, ix, iy, ieta, FNLILAMBDAS2{

        double pressureP1 = eos.get_pressure(p1.e, p1.rhob, p1.rhoq, p1.rhos);
        double pressureP2 = eos.get_pressure(p2.e, p2.rhob, p2.rhoq, p2.rhos);
        double pressureM1 = eos.get_pressure(m1.e, m1.rhob, m1.rhoq, m1.rhos);
        double pressureM2 = eos.get_pressure(m2.e, m2.rhob, m2.rhoq, m2.rhos);

        for (int alpha = 0; alpha < 7; alpha++) {
            const double gphL = qi[alpha];
            const double gphR = tauFactor*get_TJb(p1, alpha, 0, pressureP1);
            const double gmhL = tauFactor*get_TJb(m1, alpha, 0, pressureM1);
            const double gmhR = qi[alpha];
            const double fphL =  0.5*minmod.minmod_dx(gphR, qi[alpha], gmhL);
            const double fphR = -0.5*minmod.minmod_dx(
                                    tauFactor*get_TJb(p2, alpha, 0, pressureP2),
                                    gphR, qi[alpha]);
            const double fmhL =  0.5*minmod.minmod_dx(
                                    qi[alpha], gmhL,
                                    tauFactor*get_TJb(m2, alpha, 0, pressureM2));
            const double fmhR = -fphL;
            qiphL[alpha] = gphL + fphL;
            qiphR[alpha] = gphR + fphR;
            qimhL[alpha] = gmhL + fmhL;
            qimhR[alpha] = gmhR + fmhR;
        }

        // for each direction, reconstruct half-way cells
        // reconstruct e, rhob, and u[4] for half way cells
        auto grid_phL = reconst_helper.ReconstIt_shell(tauFactor, qiphL, c);
        auto grid_phR = reconst_helper.ReconstIt_shell(tauFactor, qiphR, c);
        auto grid_mhL = reconst_helper.ReconstIt_shell(tauFactor, qimhL, c);
        auto grid_mhR = reconst_helper.ReconstIt_shell(tauFactor, qimhR, c);

        double aiphL = MaxSpeed(tauFactor, direction, grid_phL, pressureP1);
        double aiphR = MaxSpeed(tauFactor, direction, grid_phR, pressureP2);
        double aimhL = MaxSpeed(tauFactor, direction, grid_mhL, pressureM1);
        double aimhR = MaxSpeed(tauFactor, direction, grid_mhR, pressureM2);

        double aiph = std::max(aiphL, aiphR);
        double aimh = std::max(aimhL, aimhR);
        for (int alpha = 0; alpha < 7; alpha++) {
            double FiphL = (tau_fac[direction]
                            *get_TJb(grid_phL, alpha, direction, pressureP1));
            double FiphR = (tau_fac[direction]
                            *get_TJb(grid_phR, alpha, direction, pressureP2));
            double FimhL = (tau_fac[direction]
                            *get_TJb(grid_mhL, alpha, direction, pressureM1));
            double FimhR = (tau_fac[direction]
                            *get_TJb(grid_mhR, alpha, direction, pressureM2));

            // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2}))/2
            //                  - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
            double Fiph = 0.5*((FiphL + FiphR)
                               - aiph*(qiphR[alpha] - qiphL[alpha]));
            double Fimh = 0.5*((FimhL + FimhR)
                               - aimh*(qimhR[alpha] - qimhL[alpha]));
            if (direction == 3 && (alpha == 0 || alpha == 3)
                && DATA.CoorType == 0) {
                T_eta_m[alpha] = Fimh;
                T_eta_p[alpha] = Fiph;
            } else {
                double DFmmp = (Fimh - Fiph)/delta[direction];
                rhs[alpha] += DFmmp*(DATA.delta_tau);
            }
        }
    });

    if (DATA.CoorType == 0) {
        // add longitudinal flux with discretized geometric terms
        double cosh_deta = cosh(delta[3]/2.)/std::max(delta[3], Util::small_eps);
        double sinh_deta = sinh(delta[3]/2.)/std::max(delta[3], Util::small_eps);
        sinh_deta = std::max(0.5, sinh_deta);
        if (DATA.boost_invariant) {
            // if the simulation is boost-invariant,
            // we directly use the limiting value at \Delta eta = 0
            // Longitudinal derivatives should be 0, we set cosh_eta = 0 here
            cosh_deta = 0.0;
            sinh_deta = 0.5;
        }
        rhs[0] += ((  (T_eta_m[0] - T_eta_p[0])*cosh_deta
                    - (T_eta_m[3] + T_eta_p[3])*sinh_deta)*DATA.delta_tau);
        rhs[3] += ((  (T_eta_m[3] - T_eta_p[3])*cosh_deta
                    - (T_eta_m[0] + T_eta_p[0])*sinh_deta)*DATA.delta_tau);
        // geometric terms
        //rhs[0] -= get_TJb(arena_current(ix, iy, ieta), 3, 3)*DATA.delta_tau;
        //rhs[3] -= get_TJb(arena_current(ix, iy, ieta), 3, 0)*DATA.delta_tau;
    }

    for (int i = 0; i < 7; i++) {
        qi[i] += rhs[i];
    }
}

// determine the maximum signal propagation speed at the given direction
double Advance::MaxSpeed(const double tau, const int direc,
                         const ReconstCell &grid_p, double &pressure) {
    double g[] = {1., 1., 1./tau};

    double utau    = grid_p.u[0];
    double utau2   = utau*utau;
    double ux      = std::abs(grid_p.u[direc]);
    double ut2mux2 = utau2 - ux*ux;

    double eps  = grid_p.e;
    double rhob = grid_p.rhob;
    double rhoq = grid_p.rhoq;
    double rhos = grid_p.rhos;

    //double vs2 = eos.get_cs2(eps, rhob);
    double dpde, dpdrhob, dpdrhoq, dpdrhos, vs2;
    eos.get_pressure_with_gradients_and_cs2(
            eps, rhob, rhoq, rhos, pressure,
            dpde, dpdrhob, dpdrhoq, dpdrhos, vs2);
    double num_temp_sqrt = (ut2mux2 - (ut2mux2 - 1.)*vs2)*vs2;
    double num;
    if (num_temp_sqrt >= 0)  {
        num = utau*ux*(1. - vs2) + sqrt(num_temp_sqrt);
    } else {
        //double dpde = eos.get_dpde(eps, rhob);
        double h = pressure + eps;
        if (dpde < 0.001) {
            num = (sqrt(-(h*dpde*h*(dpde*(-1.0 + ut2mux2) - ut2mux2)))
                   - h*(-1.0 + dpde)*utau*ux);
        } else {
          fprintf(stderr,"WARNING: in MaxSpeed. \n");
          fprintf(stderr, "Expression under sqrt in num=%lf. \n", num_temp_sqrt);
          fprintf(stderr,"at value e=%lf. \n",eps);
          fprintf(stderr,"at value p=%lf. \n",pressure);
          fprintf(stderr,"at value h=%lf. \n",h);
          fprintf(stderr,"at value rhob=%lf. \n",rhob);
          fprintf(stderr,"at value rhoq=%lf. \n",rhoq);
          fprintf(stderr,"at value rhos=%lf. \n",rhos);
          fprintf(stderr,"at value utau=%lf. \n", utau);
          fprintf(stderr,"at value uk=%lf. \n", ux);
          fprintf(stderr,"at value vs^2=%lf. \n", vs2);
          fprintf(stderr,"at value dpde=%lf. \n", dpde);
          fprintf(stderr,"at value dpdrhob=%lf. \n", dpdrhob);
          fprintf(stderr,"at value dpdrhoq=%lf. \n", dpdrhoq);
          fprintf(stderr,"at value dpdrhos=%lf. \n", dpdrhos);
          fprintf(stderr, "MaxSpeed: exiting.\n");
          exit(1);
        }
    }
    double den = utau2*(1. - vs2) + vs2;
    double f = num/std::max(den, Util::small_eps);
    // check for problems
    if (f < 0.0) {
        fprintf(stderr, "SpeedMax = %e\n is negative.\n", f);
        fprintf(stderr, "Can't happen.\n");
        exit(0);
    } else if (f <  ux/utau) {
        if (num != 0.0) {
            if (fabs(f-ux/utau)<0.0001) {
                f = ux/utau;
            } else {
                fprintf(stderr, "SpeedMax-v = %lf\n", f-ux/utau);
                fprintf(stderr, "SpeedMax = %e\n is smaller than v = %e.\n", f, ux/utau);
                fprintf(stderr, "Can't happen.\n");
                exit(0);
            }
        }
    } else if (f > 1.0) {
        fprintf(stderr, "SpeedMax = %e\n is bigger than 1.\n", f);
        fprintf(stderr, "Can't happen.\n");
        fprintf(stderr, "SpeedMax = num/den, num = %e, den = %e \n", num, den);
        fprintf(stderr, "cs2 = %e \n", vs2);
        f =1.;
        exit(1);
    }
    f *= g[direc-1];
    return f;
}


double Advance::get_TJb(const ReconstCell &grid_p, const int mu, const int nu,
                        const double pressure) {
    //assert(mu < 5); assert(mu > -1);
    //assert(nu < 4); assert(nu > -1);

    const double u_nu = grid_p.u[nu];

    if (mu == 4) {
        return (grid_p.rhob)*u_nu;
    } else if (mu == 5) {
        return (grid_p.rhoq)*u_nu;
    } else if (mu == 6) {
        return (grid_p.rhos)*u_nu;
    }

    double e = grid_p.e;
    double gfac = 0.0;
    double u_mu = 0.0;
    if (mu == nu) {
        u_mu = u_nu;
        gfac = 1.0;
        if (mu == 0) {
            gfac = -1.0;
        }
    } else {
        u_mu = grid_p.u[mu];
    }
    const double T_munu = (e + pressure)*u_mu*u_nu + pressure*gfac;
    return(T_munu);
}
