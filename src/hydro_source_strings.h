// Copyright 2019 Chun Shen

#ifndef SRC_HYDRO_SOURCE_STRINGS_H_
#define SRC_HYDRO_SOURCE_STRINGS_H_

#include <array>
#include <cmath>
#include <vector>
#include <memory>
#include "hydro_source_base.h"

//! This data structure contains a QCD string object
struct QCD_string {
    double norm;              // normalization for the string energy
    double E_remnant_norm_L, E_remnant_norm_R;
    double m_over_sigma;      // m/sigma [fm] sigma is the string tension

    double mass;
    double tau_form;
    double sigma_x, sigma_eta;
    double tau_start, eta_s_start;
    double tau_0, eta_s_0;
    double x_perp, y_perp;    // transverse position of the string
    double x_pl, y_pl, x_pr, y_pr;
    double tau_end_left, tau_end_right;
    double eta_s_left, eta_s_right;
    double y_l, y_r;          // rapidity of the two ends of the string
    double remnant_l, remnant_r;
    double y_l_i, y_r_i;
    double tau_baryon_left, tau_baryon_right;
    double eta_s_baryon_left, eta_s_baryon_right;
    double y_l_baryon, y_r_baryon;
    double baryon_frac_l, baryon_frac_r;
    double tau_Qe_left, tau_Qe_right;
    double Qe_left, Qe_right;
    double eta_s_Qe_left, eta_s_Qe_right;
    double px_i, py_i;       // px and py of the strings 
};


//! The strings of one current-tau list, binned by the transverse box each can
//! reach.  The per-cell source loops skip a string that is more than
//! n_sigma_skip * sigma_x away from it in x or in y before adding anything, so
//! a cell needs only the strings whose box contains it.  Each bin lists them in
//! list order, so a cell adds the same terms in the same order as a loop over
//! the whole list: the result is bit-identical.
struct StringTransverseBins {
    double x0 = 0., y0 = 0.;     //!< lower edge of bin (0, 0) [fm]
    double bx = 1., by = 1.;     //!< bin widths [fm]
    int nx = 0, ny = 0;
    std::vector<std::vector<int>> bins;   //!< [ix * ny + iy] -> list indices

    void clear() { nx = ny = 0; bins.clear(); }

    //! (Re)build for boxes[i] = {x_lo, x_hi, y_lo, y_hi} of list entry i, on
    //! bins of width (bx_in, by_in) covering [x_min, x_max] x [y_min, y_max].
    void build(const std::vector<std::array<double, 4>> &boxes,
               double x_min, double x_max, double y_min, double y_max,
               double bx_in, double by_in);

    //! The list indices to visit at (x, y), or nullptr for "the whole list"
    //! (not built, or a point outside the binned area).
    const std::vector<int> *lookup(const double x, const double y) const {
        if (nx == 0) return nullptr;
        const double fx = std::floor((x - x0)/bx);
        const double fy = std::floor((y - y0)/by);
        if (!(fx >= 0. && fx < nx && fy >= 0. && fy < ny)) return nullptr;
        return &bins[static_cast<int>(fx)*ny + static_cast<int>(fy)];
    }
};


class HydroSourceStrings : public HydroSourceBase {
 private:
    InitData &DATA;
    int string_dump_mode;
    double string_quench_factor;
    double parton_quench_factor;
    double stringTransverseShiftFrac_;
    double preEqFlowFactor_;
    std::vector<std::shared_ptr<QCD_string>> QCD_strings_list;
    std::vector<std::shared_ptr<QCD_string>> QCD_strings_list_current_tau;
    std::vector<std::shared_ptr<QCD_string>> QCD_strings_remnant_list_current_tau;
    std::vector<std::shared_ptr<QCD_string>> QCD_strings_baryon_list_current_tau;
    std::vector<std::shared_ptr<QCD_string>> QCD_strings_electric_list_current_tau;

    //! The per-cell loops skip a string beyond this many sigma_x / sigma_eta.
    static constexpr double n_sigma_skip_ = 8.;
    //! Transverse bins of the three current-tau lists the per-cell loops walk,
    //! rebuilt by prepare_list_for_current_tau_frame().
    StringTransverseBins string_bins_, remnant_bins_, baryon_bins_;
    void build_transverse_bins();

 public:
    HydroSourceStrings() = delete;
    HydroSourceStrings(InitData &DATA_in);
    HydroSourceStrings(InitData &DATA_in,
                       std::vector< std::vector<double> > QCDStringList);
    ~HydroSourceStrings();

    //! This function reads in the spatal information of the strings
    //! and partons which are produced from the MC-Glauber-LEXUS model
    void read_in_QCD_strings_and_partons();
    void read_in_QCD_strings_and_partons(
        std::vector< std::vector<double> > QCDStringList);

    double getStringEndTau(const double tau0, const double tau_form,
                           const double eta_s_0, const double eta_s) const;

    //! this function returns the energy source term J^\mu at a given point
    //! (tau, x, y, eta_s)
    //! The energy source reads the string and remnant lists, rhob the baryon
    //! list, rhoq the baryon or electric list; with all four empty every one
    //! of them returns exactly zero.  With evolve_QCD_string_mode 4 that is
    //! every step after the strings are deposited at the start.
    bool has_active_sources_current_tau() const override {
        return(!(QCD_strings_list_current_tau.empty()
                 && QCD_strings_remnant_list_current_tau.empty()
                 && QCD_strings_baryon_list_current_tau.empty()
                 && QCD_strings_electric_list_current_tau.empty()));
    }

    void get_hydro_energy_source(const double tau, const double x,
                                 const double y, const double eta_s,
                                 const FlowVec &u_mu,
                                 EnergyFlowVec &j_mu) const;

    //! this function returns the net baryon density source term rho
    //! at a given point (tau, x, y, eta_s)
    double get_hydro_rhob_source(const double tau, const double x,
                                 const double y, const double eta_s,
                                 const FlowVec &u_mu) const;
    double get_hydro_rhoq_source(const double tau, const double x,
                                 const double y, const double eta_s,
                                 const FlowVec &u_mu) const;

    void prepare_list_for_current_tau_frame(const double tau_local);
    void compute_norm_for_strings();
    double getStringTransverseCoord(const double xl, const double xr,
                                    const double etaFrac) const;
};

#endif  // SRC_HYDRO_SOURCE_STRINGS_H_
