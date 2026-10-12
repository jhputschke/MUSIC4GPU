// Copyright Chun Shen @ 2018

#ifndef SRC_HYDROINFOMUSIC_H_
#define SRC_HYDROINFOMUSIC_H_

#include <cstddef>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "data_struct.h"
#include "data.h"
#include "pretty_ostream.h"

//! std::allocator, except that resize() leaves new elements default- rather
//! than value-initialised: a frame of the store is written by the pack right
//! after it is added, so zeroing it first would be one more pass over 19 MB.
template <class T>
struct DefaultInitAllocator : std::allocator<T> {
    template <class U> struct rebind { using other = DefaultInitAllocator<U>; };
    using std::allocator<T>::allocator;
    template <class U>
    void construct(U *p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void *>(p)) U;
    }
    template <class U, class... Args>
    void construct(U *p, Args &&... args) {
        ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }
};

class HydroinfoMUSIC {
 private:
    double hydroTau0;       // tau_0 in the hydro data files
    double hydroTauMax;     // tau_max in the hydro data files
    double hydroDtau;       // step dtau in fm/c in the hydro data files
    double hydroXmax;       // maximum x in fm in the hydro data files
                            // [-xmax, +xmax] for both x and y
    double hydroYmax;       // maximum y in fm in the hydro data files
                            // [-ymax, +ymax] for both x and y
    double hydro_eta_max;   // maximum z in fm in the hydro data files
                            // [-zmax, +zmax] for 3D hydro
    double hydroDx;         // step dx in fm in the hydro data files
    double hydroDy;         // step dy in fm in the hydro data files
    double hydroDeta;       // step dz in fm in the hydro data files in
                            // the z-direction for 3D hydro

    int use_tau_eta_coordinate;
    bool boost_invariant;

    int itaumax, ixmax, iymax, ietamax;

    std::vector<fluidCell_ideal, DefaultInitAllocator<fluidCell_ideal>>
        lattice_ideal;
    size_t pending_frame_start_ = 0;    // begin_frame .. commit/drop_frame
    pretty_ostream music_message;

 public:
    HydroinfoMUSIC();       // constructor
    ~HydroinfoMUSIC();      // destructor

    void clean_hydro_event();
    double get_hydro_tau_max() const {return(hydroTauMax);}
    double get_hydro_tau0() const    {return(hydroTau0);}
    double get_hydro_dtau() const    {return(hydroDtau);}
    double get_hydro_dx() const      {return(hydroDx);}
    double get_hydro_dy() const      {return(hydroDy);}
    double get_hydro_deta() const    {return(hydroDeta);}
    double get_hydro_eta_max() const {return(hydro_eta_max);}
    double get_hydro_x_max() const   {return(hydroXmax);}
    double get_hydro_y_max() const   {return(hydroYmax);}
    int get_ntau() const {return(itaumax);}
    int get_nx()   const {return(ixmax  );}
    int get_ny()   const {return(iymax  );}
    int get_neta() const {return(ietamax);}
    bool is_boost_invariant() const {return(boost_invariant);}

    void getHydroValues(const double x, const double y,
                        const double z, const double t,
                        fluidCell *info);
    void set_grid_infomatioin(const InitData &DATA);
    void print_grid_information();
    void dump_ideal_info_to_memory(double tau, float eta, float epsilon,
                                   float pressure, float entropy, float T,
                                   float ux, float uy, float ueta);
    //! Add one frame of n cells to the store and return where it starts, for
    //! the caller to fill in place (ix outer, iy middle, ieta inner, as the
    //! per-cell dump_ideal_info_to_memory sequence); then commit_frame(tau),
    //! or drop_frame() if it could not be filled.
    fluidCell_ideal *begin_frame(size_t n);
    void commit_frame(double tau);
    void drop_frame();

    int get_number_of_fluid_cells() const {return(lattice_ideal.size());}
    void get_fluid_cell_with_index(const int idx, fluidCell *info) const;

    // Validation hook: when MUSIC_PACK_DUMP=<path> is set, write the raw
    // lattice_ideal (int64 count, then count*8 floats in struct field order)
    // to that path.  Used to A/B the GPU pack kernel against the host loop
    // (set MUSIC_GPU_NO_PACK=1 for the baseline run).  No-op otherwise.
    void dump_ideal_lattice_if_requested() const;
};

#endif  // SRC_HYDROINFO_MUSIC_H_

