// Kokkos backing for the GPUGrid SoA storage class (declared in GPUGrid.h,
// shared with the CUDA/Metal backends).  Counterpart of GPUGrid_cuda.cu /
// GPUGrid.mm; compiled only when USE_KOKKOS is set.
//
// Memory model (Stage 1): every snapshot/scratch/EOS buffer is a
// Kokkos::View<float*, Kokkos::SharedSpace>.  SharedSpace resolves to
//   - CudaUVMSpace / HIPManagedSpace / SYCLSharedUSMSpace on a GPU build
//     (host+device-accessible managed memory — the GB10/coherent zero-copy
//     path, and migrate-on-touch on a discrete GPU), and
//   - HostSpace on a Serial/OpenMP host build.
// So the host AoS<->SoA pack/unpack loops below write/read the Views' data()
// pointer directly (no staging, no explicit deep_copy) on every backend — the
// same in-place model the CUDA coherent path uses.  The discrete device-View +
// pinned-mirror optimisation is a Stage-2 refinement; UVM is correct everywhere
// and is exactly what the native CUDA backend uses on this GB10 box.
//
// GPUGrid.h stays Kokkos-free (PIMPL boundary, D7): the owning Views live in a
// TU-static store keyed by the GPUGrid instance, and only the raw float* aliases
// (used by every backend) are kept on the struct.  Snapshot rotation permutes
// those aliases without moving any data (View handles are untouched).

#include "GPUGrid.h"
#include "../grid.h"    // SCGrid, Cell_small
#include "../fields.h"  // Fields (XSCAPE SoA arena)

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using KView = Kokkos::View<float*, Kokkos::SharedSpace>;

// Owns the lifetime of every Kokkos allocation for one GPUGrid.  The raw float*
// on the struct point into these Views' data().
struct KStore {
    std::vector<KView> owned;   // snapshots + scratch + EOS tables
    KView              evo_pack;  // lazily (re)sized pack-evolution scratch

    float* new_buf(size_t n) {
        // Default-initialised (zero) — allocate-once, so the one-time cost is
        // negligible and it avoids any read-before-write surprise in the
        // partial-GPU fallback configs.
        owned.emplace_back("music_kokkos_buf", n);
        return owned.back().data();
    }
};

std::unordered_map<const GPUGrid*, KStore>& store_map() {
    static std::unordered_map<const GPUGrid*, KStore> m;
    return m;
}

inline int cell_idx(int ix, int iy, int ieta, int Nx, int Ny) {
    return Nx * (Ny * ieta + iy) + ix;
}

}  // namespace

// Accessor used by KokkosPipelines::pack_evolution_ideal to (re)size and fetch
// the device pack scratch without exposing Kokkos types in GPUGrid.h.
float* kokkos_grid_evo_pack(GPUGrid& gpu, size_t floats);

// ── Allocation / teardown ────────────────────────────────────────────────────

bool GPUGrid::alloc_snapshot(GPUSnapshot& /*s*/, int /*Ncells*/) {
    // Snapshots are allocated inline in allocate() so they share the per-grid
    // KStore; this stub stays for the GPUGrid.h interface.
    return true;
}

bool GPUGrid::allocate(int Nx, int Ny, int Neta) {
    Nx_     = Nx;
    Ny_     = Ny;
    Neta_   = Neta;
    Ncells_ = Nx * Ny * Neta;
    n_handles_ = 0;

    if (!Kokkos::is_initialized()) {
        // The KokkosRuntimeGuard in main() should have brought Kokkos up; if not,
        // refuse so advance.cpp falls back to the CPU reference.
        return false;
    }

    KStore& st = store_map()[this];
    st.owned.clear();
    st.owned.reserve(40);

    const size_t nc = static_cast<size_t>(Ncells_);

    auto alloc_snap = [&](GPUSnapshot& s) -> bool {
        s.epsilon = st.new_buf(nc);
        s.rhob    = st.new_buf(nc);
        s.u       = st.new_buf(GPU_U_COMPS     * nc);
        s.Wmunu   = st.new_buf(GPU_WMUNU_COMPS * nc);
        s.pi_b    = st.new_buf(nc);
        // *_stage stay null: SharedSpace is host-accessible (coherent/UVM path).
        return s.epsilon && s.rhob && s.u && s.Wmunu && s.pi_b;
    };

    bool ok = alloc_snap(snap_prev)
           && alloc_snap(snap_curr)
           && alloc_snap(snap_future);

    dwmn          = st.new_buf(5  * nc);
    qi_out        = st.new_buf(5  * nc);
    uwrhs_out     = st.new_buf(5  * nc);
    uprhs_out     = st.new_buf(     nc);
    qi_source_buf = st.new_buf(5  * nc);
    theta_buf     = st.new_buf(     nc);
    a_buf         = st.new_buf(4  * nc);
    sigma_buf     = st.new_buf(10 * nc);
    reduce_eps_out  = st.new_buf(1);
    reduce_rhob_out = st.new_buf(1);

    ok = ok && dwmn && qi_out && uwrhs_out && uprhs_out && qi_source_buf
            && theta_buf && a_buf && sigma_buf
            && reduce_eps_out && reduce_rhob_out;

    allocated_ = ok;
    return ok;
}

bool GPUGrid::upload_eos(const float* P_data, const float* dPde_data,
                         const float* s_data, const float* T_data,
                         int n_pts, float e_min, float e_max) {
    if (!allocated_) return false;
    auto it = store_map().find(this);
    if (it == store_map().end()) return false;
    KStore& st = it->second;

    const size_t nc = static_cast<size_t>(n_pts);
    eos_P    = st.new_buf(nc);
    eos_dPde = st.new_buf(nc);
    eos_s    = st.new_buf(nc);
    eos_T    = st.new_buf(nc);
    if (!eos_P || !eos_dPde || !eos_s || !eos_T) return false;

    std::memcpy(eos_P,    P_data,    nc * sizeof(float));
    std::memcpy(eos_dPde, dPde_data, nc * sizeof(float));
    std::memcpy(eos_s,    s_data,    nc * sizeof(float));
    std::memcpy(eos_T,    T_data,    nc * sizeof(float));

    eos_params.e_min   = e_min;
    eos_params.e_max   = e_max;
    eos_params.n_pts   = n_pts;
    eos_params.delta_e = (n_pts > 1)
                         ? (e_max - e_min) / static_cast<float>(n_pts - 1)
                         : 1.f;

    constexpr float s_log_e_floor = 1.e-6f;
    eos_params.log_e_min   = std::log(s_log_e_floor);
    eos_params.log_e_max   = std::log(std::max(e_max, s_log_e_floor * 1.01f));
    eos_params.log_delta_e = (n_pts > 1)
                             ? (eos_params.log_e_max - eos_params.log_e_min)
                                 / static_cast<float>(n_pts - 1)
                             : 1.f;
    return true;
}

void GPUGrid::release() {
    auto it = store_map().find(this);
    if (it != store_map().end()) {
        // Only touch Kokkos while the runtime is alive (the KokkosRuntimeGuard in
        // main() outlives every GPUGrid; this guards the pathological teardown
        // order so a stray destruction post-finalize can't crash).
        if (Kokkos::is_initialized()) {
            Kokkos::fence();          // no kernel may still reference the buffers
            store_map().erase(it);    // drops Views -> frees managed memory
        } else {
            // Runtime already gone: the allocations are reclaimed at process
            // exit; just drop our bookkeeping pointer.
            it->second.owned.clear();
            store_map().erase(it);
        }
    }

    allocated_ = false;
    n_handles_ = 0;
    snap_prev = snap_curr = snap_future = GPUSnapshot{};
    dwmn      = nullptr;
    qi_out    = nullptr;
    uwrhs_out = nullptr;
    uprhs_out = nullptr;
    qi_source_buf = nullptr;
    theta_buf = nullptr;
    a_buf     = nullptr;
    sigma_buf = nullptr;
    eos_P     = nullptr;
    eos_dPde  = nullptr;
    eos_s     = nullptr;
    eos_T     = nullptr;
    eos_params = {};
    reduce_eps_out  = nullptr;
    reduce_rhob_out = nullptr;
    evo_pack_out    = nullptr;
    evo_pack_floats = 0;
}

float* kokkos_grid_evo_pack(GPUGrid& gpu, size_t floats) {
    auto it = store_map().find(&gpu);
    if (it == store_map().end()) return nullptr;
    KStore& st = it->second;
    if (gpu.evo_pack_floats < floats || !gpu.evo_pack_out) {
        st.evo_pack = KView("music_kokkos_evo_pack", floats);
        gpu.evo_pack_out    = st.evo_pack.data();
        gpu.evo_pack_floats = floats;
    }
    return gpu.evo_pack_out;
}

// ── AoS → SoA (double → float), packed straight into the host-accessible View ─

void GPUGrid::copy_to_gpu(const SCGrid& src, GPUSnapshot& dst) const {
    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        const auto& cell = src(ix, iy, ieta);

        dst.epsilon[c] = static_cast<float>(cell.epsilon);
        dst.rhob[c]    = static_cast<float>(cell.rhob);
        dst.pi_b[c]    = static_cast<float>(cell.pi_b);

        for (int m = 0; m < GPU_U_COMPS; ++m)
            dst.u[m * Ncells_ + c] = static_cast<float>(cell.u[m]);
        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            dst.Wmunu[m * Ncells_ + c] = static_cast<float>(cell.Wmunu[m]);
    }
}

void GPUGrid::copy_wmunu_to_cpu(const GPUSnapshot& src, SCGrid& dst) const {
    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        auto& cell = dst(ix, iy, ieta);

        cell.pi_b = static_cast<double>(src.pi_b[c]);
        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            cell.Wmunu[m] = static_cast<double>(src.Wmunu[m * Ncells_ + c]);
    }
}

void GPUGrid::copy_primitives_to_cpu(const GPUSnapshot& src, SCGrid& dst) const {
    const int Nx = Nx_, Ny = Ny_, Neta = Neta_;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int ieta = 0; ieta < Neta; ++ieta)
    for (int ix   = 0; ix   < Nx;   ++ix  )
    for (int iy   = 0; iy   < Ny;   ++iy  ) {
        const int c = cell_idx(ix, iy, ieta, Nx, Ny);
        auto& cell = dst(ix, iy, ieta);

        cell.epsilon = static_cast<double>(src.epsilon[c]);
        cell.rhob    = static_cast<double>(src.rhob[c]);
        for (int m = 0; m < GPU_U_COMPS; ++m)
            cell.u[m] = static_cast<double>(src.u[m * Ncells_ + c]);
    }
}

void GPUGrid::rotate_snapshots() {
    GPUSnapshot temp = snap_prev;
    snap_prev   = snap_curr;
    snap_curr   = snap_future;
    snap_future = temp;
}

void GPUGrid::swap_curr_future() {
    GPUSnapshot tmp = snap_curr;
    snap_curr   = snap_future;
    snap_future = tmp;
}

void GPUGrid::refresh_u_curr_stage() {
    // No-op on the coherent/UVM path: snap_curr.u is itself host-accessible, so
    // host_readable_u_curr() returns it directly (no staging buffer to refresh).
}

// ── Fields (SoA double) overloads ───────────────────────────────────────────
// Mirror the SCGrid versions; the pointer-hoist keeps the inner-vector data()
// loads out of the hot loop so the cast pass runs near memory bandwidth.

void GPUGrid::copy_to_gpu(const Fields& src, GPUSnapshot& dst) const {
    const int Ncells = Ncells_;
    const double* __restrict__ src_e      = src.e_     .data();
    const double* __restrict__ src_rhob   = src.rhob_  .data();
    const double* __restrict__ src_piBulk = src.piBulk_.data();
    const double* src_u    [GPU_U_COMPS];
    const double* src_Wmunu[GPU_WMUNU_COMPS];
    for (int m = 0; m < GPU_U_COMPS; ++m)     src_u[m]     = src.u_    [m].data();
    for (int m = 0; m < GPU_WMUNU_COMPS; ++m) src_Wmunu[m] = src.Wmunu_[m].data();

    float* d_eps = dst.epsilon; float* d_rhob = dst.rhob; float* d_pib = dst.pi_b;
    float* d_u = dst.u; float* d_W = dst.Wmunu;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < Ncells; ++c) {
        d_eps [c] = static_cast<float>(src_e     [c]);
        d_rhob[c] = static_cast<float>(src_rhob  [c]);
        d_pib [c] = static_cast<float>(src_piBulk[c]);
        for (int m = 0; m < GPU_U_COMPS; ++m)
            d_u[m * Ncells + c] = static_cast<float>(src_u[m][c]);
        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            d_W[m * Ncells + c] = static_cast<float>(src_Wmunu[m][c]);
    }
}

void GPUGrid::copy_wmunu_to_cpu(const GPUSnapshot& src, Fields& dst) const {
    const int Ncells = Ncells_;
    double* __restrict__ dst_piBulk = dst.piBulk_.data();
    double* dst_Wmunu[GPU_WMUNU_COMPS];
    for (int m = 0; m < GPU_WMUNU_COMPS; ++m) dst_Wmunu[m] = dst.Wmunu_[m].data();
    const float* s_W = src.Wmunu; const float* s_pib = src.pi_b;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < Ncells; ++c) {
        dst_piBulk[c] = static_cast<double>(s_pib[c]);
        for (int m = 0; m < GPU_WMUNU_COMPS; ++m)
            dst_Wmunu[m][c] = static_cast<double>(s_W[m * Ncells + c]);
    }
}

void GPUGrid::copy_primitives_to_cpu(const GPUSnapshot& src, Fields& dst) const {
    const int Ncells = Ncells_;
    double* __restrict__ dst_e    = dst.e_   .data();
    double* __restrict__ dst_rhob = dst.rhob_.data();
    double* dst_u[GPU_U_COMPS];
    for (int m = 0; m < GPU_U_COMPS; ++m) dst_u[m] = dst.u_[m].data();
    const float* s_eps = src.epsilon; const float* s_rhob = src.rhob; const float* s_u = src.u;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < Ncells; ++c) {
        dst_e   [c] = static_cast<double>(s_eps [c]);
        dst_rhob[c] = static_cast<double>(s_rhob[c]);
        for (int m = 0; m < GPU_U_COMPS; ++m)
            dst_u[m][c] = static_cast<double>(s_u[m * Ncells + c]);
        // rhoq_ / rhos_ intentionally left alone — see PORT_GPU.md §4.1.
    }
}
