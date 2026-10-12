// C++ interface to CUDA compute pipelines (NVIDIA GPUs).
// Implementation lives in CUDAPipelines.cu (compiled by nvcc).
//
// This mirrors MetalPipelines exactly so the host dispatch logic in
// advance.cpp can be shared between the Metal and CUDA back-ends (see the
// `GPUPipelines` alias in advance.h).
//
// Usage pattern:
//   CUDAPipelines& cp = CUDAPipelines::instance();
//   if (!cp.initialize()) { /* fall back to CPU */ }
//   cp.begin_batch();
//   cp.dispatch_w_source(gpu_grid, params);
//   ...
//   cp.end_batch();
//   cp.wait();   // block until the GPU finishes

#pragma once
#include "GPUGrid.h"
#include "gpu_types.h"

class CUDAPipelines {
public:
    static CUDAPipelines& instance();

    // Select device 0 and create the compute stream.  Kernels are compiled
    // directly into the binary by nvcc, so there is no library to load.
    // The argument is accepted (and ignored) to match the MetalPipelines
    // signature used at the shared call site.
    bool initialize(const char* unused = nullptr);

    bool ready() const { return ready_; }

    // All seven dispatches mirror MetalPipelines::dispatch_*.  Each computes a
    // dim3 launch grid and enqueues its kernel on the compute stream.  They are
    // non-blocking; call wait() to synchronize.
    void dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                       const MUSICGridParams& params);
    void dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params);
    void dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params);

    // Block until all pending GPU work on the compute stream is done.
    void wait();

    // Batch mode is a no-op for CUDA: a single stream already serializes the
    // kernel launches, so begin/end_batch only exist to match the Metal API.
    void begin_batch();
    void end_batch();

    // GPU max-reduction over snap_curr: find max(epsilon) and max(rhob) across
    // all Ncells.  Results are written to gpu.reduce_eps_out[0] and
    // gpu.reduce_rhob_out[0] (managed memory, readable on the host after the
    // call).  Launches on the compute stream and synchronizes before returning.
    void reduce_max(GPUGrid& gpu, double& eps_max, double& rhob_max);

    // max(epsilon) over the transverse edge bands of snap_curr (see
    // gpu_reduce_max_eps_edge) [1/fm^4]; 0 if every edge cell has e <= 0.
    // Synchronizes before returning.
    double reduce_max_edge(GPUGrid& gpu, int fac_x, int fac_y);

    // A copy of snap_prev and snap_curr (epsilon, rhob, u, Wmunu, pi_b each,
    // prev first), for a run to resume from another run's state
    // (MusicCheckpoint).  state_floats() floats; alloc_state allocates them,
    // save_snapshots / restore_snapshots copy GPU -> state / state -> GPU and
    // synchronize, read_state copies a state to host memory.
    static size_t state_floats(const GPUGrid& gpu);
    void* alloc_state(size_t floats);
    void  free_state(void* state);
    bool  save_snapshots(GPUGrid& gpu, void* state);
    bool  restore_snapshots(GPUGrid& gpu, const void* state);
    bool  read_state(const void* state, float* host, size_t floats);

    // The freeze-out search (evolve.cpp): keep_curr copies snap_curr into
    // `state` (curr_state_floats() floats, alloc_state) on the compute
    // stream.  gather_cells copies all fields (epsilon, rhob, u[4],
    // Wmunu[14], pi_b) of the n cells cells[0..n) of snap_curr, and of `kept`
    // unless it is null, to host memory, field-major: host_curr[f*n + j],
    // host_kept[f*n + j].  Synchronizes.
    static size_t curr_state_floats(const GPUGrid& gpu);
    bool  keep_curr(GPUGrid& gpu, void* state);
    bool  gather_cells(GPUGrid& gpu, const void* kept, const int* cells,
                       int n, float* host_curr, float* host_kept);

    // Cell updates in which gpu_first_rk_step_w_full set a non-finite
    // W^{mu nu} or Pi to 0 since the last call (resets the count).
    // Synchronizes the compute stream.
    unsigned int read_and_reset_nonfinite();

    // Phase 2b: launch gpu_pack_evolution_ideal over the down-sampled grid into
    // a pipeline-owned managed scratch buffer, synchronize, and copy the packed
    // records into host_out (8 floats per cell, fluidCell_ideal layout).  The
    // scratch buffer lives here, NOT on GPUGrid, so the GPUGrid object embedded
    // in Advance/Evolve keeps its original size and layout.  Returns false
    // (no-op) if the pipeline isn't ready, so the caller falls back to the host
    // output path.
    bool pack_evolution_ideal(GPUGrid& gpu, const GPUPackParams& pp,
                              float* host_out);

    // Phase 4 (dual-stream): prefetch the freshly-packed snap_curr / snap_prev
    // SoA buffers to the device on a dedicated copy stream, then gate the
    // compute stream on completion via an event.  On a discrete GPU this moves
    // the host→device transfer onto its own stream so it can overlap prior
    // compute; on a coherent unified-memory part (GB10) it is a cheap hint.
    // Safe no-op for the Metal back-end (this method only exists here and is
    // called under a USE_CUDA guard in advance.cpp).
    void upload_snapshots_async(GPUGrid& gpu);

private:
    CUDAPipelines() = default;
    ~CUDAPipelines();

    bool  ready_     = false;
    int   device_id_ = 0;
    // cudaStream_t kept as void* so this header stays free of CUDA headers
    // and can be included by the host C++ compiler (advance.cpp).
    void* compute_stream_ = nullptr;
    // Phase 4 dual-stream: a dedicated copy stream + completion event used to
    // move/prefetch SoA snapshot data independently of the compute stream.
    void* copy_stream_ = nullptr;
    void* copy_event_  = nullptr;
    // True when the GPU can access host/managed memory coherently (integrated
    // or NVLink-C2C parts such as GB10).  On such platforms there is no
    // discrete transfer to overlap, so the dual-stream prefetch+gate is skipped
    // (it would only add migration latency).  Kept active on discrete GPUs.
    bool  coherent_memory_ = false;
    // Phase 2b: the packed-evolution scratch buffer now lives on GPUGrid
    // (gpu.evo_pack_out / gpu.evo_pack_floats), tying its lifetime to the grid
    // and freeing it in GPUGrid::release().
    // Occupancy-tuned upper bound on threads per block (Phase 2).  The 3-D
    // block is factored from this at dispatch time, adapting to Neta so 2-D
    // (Neta==1) grids don't waste the eta thread dimension.
    int   max_block_threads_ = 256;
    // gather_cells: device buffers for the cell list and the gathered
    // fields, and the pinned host buffers they are copied through; grown as
    // needed (gather_cap_ cells)
    int*   gather_cells_dev_  = nullptr;
    float* gather_out_dev_    = nullptr;
    int*   gather_cells_host_ = nullptr;
    float* gather_out_host_   = nullptr;
    size_t gather_cap_        = 0;
};
