// C++ interface to Metal compute pipelines.
// Implementation lives in MetalPipelines.mm (Objective-C++).
//
// Usage pattern:
//   MetalPipelines& mp = MetalPipelines::instance();
//   if (!mp.initialize()) { /* fall back to CPU */ }
//   mp.dispatch_w_source(gpu_grid, params);
//   mp.wait();   // block until GPU finishes

#pragma once
#include "GPUGrid.h"
#include "gpu_types.h"

class MetalPipelines {
public:
    static MetalPipelines& instance();

    // Load the precompiled Metal library and build all pipeline states.
    // Returns false if Metal is unavailable or shader loading fails.
    bool initialize(const char* metallib_path = nullptr);

    bool ready() const { return ready_; }

    // Dispatch gpu_make_w_source over the full grid.
    // Reads from gpu.snap_curr and gpu.snap_prev.
    // Writes dwmn[5 * Ncells] into gpu.dwmn.
    // Non-blocking: call wait() to ensure completion.
    void dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_make_delta_qi over the full grid.
    // Reads from gpu.snap_curr (epsilon, rhob, u) and gpu.eos_P / gpu.eos_dPde.
    // Writes qi_out[5 * Ncells] into gpu.qi_out.
    // gpu.upload_eos() must have been called before the first dispatch.
    // Non-blocking: call wait() to ensure completion.
    void dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_first_rk_step_w_full over the full grid.
    // Consumes the outputs of the four earlier viscous kernels (snap_prev,
    // snap_curr, snap_future.u, uwrhs_out, theta_buf, a_buf, sigma_buf) and
    // writes snap_future.Wmunu + snap_future.pi_b directly.  Used only when
    // the simple-config support matrix is satisfied (see kernel docs);
    // otherwise the host runs the CPU FirstRKStepW loop instead.
    void dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                       const MUSICGridParams& params);

    // Dispatch gpu_make_du over the full grid.
    // Reads gpu.snap_curr.u and gpu.snap_prev.u; writes theta_buf,
    // a_buf, sigma_buf (per-cell viscous geometry).  v1 assumes
    // include_vorticity_terms == 0 and zero net baryon (no dUoverTsup /
    // dUTsup / ∂(µ_B/T) terms).  Non-blocking; sync with wait().
    void dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_make_uwrhs over the full grid.
    // Reads gpu.snap_curr.Wmunu and gpu.snap_curr.u; writes
    // uwrhs_out[5 * Ncells] into gpu.uwrhs_out (one entry per shear index
    // 4..8 per cell).  Non-blocking; sync with wait().
    void dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_make_uprhs over the full grid.
    // Reads gpu.snap_curr.pi_b and gpu.snap_curr.u; writes
    // uprhs_out[Ncells] into gpu.uprhs_out — the scalar KT flux divergence
    // of (u^a * pi_b).  Only call when turn_on_bulk == 1.
    // Non-blocking; sync with wait().
    void dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params);

    // Dispatch gpu_finalize_ideal over the full grid.
    // Reads qi_out + dwmn (already produced by the two kernels above), the
    // current and previous snapshots, and the EOS tables; writes the
    // post-Newton primitives (epsilon, rhob, u) into gpu.snap_future.
    // Must be called after dispatch_w_source AND dispatch_delta_qi for the
    // same RK step; non-blocking, sync with wait().
    void dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params);

    // GPU max-reduction over snap_curr.  On Apple Silicon (coherent memory)
    // the managed buffers are directly readable on the CPU, so this scans
    // epsilon/rhob on the host after wait() has ensured the last kernel is done.
    void reduce_max(GPUGrid& gpu, double& eps_max, double& rhob_max);

    // max(epsilon) over the transverse edge bands of snap_curr (as
    // CUDAPipelines::reduce_max_edge), scanned on the host like reduce_max.
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

    // The freeze-out search (evolve.cpp), as CUDAPipelines::keep_curr and
    // gather_cells; on the host, like reduce_max (shared memory).
    static size_t curr_state_floats(const GPUGrid& gpu);
    bool  keep_curr(GPUGrid& gpu, void* state);
    bool  gather_cells(GPUGrid& gpu, const void* kept, const int* cells,
                       int n, float* host_curr, float* host_kept);

    // Cell updates in which gpu_first_rk_step_w_full set a non-finite
    // W^{mu nu} or Pi to 0 since the last call (resets the count).  Waits for
    // the last committed command buffer; counts from an open batch arrive at
    // the next call.  Same contract as CUDAPipelines::read_and_reset_nonfinite.
    unsigned int read_and_reset_nonfinite();

    // Phase 2b: launch gpu_pack_evolution_ideal over the down-sampled grid into
    // a pipeline-owned shared scratch buffer, synchronize, and copy the packed
    // records into host_out (8 floats per cell, fluidCell_ideal layout).
    // Mirrors CUDAPipelines::pack_evolution_ideal, but the scratch buffer lives
    // here on the singleton (NOT on GPUGrid) so the GPUGrid object embedded in
    // Advance/Evolve keeps its byte-identical layout — see docs/OOB_Bug.md.  On
    // Apple unified memory the buffer is host-readable, so the copy is a plain
    // memcpy with no D2H transfer.  Returns false (no-op) if the pipeline isn't
    // ready, so the caller falls back to the host output path.
    bool pack_evolution_ideal(GPUGrid& gpu, const GPUPackParams& pp,
                              float* host_out);

    // Block until all pending GPU work is done.
    void wait();

    // C3, the active region (opt-in; MUSIC's active_region_eps > 0), as
    // CUDAPipelines.  With gpu.active_on the seven step kernels skip the
    // 8 x 8 x 4 threadgroups whose flag in gpu.active_blk is 0: the cells there
    // are frozen (the same values in snap_prev, snap_curr and snap_future).
    // The flags live in shared MTLBuffers owned by the GPUGrid
    // (gpu.active_mtl); the fills, the rebuild and the copy of its flags for
    // active_region_count are encoded on the queue (in the open batch, if
    // any), so the calls do not wait.
    //   active_region_reset: allocate the flags if needed, every block active.
    //       false if the metallib lacks the C3 kernels (Advance then warns
    //       once and evolves the whole grid).
    //   active_region_rebuild: flag the blocks within `halo` cells of a cell
    //       of snap_curr with e > eps_act [1/fm^4], freeze the cells of the
    //       blocks that were active and are not any more.
    //   active_region_count: the number of active blocks of the last rebuild
    //       (-1 if none is pending); waits for the queue.
    //   active_region_add_sources: also flag the blocks within `halo` cells
    //       of a cell with a non-zero qi_source_buf (after the source fill).
    bool active_region_reset(GPUGrid& gpu);
    void active_region_rebuild(GPUGrid& gpu, float eps_act, int halo);
    int  active_region_count(GPUGrid& gpu);
    void active_region_add_sources(GPUGrid& gpu, int halo);

    // Batch mode: while a batch is open, all dispatch_* calls share a single
    // command buffer (one commit, one wait).  Saves N-1 command-buffer
    // allocations + commits per substep.  Outside a batch the dispatch_*
    // methods retain their legacy one-buffer-per-dispatch behavior.
    void begin_batch();
    void end_batch();

private:
    MetalPipelines() = default;
    ~MetalPipelines();

    bool   ready_ = false;

    // Opaque pointers to Metal objects (avoid ObjC in header).
    void*  device_        = nullptr;  // id<MTLDevice>
    void*  cmd_queue_     = nullptr;  // id<MTLCommandQueue>
    void*  library_       = nullptr;  // id<MTLLibrary>
    void*  pso_w_source_  = nullptr;  // id<MTLComputePipelineState>
    void*  pso_delta_qi_  = nullptr;  // id<MTLComputePipelineState>
    void*  pso_finalize_  = nullptr;  // id<MTLComputePipelineState>
    void*  pso_uwrhs_     = nullptr;  // id<MTLComputePipelineState>
    void*  pso_make_du_   = nullptr;  // id<MTLComputePipelineState>
    void*  pso_w_full_    = nullptr;  // id<MTLComputePipelineState>
    void*  pso_uprhs_     = nullptr;  // id<MTLComputePipelineState>
    void*  pso_pack_evo_  = nullptr;  // id<MTLComputePipelineState> (Phase 2b)
    void*  pso_active_mark_   = nullptr;  // id<MTLComputePipelineState> (C3)
    void*  pso_active_freeze_ = nullptr;  // id<MTLComputePipelineState> (C3)
    void*  cmd_buf_       = nullptr;  // last id<MTLCommandBuffer>
    void*  batch_cb_      = nullptr;  // open batch id<MTLCommandBuffer>, or null

    // Phase 2b: lazily-(re)allocated shared scratch buffer for the GPU-packed
    // ideal evolution output (8 floats per down-sampled cell).  Owned by the
    // singleton; freed in the destructor.  evo_pack_floats_ tracks its capacity
    // in floats so it is only reallocated when a larger frame is needed.
    void*  evo_pack_buf_    = nullptr;  // id<MTLBuffer>
    // one uint32 (atomic_uint in the kernel): see read_and_reset_nonfinite
    void*  nonfinite_buf_   = nullptr;  // id<MTLBuffer>
    size_t evo_pack_floats_ = 0;

    // The command buffer for the next encoder: the open batch, or a new one
    // (own = true) that close_cb commits and keeps in cmd_buf_ for wait().
    // void* = id<MTLCommandBuffer>.
    void* open_cb(bool& own);
    void  close_cb(void* cb, bool own);
    // C3: bind gpu.active_blk and the active_on flag at buffer(index) and
    // buffer(index + 1) of a step kernel; void* = id<MTLComputeCommandEncoder>.
    void  bind_active(void* enc, const GPUGrid& gpu, int index);
    // C3: run gpu_active_mark on snap_curr (use_eps) or qi_source_buf
    // (use_src) into `flags`, in command buffer `cb`.
    void  encode_active_mark(void* cb, GPUGrid& gpu, unsigned char* flags,
                             float eps_act, int halo, int use_eps, int use_src);
};
