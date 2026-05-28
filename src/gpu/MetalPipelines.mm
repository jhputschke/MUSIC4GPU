// Objective-C++ implementation of MetalPipelines.
// Loads the Metal shader library, creates compute pipeline states, and
// dispatches the gpu_make_w_source kernel.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include "MetalPipelines.h"
#include "GPUGrid.h"
#include "gpu_types.h"

// Shared Metal device – declared here, extern'd in GPUGrid.mm
id<MTLDevice> g_metal_device = nil;

// Directory containing libmusic.dylib (this TU is compiled into it), resolved at
// runtime via dladdr.  Lets the metallib be located relative to the shared
// library rather than the executable, so it works for any binary that links
// libmusic (MUSICTest, runJetscape, standalone MUSIChydro, ...).
static std::string music_library_dir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<const void*>(&music_library_dir), &info) &&
        info.dli_fname) {
        std::string p(info.dli_fname);
        const size_t slash = p.find_last_of('/');
        if (slash != std::string::npos) return p.substr(0, slash);
    }
    return std::string();
}

// ── singleton ─────────────────────────────────────────────────────────────────

MetalPipelines& MetalPipelines::instance() {
    static MetalPipelines inst;
    return inst;
}

MetalPipelines::~MetalPipelines() {
    if (pso_uprhs_)    { CFRelease(pso_uprhs_); }
    if (pso_w_full_)   { CFRelease(pso_w_full_); }
    if (pso_make_du_)  { CFRelease(pso_make_du_); }
    if (pso_uwrhs_)    { CFRelease(pso_uwrhs_); }
    if (pso_finalize_) { CFRelease(pso_finalize_); }
    if (pso_delta_qi_) { CFRelease(pso_delta_qi_); }
    if (pso_w_source_) { CFRelease(pso_w_source_); }
    if (library_)      { CFRelease(library_); }
    if (cmd_queue_)    { CFRelease(cmd_queue_); }
    // device is a system singleton – do not release
}

// ── initialization ────────────────────────────────────────────────────────────

bool MetalPipelines::initialize(const char* metallib_path) {
    if (ready_) return true;

    // 1. Acquire Metal device
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        fprintf(stderr, "[MUSIC-GPU] No Metal device found.\n");
        return false;
    }
    g_metal_device = dev;           // share with GPUGrid.mm
    device_ = (__bridge_retained void*)dev;

    fprintf(stderr, "[MUSIC-GPU] Metal device: %s\n",
            [[dev name] UTF8String]);

    // 2. Command queue
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (!q) { fprintf(stderr, "[MUSIC-GPU] Failed to create command queue.\n"); return false; }
    cmd_queue_ = (__bridge_retained void*)q;

    // 3. Load shader library.  Resolution order:
    //   (1) explicit path argument, (2) $MUSIC_METALLIB, (3) next to
    //   libmusic.dylib (via dladdr), (4) the main executable's bundle directory.
    id<MTLLibrary> lib = nil;
    NSError* err = nil;

    auto try_path = [&](NSString* path) {
        if (lib || path == nil || [path length] == 0) return;
        NSError* e = nil;
        id<MTLLibrary> l =
            [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
        if (l) {
            lib = l;
            fprintf(stderr, "[MUSIC-GPU] Loaded kernels from %s\n",
                    [path UTF8String]);
        } else {
            err = e;
        }
    };

    if (metallib_path && metallib_path[0] != '\0')
        try_path([NSString stringWithUTF8String:metallib_path]);

    if (const char* env = getenv("MUSIC_METALLIB"))
        try_path([NSString stringWithUTF8String:env]);

    {
        const std::string dir = music_library_dir();
        if (!dir.empty())
            try_path([NSString stringWithUTF8String:
                          (dir + "/music_kernels.metallib").c_str()]);
    }

    try_path([[NSBundle mainBundle] pathForResource:@"music_kernels"
                                             ofType:@"metallib"]);

    if (!lib) {
        fprintf(stderr, "[MUSIC-GPU] Could not load music_kernels.metallib: %s\n",
                err ? [[err localizedDescription] UTF8String] : "not found");
        fprintf(stderr, "[MUSIC-GPU] Place it next to libmusic.dylib, or set "
                        "MUSIC_METALLIB=/path/to/music_kernels.metallib\n");
        return false;
    }
    library_ = (__bridge_retained void*)lib;

    // 4. Build pipeline state for gpu_make_w_source
    id<MTLFunction> fn = [lib newFunctionWithName:@"gpu_make_w_source"];
    if (!fn) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_make_w_source' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[MUSIC-GPU] PSO creation failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_w_source_ = (__bridge_retained void*)pso;

    fprintf(stderr, "[MUSIC-GPU] Initialized. max threads/group = %lu\n",
            (unsigned long)[pso maxTotalThreadsPerThreadgroup]);

    // 5. Build pipeline state for gpu_make_delta_qi
    id<MTLFunction> fn_dqi = [lib newFunctionWithName:@"gpu_make_delta_qi"];
    if (!fn_dqi) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_make_delta_qi' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso_dqi =
        [dev newComputePipelineStateWithFunction:fn_dqi error:&err];
    if (!pso_dqi) {
        fprintf(stderr, "[MUSIC-GPU] PSO for gpu_make_delta_qi failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_delta_qi_ = (__bridge_retained void*)pso_dqi;

    // 6. Build pipeline state for gpu_finalize_ideal
    id<MTLFunction> fn_fin = [lib newFunctionWithName:@"gpu_finalize_ideal"];
    if (!fn_fin) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_finalize_ideal' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso_fin =
        [dev newComputePipelineStateWithFunction:fn_fin error:&err];
    if (!pso_fin) {
        fprintf(stderr, "[MUSIC-GPU] PSO for gpu_finalize_ideal failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_finalize_ = (__bridge_retained void*)pso_fin;

    // 7. Build pipeline state for gpu_make_uwrhs
    id<MTLFunction> fn_uw = [lib newFunctionWithName:@"gpu_make_uwrhs"];
    if (!fn_uw) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_make_uwrhs' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso_uw =
        [dev newComputePipelineStateWithFunction:fn_uw error:&err];
    if (!pso_uw) {
        fprintf(stderr, "[MUSIC-GPU] PSO for gpu_make_uwrhs failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_uwrhs_ = (__bridge_retained void*)pso_uw;

    // 8. Build pipeline state for gpu_make_du
    id<MTLFunction> fn_du = [lib newFunctionWithName:@"gpu_make_du"];
    if (!fn_du) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_make_du' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso_du =
        [dev newComputePipelineStateWithFunction:fn_du error:&err];
    if (!pso_du) {
        fprintf(stderr, "[MUSIC-GPU] PSO for gpu_make_du failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_make_du_ = (__bridge_retained void*)pso_du;

    // 9. Build pipeline state for gpu_first_rk_step_w_full
    id<MTLFunction> fn_wf = [lib newFunctionWithName:@"gpu_first_rk_step_w_full"];
    if (!fn_wf) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_first_rk_step_w_full' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso_wf =
        [dev newComputePipelineStateWithFunction:fn_wf error:&err];
    if (!pso_wf) {
        fprintf(stderr, "[MUSIC-GPU] PSO for gpu_first_rk_step_w_full failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_w_full_ = (__bridge_retained void*)pso_wf;

    // 10. Build pipeline state for gpu_make_uprhs (bulk-pressure stencil)
    id<MTLFunction> fn_up = [lib newFunctionWithName:@"gpu_make_uprhs"];
    if (!fn_up) {
        fprintf(stderr, "[MUSIC-GPU] Kernel 'gpu_make_uprhs' not found in library.\n");
        return false;
    }
    id<MTLComputePipelineState> pso_up =
        [dev newComputePipelineStateWithFunction:fn_up error:&err];
    if (!pso_up) {
        fprintf(stderr, "[MUSIC-GPU] PSO for gpu_make_uprhs failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return false;
    }
    pso_uprhs_ = (__bridge_retained void*)pso_up;

    ready_ = true;
    return true;
}

// ── reduce_max ───────────────────────────────────────────────────────────────
// Apple Silicon has coherent unified memory: snap_curr.epsilon / .rhob are
// MTLStorageModeShared buffers, so the CPU can scan them directly after
// wait() ensures the last kernel has finished writing.

void MetalPipelines::reduce_max(GPUGrid& gpu, double& eps_max, double& rhob_max) {
    eps_max  = 0.0;
    rhob_max = 0.0;
    if (!ready_ || !gpu.snap_curr.epsilon || !gpu.snap_curr.rhob) return;
    const int N         = gpu.Ncells();
    const float* e      = gpu.snap_curr.epsilon;
    const float* r      = gpu.snap_curr.rhob;
    for (int i = 0; i < N; ++i) {
        if (e[i] > static_cast<float>(eps_max))  eps_max  = static_cast<double>(e[i]);
        if (r[i] > static_cast<float>(rhob_max)) rhob_max = static_cast<double>(r[i]);
    }
}

// ── wait ──────────────────────────────────────────────────────────────────────

void MetalPipelines::wait() {
    if (!cmd_buf_) return;
    auto cb = (__bridge id<MTLCommandBuffer>)cmd_buf_;
    [cb waitUntilCompleted];
    CFRelease(cmd_buf_);
    cmd_buf_ = nullptr;
}

// ── batch_cb_ open/close ─────────────────────────────────────────────────────

void MetalPipelines::begin_batch() {
    if (!ready_ || batch_cb_) return;   // already open or not initialised
    auto q  = (__bridge id<MTLCommandQueue>)cmd_queue_;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    batch_cb_ = (__bridge_retained void*)cb;
}

void MetalPipelines::end_batch() {
    if (!batch_cb_) return;
    auto cb = (__bridge id<MTLCommandBuffer>)batch_cb_;
    [cb commit];
    // Transfer ownership of the retained CB to cmd_buf_ so wait() can join it.
    if (cmd_buf_) CFRelease(cmd_buf_);
    cmd_buf_  = batch_cb_;
    batch_cb_ = nullptr;
}

// Reverse-lookup: given a float* that lives inside one of GPUGrid's Metal
// shared buffers, return the corresponding MTLBuffer object.
// On Apple Silicon the buffer's [contents] pointer IS the CPU-visible address,
// so a range-check scan over all handles finds the right one.
// Called once per kernel dispatch (not per cell), so O(16) cost is negligible.

static id<MTLBuffer> buffer_for_ptr(void** handles, int n_handles, const void* ptr) {
    for (int i = 0; i < n_handles; ++i) {
        if (!handles[i]) continue;
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)handles[i];
        const uint8_t* base = (const uint8_t*)[b contents];
        const uint8_t* end  = base + [b length];
        const uint8_t* p    = (const uint8_t*)ptr;
        if (p >= base && p < end) return b;
    }
    return nil;
}

// Expose GPUGrid private arrays via a small accessor struct declared as friend.
// Since we can't friend a free function easily across TUs, we instead use a
// public accessor that the .mm file adds: GPUGrid exposes buf_handles_ and
// n_handles_ as public for the GPU layer only (Metal-internal detail).
// We expose them via a pair of accessors added to GPUGrid:

// ── dispatch_w_source ─────────────────────────────────────────────────────────

void MetalPipelines::dispatch_w_source(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;

    auto dev   = (__bridge id<MTLDevice>)           device_;
    auto q     = (__bridge id<MTLCommandQueue>)     cmd_queue_;
    auto pso   = (__bridge id<MTLComputePipelineState>)pso_w_source_;

    // Retrieve MTLBuffer handles from GPUGrid's public handle table.
    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();

    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    // Bind buffers – must match [[buffer(N)]] indices in the .metal file
    // Binding order:
    //   0  Wmunu_curr   [GPU_WMUNU_COMPS * Ncells]
    //   1  pi_b_curr    [Ncells]
    //   2  u_curr       [4 * Ncells]
    //   3  Wmunu_prev   [GPU_WMUNU_COMPS * Ncells]
    //   4  pi_b_prev    [Ncells]
    //   5  u_prev       [4 * Ncells]
    //   6  dwmn_out     [5 * Ncells]
    //   7  params       (constant struct, passed inline)

    [enc setBuffer:get_buf(gpu.snap_curr.Wmunu)  offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.snap_curr.pi_b)   offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.snap_curr.u)      offset:0 atIndex:2];
    [enc setBuffer:get_buf(gpu.snap_prev.Wmunu)  offset:0 atIndex:3];
    [enc setBuffer:get_buf(gpu.snap_prev.pi_b)   offset:0 atIndex:4];
    [enc setBuffer:get_buf(gpu.snap_prev.u)      offset:0 atIndex:5];
    [enc setBuffer:get_buf(gpu.dwmn)             offset:0 atIndex:6];

    [enc setBytes:&params length:sizeof(params) atIndex:7];

    // 3-D thread grid: one thread per cell (ix, iy, ieta)
    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}

// ── dispatch_delta_qi ─────────────────────────────────────────────────────────
//
// Dispatches gpu_make_delta_qi: the ideal KT flux kernel (port of MakeDeltaQI).
// Buffer layout must match [[buffer(N)]] indices in music_kernels.metal.

void MetalPipelines::dispatch_delta_qi(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_) return;

    auto dev = (__bridge id<MTLDevice>)              device_;
    auto q   = (__bridge id<MTLCommandQueue>)        cmd_queue_;
    auto pso = (__bridge id<MTLComputePipelineState>)pso_delta_qi_;

    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();
    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    // Binding order (must match gpu_make_delta_qi in .metal):
    //   0  epsilon_curr [Ncells]
    //   1  rhob_curr    [Ncells]
    //   2  u_curr       [4*Ncells]
    //   3  eos_P        [GPU_EOS_N]
    //   4  eos_dPde     [GPU_EOS_N]
    //   5  qi_out       [5*Ncells]
    //   6  params       (constant struct, passed inline)
    //   7  eos_p        (constant struct, passed inline)
    [enc setBuffer:get_buf(gpu.snap_curr.epsilon) offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.snap_curr.rhob)    offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.snap_curr.u)       offset:0 atIndex:2];
    [enc setBuffer:get_buf(gpu.eos_P)             offset:0 atIndex:3];
    [enc setBuffer:get_buf(gpu.eos_dPde)          offset:0 atIndex:4];
    [enc setBuffer:get_buf(gpu.qi_out)            offset:0 atIndex:5];
    [enc setBytes:&params          length:sizeof(params)          atIndex:6];
    [enc setBytes:&gpu.eos_params  length:sizeof(gpu.eos_params)  atIndex:7];

    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}

// ── dispatch_finalize_ideal ──────────────────────────────────────────────────
//
// Dispatches gpu_finalize_ideal: the final per-cell ideal-step RK update.
// Reads qi_out + dwmn (filled by the two earlier kernels) and snap_prev +
// snap_curr; writes primitive variables into snap_future.

void MetalPipelines::dispatch_finalize_ideal(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_ || !pso_finalize_) return;

    auto q   = (__bridge id<MTLCommandQueue>)        cmd_queue_;
    auto pso = (__bridge id<MTLComputePipelineState>)pso_finalize_;

    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();
    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    // Binding order — must match gpu_finalize_ideal in music_kernels.metal.
    [enc setBuffer:get_buf(gpu.qi_out)              offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.dwmn)                offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.snap_curr.epsilon)   offset:0 atIndex:2];
    [enc setBuffer:get_buf(gpu.snap_curr.u)         offset:0 atIndex:3];
    [enc setBuffer:get_buf(gpu.snap_prev.epsilon)   offset:0 atIndex:4];
    [enc setBuffer:get_buf(gpu.snap_prev.rhob)      offset:0 atIndex:5];
    [enc setBuffer:get_buf(gpu.snap_prev.u)         offset:0 atIndex:6];
    [enc setBuffer:get_buf(gpu.snap_future.epsilon) offset:0 atIndex:7];
    [enc setBuffer:get_buf(gpu.snap_future.rhob)    offset:0 atIndex:8];
    [enc setBuffer:get_buf(gpu.snap_future.u)       offset:0 atIndex:9];
    [enc setBuffer:get_buf(gpu.eos_P)               offset:0 atIndex:10];
    [enc setBuffer:get_buf(gpu.eos_dPde)            offset:0 atIndex:11];
    [enc setBytes:&params          length:sizeof(params)          atIndex:12];
    [enc setBytes:&gpu.eos_params  length:sizeof(gpu.eos_params)  atIndex:13];
    [enc setBuffer:get_buf(gpu.qi_source_buf)       offset:0 atIndex:14];

    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}

// ── dispatch_uwrhs ───────────────────────────────────────────────────────────
//
// Dispatches gpu_make_uwrhs: the KT flux divergence of (u^a W^{mu nu}) for
// the 5 shear-stress indices, used by the CPU viscous loop (FirstRKStepW).

void MetalPipelines::dispatch_uwrhs(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_ || !pso_uwrhs_) return;

    auto q   = (__bridge id<MTLCommandQueue>)        cmd_queue_;
    auto pso = (__bridge id<MTLComputePipelineState>)pso_uwrhs_;

    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();
    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    // Bindings must match gpu_make_uwrhs in music_kernels.metal.
    [enc setBuffer:get_buf(gpu.snap_curr.Wmunu) offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.snap_curr.u)     offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.uwrhs_out)       offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}

// ── dispatch_make_du ─────────────────────────────────────────────────────────
//
// Dispatches gpu_make_du: per-cell viscous geometry (theta, a^mu, sigma^munu).
// Outputs feed the CPU viscous loop in FirstRKStepW.

void MetalPipelines::dispatch_make_du(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_ || !pso_make_du_) return;

    auto q   = (__bridge id<MTLCommandQueue>)        cmd_queue_;
    auto pso = (__bridge id<MTLComputePipelineState>)pso_make_du_;

    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();
    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    // Bindings must match gpu_make_du in music_kernels.metal.
    [enc setBuffer:get_buf(gpu.snap_curr.u) offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.snap_prev.u) offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.theta_buf)   offset:0 atIndex:2];
    [enc setBuffer:get_buf(gpu.a_buf)       offset:0 atIndex:3];
    [enc setBuffer:get_buf(gpu.sigma_buf)   offset:0 atIndex:4];
    [enc setBytes:&params length:sizeof(params) atIndex:5];

    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}

// ── dispatch_first_rk_step_w_full ────────────────────────────────────────────
//
// Dispatches gpu_first_rk_step_w_full: full per-cell viscous RK update for
// the constant-shear / no-bulk / no-vorticity / no-diffusion configuration.
// Reads four earlier kernel outputs and writes snap_future.Wmunu + pi_b.

void MetalPipelines::dispatch_first_rk_step_w_full(GPUGrid& gpu,
                                                   const MUSICGridParams& params) {
    if (!ready_ || !pso_w_full_) return;

    auto q   = (__bridge id<MTLCommandQueue>)        cmd_queue_;
    auto pso = (__bridge id<MTLComputePipelineState>)pso_w_full_;

    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();
    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    // Binding order — must match gpu_first_rk_step_w_full in music_kernels.metal.
    [enc setBuffer:get_buf(gpu.snap_curr.Wmunu)     offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.snap_curr.pi_b)      offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.snap_curr.u)         offset:0 atIndex:2];
    [enc setBuffer:get_buf(gpu.snap_prev.Wmunu)     offset:0 atIndex:3];
    [enc setBuffer:get_buf(gpu.snap_prev.pi_b)      offset:0 atIndex:4];
    [enc setBuffer:get_buf(gpu.snap_prev.u)         offset:0 atIndex:5];
    [enc setBuffer:get_buf(gpu.snap_curr.epsilon)   offset:0 atIndex:6];
    [enc setBuffer:get_buf(gpu.snap_prev.epsilon)   offset:0 atIndex:7];
    [enc setBuffer:get_buf(gpu.snap_future.u)       offset:0 atIndex:8];
    [enc setBuffer:get_buf(gpu.uwrhs_out)           offset:0 atIndex:9];
    [enc setBuffer:get_buf(gpu.theta_buf)           offset:0 atIndex:10];
    [enc setBuffer:get_buf(gpu.a_buf)               offset:0 atIndex:11];
    [enc setBuffer:get_buf(gpu.sigma_buf)           offset:0 atIndex:12];
    [enc setBuffer:get_buf(gpu.snap_future.Wmunu)   offset:0 atIndex:13];
    [enc setBuffer:get_buf(gpu.snap_future.pi_b)    offset:0 atIndex:14];
    [enc setBuffer:get_buf(gpu.eos_P)               offset:0 atIndex:15];
    [enc setBuffer:get_buf(gpu.eos_s)               offset:0 atIndex:16];
    [enc setBuffer:get_buf(gpu.eos_T)               offset:0 atIndex:17];
    [enc setBuffer:get_buf(gpu.eos_dPde)            offset:0 atIndex:18];
    [enc setBuffer:get_buf(gpu.uprhs_out)           offset:0 atIndex:19];
    [enc setBytes:&params          length:sizeof(params)          atIndex:20];
    [enc setBytes:&gpu.eos_params  length:sizeof(gpu.eos_params)  atIndex:21];
    [enc setBuffer:get_buf(gpu.snap_future.epsilon) offset:0 atIndex:22];
    [enc setBuffer:get_buf(gpu.snap_future.rhob)    offset:0 atIndex:23];

    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}

// ── dispatch_uprhs ───────────────────────────────────────────────────────────
//
// Dispatches gpu_make_uprhs: scalar KT flux divergence of (u^a * pi_b).
// Called only when DATA.turn_on_bulk == 1.

void MetalPipelines::dispatch_uprhs(GPUGrid& gpu, const MUSICGridParams& params) {
    if (!ready_ || !pso_uprhs_) return;

    auto q   = (__bridge id<MTLCommandQueue>)        cmd_queue_;
    auto pso = (__bridge id<MTLComputePipelineState>)pso_uprhs_;

    int    n_handles = gpu.buf_handle_count();
    void** handles   = gpu.buf_handle_ptr();
    auto get_buf = [&](const float* ptr) -> id<MTLBuffer> {
        return buffer_for_ptr(handles, n_handles, ptr);
    };

    const bool own_cb = (batch_cb_ == nullptr);
    id<MTLCommandBuffer> cb = own_cb
        ? [q commandBuffer]
        : (__bridge id<MTLCommandBuffer>)batch_cb_;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    [enc setBuffer:get_buf(gpu.snap_curr.pi_b) offset:0 atIndex:0];
    [enc setBuffer:get_buf(gpu.snap_curr.u)    offset:0 atIndex:1];
    [enc setBuffer:get_buf(gpu.uprhs_out)      offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(8, 8, 4);
    MTLSize num_groups = MTLSizeMake(
        (gpu.Nx()   + 7) / 8,
        (gpu.Ny()   + 7) / 8,
        (gpu.Neta() + 3) / 4
    );

    [enc dispatchThreadgroups:num_groups threadsPerThreadgroup:threads_per_group];
    [enc endEncoding];
    if (own_cb) {
        [cb commit];
        if (cmd_buf_) CFRelease(cmd_buf_);
        cmd_buf_ = (__bridge_retained void*)cb;
    }
}
