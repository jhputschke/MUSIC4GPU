// Objective-C++ implementation of MetalPipelines.
// Loads the Metal shader library, creates compute pipeline states, and
// dispatches the gpu_make_w_source kernel.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include "MetalPipelines.h"
#include "GPUGrid.h"
#include "gpu_types.h"

// Shared Metal device – declared here, extern'd in GPUGrid.mm
id<MTLDevice> g_metal_device = nil;

// ── singleton ─────────────────────────────────────────────────────────────────

MetalPipelines& MetalPipelines::instance() {
    static MetalPipelines inst;
    return inst;
}

MetalPipelines::~MetalPipelines() {
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

    // 3. Load shader library
    id<MTLLibrary> lib = nil;
    NSError* err = nil;

    if (metallib_path && metallib_path[0] != '\0') {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        lib = [dev newLibraryWithURL:url error:&err];
    }

    if (!lib) {
        // Fall back: look for music_kernels.metallib next to the executable
        NSBundle* bundle = [NSBundle mainBundle];
        NSString* path = [bundle pathForResource:@"music_kernels" ofType:@"metallib"];
        if (path) {
            NSURL* url = [NSURL fileURLWithPath:path];
            lib = [dev newLibraryWithURL:url error:&err];
        }
    }

    if (!lib) {
        fprintf(stderr, "[MUSIC-GPU] Could not load music_kernels.metallib: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        fprintf(stderr, "[MUSIC-GPU] Run: xcrun -sdk macosx metal music_kernels.metal"
                        " -o music_kernels.metallib\n");
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

    ready_ = true;
    return true;
}

// ── wait ──────────────────────────────────────────────────────────────────────

void MetalPipelines::wait() {
    if (!cmd_buf_) return;
    auto cb = (__bridge id<MTLCommandBuffer>)cmd_buf_;
    [cb waitUntilCompleted];
    CFRelease(cmd_buf_);
    cmd_buf_ = nullptr;
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

    id<MTLCommandBuffer>      cb  = [q commandBuffer];
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
    [cb commit];

    if (cmd_buf_) CFRelease(cmd_buf_);
    cmd_buf_ = (__bridge_retained void*)cb;
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

    id<MTLCommandBuffer>         cb  = [q commandBuffer];
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
    [cb commit];

    if (cmd_buf_) CFRelease(cmd_buf_);
    cmd_buf_ = (__bridge_retained void*)cb;
}
