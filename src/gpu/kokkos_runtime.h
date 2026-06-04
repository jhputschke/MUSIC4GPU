// Host-safe RAII wrapper for the Kokkos runtime lifecycle.
//
// Declared with NO Kokkos headers so that plain host translation units — the
// stand-alone MUSIChydro main() and the X-SCAPE framework driver main() — can
// own Kokkos::initialize / Kokkos::finalize without being compiled by the
// Kokkos toolchain.  The definition lives in kokkos_runtime.cpp, the only TU
// here that includes <Kokkos_Core.hpp> (PIMPL boundary, PlanKokkosPort.md D7).
//
// Lifecycle rule (PlanKokkosPort.md D2): construct ONE guard before any object
// that allocates Kokkos Views, and let it outlive them.  In MUSIC stand-alone
// that is the top of main(); under X-SCAPE it is the framework driver main(),
// before the JetScape object.
//
// The guard is collision-safe: if Kokkos is already initialized (e.g. another
// module brought it up first) this guard does nothing and will NOT finalize on
// destruction — only the guard that actually initialized Kokkos finalizes it.

#pragma once

struct KokkosRuntimeGuard {
    // Forwards argc/argv so Kokkos can parse its --kokkos-* command-line flags.
    KokkosRuntimeGuard(int& argc, char* argv[]);
    // Convenience overload for callers without command-line arguments.
    KokkosRuntimeGuard();
    ~KokkosRuntimeGuard();

    KokkosRuntimeGuard(const KokkosRuntimeGuard&)            = delete;
    KokkosRuntimeGuard& operator=(const KokkosRuntimeGuard&) = delete;

private:
    bool owns_ = false;   // true only if THIS guard called Kokkos::initialize()
};
