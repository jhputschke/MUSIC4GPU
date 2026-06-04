// Definition of the host-safe Kokkos lifecycle guard.  This is the only place
// that includes <Kokkos_Core.hpp>; everyone else uses the header above.

#include "kokkos_runtime.h"
#include <Kokkos_Core.hpp>

KokkosRuntimeGuard::KokkosRuntimeGuard(int& argc, char* argv[]) {
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        Kokkos::initialize(argc, argv);
        owns_ = true;
    }
}

KokkosRuntimeGuard::KokkosRuntimeGuard() {
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        Kokkos::initialize();
        owns_ = true;
    }
}

KokkosRuntimeGuard::~KokkosRuntimeGuard() {
    // Only finalize if this guard initialized Kokkos, so a guard constructed
    // when Kokkos was already up (another module) never tears it down early.
    if (owns_ && Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}
