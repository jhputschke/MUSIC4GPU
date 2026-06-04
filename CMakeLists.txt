cmake_minimum_required(VERSION 3.10)
project(MUSIChydro VERSION 2.0 LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

set(libname "music")
set(exename "MUSIChydro")

# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}")

set(CMAKE_INSTALL_RPATH ${CMAKE_HOME_DIRECTORY})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find GSL and openMP for this project:
find_package(GSL)
if (GSL_FOUND)
    message("Found GSL library ${GSL_INCLUDE_DIR}, will build MUSIC with GSL")
    # NOTE: the GSL -I flag is appended below, AFTER the compiler-specific
    # set(CMAKE_CXX_FLAGS ...) blocks, which would otherwise overwrite it.
    # include_directories() is not used here because CMake strips paths it
    # considers implicit to the (conda) compiler, dropping GSL from -I.
elseif()
    message("Did not find GSL library, will build MUSIC alone")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message(FATAL_ERROR "Insufficient gcc version")
    endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    # AppleClang has no built-in OpenMP runtime; use Homebrew libomp.
    find_program(BREW_EXECUTABLE brew)
    if (BREW_EXECUTABLE)
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LIBOMP_BREW_RESULT)
    endif()
    if (LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "Found Homebrew libomp at ${LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        find_package(OpenMP)
        if (OpenMP_FOUND)
            message(STATUS "OpenMP enabled for AppleClang via Homebrew libomp")
        endif()
    else ()
        message("AppleClang: Homebrew libomp not found "
                "(install with: brew install libomp). Building serial...")
    endif()
else ()
    find_package(OpenMP COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
        set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
    endif ()
endif()


# default build type: Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_HOME_DIRECTORY}/"
        CACHE PATH "Install path prefix, prepended onto install directories."
        FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

option (KNL "Build executable on KNL" OFF)
option (unittest "Build Unit tests" OFF)
option (link_with_lib "Link executable with the libarary" ON)
option (USE_METAL "Enable Metal GPU acceleration (macOS/Apple Silicon)" OFF)
option (USE_CUDA  "Enable CUDA GPU acceleration (NVIDIA)" OFF)
option (USE_KOKKOS "Enable Kokkos backend (NVIDIA/AMD/Intel GPU + multicore CPU)" OFF)

if (USE_METAL AND NOT APPLE)
    message(FATAL_ERROR "USE_METAL is only supported on macOS (Apple Silicon).")
endif()

if (USE_METAL AND USE_CUDA)
    message(FATAL_ERROR "USE_METAL and USE_CUDA are mutually exclusive.")
endif()

if (USE_KOKKOS AND (USE_CUDA OR USE_METAL))
    message(FATAL_ERROR "USE_KOKKOS is mutually exclusive with USE_CUDA / USE_METAL.")
endif()

if (USE_CUDA)
    # Detect the local GPU's compute capability unless the user overrides it
    # (e.g. -DCMAKE_CUDA_ARCHITECTURES=80 for an A100).
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if (KNL)
        set(CMAKE_CXX_FLAGS "-vecabi=cmdtarget -debug inline-debug-info -fimf-use-svml -xmic-avx512 -qopt-report=5 -qopt-report-phase:vec ${OpenMP_CXX_FLAGS} -std=c++11 -fma -align -finline-functions -DDOCTEST_CONFIG_DISABLE -DNDEBUG")
    else ()
        set(CMAKE_CXX_FLAGS "-ipo -qopt-report=5 -qopt-report-phase:vec ${OpenMP_CXX_FLAGS} -std=c++11 -fimf-use-svml -fma -align -finline-functions -DDOCTEST_CONFIG_DISABLE -DNDEBUG")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} -std=c++11 -DDOCTEST_CONFIG_DISABLE -DNDEBUG")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-std=c++11  -stdlib=libc++ -DDOCTEST_CONFIG_DISABLE -DNDEBUG -w")
    string(APPEND CMAKE_OBJCXX_FLAGS " -w")
endif()

if (unittest)
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} -std=c++11 -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN")
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
endif()

# Append the GSL include here, after the compiler-specific set(CMAKE_CXX_FLAGS)
# overwrites above. Done as a raw -I flag (not include_directories) so CMake
# cannot strip it as an "implicit" compiler path (conda toolchain reports
# $PREFIX/include as implicit only when CPATH is set at configure time).
if (GSL_FOUND)
    string(APPEND CMAKE_CXX_FLAGS " -I${GSL_INCLUDE_DIR}")
endif()

# ── Kokkos backend (NVIDIA/AMD/Intel GPU + multicore CPU) ────────────────────
# Single integration point: this runs in BOTH the stand-alone MUSIChydro build
# and the X-SCAPE build (which add_subdirectory()'s this CMakeLists.txt).
if (USE_KOKKOS)
    # Kokkos needs CMake >= 3.16 (4.x; 5.x wants ~3.25). Raise the policy floor
    # only in the Kokkos configuration, leaving the CUDA/Metal/CPU builds on the
    # 3.10 floor untouched. Kokkos' own CMakeLists enforces its exact minimum.
    cmake_minimum_required(VERSION 3.16)

    # Kokkos 5.x requires C++20 (4.x needs C++17); use C++20 so the default
    # latest-release build works. The compiler blocks above hard-set -std=c++11,
    # so surgically bump that flag too (the last -std on the command line wins).
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    string(REPLACE "-std=c++11" "-std=c++20" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    # libmusic is a SHARED library, so Kokkos' default static archive must be
    # built position-independent or the link fails (notably on AArch64:
    # "relocation R_AARCH64_* ... recompile with -fPIC").
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    if (NOT TARGET Kokkos::kokkos)
        # Discovery order: explicit override -> in-repo source (get_kokkos.sh)
        # -> X-SCAPE sibling -> system install (Spack / module on HPC).
        if (NOT KOKKOS_SOURCE_DIR)
            foreach (_cand "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos"
                           "${CMAKE_CURRENT_SOURCE_DIR}/../kokkos")
                if (EXISTS "${_cand}/CMakeLists.txt")
                    set(KOKKOS_SOURCE_DIR "${_cand}")
                    break()
                endif()
            endforeach()
        endif()
        if (KOKKOS_SOURCE_DIR AND EXISTS "${KOKKOS_SOURCE_DIR}/CMakeLists.txt")
            message(STATUS "Kokkos: add_subdirectory(${KOKKOS_SOURCE_DIR})")
            add_subdirectory(${KOKKOS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kokkos_build)
        else()
            message(STATUS "Kokkos source not found - find_package(Kokkos) "
                           "(system / Spack / module install). Hint: run get_kokkos.sh "
                           "or pass -DKokkos_ROOT=<path>.")
            find_package(Kokkos REQUIRED)
        endif()
    endif()
    message(STATUS "Kokkos backend enabled (USE_KOKKOS=ON).")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

find_package(Git)
if(Git_FOUND)
    # Get the current working branch
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Get the latest abbreviated commit hash of the working branch
    execute_process(
        COMMAND git log -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    add_definitions(-DGIT_COMMIT_HASH="${GIT_COMMIT_HASH}")
    add_definitions(-DGIT_BRANCH="${GIT_BRANCH}")
else ()
    add_definitions(-DGIT_COMMIT_HASH="unknown")
    add_definitions(-DGIT_BRANCH="unknown")
endif()

configure_file(
  src/version.h.in
  ${CMAKE_BINARY_DIR}/generated/version.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)
add_subdirectory (src)
