set (SOURCES
    u_derivative.cpp
    dissipative.cpp
    init.cpp
    reconst.cpp
    minmod.cpp
    music.cpp
    cornelius.cpp
    hydro_source_base.cpp
    hydro_source_strings.cpp
    hydro_source_ampt.cpp
    hydro_source_TATB.cpp
    pretty_ostream.cpp
    grid_info.cpp
    util.cpp
    read_in_parameters.cpp
    advance.cpp
    fields.cpp
    eos.cpp
    eos_base.cpp
    eos_idealgas.cpp
    eos_EOSQ.cpp
    eos_s95p.cpp
    eos_WB.cpp
    eos_hotQCD.cpp
    eos_best.cpp
    eos_neos.cpp
    eos_UH.cpp
    eos_4D.cpp
    evolve.cpp
    emoji.cpp
    music_logo.cpp
    HydroinfoMUSIC.cpp
    transport_coeffs.cpp
    )
if (GSL_FOUND)
    set(SOURCES ${SOURCES}
        freeze.cpp
        freeze_pseudo.cpp
        reso_decay.cpp
        )
endif()

# ── Metal GPU sources (macOS / Apple Silicon) ────────────────────────────────
if (USE_METAL)
    set(SOURCES ${SOURCES}
        gpu/GPUGrid.mm
        gpu/MetalPipelines.mm
        )

    # Compile the Metal shader to a .metallib at build time.
    # xcrun is part of the Xcode command-line tools (required on macOS).
    set(METAL_SRC  "${CMAKE_CURRENT_SOURCE_DIR}/gpu/music_kernels.metal")
    set(METAL_AIR  "${CMAKE_CURRENT_BINARY_DIR}/music_kernels.air")
    set(METAL_LIB  "${CMAKE_CURRENT_BINARY_DIR}/music_kernels.metallib")

    find_program(XCRUN xcrun REQUIRED)

    add_custom_command(
        OUTPUT  ${METAL_AIR}
        COMMAND ${XCRUN} -sdk macosx metal
                    -c ${METAL_SRC}
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/gpu
                    -o ${METAL_AIR}
        DEPENDS ${METAL_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_types.h
        COMMENT "Compiling Metal shader music_kernels.metal -> .air"
    )
    add_custom_command(
        OUTPUT  ${METAL_LIB}
        COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal library -> music_kernels.metallib"
    )
    add_custom_target(metal_kernels ALL DEPENDS ${METAL_LIB})

    # The metallib is generated in ${CMAKE_CURRENT_BINARY_DIR}, i.e. right next to
    # the freshly built libmusic.dylib, so the dladdr-based loader in
    # MetalPipelines.mm finds it directly in the build tree (no copy needed).
    # For `make install`, place it alongside the installed library.
    install(FILES ${METAL_LIB} DESTINATION ${CMAKE_HOME_DIRECTORY})
endif()

# ── CUDA GPU sources (NVIDIA) ────────────────────────────────────────────────
if (USE_CUDA)
    set(SOURCES ${SOURCES}
        gpu/GPUGrid_cuda.cu
        gpu/CUDAPipelines.cu
        gpu/music_kernels.cu
        )
endif()

add_library(${libname} SHARED ${SOURCES})
set(CompileFlags "")
if (APPLE)
    set(CompileFlags "${CompileFlags} -DAPPLE")
endif (APPLE)
if (GSL_FOUND)
    set(CompileFlags "${CompileFlags} -DGSL")
    target_link_libraries (${libname} ${GSL_LIBRARIES})
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(${libname} OpenMP::OpenMP_CXX)
endif()

if (USE_METAL)
    set(CompileFlags "${CompileFlags} -DUSE_METAL")
    # Link Metal and Foundation frameworks
    target_link_libraries(${libname} "-framework Metal" "-framework Foundation")
    # Ensure .mm files are compiled as Objective-C++
    set_source_files_properties(
        gpu/GPUGrid.mm
        gpu/MetalPipelines.mm
        PROPERTIES COMPILE_FLAGS "-x objective-c++"
    )
    # Include gpu/ subdirectory for header search
    target_include_directories(${libname} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    )
    add_dependencies(${libname} metal_kernels)
endif()

if (USE_CUDA)
    set(CompileFlags "${CompileFlags} -DUSE_CUDA")
    set_source_files_properties(
        gpu/GPUGrid_cuda.cu
        gpu/CUDAPipelines.cu
        gpu/music_kernels.cu
        PROPERTIES LANGUAGE CUDA
    )
    # USE_CUDA / NDEBUG must reach the nvcc compilation too (COMPILE_FLAGS is
    # applied to every source, but defines are cleaner via this property, which
    # spans both CXX and CUDA).
    target_compile_definitions(${libname} PRIVATE USE_CUDA NDEBUG)
    target_link_libraries(${libname} CUDA::cudart)
    target_include_directories(${libname} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    )
    # Whole-program compilation: each .cu is self-contained (kernels and their
    # __device__ helpers live in the same TU; cross-TU references are only the
    # host launch stubs), so relocatable device code / device linking is not
    # needed and we avoid the shared-library nvlink complications it brings.
    # Forward OpenMP to nvcc's host compiler so the AoS<->SoA converters in
    # GPUGrid_cuda.cu stay parallel.
    if (OpenMP_CXX_FOUND)
        target_compile_options(${libname} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    endif()
    # Phase 5: fast-math intrinsics for the compute-bound device kernels
    # (Newton reconstruction divides/sqrtf, transport expf/powf).  The hydro
    # state is already FP32 (~1e-5 vs CPU); fast-math stays well inside the
    # 1e-3 correctness threshold while accelerating delta_qi / w_full.
    target_compile_options(${libname} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()


set_target_properties (${libname} PROPERTIES COMPILE_FLAGS "${CompileFlags}")
install(TARGETS ${libname} DESTINATION ${CMAKE_HOME_DIRECTORY})


if (unittest)
    add_executable (unittest_reconst.e reconst_unittest.cpp)
    target_link_libraries (unittest_reconst.e ${libname})
    install(TARGETS unittest_reconst.e DESTINATION ${CMAKE_HOME_DIRECTORY})

    add_executable (unittest_grid.e grid_unittest.cpp)
    target_link_libraries (unittest_grid.e ${libname})
    install(TARGETS unittest_grid.e DESTINATION ${CMAKE_HOME_DIRECTORY})

    add_executable (unittest_cell.e cell_unittest.cpp)
    target_link_libraries (unittest_cell.e ${libname})
    install(TARGETS unittest_cell.e DESTINATION ${CMAKE_HOME_DIRECTORY})

    add_executable (unittest_minmod.e minmod_unittest.cpp)
    target_link_libraries (unittest_minmod.e ${libname})
    install(TARGETS unittest_minmod.e DESTINATION ${CMAKE_HOME_DIRECTORY})
else (unittest)
    if (link_with_lib)
        add_executable (${exename} main.cpp)
        target_link_libraries (${exename} ${libname})
    else (link_with_lib)
        add_executable (${exename} main.cpp ${SOURCES})
        if (GSL_FOUND)
            target_link_libraries (${exename} ${GSL_LIBRARIES})
        endif (GSL_FOUND)
    endif (link_with_lib)
    set_target_properties (${exename} PROPERTIES COMPILE_FLAGS "${CompileFlags}")
    install(TARGETS ${exename} DESTINATION ${CMAKE_HOME_DIRECTORY})
endif (unittest)
