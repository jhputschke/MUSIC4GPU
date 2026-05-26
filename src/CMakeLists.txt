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

    # Copy metallib next to the final executable/library so it can be found
    # at runtime via [NSBundle mainBundle].
    add_custom_command(
        TARGET metal_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${METAL_LIB}
            ${CMAKE_HOME_DIRECTORY}/music_kernels.metallib
        COMMENT "Copying music_kernels.metallib to install prefix"
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
