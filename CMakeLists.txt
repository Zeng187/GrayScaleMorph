
cmake_minimum_required(VERSION 3.16)

project(GrayScaleMorph LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBIGL_OPENGL         ON  CACHE BOOL "Use ImGui"          FORCE)
set(LIBIGL_STB            ON  CACHE BOOL "Use ImGui"          FORCE)
set(LIBIGL_GLFW           ON  CACHE BOOL "Use ImGui"          FORCE)
set(LIBIGL_WITH_VIEWER    ON  CACHE BOOL "Use OpenGL viewer"  FORCE)

add_subdirectory("${CMAKE_SOURCE_DIR}/deps/eigen-src" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_SOURCE_DIR}/deps/libigl-src" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_SOURCE_DIR}/deps/geometry-central-src" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_SOURCE_DIR}/deps/TinyAD-src" EXCLUDE_FROM_ALL)

# ═══════════════════════════════════════════════════════════════════════
# Shared core library — all algorithms, no entry points
# ═══════════════════════════════════════════════════════════════════════
add_library(grayscale_core STATIC
  core/config.hpp             core/config.cpp
  core/material.hpp           core/material.cpp
  core/output.hpp             core/output.cpp
  core/patch_utils.h          core/patch_utils.cpp
  core/common.hpp             core/solvers.h            core/timer.h
  core/parameterization.h     core/parameterization.cpp
  core/parameterize_pipeline.h core/parameterize_pipeline.cpp
  core/LocalGlobalSolver.h    core/LocalGlobalSolver.cpp
  core/morphmesh.hpp          core/morphmesh.cpp
  core/morph_functions.hpp    core/morph_functions.cpp
  core/functions.h            core/functions.cpp
  core/newton.h               core/newton.cpp
  core/simulation_utils.h     core/simulation_utils.cpp
  core/boundary_utils.h       core/boundary_utils.cpp
  inverse/inverse_design.h    inverse/inverse_design.cpp
)

target_link_libraries(grayscale_core PUBLIC
  igl::core
  igl::opengl
  igl::stb
  glfw::glfw
  geometry-central
  TinyAD
)

target_include_directories(grayscale_core PUBLIC
  "${CMAKE_SOURCE_DIR}/core"
  "${CMAKE_SOURCE_DIR}/inverse"
  "${CMAKE_SOURCE_DIR}/include"
)

# ═══════════════════════════════════════════════════════════════════════
# Executable targets
# ═══════════════════════════════════════════════════════════════════════
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Inverse design — single patch (reads inverse_cfg.json)
add_executable(Inverse inverse/main.cpp)
target_link_libraries(Inverse PRIVATE grayscale_core)

# Inverse design — all patches (reads inverse_whole_cfg.json)
add_executable(InverseWhole inverse_whole/main.cpp)
target_link_libraries(InverseWhole PRIVATE grayscale_core)

# Backward-compat alias
add_executable(GrayScaleMorph inverse/main.cpp)
target_link_libraries(GrayScaleMorph PRIVATE grayscale_core)

# Parameterize-only entry point
add_executable(Parameterize parameterize/main.cpp)
target_link_libraries(Parameterize PRIVATE grayscale_core)

# Forward-simulation entry point
add_executable(Forward forward/main.cpp)
target_link_libraries(Forward PRIVATE grayscale_core)

# Evaluation entry point (replaces FeasEval + PatchMetrics)
add_executable(Evaluate eval/main.cpp)
target_link_libraries(Evaluate PRIVATE grayscale_core)

# Verify: compute per-face (lambda, kappa) on an arbitrary 3D mesh
add_executable(Verify verify/main.cpp)
target_link_libraries(Verify PRIVATE grayscale_core)

if (MSVC)
    target_compile_options(Inverse PRIVATE /utf-8)
    target_compile_options(GrayScaleMorph PRIVATE /utf-8)
    target_compile_options(Forward PRIVATE /utf-8)
    target_compile_options(Evaluate PRIVATE /utf-8)
    set_target_properties(Inverse PROPERTIES LINK_FLAGS "/STACK:10485760")
    set_target_properties(GrayScaleMorph PROPERTIES LINK_FLAGS "/STACK:10485760")
    set_target_properties(Forward PROPERTIES LINK_FLAGS "/STACK:10485760")
    set_target_properties(Evaluate PROPERTIES LINK_FLAGS "/STACK:10485760")
endif()
