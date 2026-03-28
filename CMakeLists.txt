
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
  common/config.hpp       common/config.cpp
  common/material.hpp     common/material.cpp
  common/output.hpp       common/output.cpp
  common/common.hpp       common/solvers.h      common/timer.h
  morph/parameterization.h      morph/parameterization.cpp
  morph/parameterize_pipeline.h morph/parameterize_pipeline.cpp
  morph/LocalGlobalSolver.h     morph/LocalGlobalSolver.cpp
  morph/morphmesh.hpp           morph/morphmesh.cpp
  morph/morph_functions.hpp     morph/morph_functions.cpp
  morph/functions.h             morph/functions.cpp
  morph/newton.h                morph/newton.cpp
  morph/simulation_utils.h      morph/simulation_utils.cpp
  morph/inverse_design.h        morph/inverse_design.cpp
  metric/patch_utils.h          metric/patch_utils.cpp
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
  "${CMAKE_SOURCE_DIR}/common"
  "${CMAKE_SOURCE_DIR}/morph"
  "${CMAKE_SOURCE_DIR}/metric"
  "${CMAKE_SOURCE_DIR}/include"
)

# ═══════════════════════════════════════════════════════════════════════
# Executable targets
# ═══════════════════════════════════════════════════════════════════════
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Inverse design (current main entry point)
add_executable(Inverse morph/main.cpp)
target_link_libraries(Inverse PRIVATE grayscale_core)

# Backward-compat alias
add_executable(GrayScaleMorph morph/main.cpp)
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
