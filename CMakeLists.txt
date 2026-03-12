
cmake_minimum_required(VERSION 3.16)

set(ProjectName "GrayScaleMorph")

project(${ProjectName} LANGUAGES CXX)
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

file(GLOB RSC_FILES
    "${CMAKE_SOURCE_DIR}/*.json"
)

add_executable(
  ${ProjectName}
  morph/main.cpp
  common/common.hpp
  common/config.hpp
  common/config.cpp
  common/material.hpp
  common/material.cpp
  common/timer.h
  common/solvers.h
  morph/newton.h
  morph/newton.cpp
  morph/parameterization.h
  morph/parameterization.cpp
  morph/LocalGlobalSolver.h
  morph/LocalGlobalSolver.cpp
  morph/simulation_utils.h
  morph/simulation_utils.cpp
  morph/functions.h
  morph/functions.cpp
  morph/morphmesh.hpp
  morph/morphmesh.cpp
  morph/morph_functions.hpp
  morph/morph_functions.cpp
  common/output.hpp
  common/output.cpp
  .gitignore
  ${RSC_FILES}
)

target_link_libraries(
  ${ProjectName}
  PRIVATE
  igl::core
  igl::opengl
  igl::stb
  glfw::glfw
  geometry-central
  TinyAD
)

target_include_directories(
  ${ProjectName}
  PRIVATE
  "${CMAKE_SOURCE_DIR}/common"
  "${CMAKE_SOURCE_DIR}/morph"
  "${CMAKE_SOURCE_DIR}/metric"
  "${CMAKE_SOURCE_DIR}/include"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set_target_properties(${ProjectName} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
if (MSVC)
    target_compile_options(${ProjectName} PRIVATE /utf-8)
    set_target_properties(${ProjectName} PROPERTIES LINK_FLAGS "/STACK:10485760")
endif()

# PatchMetrics — per-patch material feasibility metrics tool
add_executable(PatchMetrics
  metric/patch_metrics_main.cpp
  metric/patch_utils.h
  metric/patch_utils.cpp
  common/common.hpp
  common/material.hpp
  common/material.cpp
  morph/parameterization.h
  morph/parameterization.cpp
  morph/LocalGlobalSolver.h
  morph/LocalGlobalSolver.cpp
  morph/morphmesh.hpp
  morph/morphmesh.cpp
  morph/morph_functions.hpp
  morph/morph_functions.cpp
  morph/newton.h
  morph/newton.cpp
  morph/functions.h
  morph/functions.cpp
  morph/simulation_utils.h
  morph/simulation_utils.cpp
)

target_link_libraries(PatchMetrics PRIVATE
  igl::core
  igl::opengl
  igl::stb
  glfw::glfw
  geometry-central
  TinyAD
)

target_include_directories(PatchMetrics PRIVATE
  "${CMAKE_SOURCE_DIR}/common"
  "${CMAKE_SOURCE_DIR}/morph"
  "${CMAKE_SOURCE_DIR}/metric"
  "${CMAKE_SOURCE_DIR}/include"
)

if (MSVC)
    target_compile_options(PatchMetrics PRIVATE /utf-8)
endif()