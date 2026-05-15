
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


# ---------------------------------------------------------------------------
# libmorph: shared static library used by every sub-binary
# ---------------------------------------------------------------------------

add_library(morph STATIC
  morph/common.hpp
  morph/config.hpp
  morph/config.cpp
  morph/material.hpp
  morph/material.cpp
  morph/timer.h
  morph/solvers.h
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
  morph/output.hpp
  morph/output.cpp
  morph/boundary_utils.h
  morph/boundary_utils.cpp
)

target_link_libraries(morph
  PUBLIC
    igl::core
    igl::opengl
    igl::stb
    glfw::glfw
    geometry-central
    TinyAD
)

target_include_directories(morph
  PUBLIC
    "${CMAKE_SOURCE_DIR}/morph"
    "${CMAKE_SOURCE_DIR}/include"
)

if (MSVC)
  target_compile_options(morph PRIVATE /utf-8)
endif()


# ---------------------------------------------------------------------------
# Sub-binaries: param / forward / inverse
# ---------------------------------------------------------------------------

function(add_sub_binary BIN_NAME SUBDIR)
  add_executable(${BIN_NAME} ${SUBDIR}/main.cpp)
  target_link_libraries(${BIN_NAME} PRIVATE morph)
  set_target_properties(${BIN_NAME} PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/${SUBDIR}")
  if (MSVC)
    target_compile_options(${BIN_NAME} PRIVATE /utf-8)
    set_target_properties(${BIN_NAME} PROPERTIES LINK_FLAGS "/STACK:10485760")
  endif()
endfunction()

add_sub_binary(Param      param)
add_sub_binary(ParamAll   param_all)
add_sub_binary(Forward    forward)
add_sub_binary(Inverse    inverse)
add_sub_binary(InverseAll inverse_all)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
