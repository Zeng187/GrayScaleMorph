
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



add_executable(
  ${ProjectName}
  src/main.cpp
  src/config.hpp
  src/config.cpp
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
  "${CMAKE_SOURCE_DIR}/include"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set_target_properties(${ProjectName} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/src"
)
if (MSVC)
    target_compile_options(${ProjectName} PRIVATE /utf-8)
endif()