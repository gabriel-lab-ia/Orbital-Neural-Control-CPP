cmake_minimum_required(VERSION 3.24)

project(OrbitalPybind LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(py_orbital_core py_orbital_core.cpp)

target_include_directories(py_orbital_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../control/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sim/include
)

target_sources(py_orbital_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/core_api.cpp
)

target_compile_features(py_orbital_core PRIVATE cxx_std_20)
