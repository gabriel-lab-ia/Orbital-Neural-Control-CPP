cmake_minimum_required(VERSION 3.24)

project(OrbitalCore LANGUAGES CXX)

add_library(orbital_core STATIC
    src/core_api.cpp
)

target_include_directories(orbital_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../control/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../rl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim/include
)

target_compile_features(orbital_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(orbital_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()

option(ORBITAL_CORE_BUILD_TESTS "Build orbital core tests" ON)
option(ORBITAL_CORE_BUILD_BENCHMARKS "Build orbital core benchmarks" ON)

if(ORBITAL_CORE_BUILD_TESTS)
    enable_testing()

    add_executable(orbital_core_tests tests/test_orbital_core.cpp)
    target_link_libraries(orbital_core_tests PRIVATE orbital_core)
    target_compile_features(orbital_core_tests PRIVATE cxx_std_20)
    add_test(NAME orbital_core_tests COMMAND orbital_core_tests)

    add_executable(orbital_baseline_tests tests/test_baselines.cpp)
    target_link_libraries(orbital_baseline_tests PRIVATE orbital_core)
    target_compile_features(orbital_baseline_tests PRIVATE cxx_std_20)
    add_test(NAME orbital_baseline_tests COMMAND orbital_baseline_tests)
endif()

if(ORBITAL_CORE_BUILD_BENCHMARKS)
    add_executable(orbital_control_benchmark benchmarks/benchmark_control_baselines.cpp)
    target_link_libraries(orbital_control_benchmark PRIVATE orbital_core)
    target_compile_features(orbital_control_benchmark PRIVATE cxx_std_20)
endif()
