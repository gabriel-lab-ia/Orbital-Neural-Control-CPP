cmake_minimum_required(VERSION 3.24)

project(OrbitalBackend LANGUAGES CXX)

find_package(Boost QUIET COMPONENTS system)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost::system is required for orbital_backend. Install libboost-all-dev.")
endif()

add_executable(orbital_backend src/main.cpp)

target_compile_features(orbital_backend PRIVATE cxx_std_20)

target_include_directories(
    orbital_backend PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../control/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../rl/include
)

target_link_libraries(orbital_backend PRIVATE Boost::system pthread)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(orbital_backend PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()
