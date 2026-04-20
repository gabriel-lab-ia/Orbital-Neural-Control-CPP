cmake_minimum_required(VERSION 3.26)

project(OrbitalBackend LANGUAGES CXX)

find_package(Boost CONFIG REQUIRED COMPONENTS system json)
set(ORBITAL_SQLITE_TARGET "")
find_package(SQLite3 CONFIG QUIET)
if(TARGET SQLite::SQLite3)
    set(ORBITAL_SQLITE_TARGET SQLite::SQLite3)
else()
    find_package(unofficial-sqlite3 CONFIG REQUIRED)
    if(TARGET unofficial::sqlite3::sqlite3)
        set(ORBITAL_SQLITE_TARGET unofficial::sqlite3::sqlite3)
    endif()
endif()
if(NOT ORBITAL_SQLITE_TARGET)
    message(FATAL_ERROR "SQLite3 target not found (expected SQLite::SQLite3 or unofficial::sqlite3::sqlite3)")
endif()

set(ORBITAL_BACKEND_SOURCES
    src/main.cpp
    src/common/http_utils.cpp
    src/common/logger.cpp
    src/common/run_id.cpp
    src/common/time_utils.cpp
    src/persistence/sqlite_store.cpp
    src/telemetry/csv_telemetry_store.cpp
    src/replay/replay_service.cpp
    src/application/mission_service.cpp
    src/application/job_service.cpp
    src/transport/json_serialization.cpp
    src/transport/http_server.cpp
)

add_executable(orbital_backend ${ORBITAL_BACKEND_SOURCES})

target_compile_features(orbital_backend PRIVATE cxx_std_20)

target_include_directories(
    orbital_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(orbital_backend PRIVATE Boost::system Boost::json ${ORBITAL_SQLITE_TARGET} pthread)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        orbital_backend PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wformat=2
    )
endif()
