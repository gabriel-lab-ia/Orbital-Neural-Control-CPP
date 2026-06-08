#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace orbital::backend::persistence {

enum class DatabaseBackend {
    SQLite,
    Postgres
};

struct PostgresConfig {
    std::string host = "localhost";
    std::uint16_t port = 5432;
    std::string database;
    std::string user;
    std::string password;
    int connect_timeout_seconds = 5;
    int statement_timeout_ms = 5000;
};

struct DatabaseConfig {
    DatabaseBackend backend = DatabaseBackend::SQLite;
    std::filesystem::path sqlite_path = "artifacts/experiments.sqlite";
    PostgresConfig postgres;
};

[[nodiscard]] DatabaseConfig database_config_from_env(const std::filesystem::path& artifact_root);
[[nodiscard]] const char* to_string(DatabaseBackend backend);

}  // namespace orbital::backend::persistence
