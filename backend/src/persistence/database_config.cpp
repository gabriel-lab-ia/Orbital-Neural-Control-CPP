#include "persistence/database_config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace orbital::backend::persistence {
namespace {

std::string env_string(const char* key, const std::string& fallback = "") {
    const char* raw = std::getenv(key);
    if (raw == nullptr || std::string(raw).empty()) {
        return fallback;
    }
    return raw;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::uint16_t parse_port(const std::string& raw, const std::uint16_t fallback) {
    if (raw.empty()) {
        return fallback;
    }
    try {
        const int value = std::stoi(raw);
        if (value <= 0 || value > 65535) {
            throw std::out_of_range("port out of range");
        }
        return static_cast<std::uint16_t>(value);
    } catch (const std::exception& ex) {
        throw std::runtime_error("invalid POSTGRES_PORT '" + raw + "': " + ex.what());
    }
}

int parse_positive_int(const char* key, const int fallback) {
    const std::string raw = env_string(key);
    if (raw.empty()) {
        return fallback;
    }
    try {
        const int value = std::stoi(raw);
        if (value <= 0) {
            throw std::out_of_range("value must be positive");
        }
        return value;
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("invalid ") + key + " '" + raw + "': " + ex.what());
    }
}

std::filesystem::path sqlite_path_from_env(const std::filesystem::path& artifact_root) {
    const std::string standard = env_string("SQLITE_PATH");
    if (!standard.empty()) {
        return standard;
    }
    const std::string legacy = env_string("ORBITAL_SQLITE_PATH");
    if (!legacy.empty()) {
        return legacy;
    }
    return artifact_root / "experiments.sqlite";
}

}  // namespace

const char* to_string(const DatabaseBackend backend) {
    switch (backend) {
        case DatabaseBackend::SQLite:
            return "sqlite";
        case DatabaseBackend::Postgres:
            return "postgres";
    }
    return "unknown";
}

DatabaseConfig database_config_from_env(const std::filesystem::path& artifact_root) {
    DatabaseConfig config;
    config.sqlite_path = sqlite_path_from_env(artifact_root);

    const std::string backend = lowercase(env_string("DB_BACKEND", "sqlite"));
    if (backend == "sqlite") {
        config.backend = DatabaseBackend::SQLite;
        return config;
    }
    if (backend != "postgres") {
        throw std::runtime_error("unsupported DB_BACKEND '" + backend + "'; expected sqlite or postgres");
    }

    config.backend = DatabaseBackend::Postgres;
    config.postgres.host = env_string("POSTGRES_HOST", "localhost");
    config.postgres.port = parse_port(env_string("POSTGRES_PORT", "5432"), 5432);
    config.postgres.database = env_string("POSTGRES_DB");
    config.postgres.user = env_string("POSTGRES_USER");
    config.postgres.password = env_string("POSTGRES_PASSWORD");
    config.postgres.connect_timeout_seconds = parse_positive_int("POSTGRES_CONNECT_TIMEOUT_SECONDS", 5);
    config.postgres.statement_timeout_ms = parse_positive_int("POSTGRES_STATEMENT_TIMEOUT_MS", 5000);

    if (config.postgres.database.empty()) {
        throw std::runtime_error("POSTGRES_DB is required when DB_BACKEND=postgres");
    }
    if (config.postgres.user.empty()) {
        throw std::runtime_error("POSTGRES_USER is required when DB_BACKEND=postgres");
    }
    if (config.postgres.password.empty()) {
        throw std::runtime_error("POSTGRES_PASSWORD is required when DB_BACKEND=postgres");
    }

    return config;
}

}  // namespace orbital::backend::persistence
