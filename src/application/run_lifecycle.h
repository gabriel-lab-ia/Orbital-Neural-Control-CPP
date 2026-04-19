#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "infrastructure/artifacts/artifact_layout.h"
#include "infrastructure/artifacts/run_manifest.h"
#include "infrastructure/persistence/sqlite_experiment_store.h"

namespace nmc::application {

struct RunStartContext {
    std::string run_id;
    std::string mode;
    std::string environment;
    int64_t seed = 0;
    std::string started_at;
    std::filesystem::path artifact_dir;
    std::string config_json;
};

struct RunFailureContext {
    std::string run_id;
    std::string mode;
    std::string environment;
    std::string started_at;
    std::string failed_at;
    std::string config_json;
    std::vector<infrastructure::artifacts::RunManifestArtifact> artifacts;
    std::optional<std::string> checkpoint_path;
    std::string error_message;
};

void record_run_start(
    infrastructure::persistence::SQLiteExperimentStore& db,
    const RunStartContext& context
);

void record_run_success(
    infrastructure::persistence::SQLiteExperimentStore& db,
    const std::string& run_id,
    const std::string& ended_at,
    const std::string& summary_json
);

void record_run_failure(
    infrastructure::persistence::SQLiteExperimentStore& db,
    const infrastructure::artifacts::ArtifactLayout& layout,
    const RunFailureContext& context
);

}  // namespace nmc::application
