#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nmc::infrastructure::artifacts {

struct RunManifestArtifact {
    std::string name;
    std::filesystem::path path;
};

struct RunManifest {
    std::string run_id;
    std::string mode;
    std::string environment;
    std::string started_at;
    std::string ended_at;
    std::string status;
    std::string config_json;
    std::vector<RunManifestArtifact> artifacts;
    std::optional<std::string> checkpoint_path;
    std::optional<std::string> error_message;
};

std::string render_run_manifest_json(const RunManifest& manifest);

}  // namespace nmc::infrastructure::artifacts
