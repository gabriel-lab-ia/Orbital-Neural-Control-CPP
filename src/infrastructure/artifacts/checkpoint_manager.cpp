#include "infrastructure/artifacts/checkpoint_manager.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nmc::infrastructure::artifacts {

std::filesystem::path metadata_path_for(const std::filesystem::path& checkpoint_path) {
    return checkpoint_path.string() + ".meta";
}

void save_policy_checkpoint(
    const std::filesystem::path& checkpoint_path,
    const std::filesystem::path& metadata_path,
    domain::ppo::PolicyValueModel& model,
    const CheckpointMetadata& metadata
) {
    std::filesystem::create_directories(checkpoint_path.parent_path());
    std::filesystem::create_directories(metadata_path.parent_path());

    torch::save(model, checkpoint_path.string());

    std::ofstream stream(metadata_path, std::ios::out | std::ios::trunc);
    if (!stream.is_open()) {
        throw std::runtime_error("unable to write checkpoint metadata: " + metadata_path.string());
    }

    stream << "run_id=" << metadata.run_id << '\n';
    stream << "environment=" << metadata.environment << '\n';
    stream << "observation_dim=" << metadata.observation_dim << '\n';
    stream << "action_dim=" << metadata.action_dim << '\n';
    stream << "hidden_dim=" << metadata.hidden_dim << '\n';
    stream << "seed=" << metadata.seed << '\n';
    stream << "created_at=" << metadata.created_at << '\n';
}

void load_policy_checkpoint(const std::filesystem::path& checkpoint_path, domain::ppo::PolicyValueModel& model) {
    if (!std::filesystem::exists(checkpoint_path)) {
        throw std::runtime_error("checkpoint not found: " + checkpoint_path.string());
    }
    torch::load(model, checkpoint_path.string());
}

CheckpointMetadata load_checkpoint_metadata(const std::filesystem::path& metadata_path) {
    std::ifstream stream(metadata_path);
    if (!stream.is_open()) {
        throw std::runtime_error("checkpoint metadata not found: " + metadata_path.string());
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(stream, line)) {
        const auto split = line.find('=');
        if (split == std::string::npos) {
            continue;
        }
        kv[line.substr(0, split)] = line.substr(split + 1);
    }

    const auto read_required = [&](const char* key) -> std::string {
        const auto it = kv.find(key);
        if (it == kv.end() || it->second.empty()) {
            throw std::runtime_error(
                std::string("checkpoint metadata missing required field '") + key +
                "' in " + metadata_path.string()
            );
        }
        return it->second;
    };

    const auto parse_int64 = [&](const char* key) -> int64_t {
        const auto raw = read_required(key);
        try {
            return std::stoll(raw);
        } catch (const std::exception&) {
            throw std::runtime_error(
                std::string("checkpoint metadata field '") + key +
                "' is not a valid integer in " + metadata_path.string() + ": " + raw
            );
        }
    };

    CheckpointMetadata metadata;
    metadata.run_id = read_required("run_id");
    metadata.environment = read_required("environment");
    metadata.observation_dim = parse_int64("observation_dim");
    metadata.action_dim = parse_int64("action_dim");
    metadata.hidden_dim = parse_int64("hidden_dim");
    metadata.seed = parse_int64("seed");
    metadata.created_at = read_required("created_at");
    return metadata;
}

}  // namespace nmc::infrastructure::artifacts
