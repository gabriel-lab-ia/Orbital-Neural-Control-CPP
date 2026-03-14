#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "env/environment.h"

namespace nmc {

struct EnvironmentSelection {
    std::string kind = "point_mass";
    std::filesystem::path mujoco_model_path;
};

struct EnvironmentPack {
    std::string display_name;
    int64_t observation_dim = 0;
    int64_t action_dim = 0;
    std::vector<std::unique_ptr<Environment>> environments;
};

EnvironmentPack make_environment_pack(const EnvironmentSelection& selection, int64_t num_envs);
bool mujoco_support_enabled();

}  // namespace nmc
