#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "model/ppo_agent.h"
#include "train/ppo_trainer.h"

namespace nmc {

void write_neural_3d_visualization(
    const std::filesystem::path& html_path,
    const std::filesystem::path& json_path,
    const std::string& environment_name,
    PPOAgent& agent,
    const std::vector<PPOTrainer::LiveStep>& live_steps
);

}  // namespace nmc
