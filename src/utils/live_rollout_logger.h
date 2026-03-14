#pragma once

#include <filesystem>
#include <vector>

#include "train/ppo_trainer.h"

namespace nmc {

void write_live_rollout_csv(
    const std::filesystem::path& path,
    const std::vector<PPOTrainer::LiveStep>& steps
);

}  // namespace nmc
