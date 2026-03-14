#pragma once

#include <filesystem>

#include <mujoco/mujoco.h>

#include "env/environment.h"

namespace nmc {

class MuJoCoCartPoleEnv final : public Environment {
public:
    explicit MuJoCoCartPoleEnv(const std::filesystem::path& model_path);
    ~MuJoCoCartPoleEnv() override;

    MuJoCoCartPoleEnv(const MuJoCoCartPoleEnv&) = delete;
    MuJoCoCartPoleEnv& operator=(const MuJoCoCartPoleEnv&) = delete;

    torch::Tensor reset() override;
    StepResult step(const torch::Tensor& action) override;
    int64_t observation_dim() const override;
    int64_t action_dim() const override;

private:
    torch::Tensor make_observation() const;

    std::filesystem::path model_path_;
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int64_t step_count_ = 0;
};

}  // namespace nmc
