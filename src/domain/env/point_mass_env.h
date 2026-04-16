#pragma once

#include "domain/env/environment.h"
#include "domain/env/point_mass_reward.h"

namespace nmc::domain::env {

class PointMassEnv final : public Environment {
public:
    explicit PointMassEnv(PointMassRewardConfig reward_config = {});

    torch::Tensor reset() override;
    StepResult step(const torch::Tensor& action) override;
    int64_t observation_dim() const override;
    int64_t action_dim() const override;
    std::string name() const override;
    float success_signal(const StepResult& result) const override;

private:
    float compute_reward(float force, float position_error, float previous_potential);
    float potential(float position_error, float velocity) const;

    torch::Tensor make_observation() const;

    PointMassRewardConfig reward_config_{};
    float position_ = 0.0f;
    float velocity_ = 0.0f;
    float target_ = 0.0f;
    float previous_potential_ = 0.0f;
    int64_t step_count_ = 0;
    int64_t stable_steps_ = 0;
};

}  // namespace nmc::domain::env
