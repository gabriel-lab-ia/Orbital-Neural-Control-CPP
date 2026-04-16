#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace orbital::rl::ppo {

inline torch::Tensor compute_gae(
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    const torch::Tensor& values,
    const torch::Tensor& last_values,
    const float gamma,
    const float gae_lambda
) {
    const auto rollout_steps = rewards.size(0);
    auto advantages = torch::zeros_like(values);
    auto gae = torch::zeros_like(last_values);

    for (int64_t step = rollout_steps - 1; step >= 0; --step) {
        const auto mask = 1.0f - dones.select(0, step);
        const auto next_values = (step == rollout_steps - 1) ? last_values : values.select(0, step + 1);
        const auto delta = rewards.select(0, step) + gamma * next_values * mask - values.select(0, step);
        gae = delta + gamma * gae_lambda * mask * gae;
        advantages.select(0, step).copy_(gae);
    }

    return advantages;
}

}  // namespace orbital::rl::ppo
