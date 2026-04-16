#pragma once

#include <cstdint>
#include <span>

#include <torch/torch.h>

#include "domain/ppo/ppo_types.h"

namespace nmc::domain::ppo {

class RolloutBuffer {
public:
    RolloutBuffer(
        int64_t rollout_steps,
        int64_t num_envs,
        int64_t observation_dim,
        int64_t action_dim,
        torch::Device device
    );

    void reset();
    void add_step(
        int64_t step_index,
        const torch::Tensor& observations,
        const torch::Tensor& actions,
        const torch::Tensor& log_probs,
        const torch::Tensor& values,
        std::span<const float> rewards,
        std::span<const float> dones
    );

    RolloutBatch build_batch(const torch::Tensor& last_values, float gamma, float gae_lambda) const;

private:
    static torch::Tensor normalize_advantages(const torch::Tensor& advantages);

    int64_t rollout_steps_ = 0;
    int64_t num_envs_ = 0;
    int64_t observation_dim_ = 0;
    int64_t action_dim_ = 0;
    torch::Device device_{torch::kCPU};

    torch::Tensor observations_;
    torch::Tensor actions_;
    torch::Tensor log_probs_;
    torch::Tensor rewards_;
    torch::Tensor dones_;
    torch::Tensor values_;
};

}  // namespace nmc::domain::ppo
