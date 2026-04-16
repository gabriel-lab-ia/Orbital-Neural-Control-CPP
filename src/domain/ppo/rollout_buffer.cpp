#include "domain/ppo/rollout_buffer.h"

#include <stdexcept>

namespace nmc::domain::ppo {

RolloutBuffer::RolloutBuffer(
    const int64_t rollout_steps,
    const int64_t num_envs,
    const int64_t observation_dim,
    const int64_t action_dim,
    const torch::Device device
) : rollout_steps_(rollout_steps),
    num_envs_(num_envs),
    observation_dim_(observation_dim),
    action_dim_(action_dim),
    device_(device) {
    if (rollout_steps_ <= 0 || num_envs_ <= 0 || observation_dim_ <= 0 || action_dim_ <= 0) {
        throw std::runtime_error("invalid rollout buffer dimensions");
    }
    reset();
}

void RolloutBuffer::reset() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    observations_ = torch::zeros({rollout_steps_, num_envs_, observation_dim_}, options);
    actions_ = torch::zeros({rollout_steps_, num_envs_, action_dim_}, options);
    log_probs_ = torch::zeros({rollout_steps_, num_envs_}, options);
    rewards_ = torch::zeros({rollout_steps_, num_envs_}, options);
    dones_ = torch::zeros({rollout_steps_, num_envs_}, options);
    values_ = torch::zeros({rollout_steps_, num_envs_}, options);
}

void RolloutBuffer::add_step(
    const int64_t step_index,
    const torch::Tensor& observations,
    const torch::Tensor& actions,
    const torch::Tensor& log_probs,
    const torch::Tensor& values,
    const std::span<const float> rewards,
    const std::span<const float> dones
) {
    if (step_index < 0 || step_index >= rollout_steps_) {
        throw std::runtime_error("rollout buffer step index out of bounds");
    }
    if (static_cast<int64_t>(rewards.size()) != num_envs_ || static_cast<int64_t>(dones.size()) != num_envs_) {
        throw std::runtime_error("rollout buffer rewards/dones size mismatch");
    }

    observations_.select(0, step_index).copy_(observations);
    actions_.select(0, step_index).copy_(actions);
    log_probs_.select(0, step_index).copy_(log_probs);
    values_.select(0, step_index).copy_(values);

    auto rewards_tensor = torch::from_blob(
        const_cast<float*>(rewards.data()),
        {num_envs_},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(device_);
    auto dones_tensor = torch::from_blob(
        const_cast<float*>(dones.data()),
        {num_envs_},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(device_);

    rewards_.select(0, step_index).copy_(rewards_tensor);
    dones_.select(0, step_index).copy_(dones_tensor);
}

RolloutBatch RolloutBuffer::build_batch(
    const torch::Tensor& last_values,
    const float gamma,
    const float gae_lambda
) const {
    auto advantages = torch::zeros({rollout_steps_, num_envs_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto gae = torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    for (int64_t step = rollout_steps_ - 1; step >= 0; --step) {
        const auto mask = 1.0f - dones_.select(0, step);
        const auto next_values = (step == rollout_steps_ - 1) ? last_values : values_.select(0, step + 1);
        const auto delta = rewards_.select(0, step) + gamma * next_values * mask - values_.select(0, step);
        gae = delta + (gamma * gae_lambda) * mask * gae;
        advantages.select(0, step).copy_(gae);
    }

    const auto returns = advantages + values_;
    const auto flat_size = rollout_steps_ * num_envs_;

    return {
        observations_.reshape({flat_size, observation_dim_}).detach(),
        actions_.reshape({flat_size, action_dim_}).detach(),
        log_probs_.reshape({flat_size}).detach(),
        rewards_.reshape({flat_size}).detach(),
        returns.reshape({flat_size}).detach(),
        normalize_advantages(advantages.reshape({flat_size}).detach()),
        values_.reshape({flat_size}).detach()
    };
}

torch::Tensor RolloutBuffer::normalize_advantages(const torch::Tensor& advantages) {
    const auto centered = advantages - advantages.mean();
    const auto scale = centered.std(false).clamp_min(1.0e-6);
    return centered / scale;
}

}  // namespace nmc::domain::ppo
