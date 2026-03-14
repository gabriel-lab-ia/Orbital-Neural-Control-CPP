#include "train/ppo_trainer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>

namespace nmc {
namespace {

float mean_or_zero(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }

    const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / static_cast<float>(values.size());
}

float mean_or_zero(const std::vector<int64_t>& values) {
    if (values.empty()) {
        return 0.0f;
    }

    const auto sum = std::accumulate(values.begin(), values.end(), int64_t{0});
    return static_cast<float>(sum) / static_cast<float>(values.size());
}

torch::Tensor normalize_advantages(const torch::Tensor& advantages) {
    const auto centered = advantages - advantages.mean();
    const auto scale = centered.std(false).clamp_min(1.0e-6);
    return centered / scale;
}

}  // namespace

PPOTrainer::PPOTrainer(
    TrainerConfig config,
    std::filesystem::path artifact_dir,
    EnvironmentPack environment_pack
)
    : config_(config),
      artifact_dir_(std::move(artifact_dir)),
      agent_(nullptr) {
    torch::manual_seed(static_cast<uint64_t>(config_.seed));

    agent_ = PPOAgent(
        environment_pack.observation_dim,
        environment_pack.action_dim,
        config_.hidden_dim
    );
    agent_->to(device_);
    optimizer_ =
        std::make_unique<torch::optim::Adam>(
            agent_->parameters(),
            torch::optim::AdamOptions(config_.learning_rate)
        );

    environments_ = std::move(environment_pack.environments);
    environments_.reserve(static_cast<std::size_t>(config_.num_envs));
    current_observations_.reserve(static_cast<std::size_t>(config_.num_envs));
    episode_returns_.assign(static_cast<std::size_t>(config_.num_envs), 0.0f);
    episode_lengths_.assign(static_cast<std::size_t>(config_.num_envs), 0);
    finished_successes_.reserve(static_cast<std::size_t>(config_.num_envs * config_.total_updates));

    for (auto& environment : environments_) {
        current_observations_.push_back(environment->reset());
    }
}

std::vector<TrainingMetrics> PPOTrainer::train() {
    std::vector<TrainingMetrics> metrics;
    metrics.reserve(static_cast<std::size_t>(config_.total_updates));

    for (int64_t update = 1; update <= config_.total_updates; ++update) {
        auto batch = collect_rollout();
        metrics.push_back(update_policy(batch, update));
    }

    return metrics;
}

std::vector<PPOTrainer::LiveStep> PPOTrainer::run_live_episode(int64_t max_steps, std::ostream& stream) {
    if (environments_.empty()) {
        return {};
    }

    agent_->eval();
    std::vector<LiveStep> steps;
    steps.reserve(static_cast<std::size_t>(std::max<int64_t>(max_steps, 0)));

    auto observation = environments_.front()->reset();
    float episode_return = 0.0f;

    stream << "\nLive neural control\n";
    stream << "-------------------\n";

    for (int64_t step_index = 0; step_index < max_steps; ++step_index) {
        const auto batch_observation = observation.unsqueeze(0).to(device_);
        const auto policy = agent_->act(batch_observation, true);
        const auto result = environments_.front()->step(policy.action[0].to(torch::kCPU));
        const auto next_observation = result.observation.to(torch::kCPU);
        episode_return += result.reward;

        LiveStep step;
        step.step = step_index + 1;
        step.reward = result.reward;
        step.action = policy.action[0][0].item<float>();
        step.value = policy.value[0].item<float>();
        step.observation.reserve(static_cast<std::size_t>(next_observation.numel()));
        for (int64_t index = 0; index < next_observation.numel(); ++index) {
            step.observation.push_back(next_observation[index].item<float>());
        }
        step.terminated = result.terminated;
        step.truncated = result.truncated;
        steps.push_back(step);

        const auto observation_0 = step.observation.size() > 0 ? step.observation[0] : 0.0f;
        const auto observation_1 = step.observation.size() > 1 ? step.observation[1] : 0.0f;
        const auto observation_2 = step.observation.size() > 2 ? step.observation[2] : 0.0f;
        const auto observation_3 = step.observation.size() > 3 ? step.observation[3] : 0.0f;

        stream
            << "step " << std::setw(3) << step.step
            << " | action=" << std::setw(8) << std::fixed << std::setprecision(4) << step.action
            << " | value=" << std::setw(8) << step.value
            << " | reward=" << std::setw(8) << step.reward
            << " | obs=[" << observation_0
            << ", " << observation_1
            << ", " << observation_2
            << ", " << observation_3 << "]"
            << '\n';

        observation = next_observation;
        if (result.terminated || result.truncated) {
            stream << "episode finished | return=" << episode_return << '\n';
            break;
        }
    }

    return steps;
}

PPOAgent& PPOTrainer::agent() {
    return agent_;
}

RolloutBatch PPOTrainer::collect_rollout() {
    std::vector<torch::Tensor> observations;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> log_probs;
    std::vector<torch::Tensor> rewards;
    std::vector<torch::Tensor> dones;
    std::vector<torch::Tensor> values;

    observations.reserve(static_cast<std::size_t>(config_.rollout_steps));
    actions.reserve(static_cast<std::size_t>(config_.rollout_steps));
    log_probs.reserve(static_cast<std::size_t>(config_.rollout_steps));
    rewards.reserve(static_cast<std::size_t>(config_.rollout_steps));
    dones.reserve(static_cast<std::size_t>(config_.rollout_steps));
    values.reserve(static_cast<std::size_t>(config_.rollout_steps));

    agent_->eval();

    for (int64_t step = 0; step < config_.rollout_steps; ++step) {
        auto observation_batch = stack_observations().to(device_);
        const auto policy = agent_->act(observation_batch, false);

        observations.push_back(observation_batch);
        actions.push_back(policy.action.detach());
        log_probs.push_back(policy.log_prob.detach());
        values.push_back(policy.value.detach());

        std::vector<torch::Tensor> next_observations;
        std::vector<float> reward_values;
        std::vector<float> done_values;

        next_observations.reserve(static_cast<std::size_t>(config_.num_envs));
        reward_values.reserve(static_cast<std::size_t>(config_.num_envs));
        done_values.reserve(static_cast<std::size_t>(config_.num_envs));

        for (int64_t env_index = 0; env_index < config_.num_envs; ++env_index) {
            auto result = environments_[static_cast<std::size_t>(env_index)]->step(
                policy.action[env_index].to(torch::kCPU)
            );

            const bool done = result.terminated || result.truncated;
            reward_values.push_back(result.reward);
            done_values.push_back(done ? 1.0f : 0.0f);
            episode_returns_[static_cast<std::size_t>(env_index)] += result.reward;
            episode_lengths_[static_cast<std::size_t>(env_index)] += 1;

            if (done) {
                finished_returns_.push_back(episode_returns_[static_cast<std::size_t>(env_index)]);
                finished_lengths_.push_back(episode_lengths_[static_cast<std::size_t>(env_index)]);
                finished_successes_.push_back(result.terminated ? 1.0f : 0.0f);
                episode_returns_[static_cast<std::size_t>(env_index)] = 0.0f;
                episode_lengths_[static_cast<std::size_t>(env_index)] = 0;
                next_observations.push_back(environments_[static_cast<std::size_t>(env_index)]->reset());
            } else {
                next_observations.push_back(result.observation);
            }
        }

        current_observations_ = next_observations;
        rewards.push_back(torch::tensor(reward_values, torch::TensorOptions().dtype(torch::kFloat32)).to(device_));
        dones.push_back(torch::tensor(done_values, torch::TensorOptions().dtype(torch::kFloat32)).to(device_));
        total_env_steps_ += config_.num_envs;
    }

    const auto last_values = agent_->values(stack_observations().to(device_)).detach();
    auto advantages = torch::zeros_like(last_values);
    std::vector<torch::Tensor> advantage_steps(static_cast<std::size_t>(config_.rollout_steps));
    std::vector<torch::Tensor> return_steps(static_cast<std::size_t>(config_.rollout_steps));

    for (int64_t step = config_.rollout_steps - 1; step >= 0; --step) {
        const auto mask = 1.0f - dones[static_cast<std::size_t>(step)];
        const auto next_value =
            step == config_.rollout_steps - 1
                ? last_values
                : values[static_cast<std::size_t>(step + 1)];
        const auto delta =
            rewards[static_cast<std::size_t>(step)] +
            config_.gamma * next_value * mask -
            values[static_cast<std::size_t>(step)];

        advantages =
            delta +
            (config_.gamma * config_.gae_lambda) * mask * advantages;

        advantage_steps[static_cast<std::size_t>(step)] = advantages;
        return_steps[static_cast<std::size_t>(step)] =
            advantages + values[static_cast<std::size_t>(step)];
    }

    auto batch = RolloutBatch{
        torch::cat(observations).detach(),
        torch::cat(actions).detach(),
        torch::cat(log_probs).detach(),
        torch::cat(rewards).detach(),
        torch::cat(return_steps).detach(),
        normalize_advantages(torch::cat(advantage_steps).detach()),
        torch::cat(values).detach()
    };

    return batch;
}

TrainingMetrics PPOTrainer::update_policy(const RolloutBatch& batch, int64_t update_index) {
    agent_->train();

    const auto sample_count = batch.observations.size(0);
    auto permutation = torch::randperm(sample_count, torch::TensorOptions().dtype(torch::kInt64));

    float last_policy_loss = 0.0f;
    float last_value_loss = 0.0f;
    float last_entropy = 0.0f;
    float last_approx_kl = 0.0f;
    float last_clip_fraction = 0.0f;
    float last_action_std = 0.0f;

    for (int64_t epoch = 0; epoch < config_.ppo_epochs; ++epoch) {
        permutation = torch::randperm(sample_count, torch::TensorOptions().dtype(torch::kInt64));

        for (int64_t start = 0; start < sample_count; start += config_.minibatch_size) {
            const auto stop = std::min(start + config_.minibatch_size, sample_count);
            const auto indices = permutation.slice(0, start, stop);

            const auto obs = batch.observations.index_select(0, indices);
            const auto actions = batch.actions.index_select(0, indices);
            const auto old_log_probs = batch.log_probs.index_select(0, indices);
            const auto returns = batch.returns.index_select(0, indices);
            const auto advantages = batch.advantages.index_select(0, indices);

            const auto [new_log_probs, entropy] = agent_->evaluate_actions(obs, actions);
            const auto values = agent_->values(obs);
            const auto log_ratio = new_log_probs - old_log_probs;
            const auto ratio = torch::exp(log_ratio);
            const auto clipped_ratio =
                torch::clamp(ratio, 1.0f - config_.clip_epsilon, 1.0f + config_.clip_epsilon);

            const auto surrogate_1 = ratio * advantages;
            const auto surrogate_2 = clipped_ratio * advantages;
            const auto policy_loss = -torch::minimum(surrogate_1, surrogate_2).mean();
            const auto value_loss = torch::mse_loss(values, returns);
            const auto entropy_bonus = entropy.mean();
            const auto total_loss =
                policy_loss +
                config_.value_loss_weight * value_loss -
                config_.entropy_weight * entropy_bonus;

            optimizer_->zero_grad();
            total_loss.backward();
            torch::nn::utils::clip_grad_norm_(agent_->parameters(), config_.max_grad_norm);
            optimizer_->step();

            last_policy_loss = policy_loss.item<float>();
            last_value_loss = value_loss.item<float>();
            last_entropy = entropy_bonus.item<float>();
            last_approx_kl = ((ratio - 1.0f) - log_ratio).mean().item<float>();
            last_clip_fraction =
                ((ratio < (1.0f - config_.clip_epsilon)) | (ratio > (1.0f + config_.clip_epsilon)))
                    .to(torch::kFloat32)
                    .mean()
                    .item<float>();
            last_action_std = agent_->act(obs, true).std_action.mean().item<float>();
        }
    }

    const int64_t metric_window = std::min<int64_t>(20, static_cast<int64_t>(finished_returns_.size()));
    std::vector<float> reward_window;
    std::vector<int64_t> length_window;
    std::vector<float> success_window;

    if (metric_window > 0) {
        reward_window.assign(
            finished_returns_.end() - metric_window,
            finished_returns_.end()
        );
        length_window.assign(
            finished_lengths_.end() - metric_window,
            finished_lengths_.end()
        );
        success_window.assign(
            finished_successes_.end() - metric_window,
            finished_successes_.end()
        );
    }

    const float step_reward = batch.rewards.mean().item<float>();
    const float success_rate = mean_or_zero(success_window);

    return {
        update_index,
        total_env_steps_,
        last_policy_loss,
        last_value_loss,
        last_entropy,
        last_approx_kl,
        last_clip_fraction,
        step_reward,
        mean_or_zero(reward_window),
        mean_or_zero(length_window),
        success_rate,
        last_action_std
    };
}

torch::Tensor PPOTrainer::stack_observations() const {
    return torch::stack(current_observations_);
}

}  // namespace nmc
