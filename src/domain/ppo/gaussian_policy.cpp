#include "domain/ppo/gaussian_policy.h"

#include <cmath>

namespace nmc::domain::ppo {
namespace {

constexpr float kMinLogStd = -1.2f;
constexpr float kMaxLogStd = 0.35f;
constexpr double kLogTwoPi = 1.8378770664093453;

void init_linear(const torch::nn::Linear& layer, const double gain) {
    torch::NoGradGuard no_grad;
    torch::nn::init::orthogonal_(layer->weight, gain);
    torch::nn::init::constant_(layer->bias, 0.0);
}

}  // namespace

GaussianPolicyImpl::GaussianPolicyImpl(
    const int64_t observation_dim,
    const int64_t action_dim,
    const int64_t hidden_dim
) {
    encoder_input_ = torch::nn::Linear(observation_dim, hidden_dim);
    encoder_hidden_ = torch::nn::Linear(hidden_dim, hidden_dim);
    actor_mean_ = torch::nn::Linear(hidden_dim, action_dim);
    log_std_ = register_parameter("log_std", torch::full({action_dim}, -0.10f));

    register_module("encoder_input", encoder_input_);
    register_module("encoder_hidden", encoder_hidden_);
    register_module("actor_mean", actor_mean_);

    init_linear(encoder_input_, std::sqrt(2.0));
    init_linear(encoder_hidden_, std::sqrt(2.0));
    init_linear(actor_mean_, 0.01);
}

GaussianPolicyDistribution GaussianPolicyImpl::distribution(const torch::Tensor& observations) {
    const auto latent = encode(observations);
    const auto mean = torch::tanh(actor_mean_->forward(latent));
    const auto std = torch::exp(torch::clamp(log_std_, kMinLogStd, kMaxLogStd)).expand_as(mean);
    return {mean, std};
}

torch::Tensor GaussianPolicyImpl::sample_actions(
    const GaussianPolicyDistribution& distribution,
    const bool deterministic
) const {
    auto actions = distribution.mean;
    if (!deterministic) {
        actions = distribution.mean + distribution.std * torch::randn_like(distribution.mean);
    }
    return torch::clamp(actions, -1.0f, 1.0f);
}

torch::Tensor GaussianPolicyImpl::log_prob(
    const torch::Tensor& actions,
    const GaussianPolicyDistribution& distribution
) const {
    const auto variance = distribution.std.pow(2);
    const auto centered = actions - distribution.mean;
    const auto log_probability =
        -0.5 * ((centered.pow(2) / variance) + 2.0 * distribution.std.log() + kLogTwoPi);
    return log_probability.sum(-1);
}

torch::Tensor GaussianPolicyImpl::entropy(const GaussianPolicyDistribution& distribution) const {
    const auto entropy_values = 0.5 + 0.5 * kLogTwoPi + distribution.std.log();
    return entropy_values.sum(-1);
}

torch::Tensor GaussianPolicyImpl::encode(const torch::Tensor& observations) {
    const auto hidden_1 = torch::silu(encoder_input_->forward(observations));
    return torch::silu(encoder_hidden_->forward(hidden_1) + hidden_1);
}

int64_t GaussianPolicyImpl::parameter_count() const {
    int64_t total = 0;
    for (const auto& parameter : parameters()) {
        total += parameter.numel();
    }
    return total;
}

}  // namespace nmc::domain::ppo
