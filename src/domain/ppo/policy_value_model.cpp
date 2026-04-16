#include "domain/ppo/policy_value_model.h"

#include <string>
#include <vector>

namespace nmc::domain::ppo {

PolicyValueModelImpl::PolicyValueModelImpl(
    const int64_t observation_dim,
    const int64_t action_dim,
    const int64_t hidden_dim
) {
    observation_dim_ = observation_dim;
    action_dim_ = action_dim;
    hidden_dim_ = hidden_dim;

    policy_ = GaussianPolicy(observation_dim, action_dim, hidden_dim);
    value_network_ = ValueNetwork(observation_dim, hidden_dim);

    register_module("policy", policy_);
    register_module("value_network", value_network_);
}

PolicyOutput PolicyValueModelImpl::act(const torch::Tensor& observations, const bool deterministic) {
    const auto distribution = policy_->distribution(observations);
    const auto action = policy_->sample_actions(distribution, deterministic);
    const auto log_prob = policy_->log_prob(action, distribution);
    const auto value = value_network_->forward_values(observations);

    return {
        action,
        log_prob,
        value,
        distribution.mean,
        distribution.std
    };
}

std::pair<torch::Tensor, torch::Tensor> PolicyValueModelImpl::evaluate_actions(
    const torch::Tensor& observations,
    const torch::Tensor& actions
) {
    const auto distribution = policy_->distribution(observations);
    return {
        policy_->log_prob(actions, distribution),
        policy_->entropy(distribution)
    };
}

torch::Tensor PolicyValueModelImpl::values(const torch::Tensor& observations) {
    return value_network_->forward_values(observations);
}

torch::Tensor PolicyValueModelImpl::policy_std(const torch::Tensor& observations) {
    return policy_->distribution(observations).std;
}

int64_t PolicyValueModelImpl::parameter_count() const {
    return policy_->parameter_count() + value_network_->parameter_count();
}

std::vector<std::string> PolicyValueModelImpl::visualization_layer_names() const {
    return {
        "observation",
        "policy_hidden",
        "value_hidden",
        "policy_mean",
        "value"
    };
}

std::vector<int64_t> PolicyValueModelImpl::visualization_layer_sizes() const {
    return {
        observation_dim_,
        hidden_dim_,
        hidden_dim_,
        action_dim_,
        1
    };
}

std::vector<torch::Tensor> PolicyValueModelImpl::visualization_weights() {
    std::vector<torch::Tensor> weights;
    for (const auto& parameter : named_parameters()) {
        if (parameter.key().find("weight") != std::string::npos) {
            weights.push_back(parameter.value().detach().to(torch::kCPU).clone());
        }
    }
    return weights;
}

std::vector<torch::Tensor> PolicyValueModelImpl::visualization_activations(const torch::Tensor& observations) {
    auto batch = observations;
    if (batch.dim() == 1) {
        batch = batch.unsqueeze(0);
    }

    const auto input = batch.squeeze(0).detach().to(torch::kCPU).clone();
    const auto policy_hidden = policy_->encode(batch).squeeze(0);
    const auto value_hidden = value_network_->encode(batch).squeeze(0);
    const auto policy_mean = policy_->distribution(batch).mean.squeeze(0);
    const auto value = value_network_->forward_values(batch).squeeze(0);

    return {
        input,
        policy_hidden.detach().to(torch::kCPU).clone(),
        value_hidden.detach().to(torch::kCPU).clone(),
        policy_mean.detach().to(torch::kCPU).clone(),
        value.detach().to(torch::kCPU).clone()
    };
}

float PolicyValueModelImpl::policy_std_scalar() {
    const auto std = policy_->distribution(torch::zeros({1, observation_dim_}, torch::TensorOptions().dtype(torch::kFloat32))).std;
    return std.mean().item<float>();
}

}  // namespace nmc::domain::ppo
