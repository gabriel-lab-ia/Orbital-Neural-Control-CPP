#include "domain/ppo/value_network.h"

#include <cmath>

namespace nmc::domain::ppo {
namespace {

void init_linear(const torch::nn::Linear& layer, const double gain) {
    torch::NoGradGuard no_grad;
    torch::nn::init::orthogonal_(layer->weight, gain);
    torch::nn::init::constant_(layer->bias, 0.0);
}

}  // namespace

ValueNetworkImpl::ValueNetworkImpl(const int64_t observation_dim, const int64_t hidden_dim) {
    encoder_input_ = torch::nn::Linear(observation_dim, hidden_dim);
    encoder_hidden_ = torch::nn::Linear(hidden_dim, hidden_dim);
    critic_ = torch::nn::Linear(hidden_dim, 1);

    register_module("encoder_input", encoder_input_);
    register_module("encoder_hidden", encoder_hidden_);
    register_module("critic", critic_);

    init_linear(encoder_input_, std::sqrt(2.0));
    init_linear(encoder_hidden_, std::sqrt(2.0));
    init_linear(critic_, 1.0);
}

torch::Tensor ValueNetworkImpl::forward_values(const torch::Tensor& observations) {
    return critic_->forward(encode(observations)).squeeze(-1);
}

torch::Tensor ValueNetworkImpl::encode(const torch::Tensor& observations) {
    const auto hidden_1 = torch::silu(encoder_input_->forward(observations));
    return torch::silu(encoder_hidden_->forward(hidden_1) + hidden_1);
}

int64_t ValueNetworkImpl::parameter_count() const {
    int64_t total = 0;
    for (const auto& parameter : parameters()) {
        total += parameter.numel();
    }
    return total;
}

}  // namespace nmc::domain::ppo
