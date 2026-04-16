#pragma once

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "domain/ppo/gaussian_policy.h"
#include "domain/ppo/value_network.h"

namespace nmc::domain::ppo {

struct PolicyOutput {
    torch::Tensor action;
    torch::Tensor log_prob;
    torch::Tensor value;
    torch::Tensor mean_action;
    torch::Tensor std_action;
};

class PolicyValueModelImpl : public torch::nn::Module {
public:
    PolicyValueModelImpl(int64_t observation_dim, int64_t action_dim, int64_t hidden_dim = 128);

    PolicyOutput act(const torch::Tensor& observations, bool deterministic);
    std::pair<torch::Tensor, torch::Tensor> evaluate_actions(
        const torch::Tensor& observations,
        const torch::Tensor& actions
    );
    torch::Tensor values(const torch::Tensor& observations);
    torch::Tensor policy_std(const torch::Tensor& observations);
    int64_t parameter_count() const;

    std::vector<std::string> visualization_layer_names() const;
    std::vector<int64_t> visualization_layer_sizes() const;
    std::vector<torch::Tensor> visualization_weights();
    std::vector<torch::Tensor> visualization_activations(const torch::Tensor& observations);
    float policy_std_scalar();

private:
    GaussianPolicy policy_{nullptr};
    ValueNetwork value_network_{nullptr};
    int64_t observation_dim_ = 0;
    int64_t action_dim_ = 0;
    int64_t hidden_dim_ = 0;
};

TORCH_MODULE(PolicyValueModel);

}  // namespace nmc::domain::ppo
