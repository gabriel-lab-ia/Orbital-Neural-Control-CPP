#pragma once

#include <cstdint>
#include <utility>

#include <torch/torch.h>

namespace nmc::domain::ppo {

struct GaussianPolicyDistribution {
    torch::Tensor mean;
    torch::Tensor std;
};

class GaussianPolicyImpl : public torch::nn::Module {
public:
    GaussianPolicyImpl(int64_t observation_dim, int64_t action_dim, int64_t hidden_dim = 128);

    GaussianPolicyDistribution distribution(const torch::Tensor& observations);
    torch::Tensor sample_actions(const GaussianPolicyDistribution& distribution, bool deterministic) const;
    torch::Tensor log_prob(const torch::Tensor& actions, const GaussianPolicyDistribution& distribution) const;
    torch::Tensor entropy(const GaussianPolicyDistribution& distribution) const;

    torch::Tensor encode(const torch::Tensor& observations);
    int64_t parameter_count() const;

private:
    torch::nn::Linear encoder_input_{nullptr};
    torch::nn::Linear encoder_hidden_{nullptr};
    torch::nn::Linear actor_mean_{nullptr};
    torch::Tensor log_std_;
};

TORCH_MODULE(GaussianPolicy);

}  // namespace nmc::domain::ppo
