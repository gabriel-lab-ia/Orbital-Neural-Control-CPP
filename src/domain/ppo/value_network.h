#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nmc::domain::ppo {

class ValueNetworkImpl : public torch::nn::Module {
public:
    ValueNetworkImpl(int64_t observation_dim, int64_t hidden_dim = 128);

    torch::Tensor forward_values(const torch::Tensor& observations);
    torch::Tensor encode(const torch::Tensor& observations);
    int64_t parameter_count() const;

private:
    torch::nn::Linear encoder_input_{nullptr};
    torch::nn::Linear encoder_hidden_{nullptr};
    torch::nn::Linear critic_{nullptr};
};

TORCH_MODULE(ValueNetwork);

}  // namespace nmc::domain::ppo
