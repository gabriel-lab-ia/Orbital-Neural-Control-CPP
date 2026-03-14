#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nmc {

struct StepResult {
    torch::Tensor observation;
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
};

class Environment {
public:
    virtual ~Environment() = default;

    virtual torch::Tensor reset() = 0;
    virtual StepResult step(const torch::Tensor& action) = 0;
    virtual int64_t observation_dim() const = 0;
    virtual int64_t action_dim() const = 0;
};

}  // namespace nmc
