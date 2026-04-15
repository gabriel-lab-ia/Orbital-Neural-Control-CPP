#pragma once

#include <array>

namespace orbital::sim {

struct DisturbanceProfile {
    std::array<double, 3> constant_bias_mps2{0.0, 0.0, 0.0};
};

class PerturbationModel3DOF {
public:
    explicit PerturbationModel3DOF(DisturbanceProfile profile = {}) : profile_(profile) {}

    [[nodiscard]] std::array<double, 3> acceleration_bias() const {
        return profile_.constant_bias_mps2;
    }

private:
    DisturbanceProfile profile_{};
};

}  // namespace orbital::sim
