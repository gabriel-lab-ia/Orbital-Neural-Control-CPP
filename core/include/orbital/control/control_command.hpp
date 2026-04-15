#pragma once

#include <array>

namespace orbital::control {

struct ThrusterCommand {
    // Normalized thrust command per axis in [-1, 1].
    std::array<double, 3> thrust_axis{0.0, 0.0, 0.0};
};

}  // namespace orbital::control
