#pragma once

#include <algorithm>
#include <array>

namespace orbital::control {

class LqrBaselineController3DOF {
public:
    struct Gains {
        double k_position = 0.0018;
        double k_velocity = 0.03;
    };

    LqrBaselineController3DOF() = default;
    explicit LqrBaselineController3DOF(const Gains& gains) : gains_(gains) {}

    [[nodiscard]] std::array<double, 3> compute(
        const std::array<double, 3>& position_error,
        const std::array<double, 3>& velocity_error
    ) const {
        std::array<double, 3> command{};

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double value =
                gains_.k_position * position_error[axis] +
                gains_.k_velocity * velocity_error[axis];
            command[axis] = std::clamp(value, -1.0, 1.0);
        }

        return command;
    }

private:
    Gains gains_{};
};

}  // namespace orbital::control
