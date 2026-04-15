#pragma once

#include <algorithm>
#include <array>

namespace orbital::control {

class PidController3DOF {
public:
    struct Gains {
        double kp = 0.002;
        double ki = 0.00001;
        double kd = 0.02;
    };

    PidController3DOF() = default;
    explicit PidController3DOF(const Gains& gains) : gains_(gains) {}

    [[nodiscard]] std::array<double, 3> compute(
        const std::array<double, 3>& position_error,
        const std::array<double, 3>& velocity_error,
        const double dt_s
    ) {
        std::array<double, 3> output{};

        for (std::size_t axis = 0; axis < 3; ++axis) {
            integral_error_[axis] += position_error[axis] * dt_s;
            const double derivative = velocity_error[axis];
            const double command =
                gains_.kp * position_error[axis] +
                gains_.ki * integral_error_[axis] +
                gains_.kd * derivative;
            output[axis] = std::clamp(command, -1.0, 1.0);
        }

        return output;
    }

private:
    Gains gains_{};
    std::array<double, 3> integral_error_{0.0, 0.0, 0.0};
};

}  // namespace orbital::control
