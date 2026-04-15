#include <array>
#include <cassert>

#include "orbital/control/lqr_controller.hpp"
#include "orbital/control/pid_controller.hpp"

int main() {
    orbital::control::LqrBaselineController3DOF lqr;
    orbital::control::PidController3DOF pid;

    const std::array<double, 3> position_error{100.0, -50.0, 25.0};
    const std::array<double, 3> velocity_error{-5.0, 2.5, -1.25};

    const auto lqr_command = lqr.compute(position_error, velocity_error);
    const auto pid_command = pid.compute(position_error, velocity_error, 0.1);

    for (std::size_t axis = 0; axis < 3; ++axis) {
        assert(lqr_command[axis] <= 1.0 && lqr_command[axis] >= -1.0);
        assert(pid_command[axis] <= 1.0 && pid_command[axis] >= -1.0);
    }

    return 0;
}
