#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "orbital/control/control_command.hpp"
#include "orbital/simulation/orbital_state.hpp"

namespace orbital::simulation {

struct DynamicsConfig {
    double thrust_accel_scale_mps2 = 0.15;
    double drag_damping = 0.001;
    double max_speed_mps = 250.0;
};

class OrbitalDynamics3DOF {
public:
    explicit OrbitalDynamics3DOF(DynamicsConfig config = {}) : config_(config) {}

    [[nodiscard]] OrbitalState3DOF propagate(
        const OrbitalState3DOF& state,
        const control::ThrusterCommand& command,
        const double dt_s
    ) const {
        OrbitalState3DOF next = state;

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double bounded_thrust = std::clamp(command.thrust_axis[axis], -1.0, 1.0);
            const double acceleration = config_.thrust_accel_scale_mps2 * bounded_thrust;
            next.velocity_mps[axis] += acceleration * dt_s;
            next.velocity_mps[axis] -= config_.drag_damping * next.velocity_mps[axis] * dt_s;
            next.velocity_mps[axis] = std::clamp(
                next.velocity_mps[axis],
                -config_.max_speed_mps,
                config_.max_speed_mps
            );
            next.position_m[axis] += next.velocity_mps[axis] * dt_s;
        }

        next.mission_time_s += dt_s;
        return next;
    }

private:
    DynamicsConfig config_{};
};

}  // namespace orbital::simulation
