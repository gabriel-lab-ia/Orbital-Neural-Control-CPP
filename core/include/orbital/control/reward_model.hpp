#pragma once

#include <array>
#include <cmath>

#include "orbital/control/control_command.hpp"
#include "orbital/simulation/orbital_state.hpp"

namespace orbital::control {

struct MissionTarget {
    std::array<double, 3> target_position_m{0.0, 0.0, 0.0};
    std::array<double, 3> target_velocity_mps{0.0, 0.0, 0.0};
};

struct RewardWeights {
    double position_weight = 1.0;
    double velocity_weight = 0.2;
    double control_weight = 0.01;
    double terminal_bonus = 5.0;
};

class RewardModel {
public:
    explicit RewardModel(RewardWeights weights = {}) : weights_(weights) {}

    [[nodiscard]] double compute_step_reward(
        const simulation::OrbitalState3DOF& state,
        const ThrusterCommand& command,
        const MissionTarget& target,
        const bool terminal_success
    ) const {
        const double position_error = l2_norm(delta(state.position_m, target.target_position_m));
        const double velocity_error = l2_norm(delta(state.velocity_mps, target.target_velocity_mps));
        const double control_cost = l2_norm(command.thrust_axis);

        const double shaping =
            -weights_.position_weight * position_error
            -weights_.velocity_weight * velocity_error
            -weights_.control_weight * control_cost;

        return shaping + (terminal_success ? weights_.terminal_bonus : 0.0);
    }

private:
    static std::array<double, 3> delta(
        const std::array<double, 3>& lhs,
        const std::array<double, 3>& rhs
    ) {
        return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
    }

    static double l2_norm(const std::array<double, 3>& values) {
        return std::sqrt(
            values[0] * values[0] +
            values[1] * values[1] +
            values[2] * values[2]
        );
    }

    RewardWeights weights_{};
};

}  // namespace orbital::control
