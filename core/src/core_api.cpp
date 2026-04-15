#include "orbital/core_api.hpp"

#include <algorithm>
#include <cmath>

namespace orbital {
namespace {

control::ThrusterCommand proportional_guidance(
    const simulation::OrbitalState3DOF& state,
    const control::MissionTarget& target
) {
    control::ThrusterCommand command;
    constexpr double kPositionGain = 0.0008;
    constexpr double kVelocityGain = 0.015;

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double position_error = target.target_position_m[axis] - state.position_m[axis];
        const double velocity_error = target.target_velocity_mps[axis] - state.velocity_mps[axis];
        const double control_signal = (kPositionGain * position_error) + (kVelocityGain * velocity_error);
        command.thrust_axis[axis] = std::clamp(control_signal, -1.0, 1.0);
    }

    return command;
}

bool is_terminal_success(
    const simulation::OrbitalState3DOF& state,
    const control::MissionTarget& target
) {
    constexpr double kPositionToleranceM = 3.5;
    constexpr double kVelocityToleranceMps = 0.25;

    const auto position_error = std::sqrt(
        std::pow(state.position_m[0] - target.target_position_m[0], 2.0) +
        std::pow(state.position_m[1] - target.target_position_m[1], 2.0) +
        std::pow(state.position_m[2] - target.target_position_m[2], 2.0)
    );

    const auto velocity_error = std::sqrt(
        std::pow(state.velocity_mps[0] - target.target_velocity_mps[0], 2.0) +
        std::pow(state.velocity_mps[1] - target.target_velocity_mps[1], 2.0) +
        std::pow(state.velocity_mps[2] - target.target_velocity_mps[2], 2.0)
    );

    return position_error <= kPositionToleranceM && velocity_error <= kVelocityToleranceMps;
}

}  // namespace

OrbitalControlCore::OrbitalControlCore(simulation::DynamicsConfig dynamics, control::RewardWeights reward)
    : dynamics_(dynamics), reward_model_(reward) {}

MissionRolloutResult OrbitalControlCore::run_open_loop_rollout(
    const std::string& mission_id,
    const simulation::OrbitalState3DOF& initial_state,
    const control::MissionTarget& target,
    const std::int64_t total_steps,
    const double dt_s
) const {
    MissionRolloutResult result;
    result.mission_id = mission_id;
    result.timeline.reserve(static_cast<std::size_t>(std::max<std::int64_t>(0, total_steps)));

    auto state = initial_state;

    for (std::int64_t step = 0; step < total_steps; ++step) {
        const auto command = proportional_guidance(state, target);
        state = dynamics_.propagate(state, command, dt_s);

        const bool success = is_terminal_success(state, target);
        const double reward = reward_model_.compute_step_reward(state, command, target, success);
        result.cumulative_reward += reward;

        result.timeline.push_back(
            {
                step + 1,
                state.mission_time_s,
                state.position_m,
                state.velocity_mps,
                command.thrust_axis,
                reward
            }
        );

        if (success) {
            break;
        }
    }

    return result;
}

}  // namespace orbital
