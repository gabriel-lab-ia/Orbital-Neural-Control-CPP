#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "orbital/control/control_command.hpp"
#include "orbital/control/reward_model.hpp"
#include "orbital/simulation/orbital_dynamics.hpp"
#include "orbital/simulation/orbital_state.hpp"

namespace orbital {

struct MissionStepTelemetry {
    std::int64_t step_index = 0;
    double mission_time_s = 0.0;
    std::array<double, 3> position_m{0.0, 0.0, 0.0};
    std::array<double, 3> velocity_mps{0.0, 0.0, 0.0};
    std::array<double, 3> thrust_axis{0.0, 0.0, 0.0};
    double reward = 0.0;
};

struct MissionRolloutResult {
    std::string mission_id;
    double cumulative_reward = 0.0;
    std::vector<MissionStepTelemetry> timeline;
};

class OrbitalControlCore {
public:
    OrbitalControlCore(simulation::DynamicsConfig dynamics, control::RewardWeights reward);

    [[nodiscard]] MissionRolloutResult run_open_loop_rollout(
        const std::string& mission_id,
        const simulation::OrbitalState3DOF& initial_state,
        const control::MissionTarget& target,
        std::int64_t total_steps,
        double dt_s
    ) const;

private:
    simulation::OrbitalDynamics3DOF dynamics_;
    control::RewardModel reward_model_;
};

}  // namespace orbital
