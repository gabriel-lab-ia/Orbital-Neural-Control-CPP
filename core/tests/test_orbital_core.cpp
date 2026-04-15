#include <cassert>

#include "orbital/core_api.hpp"

int main() {
    orbital::simulation::DynamicsConfig dynamics;
    dynamics.thrust_accel_scale_mps2 = 0.2;

    orbital::control::RewardWeights reward;
    reward.position_weight = 0.4;
    reward.velocity_weight = 0.1;
    reward.control_weight = 0.01;

    const orbital::OrbitalControlCore core(dynamics, reward);

    orbital::simulation::OrbitalState3DOF initial;
    initial.position_m = {120.0, -80.0, 40.0};
    initial.velocity_mps = {0.0, 0.0, 0.0};

    orbital::control::MissionTarget target;
    target.target_position_m = {0.0, 0.0, 0.0};
    target.target_velocity_mps = {0.0, 0.0, 0.0};

    const auto rollout = core.run_open_loop_rollout("unit-test", initial, target, 150, 0.1);

    assert(!rollout.timeline.empty());
    assert(rollout.timeline.size() <= 150);
    assert(rollout.timeline.front().step_index == 1);
    assert(rollout.cumulative_reward < 0.0);

    return 0;
}
