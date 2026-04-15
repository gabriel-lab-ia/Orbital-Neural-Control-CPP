#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "orbital/control/lqr_controller.hpp"
#include "orbital/control/pid_controller.hpp"
#include "orbital/core_api.hpp"

namespace {

double l2(const std::array<double, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

struct BaselineMetrics {
    std::string name;
    double final_position_error = 0.0;
    double final_velocity_error = 0.0;
};

BaselineMetrics evaluate_lqr() {
    orbital::simulation::OrbitalState3DOF state;
    state.position_m = {250.0, -180.0, 90.0};

    orbital::control::MissionTarget target;
    orbital::control::LqrBaselineController3DOF controller;
    orbital::simulation::OrbitalDynamics3DOF dynamics;

    constexpr double dt = 0.1;
    for (int step = 0; step < 500; ++step) {
        const std::array<double, 3> p_err{
            target.target_position_m[0] - state.position_m[0],
            target.target_position_m[1] - state.position_m[1],
            target.target_position_m[2] - state.position_m[2]
        };
        const std::array<double, 3> v_err{
            target.target_velocity_mps[0] - state.velocity_mps[0],
            target.target_velocity_mps[1] - state.velocity_mps[1],
            target.target_velocity_mps[2] - state.velocity_mps[2]
        };

        orbital::control::ThrusterCommand command;
        command.thrust_axis = controller.compute(p_err, v_err);
        state = dynamics.propagate(state, command, dt);
    }

    return {"lqr", l2(state.position_m), l2(state.velocity_mps)};
}

BaselineMetrics evaluate_pid() {
    orbital::simulation::OrbitalState3DOF state;
    state.position_m = {250.0, -180.0, 90.0};

    orbital::control::MissionTarget target;
    orbital::control::PidController3DOF controller;
    orbital::simulation::OrbitalDynamics3DOF dynamics;

    constexpr double dt = 0.1;
    for (int step = 0; step < 500; ++step) {
        const std::array<double, 3> p_err{
            target.target_position_m[0] - state.position_m[0],
            target.target_position_m[1] - state.position_m[1],
            target.target_position_m[2] - state.position_m[2]
        };
        const std::array<double, 3> v_err{
            target.target_velocity_mps[0] - state.velocity_mps[0],
            target.target_velocity_mps[1] - state.velocity_mps[1],
            target.target_velocity_mps[2] - state.velocity_mps[2]
        };

        orbital::control::ThrusterCommand command;
        command.thrust_axis = controller.compute(p_err, v_err, dt);
        state = dynamics.propagate(state, command, dt);
    }

    return {"pid", l2(state.position_m), l2(state.velocity_mps)};
}

}  // namespace

int main() {
    const auto lqr = evaluate_lqr();
    const auto pid = evaluate_pid();

    std::ofstream csv("artifacts/benchmarks/control_baseline_comparison.csv", std::ios::out | std::ios::trunc);
    csv << "controller,final_position_error,final_velocity_error\n";
    csv << lqr.name << ',' << lqr.final_position_error << ',' << lqr.final_velocity_error << '\n';
    csv << pid.name << ',' << pid.final_position_error << ',' << pid.final_velocity_error << '\n';

    std::cout << "LQR final position error: " << lqr.final_position_error << '\n';
    std::cout << "PID final position error: " << pid.final_position_error << '\n';

    return 0;
}
