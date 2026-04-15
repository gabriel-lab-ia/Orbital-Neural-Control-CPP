#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orbital/core_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_orbital_core, module) {
    module.doc() = "Pybind11 bridge for orbital core mission rollout APIs";

    py::class_<orbital::simulation::DynamicsConfig>(module, "DynamicsConfig")
        .def(py::init<>())
        .def_readwrite("thrust_accel_scale_mps2", &orbital::simulation::DynamicsConfig::thrust_accel_scale_mps2)
        .def_readwrite("drag_damping", &orbital::simulation::DynamicsConfig::drag_damping)
        .def_readwrite("max_speed_mps", &orbital::simulation::DynamicsConfig::max_speed_mps);

    py::class_<orbital::control::RewardWeights>(module, "RewardWeights")
        .def(py::init<>())
        .def_readwrite("position_weight", &orbital::control::RewardWeights::position_weight)
        .def_readwrite("velocity_weight", &orbital::control::RewardWeights::velocity_weight)
        .def_readwrite("control_weight", &orbital::control::RewardWeights::control_weight)
        .def_readwrite("terminal_bonus", &orbital::control::RewardWeights::terminal_bonus);

    py::class_<orbital::simulation::OrbitalState3DOF>(module, "OrbitalState3DOF")
        .def(py::init<>())
        .def_readwrite("position_m", &orbital::simulation::OrbitalState3DOF::position_m)
        .def_readwrite("velocity_mps", &orbital::simulation::OrbitalState3DOF::velocity_mps)
        .def_readwrite("mission_time_s", &orbital::simulation::OrbitalState3DOF::mission_time_s);

    py::class_<orbital::control::MissionTarget>(module, "MissionTarget")
        .def(py::init<>())
        .def_readwrite("target_position_m", &orbital::control::MissionTarget::target_position_m)
        .def_readwrite("target_velocity_mps", &orbital::control::MissionTarget::target_velocity_mps);

    py::class_<orbital::MissionStepTelemetry>(module, "MissionStepTelemetry")
        .def_readonly("step_index", &orbital::MissionStepTelemetry::step_index)
        .def_readonly("mission_time_s", &orbital::MissionStepTelemetry::mission_time_s)
        .def_readonly("position_m", &orbital::MissionStepTelemetry::position_m)
        .def_readonly("velocity_mps", &orbital::MissionStepTelemetry::velocity_mps)
        .def_readonly("thrust_axis", &orbital::MissionStepTelemetry::thrust_axis)
        .def_readonly("reward", &orbital::MissionStepTelemetry::reward);

    py::class_<orbital::MissionRolloutResult>(module, "MissionRolloutResult")
        .def_readonly("mission_id", &orbital::MissionRolloutResult::mission_id)
        .def_readonly("cumulative_reward", &orbital::MissionRolloutResult::cumulative_reward)
        .def_readonly("timeline", &orbital::MissionRolloutResult::timeline);

    py::class_<orbital::OrbitalControlCore>(module, "OrbitalControlCore")
        .def(py::init<orbital::simulation::DynamicsConfig, orbital::control::RewardWeights>())
        .def("run_open_loop_rollout", &orbital::OrbitalControlCore::run_open_loop_rollout);
}
