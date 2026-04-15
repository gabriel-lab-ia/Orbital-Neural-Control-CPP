#pragma once

#include <array>

namespace orbital::simulation {

struct OrbitalState3DOF {
    // Position in meters in a local orbital frame.
    std::array<double, 3> position_m{0.0, 0.0, 0.0};
    // Velocity in meters per second in the same frame.
    std::array<double, 3> velocity_mps{0.0, 0.0, 0.0};
    // Mission elapsed time in seconds.
    double mission_time_s = 0.0;
};

}  // namespace orbital::simulation
