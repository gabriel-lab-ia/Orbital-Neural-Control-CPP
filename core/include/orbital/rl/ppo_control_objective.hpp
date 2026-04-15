#pragma once

#include <algorithm>
#include <cmath>

namespace orbital::rl {

struct PpoObjectiveTerms {
    double policy_ratio = 1.0;
    double advantage = 0.0;
    double clip_epsilon = 0.2;
    double entropy = 0.0;
    double entropy_weight = 0.01;
};

[[nodiscard]] inline double clipped_surrogate(const PpoObjectiveTerms& terms) {
    const double clipped_ratio = std::clamp(
        terms.policy_ratio,
        1.0 - terms.clip_epsilon,
        1.0 + terms.clip_epsilon
    );

    const double unclipped = terms.policy_ratio * terms.advantage;
    const double clipped = clipped_ratio * terms.advantage;
    const double surrogate = std::min(unclipped, clipped);

    return surrogate + terms.entropy_weight * terms.entropy;
}

[[nodiscard]] inline double generalized_advantage_estimate(
    const double delta_t,
    const double gamma,
    const double lambda,
    const double next_advantage
) {
    return delta_t + (gamma * lambda * next_advantage);
}

}  // namespace orbital::rl
