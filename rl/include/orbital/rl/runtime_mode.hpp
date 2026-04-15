#pragma once

namespace orbital::rl {

enum class RuntimeMode {
    Training,
    Evaluation,
    Production
};

struct RuntimePolicyConfig {
    RuntimeMode mode = RuntimeMode::Training;
    bool deterministic_actions = false;
    double exploration_std_scale = 1.0;

    [[nodiscard]] static RuntimePolicyConfig production() {
        RuntimePolicyConfig config;
        config.mode = RuntimeMode::Production;
        config.deterministic_actions = true;
        config.exploration_std_scale = 0.15;
        return config;
    }
};

}  // namespace orbital::rl
