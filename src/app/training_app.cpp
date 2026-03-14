#include "app/training_app.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "env/environment_registry.h"
#include "train/ppo_trainer.h"
#include "utils/csv_logger.h"
#include "utils/live_rollout_logger.h"
#include "utils/neural_3d_export.h"

namespace nmc {
namespace {

std::string get_env_or_default(const char* key, const std::string& fallback) {
    if (const char* value = std::getenv(key)) {
        return value;
    }
    return fallback;
}

EnvironmentSelection load_environment_selection() {
    EnvironmentSelection selection;
    selection.kind = get_env_or_default("NMC_ENV", "point_mass");

    if (selection.kind == "mujoco_cartpole") {
        const auto model_override = get_env_or_default("NMC_MUJOCO_XML", "");
        if (!model_override.empty()) {
            selection.mujoco_model_path = model_override;
        } else {
            selection.mujoco_model_path = std::filesystem::path("assets/mujoco/cartpole.xml");
        }
    }

    return selection;
}

bool run_live_after_training() {
    const auto value = get_env_or_default("NMC_LIVE_POLICY", "0");
    return value == "1" || value == "true" || value == "TRUE" || value == "on";
}

int64_t live_step_limit() {
    const auto raw = get_env_or_default("NMC_LIVE_STEPS", "240");
    try {
        return std::max<int64_t>(1, std::stoll(raw));
    } catch (const std::exception&) {
        return 240;
    }
}

}  // namespace

int run_training_app() {
    const auto artifact_dir = std::filesystem::path("artifacts");
    const auto metrics_path = artifact_dir / "learning_curve.csv";

    const TrainerConfig config{};
    const auto environment_selection = load_environment_selection();
    auto environment_pack = make_environment_pack(environment_selection, config.num_envs);
    const auto environment_name = environment_pack.display_name;

    PPOTrainer trainer(config, artifact_dir, std::move(environment_pack));
    CsvLogger logger(metrics_path);

    std::cout << "=============================================================\n";
    std::cout << "  NeuroMotor PPO Foundation\n";
    std::cout << "  C++20 + LibTorch | continuous control baseline for MuJoCo\n";
    std::cout << "=============================================================\n";
    std::cout << "Environment: " << environment_name << '\n';
    std::cout << "Goal: replace the old synthetic AGI prototype with a clean PPO core.\n";
    if (mujoco_support_enabled()) {
        std::cout << "MuJoCo support: enabled at compile time.\n\n";
    } else {
        std::cout << "MuJoCo support: not compiled in yet. Set up the library and rebuild with NMC_ENABLE_MUJOCO=ON.\n\n";
    }

    auto metrics = trainer.train();

    std::cout << "Training progress\n";
    std::cout << "-----------------\n";
    for (const auto& metric : metrics) {
        logger.log(metric);
        std::cout
            << "update " << std::setw(2) << metric.update
            << " | steps=" << std::setw(6) << metric.env_steps
            << " | policy=" << std::setw(8) << std::fixed << std::setprecision(4) << metric.policy_loss
            << " | value=" << std::setw(8) << metric.value_loss
            << " | reward=" << std::setw(8) << metric.avg_episode_return
            << " | success=" << std::setw(7) << metric.success_rate
            << " | len=" << std::setw(6) << metric.avg_episode_length
            << '\n';
    }

    if (!metrics.empty()) {
        const auto& final = metrics.back();
        std::cout << "\nSummary\n";
        std::cout << "-------\n";
        std::cout << "Final avg episode return : " << final.avg_episode_return << '\n';
        std::cout << "Final success rate       : " << final.success_rate << '\n';
        std::cout << "Final avg episode length : " << final.avg_episode_length << '\n';
        std::cout << "Final policy entropy     : " << final.entropy << '\n';
        std::cout << "Metrics exported to      : " << metrics_path << '\n';
    }

    {
        const auto live_path = artifact_dir / "live_rollout.csv";
        const auto neural_html_path = artifact_dir / "neural_network_3d.html";
        const auto neural_json_path = artifact_dir / "neural_network_3d.json";
        std::ostringstream silent_stream;
        std::ostream& live_stream = run_live_after_training() ? std::cout : silent_stream;
        auto live_steps = trainer.run_live_episode(live_step_limit(), live_stream);
        write_live_rollout_csv(live_path, live_steps);
        write_neural_3d_visualization(
            neural_html_path,
            neural_json_path,
            environment_name,
            trainer.agent(),
            live_steps
        );
        std::cout << "Live rollout exported to : " << live_path << '\n';
        std::cout << "3D neural viewer saved to: " << neural_html_path << '\n';
    }

    return 0;
}

}  // namespace nmc
