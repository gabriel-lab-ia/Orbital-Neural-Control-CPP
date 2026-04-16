#include "interfaces/cli/command_line.h"

#include <charconv>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "application/benchmark_runner.h"
#include "application/evaluation_runner.h"
#include "application/training_runner.h"
#include "domain/config/config_validation.h"
#include "domain/env/environment_factory.h"
#include "domain/inference/inference_backend_factory.h"

namespace nmc::interfaces::cli {
namespace {

int64_t parse_int64(const std::string& name, const std::string& value) {
    int64_t out = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto [ptr, error] = std::from_chars(begin, end, out);
    if (error != std::errc{} || ptr != end) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
    return out;
}

float parse_float(const std::string& name, const std::string& value) {
    std::size_t consumed = 0;
    try {
        const float parsed = std::stof(value, &consumed);
        if (consumed != value.size()) {
            throw std::runtime_error("invalid float for " + name + ": " + value);
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid float for " + name + ": " + value);
    }
}

std::string lowercase(const std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (const char ch : value) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return normalized;
}

bool parse_bool(const std::string& name, const std::string& value) {
    const auto normalized = lowercase(value);
    if (normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no") {
        return false;
    }
    throw std::runtime_error("invalid bool for " + name + ": " + value);
}

std::string require_value(const int argc, char** argv, int& index, const std::string& name) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for option: " + name);
    }
    ++index;
    return argv[index];
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  nmc train [options]\n"
        << "  nmc eval [options]\n"
        << "  nmc benchmark [--quick|--full] [--seed N] [--name NAME]\n\n"
        << "Train options:\n"
        << "  --env <" << domain::env::supported_environment_kinds() << ">\n"
        << "  --quick\n"
        << "  --seed <int>\n"
        << "  --num-envs <int>\n"
        << "  --updates <int>\n"
        << "  --rollout-steps <int>\n"
        << "  --ppo-epochs <int>\n"
        << "  --minibatch-size <int>\n"
        << "  --hidden-dim <int>\n"
        << "  --learning-rate <float>\n"
        << "  --run-id <string>\n"
        << "  --resume-checkpoint <path>\n"
        << "  --live-steps <int>\n"
        << "  --deterministic-live <bool>\n"
        << "  --mujoco-model <path>\n"
        << "  --pm-pos-log-w <float>\n"
        << "  --pm-pos-exp-w <float>\n"
        << "  --pm-vel-align-w <float>\n"
        << "  --pm-vel-error-w <float>\n"
        << "  --pm-control-w <float>\n"
        << "  --pm-risk-w <float>\n"
        << "  --pm-potential-shaping <bool>\n"
        << "  --help\n\n"
        << "Eval options:\n"
        << "  --checkpoint <path>\n"
        << "  --env <" << domain::env::supported_environment_kinds() << ">\n"
        << "  --episodes <int>\n"
        << "  --max-steps <int>\n"
        << "  --seed <int>\n"
        << "  --backend <" << domain::inference::supported_inference_backends() << ">\n"
        << "  --deterministic <bool>\n"
        << "  --run-id <string>\n"
        << "  --mujoco-model <path>\n"
        << "  --pm-pos-log-w <float>\n"
        << "  --pm-pos-exp-w <float>\n"
        << "  --pm-vel-align-w <float>\n"
        << "  --pm-vel-error-w <float>\n"
        << "  --pm-control-w <float>\n"
        << "  --pm-risk-w <float>\n"
        << "  --pm-potential-shaping <bool>\n"
        << "  --help\n\n"
        << "Benchmark options:\n"
        << "  --quick | --smoke\n"
        << "  --full\n"
        << "  --seed <int>\n"
        << "  --name <string>\n"
        << "  --help\n";
}

int run_train(const int argc, char** argv) {
    domain::config::TrainConfig config;

    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--env") {
            config.environment = require_value(argc, argv, index, arg);
        } else if (arg == "--quick") {
            config.trainer.num_envs = 4;
            config.trainer.total_updates = 3;
            config.trainer.ppo.rollout_steps = 32;
            config.trainer.ppo.ppo_epochs = 2;
            config.trainer.ppo.minibatch_size = 64;
            config.live_rollout_steps = 48;
        } else if (arg == "--seed") {
            config.trainer.seed = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--num-envs") {
            config.trainer.num_envs = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--updates") {
            config.trainer.total_updates = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--rollout-steps") {
            config.trainer.ppo.rollout_steps = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--ppo-epochs") {
            config.trainer.ppo.ppo_epochs = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--minibatch-size") {
            config.trainer.ppo.minibatch_size = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--hidden-dim") {
            config.trainer.hidden_dim = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--learning-rate") {
            config.trainer.ppo.learning_rate = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--run-id") {
            config.run_id = require_value(argc, argv, index, arg);
        } else if (arg == "--resume-checkpoint") {
            config.resume_checkpoint = std::filesystem::path(require_value(argc, argv, index, arg));
        } else if (arg == "--live-steps") {
            config.live_rollout_steps = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--deterministic-live") {
            config.deterministic_live_rollout = parse_bool(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--mujoco-model") {
            config.mujoco_model_path = std::filesystem::path(require_value(argc, argv, index, arg));
        } else if (arg == "--pm-pos-log-w") {
            config.point_mass_reward.position_log_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-pos-exp-w") {
            config.point_mass_reward.position_exp_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-vel-align-w") {
            config.point_mass_reward.velocity_alignment_weight = parse_float(
                arg,
                require_value(argc, argv, index, arg)
            );
        } else if (arg == "--pm-vel-error-w") {
            config.point_mass_reward.velocity_error_weight = parse_float(
                arg,
                require_value(argc, argv, index, arg)
            );
        } else if (arg == "--pm-control-w") {
            config.point_mass_reward.control_quadratic_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-risk-w") {
            config.point_mass_reward.corridor_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-potential-shaping") {
            config.point_mass_reward.potential_shaping_enabled = parse_bool(
                arg,
                require_value(argc, argv, index, arg)
            );
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            throw std::runtime_error("unknown train option: " + arg);
        }
    }

    domain::config::validate_train_config_or_throw(config);
    application::TrainingRunner runner;
    static_cast<void>(runner.run(config));
    return 0;
}

int run_eval(const int argc, char** argv) {
    domain::config::EvalConfig config;

    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--checkpoint") {
            config.checkpoint_path = std::filesystem::path(require_value(argc, argv, index, arg));
        } else if (arg == "--env") {
            config.environment = require_value(argc, argv, index, arg);
        } else if (arg == "--episodes") {
            config.episodes = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--max-steps") {
            config.max_steps = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--seed") {
            config.seed = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--backend") {
            config.inference_backend = require_value(argc, argv, index, arg);
        } else if (arg == "--deterministic") {
            config.deterministic_policy = parse_bool(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--run-id") {
            config.run_id = require_value(argc, argv, index, arg);
        } else if (arg == "--mujoco-model") {
            config.mujoco_model_path = std::filesystem::path(require_value(argc, argv, index, arg));
        } else if (arg == "--pm-pos-log-w") {
            config.point_mass_reward.position_log_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-pos-exp-w") {
            config.point_mass_reward.position_exp_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-vel-align-w") {
            config.point_mass_reward.velocity_alignment_weight = parse_float(
                arg,
                require_value(argc, argv, index, arg)
            );
        } else if (arg == "--pm-vel-error-w") {
            config.point_mass_reward.velocity_error_weight = parse_float(
                arg,
                require_value(argc, argv, index, arg)
            );
        } else if (arg == "--pm-control-w") {
            config.point_mass_reward.control_quadratic_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-risk-w") {
            config.point_mass_reward.corridor_weight = parse_float(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--pm-potential-shaping") {
            config.point_mass_reward.potential_shaping_enabled = parse_bool(
                arg,
                require_value(argc, argv, index, arg)
            );
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            throw std::runtime_error("unknown eval option: " + arg);
        }
    }

    domain::config::validate_eval_config_or_throw(config);
    application::EvaluationRunner runner;
    static_cast<void>(runner.run(config));
    return 0;
}

int run_benchmark(const int argc, char** argv) {
    domain::config::BenchmarkConfig config;

    for (int index = 2; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--quick" || arg == "--smoke") {
            config.quick = true;
        } else if (arg == "--full") {
            config.quick = false;
        } else if (arg == "--seed") {
            config.seed = parse_int64(arg, require_value(argc, argv, index, arg));
        } else if (arg == "--name") {
            config.benchmark_name = require_value(argc, argv, index, arg);
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            throw std::runtime_error("unknown benchmark option: " + arg);
        }
    }

    domain::config::validate_benchmark_config_or_throw(config);
    application::BenchmarkRunner runner;
    static_cast<void>(runner.run(config));
    return 0;
}

}  // namespace

int run_cli(const int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string command = argv[1];
    if (command == "train") {
        return run_train(argc, argv);
    }
    if (command == "eval") {
        return run_eval(argc, argv);
    }
    if (command == "benchmark") {
        return run_benchmark(argc, argv);
    }
    if (command == "help" || command == "--help" || command == "-h") {
        print_usage();
        return 0;
    }

    throw std::runtime_error("unknown command: " + command);
}

}  // namespace nmc::interfaces::cli
