#pragma once

#include "domain/config/experiment_config.h"

namespace nmc::domain::config {

void validate_train_config_or_throw(const TrainConfig& config);
void validate_eval_config_or_throw(const EvalConfig& config);
void validate_benchmark_config_or_throw(const BenchmarkConfig& config);

}  // namespace nmc::domain::config
