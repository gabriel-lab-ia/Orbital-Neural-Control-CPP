#pragma once

#include <cstdint>

namespace orbital::rl {

struct DeterminismConfig {
    std::int64_t seed = 7;
    std::int64_t torch_threads = 1;
    bool deterministic_algorithms = true;
};

}  // namespace orbital::rl
