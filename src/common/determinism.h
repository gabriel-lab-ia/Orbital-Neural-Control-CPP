#pragma once

#include <cstdint>

namespace nmc::common {

void validate_deterministic_cuda_environment_or_throw(bool uses_cuda);
void configure_determinism(std::uint64_t seed, int64_t torch_num_threads);

}  // namespace nmc::common
