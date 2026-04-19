#pragma once

#include <string_view>

namespace orbital::backend::common {

constexpr int kMaxRunIdLength = 64;

bool is_valid_run_id(std::string_view run_id);
void validate_run_id_or_throw(std::string_view run_id, std::string_view field_name);

}  // namespace orbital::backend::common
