#pragma once

#include <string_view>

namespace nmc::common {

// Run identifiers are part of filesystem paths, DB keys, and API payloads.
// Keep a conservative character policy to avoid traversal/injection ambiguity.
constexpr int kMaxRunIdLength = 64;

bool is_valid_run_id(std::string_view run_id);
void validate_run_id_or_throw(std::string_view run_id, std::string_view field_name);

}  // namespace nmc::common
