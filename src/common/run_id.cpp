#include "common/run_id.h"

#include <cctype>
#include <stdexcept>
#include <string>

namespace nmc::common {
namespace {

bool is_allowed_char(const char value) {
    const unsigned char ch = static_cast<unsigned char>(value);
    return std::isalnum(ch) != 0 || ch == '_' || ch == '-' || ch == '.';
}

}  // namespace

bool is_valid_run_id(const std::string_view run_id) {
    if (run_id.empty() || run_id.size() > static_cast<std::size_t>(kMaxRunIdLength)) {
        return false;
    }
    if (!std::isalnum(static_cast<unsigned char>(run_id.front()))) {
        return false;
    }
    if (run_id == "." || run_id == "..") {
        return false;
    }
    for (const char ch : run_id) {
        if (!is_allowed_char(ch)) {
            return false;
        }
    }
    return true;
}

void validate_run_id_or_throw(const std::string_view run_id, const std::string_view field_name) {
    if (is_valid_run_id(run_id)) {
        return;
    }
    throw std::runtime_error(
        std::string(field_name) + " must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}; got: '" + std::string(run_id) + "'"
    );
}

}  // namespace nmc::common
