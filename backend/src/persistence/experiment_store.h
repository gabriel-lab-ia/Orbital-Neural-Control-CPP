#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "domain/types.h"

namespace orbital::backend::persistence {

class ExperimentStore {
public:
    virtual ~ExperimentStore() = default;

    [[nodiscard]] virtual std::vector<domain::RunRecord> list_runs(std::int64_t limit, std::int64_t offset) const = 0;
    [[nodiscard]] virtual std::optional<domain::RunRecord> get_run(const std::string& run_id) const = 0;

    [[nodiscard]] virtual std::vector<domain::EventRecord> list_events(
        const std::string& run_id,
        std::int64_t limit,
        std::int64_t offset
    ) const = 0;

    [[nodiscard]] virtual std::vector<domain::BenchmarkRecord> list_benchmarks(
        std::int64_t limit,
        std::int64_t offset
    ) const = 0;
    [[nodiscard]] virtual std::optional<domain::BenchmarkRecord> get_benchmark(
        const std::string& benchmark_id_or_name
    ) const = 0;
};

}  // namespace orbital::backend::persistence
