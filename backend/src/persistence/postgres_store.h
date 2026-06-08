#pragma once

#include "persistence/database_config.h"
#include "persistence/experiment_store.h"

struct pg_conn;

namespace orbital::backend::persistence {

class PostgresStore final : public ExperimentStore {
public:
    explicit PostgresStore(PostgresConfig config);
    ~PostgresStore() override;

    PostgresStore(const PostgresStore&) = delete;
    PostgresStore& operator=(const PostgresStore&) = delete;

    [[nodiscard]] std::vector<domain::RunRecord> list_runs(std::int64_t limit, std::int64_t offset) const override;
    [[nodiscard]] std::optional<domain::RunRecord> get_run(const std::string& run_id) const override;

    [[nodiscard]] std::vector<domain::EventRecord> list_events(
        const std::string& run_id,
        std::int64_t limit,
        std::int64_t offset
    ) const override;

    [[nodiscard]] std::vector<domain::BenchmarkRecord> list_benchmarks(
        std::int64_t limit,
        std::int64_t offset
    ) const override;
    [[nodiscard]] std::optional<domain::BenchmarkRecord> get_benchmark(
        const std::string& benchmark_id_or_name
    ) const override;

private:
    void connect();
    void initialize_schema() const;

    PostgresConfig config_;
    pg_conn* conn_ = nullptr;
};

}  // namespace orbital::backend::persistence
