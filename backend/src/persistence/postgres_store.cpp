#include "persistence/postgres_store.h"

#ifdef ORBITAL_ENABLE_POSTGRES
#include <libpq-fe.h>
#endif

#include <array>
#include <charconv>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace orbital::backend::persistence {
namespace {

#ifdef ORBITAL_ENABLE_POSTGRES
struct PgResultDeleter {
    void operator()(PGresult* result) const {
        if (result != nullptr) {
            PQclear(result);
        }
    }
};

using ResultPtr = std::unique_ptr<PGresult, PgResultDeleter>;

std::string int64_to_string(const std::int64_t value) {
    std::array<char, 32> buffer{};
    const auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (ec != std::errc{}) {
        throw std::runtime_error("failed to format integer SQL parameter");
    }
    return std::string(buffer.data(), ptr);
}

std::string text_value(PGresult* result, const int row, const int column) {
    if (PQgetisnull(result, row, column) != 0) {
        return "";
    }
    return PQgetvalue(result, row, column);
}

std::int64_t int64_value(PGresult* result, const int row, const int column) {
    const std::string raw = text_value(result, row, column);
    if (raw.empty()) {
        return 0;
    }
    return std::stoll(raw);
}

void check_status(PGconn* conn, PGresult* result, const ExecStatusType expected, const char* context) {
    if (result == nullptr || PQresultStatus(result) != expected) {
        const std::string detail = conn != nullptr ? PQerrorMessage(conn) : "no connection";
        throw std::runtime_error(std::string(context) + " failed: " + detail);
    }
}

ResultPtr exec(PGconn* conn, const char* sql, const char* context) {
    ResultPtr result(PQexec(conn, sql));
    check_status(conn, result.get(), PGRES_COMMAND_OK, context);
    return result;
}

ResultPtr query(
    PGconn* conn,
    const char* sql,
    const std::vector<std::string>& params,
    const char* context
) {
    std::vector<const char*> values;
    values.reserve(params.size());
    for (const auto& param : params) {
        values.push_back(param.c_str());
    }

    ResultPtr result(PQexecParams(
        conn,
        sql,
        static_cast<int>(values.size()),
        nullptr,
        values.data(),
        nullptr,
        nullptr,
        0
    ));
    check_status(conn, result.get(), PGRES_TUPLES_OK, context);
    return result;
}

#endif

}  // namespace

PostgresStore::PostgresStore(PostgresConfig config)
    : config_(std::move(config)) {
    connect();
    initialize_schema();
}

PostgresStore::~PostgresStore() {
#ifdef ORBITAL_ENABLE_POSTGRES
    if (conn_ != nullptr) {
        PQfinish(conn_);
        conn_ = nullptr;
    }
#endif
}

void PostgresStore::connect() {
#ifndef ORBITAL_ENABLE_POSTGRES
    throw std::runtime_error(
        "DB_BACKEND=postgres requested, but this backend binary was built without PostgreSQL/libpq support"
    );
#else
    const std::string port = std::to_string(config_.port);
    const std::string connect_timeout = std::to_string(config_.connect_timeout_seconds);
    const char* keywords[] = {
        "host",
        "port",
        "dbname",
        "user",
        "password",
        "connect_timeout",
        "application_name",
        nullptr
    };
    const char* values[] = {
        config_.host.c_str(),
        port.c_str(),
        config_.database.c_str(),
        config_.user.c_str(),
        config_.password.c_str(),
        connect_timeout.c_str(),
        "orbital-backend",
        nullptr
    };

    conn_ = PQconnectdbParams(keywords, values, 0);
    if (conn_ == nullptr || PQstatus(conn_) != CONNECTION_OK) {
        const std::string detail = conn_ != nullptr ? PQerrorMessage(conn_) : "unable to allocate libpq connection";
        throw std::runtime_error(
            "unable to connect to PostgreSQL at " + config_.host + ":" + port +
            " database '" + config_.database + "': " + detail
        );
    }

    const std::string statement_timeout = "SET statement_timeout = " + std::to_string(config_.statement_timeout_ms) + ";";
    exec(conn_, statement_timeout.c_str(), "set statement_timeout");
#endif
}

void PostgresStore::initialize_schema() const {
#ifdef ORBITAL_ENABLE_POSTGRES
    exec(
        conn_,
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version INTEGER PRIMARY KEY,"
        "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ");",
        "create schema_migrations"
    );
    exec(
        conn_,
        "CREATE TABLE IF NOT EXISTS runs ("
        "  run_id TEXT PRIMARY KEY,"
        "  mode TEXT NOT NULL,"
        "  environment TEXT NOT NULL,"
        "  seed BIGINT NOT NULL,"
        "  started_at TEXT NOT NULL,"
        "  ended_at TEXT,"
        "  status TEXT NOT NULL,"
        "  artifact_dir TEXT NOT NULL,"
        "  config_json TEXT NOT NULL,"
        "  summary_json TEXT"
        ");",
        "create runs"
    );
    exec(
        conn_,
        "CREATE TABLE IF NOT EXISTS events ("
        "  id BIGSERIAL PRIMARY KEY,"
        "  run_id TEXT NOT NULL REFERENCES runs(run_id),"
        "  level TEXT NOT NULL,"
        "  event_type TEXT NOT NULL,"
        "  message TEXT NOT NULL,"
        "  payload_json TEXT,"
        "  created_at TEXT NOT NULL"
        ");",
        "create events"
    );
    exec(
        conn_,
        "CREATE TABLE IF NOT EXISTS benchmarks ("
        "  id BIGSERIAL PRIMARY KEY,"
        "  benchmark_name TEXT NOT NULL,"
        "  run_id TEXT,"
        "  summary_json TEXT NOT NULL,"
        "  created_at TEXT NOT NULL"
        ");",
        "create benchmarks"
    );
    exec(conn_, "CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);", "index runs");
    exec(conn_, "CREATE INDEX IF NOT EXISTS idx_events_run_created_at ON events(run_id, created_at);", "index events");
    exec(conn_, "CREATE INDEX IF NOT EXISTS idx_benchmarks_created_at ON benchmarks(created_at DESC);", "index benchmarks");
    exec(
        conn_,
        "INSERT INTO schema_migrations(version, applied_at) VALUES (1, now()) ON CONFLICT(version) DO NOTHING;",
        "seed schema_migrations"
    );
#endif
}

std::vector<domain::RunRecord> PostgresStore::list_runs(const std::int64_t limit, const std::int64_t offset) const {
#ifndef ORBITAL_ENABLE_POSTGRES
    (void)limit;
    (void)offset;
    throw std::runtime_error("PostgreSQL support is not compiled into this binary");
#else
    const auto result = query(
        conn_,
        "SELECT run_id, mode, environment, seed, started_at, COALESCE(ended_at,''), status, artifact_dir, "
        "config_json, COALESCE(summary_json,'') "
        "FROM runs ORDER BY started_at DESC LIMIT $1 OFFSET $2;",
        {int64_to_string(limit), int64_to_string(offset)},
        "list runs"
    );

    std::vector<domain::RunRecord> runs;
    runs.reserve(static_cast<std::size_t>(PQntuples(result.get())));
    for (int row = 0; row < PQntuples(result.get()); ++row) {
        domain::RunRecord run;
        run.run_id = text_value(result.get(), row, 0);
        run.mode = text_value(result.get(), row, 1);
        run.environment = text_value(result.get(), row, 2);
        run.seed = int64_value(result.get(), row, 3);
        run.started_at = text_value(result.get(), row, 4);
        run.ended_at = text_value(result.get(), row, 5);
        run.status = text_value(result.get(), row, 6);
        run.artifact_dir = text_value(result.get(), row, 7);
        run.config_json = text_value(result.get(), row, 8);
        run.summary_json = text_value(result.get(), row, 9);
        runs.emplace_back(std::move(run));
    }
    return runs;
#endif
}

std::optional<domain::RunRecord> PostgresStore::get_run(const std::string& run_id) const {
#ifndef ORBITAL_ENABLE_POSTGRES
    (void)run_id;
    throw std::runtime_error("PostgreSQL support is not compiled into this binary");
#else
    const auto result = query(
        conn_,
        "SELECT run_id, mode, environment, seed, started_at, COALESCE(ended_at,''), status, artifact_dir, "
        "config_json, COALESCE(summary_json,'') FROM runs WHERE run_id = $1;",
        {run_id},
        "get run"
    );
    if (PQntuples(result.get()) == 0) {
        return std::nullopt;
    }

    domain::RunRecord run;
    run.run_id = text_value(result.get(), 0, 0);
    run.mode = text_value(result.get(), 0, 1);
    run.environment = text_value(result.get(), 0, 2);
    run.seed = int64_value(result.get(), 0, 3);
    run.started_at = text_value(result.get(), 0, 4);
    run.ended_at = text_value(result.get(), 0, 5);
    run.status = text_value(result.get(), 0, 6);
    run.artifact_dir = text_value(result.get(), 0, 7);
    run.config_json = text_value(result.get(), 0, 8);
    run.summary_json = text_value(result.get(), 0, 9);
    return run;
#endif
}

std::vector<domain::EventRecord> PostgresStore::list_events(
    const std::string& run_id,
    const std::int64_t limit,
    const std::int64_t offset
) const {
#ifndef ORBITAL_ENABLE_POSTGRES
    (void)run_id;
    (void)limit;
    (void)offset;
    throw std::runtime_error("PostgreSQL support is not compiled into this binary");
#else
    const auto result = query(
        conn_,
        "SELECT id, run_id, level, event_type, message, COALESCE(payload_json,''), created_at "
        "FROM events WHERE run_id = $1 ORDER BY id ASC LIMIT $2 OFFSET $3;",
        {run_id, int64_to_string(limit), int64_to_string(offset)},
        "list events"
    );

    std::vector<domain::EventRecord> events;
    events.reserve(static_cast<std::size_t>(PQntuples(result.get())));
    for (int row = 0; row < PQntuples(result.get()); ++row) {
        domain::EventRecord event;
        event.id = int64_value(result.get(), row, 0);
        event.run_id = text_value(result.get(), row, 1);
        event.level = text_value(result.get(), row, 2);
        event.event_type = text_value(result.get(), row, 3);
        event.message = text_value(result.get(), row, 4);
        event.payload_json = text_value(result.get(), row, 5);
        event.created_at = text_value(result.get(), row, 6);
        events.emplace_back(std::move(event));
    }
    return events;
#endif
}

std::vector<domain::BenchmarkRecord> PostgresStore::list_benchmarks(
    const std::int64_t limit,
    const std::int64_t offset
) const {
#ifndef ORBITAL_ENABLE_POSTGRES
    (void)limit;
    (void)offset;
    throw std::runtime_error("PostgreSQL support is not compiled into this binary");
#else
    const auto result = query(
        conn_,
        "SELECT id, benchmark_name, COALESCE(run_id,''), summary_json, created_at "
        "FROM benchmarks ORDER BY id DESC LIMIT $1 OFFSET $2;",
        {int64_to_string(limit), int64_to_string(offset)},
        "list benchmarks"
    );

    std::vector<domain::BenchmarkRecord> benchmarks;
    benchmarks.reserve(static_cast<std::size_t>(PQntuples(result.get())));
    for (int row = 0; row < PQntuples(result.get()); ++row) {
        domain::BenchmarkRecord record;
        record.id = int64_value(result.get(), row, 0);
        record.benchmark_name = text_value(result.get(), row, 1);
        record.run_id = text_value(result.get(), row, 2);
        record.summary_json = text_value(result.get(), row, 3);
        record.created_at = text_value(result.get(), row, 4);
        benchmarks.emplace_back(std::move(record));
    }
    return benchmarks;
#endif
}

std::optional<domain::BenchmarkRecord> PostgresStore::get_benchmark(
    const std::string& benchmark_id_or_name
) const {
#ifndef ORBITAL_ENABLE_POSTGRES
    (void)benchmark_id_or_name;
    throw std::runtime_error("PostgreSQL support is not compiled into this binary");
#else
    const auto result = query(
        conn_,
        "SELECT id, benchmark_name, COALESCE(run_id,''), summary_json, created_at "
        "FROM benchmarks WHERE id::text = $1 OR benchmark_name = $1 LIMIT 1;",
        {benchmark_id_or_name},
        "get benchmark"
    );
    if (PQntuples(result.get()) == 0) {
        return std::nullopt;
    }

    domain::BenchmarkRecord record;
    record.id = int64_value(result.get(), 0, 0);
    record.benchmark_name = text_value(result.get(), 0, 1);
    record.run_id = text_value(result.get(), 0, 2);
    record.summary_json = text_value(result.get(), 0, 3);
    record.created_at = text_value(result.get(), 0, 4);
    return record;
#endif
}

}  // namespace orbital::backend::persistence
