# backend/

Mission telemetry and replay service for Orbital Neural Control CPP.

Status: optional stack module (not required for baseline `nmc` CI path), but now with versioned REST and WebSocket contracts aligned to `docs/openapi/orbital-api.yaml`.

## Responsibilities

- Serve run, benchmark, event, artifact, and replay data from SQLite or PostgreSQL plus `artifacts/runs/*`.
- Stream live/replay telemetry with typed message envelopes.
- Expose train/eval/benchmark job submission API (dry-run by default, executable mode behind `ORBITAL_JOB_EXECUTOR=1`).

## API Surface

REST:

- `GET /health`
- `GET /version`
- `GET /runs`
- `GET /runs/{runId}`
- `GET /runs/{runId}/summary`
- `GET /runs/{runId}/telemetry`
- `GET /runs/{runId}/telemetry/window`
- `GET /runs/{runId}/events`
- `GET /runs/{runId}/artifacts`
- `GET /runs/{runId}/replay`
- `GET /benchmarks`
- `GET /benchmarks/{benchmarkId}`
- `POST /train/jobs`
- `POST /eval/jobs`
- `POST /benchmark/jobs`
- `GET /jobs/{jobId}`
- `GET /config/presets`

WebSocket:

- `WS /ws/telemetry/live`
- `WS /ws/runs/{runId}/stream`

## Build

From repository root:

```bash
cmake --preset orbital-stack
cmake --build --preset build-orbital-backend
./build-orbital/backend/orbital_backend
```

Or directly:

```bash
cmake -S backend -B build/backend -G Ninja
cmake --build build/backend --target orbital_backend
./build/backend/orbital_backend
```

## Runtime Environment

- `ORBITAL_BACKEND_PORT` (default: `8080`)
- `ORBITAL_ARTIFACT_ROOT` (default: `artifacts`)
- `DB_BACKEND` (`sqlite` default, or `postgres`)
- `SQLITE_PATH` (default: `artifacts/experiments.sqlite`)
- `ORBITAL_SQLITE_PATH` (default: `artifacts/experiments.sqlite`)
- `POSTGRES_HOST`
- `POSTGRES_PORT` (default: `5432`)
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_CONNECT_TIMEOUT_SECONDS` (default: `5`)
- `POSTGRES_STATEMENT_TIMEOUT_MS` (default: `5000`)
- `ORBITAL_REPO_ROOT` (default: current working directory)
- `ORBITAL_JOB_EXECUTOR` (`0` dry-run default, `1` to execute `./build/nmc ...` commands)

## Notes

- SQLite is opened with WAL and read-path indexes for replay/event queries.
- PostgreSQL uses libpq when available at build time, environment-only credentials, connection timeouts, statement timeouts, and parameterized runtime queries.
- Telemetry is read from `live_rollout.csv` per run and converted to orbital-friendly vectors for frontend rendering.
- Job execution is intentionally opt-in to keep backend deterministic/safe by default.
