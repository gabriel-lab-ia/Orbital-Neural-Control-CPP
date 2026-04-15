# Architecture Overview

This repository now has two complementary tracks:

1. **Production baseline (`src/`)**: CPU-first PPO train/eval/benchmark pipeline in C++20 + LibTorch.
2. **Orbital expansion stack (`core/control/sim/rl + backend/frontend + mlops`)**: mission-oriented evolution path for orbital autonomy, telemetry streaming, and experiment operations.

## Layered Core (`src/`)

- `src/domain/`
  - `ppo/`: actor-critic model, PPO objective/training loop.
  - `env/`: environment interface and environment factory.
  - `inference/`: inference backend abstraction (`libtorch` active, TensorRT stub).
  - `config/`: explicit train/eval/benchmark config objects.
- `src/application/`
  - `training_runner`, `evaluation_runner`, `benchmark_runner`.
- `src/infrastructure/`
  - `artifacts/`: run layout, manifests, checkpoint management.
  - `persistence/`: SQLite store (`runs`, `episodes`, `events`, `benchmarks`).
  - `reporting/`: CSV/live rollout reports.
- `src/interfaces/`
  - CLI command surface: `train`, `eval`, `benchmark`.
- `src/common/`
  - time/JSON helpers and run-id generation.

## Orbital Expansion Modules

- `core/`
  - Header-focused orbital control kernel (`orbital::` namespace).
  - Deterministic 3DOF dynamics and reward model primitives.
  - `OrbitalControlCore` mission rollout API.
- `control/`
  - baseline LQR/PID controllers used for benchmark reference.
- `sim/`
  - perturbation/disturbance model interfaces.
- `rl/`
  - runtime modes (training/eval/production) and determinism profiles.
- `training/`
  - Python orchestration + `pybind11` bridge (`py_orbital_core`).
- `mlops/`
  - MLflow tracking pipeline, ONNX export, model registry scripts.
- `backend/`
  - C++ REST/WebSocket telemetry stream service.
- `frontend/`
  - Next.js mission dashboard (3D orbit + live telemetry charts).

## Artifact and Persistence Model

```text
artifacts/
  runs/<run_id>/
    manifest.json
    training_metrics.csv
    training_summary.json
    evaluation_summary.json
    checkpoints/policy_last.pt
  latest/
  reports/
  benchmarks/
  experiments.sqlite
  mlflow/
  mlflow-artifacts/
```

SQLite schema (baseline runtime store):

- `runs`
- `episodes`
- `events`
- `benchmarks`

MLflow store (MLOps):

- experiment params/metrics/tags
- run artifacts
- ONNX model artifacts and registry integration path

## Runtime Modes

### Baseline CLI

- `./build/nmc train ...`
- `./build/nmc eval ...`
- `./build/nmc benchmark --quick ...`

### Mission Telemetry Demo

- backend emits telemetry through `/ws/telemetry`
- frontend consumes stream for 3D mission visualization and control diagnostics

### MLOps

- tracked training with MLflow tags (`orbital_dynamics`, `perturbation_level`, `reward_shaping`)
- ONNX export for embedded inference contract

## Optional Integrations and Guardrails

- MuJoCo remains optional (`NMC_ENABLE_MUJOCO=ON` only when available).
- TensorRT backend remains a stub and is intentionally disabled by default.
- CPU-first path is the default CI and local baseline.

## Determinism and Reproducibility

- seed control in train/eval/benchmark configs
- run manifests and structured summaries for every run
- benchmark smoke path with artifact validation in CI
- deterministic mission rollout path in orbital core tests

