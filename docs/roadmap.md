# Roadmap

## Implemented Baseline

- CPU-first PPO train/eval/benchmark pipeline in C++20 + LibTorch.
- Layered architecture (`domain`, `application`, `infrastructure`, `interfaces`, `common`).
- Reproducible artifact model with run manifests and benchmark summaries.
- SQLite telemetry persistence (`runs`, `episodes`, `events`, `benchmarks`).
- Optional MuJoCo integration path.
- Inference backend abstraction with TensorRT stub.
- Orbital expansion modules (`core/control/sim/rl`) with deterministic mission rollout primitives.
- MLflow tracking scripts and ONNX export/registry workflow.
- C++ telemetry backend and Next.js mission dashboard scaffold.

## Next 30 Days

1. Add config-file ingestion in `nmc` (`--config path.json`) with CLI override precedence.
2. Add unit tests for SQLite persistence invariants and artifact schema validation.
3. Add regression thresholds in smoke benchmark (return floor, latency ceiling).
4. Add dashboard panels for advantage traces and policy-ratio drift from persisted reports.
5. Add MLflow run-link generation directly into run manifests.

## Next 90 Days

1. Introduce 6DOF orbital environment interface and perturbation packs.
2. Add LQR/MPC comparison harness with standardized metrics and confidence intervals.
3. Add safety envelope checks (actuation limits, residual bounds, unstable trajectory flags).
4. Add ARM cross-compilation profile and embedded latency profiling scripts.
5. Add model-serving contract tests between LibTorch and ONNX runtimes.

## Long-Term Direction (Orbital Autonomy)

1. Mission-level objective packs (station-keeping, rendezvous, collision avoidance).
2. Hierarchical policy structure for tactical vs strategic orbital maneuvers.
3. Formalized control-theoretic validation with Lyapunov-style empirical certificates.
4. Optional TensorRT inference backend with strict parity checks and fallback to LibTorch.
5. Digital-twin replay workflows for failure analysis and mission auditability.

## Deferred by Design

- CUDA acceleration (intentionally not required in baseline).
- Active TensorRT dependency (kept optional and future-scoped).
- Distributed training infrastructure (deferred until orbital environment fidelity increases).

