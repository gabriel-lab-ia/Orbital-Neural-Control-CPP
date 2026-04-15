<p align="center">
  <img src="docs/assets/orbital-hero-banner.svg" alt="Orbital Neural Control CPP hero" width="100%" />
</p>

# Orbital Neural Control CPP

**C++20 orbital autonomy systems platform for reinforcement learning, mission simulation, telemetry tracking, and extensible control architecture.**

CPU-first engineering baseline with PPO + LibTorch, SQLite mission telemetry persistence, reproducible benchmark workflows, and layered software architecture designed for advanced control-system evolution.

## Stack and Engineering Identity

<p align="center">
  <img src="https://img.shields.io/badge/Linux-First-1E3A52.svg" alt="Linux first" />
  <img src="https://img.shields.io/badge/C%2B%2B-20-0C63A7.svg" alt="C++20" />
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-PPO-1F6ED4.svg" alt="RL PPO" />
  <img src="https://img.shields.io/badge/Backend-LibTorch%20CPU-1D6A3A.svg" alt="LibTorch CPU" />
  <img src="https://img.shields.io/badge/Tracking-SQLite-0B7285.svg" alt="SQLite" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Simulation-Orbital%20Mission%20Direction-1C7ED6.svg" alt="Orbital simulation direction" />
  <img src="https://img.shields.io/badge/Telemetry-Run%2FEpisode%2FEvent-1864AB.svg" alt="Mission telemetry" />
  <img src="https://img.shields.io/badge/Benchmarking-Reproducible-2B8A3E.svg" alt="Repro benchmarking" />
  <img src="https://img.shields.io/badge/UML-Architecture%20Discipline-364FC7.svg" alt="UML architecture" />
  <img src="https://img.shields.io/badge/Systems-Engineering-0A6CA8.svg" alt="Systems engineering" />
</p>

<p align="center">
  <a href="https://github.com/gabriel-lab-ia/PPO_Neural-Control-cpp/actions/workflows/ci.yml"><img src="https://github.com/gabriel-lab-ia/PPO_Neural-Control-cpp/actions/workflows/ci.yml/badge.svg" alt="CI build" /></a>
  <img src="https://img.shields.io/badge/CI-Smoke%20Benchmark-2B8A3E.svg" alt="Smoke benchmark" />
  <img src="https://img.shields.io/badge/Baseline-CPU%20Only-495057.svg" alt="CPU only" />
  <img src="https://img.shields.io/badge/Optional-MuJoCo-495057.svg" alt="MuJoCo optional" />
  <img src="https://img.shields.io/badge/Future-TensorRT%20Path-495057.svg" alt="TensorRT future path" />
</p>

<p align="center">
  <img src="docs/assets/mission-capabilities-strip.svg" alt="Mission capabilities" width="100%" />
</p>

## Why This Repository Exists

This repository exists to bridge **software engineering rigor** and **AI control-system development**:

- reproducible RL workflows that can survive CI and long-term maintenance
- architecture boundaries that support growth from toy environments to mission-scale simulations
- telemetry and persistence that make experiments auditable, not anecdotal

## Orbital Systems Vision

The near-term baseline is PPO continuous control in C++20.

The strategic direction is a mission-oriented autonomy stack:

- orbital environment adapters under a stable `Environment` contract
- policy optimization loops that can incorporate mission-level objectives
- telemetry pipelines for mission replay, failure analysis, and benchmark comparisons
- software architecture that supports advanced simulation domains without rewriting the RL core

## Architecture Overview

<p align="center">
  <img src="docs/assets/architecture-mission-strip.svg" alt="Layered architecture" width="100%" />
</p>

Core code layout:

- `src/domain/`: PPO logic, model, environment interfaces, inference contracts
- `src/application/`: train/eval/benchmark orchestration
- `src/infrastructure/`: artifacts, checkpoints, SQLite persistence, reporting
- `src/interfaces/`: CLI surface and entrypoints
- `src/common/`: shared cross-cutting utilities

UML references:

- `docs/uml/component-diagram.md`
- `docs/uml/class-diagram.md`
- `docs/uml/sequence-training.md`

## Mathematical Control Foundation

<p align="center">
  <img src="docs/assets/math-control-foundation.svg" alt="PPO control math foundation" width="100%" />
</p>

Practical interpretation in this project:

- clipped objective constrains policy ratio updates to reduce destructive jumps
- actor-critic split combines stochastic control policy and value stabilization
- GAE(lambda) improves advantage quality for training stability
- entropy regularization keeps exploration active in continuous action spaces
- policy outputs follow Gaussian control parameterization (mean + log std)

This keeps optimization behavior aligned with control-system reliability instead of raw reward chasing.

## Simulation and Telemetry Pipeline

```text
Build -> Train -> Evaluate -> Persist -> Benchmark -> Inspect
```

Operational commands:

```bash
./build/nmc train --env point_mass --seed 7 --updates 30
./build/nmc eval --checkpoint artifacts/latest/checkpoint.pt --episodes 10 --backend libtorch
./build/nmc benchmark --quick --name smoke
```

Artifacts produced under `artifacts/`:

```text
runs/<run_id>/manifest.json
runs/<run_id>/training_metrics.csv
runs/<run_id>/training_summary.json
runs/<run_id>/evaluation_summary.json
runs/<run_id>/checkpoints/policy_last.pt
benchmarks/latest.json
experiments.sqlite
```

## Experiment Tracking and Benchmarking

SQLite (`artifacts/experiments.sqlite`) stores:

- `runs`: lifecycle, config, status, summary
- `episodes`: train/eval episode telemetry
- `events`: runtime events and important transitions
- `benchmarks`: benchmark summaries

This provides local-first experiment traceability suitable for engineering iteration and review.

## CI and Reproducibility

CI workflow validates a meaningful baseline:

1. configure + build (CPU-first path)
2. CTest smoke benchmark execution
3. artifact existence validation (`benchmark`, `manifest`, `checkpoint`)

This repository treats reproducibility and validation as product-level requirements.

## Roadmap Toward Advanced Orbital Autonomy

- add mission-grade orbital environments under existing env interface
- extend reward/control objectives toward mission constraints and safety envelopes
- scale telemetry and benchmark diagnostics for comparative mission studies
- keep backend abstraction ready for future inference acceleration without breaking CPU-first baseline

See full roadmap: `docs/roadmap.md`

## Build Baseline

```bash
bash tools/setup_libtorch_cpu.sh
cmake --preset dev
cmake --build --preset build
```

Optional MuJoCo remains gated behind `NMC_ENABLE_MUJOCO=ON` and is not required for baseline CI path.
