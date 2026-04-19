# Contributing

Thanks for contributing to **Orbital Neural Control CPP**.

This repository is a C++20 RL systems baseline with an orbital-autonomy direction. Contributions should improve reliability, reproducibility, and maintainability.

Baseline vs optional scope:

- Baseline (required): `src/` + `nmc` (`train`, `eval`, `benchmark`)
- Optional modules: `core`, `control`, `sim`, `rl`, `training`, `mlops`, `backend`, `frontend`

## Engineering Principles

- Keep the CPU-first baseline (`nmc`) working.
- Prefer explicit interfaces and low coupling.
- Keep optional integrations optional (MuJoCo, backend/frontend, MLflow).
- Do not add hidden runtime dependencies.
- Keep documentation aligned with real behavior.

## Development Setup

```bash
./tools/setup_vcpkg.sh
export VCPKG_ROOT="$HOME/.vcpkg"
cmake --preset dev
cmake --build --preset build
```

Primary executable:

```bash
./build/nmc help
```

## Validation Before PR

Run at least these checks locally:

```bash
./build/nmc benchmark --quick --name pre_pr_smoke --seed 7
./build/nmc train --quick --run-id pre_pr_train_001 --seed 7
./build/nmc eval --checkpoint artifacts/latest/checkpoint.pt --episodes 10 --backend libtorch --run-id pre_pr_eval_001 --seed 7
ctest --test-dir build --output-on-failure --verbose -R nmc_smoke_benchmark
python3 scripts/validate_artifacts.py --root artifacts --strict
```

CI-equivalent local path:

```bash
export VCPKG_ROOT="$HOME/.vcpkg"
cmake --preset ci
cmake --build --preset build-ci --verbose
./build-ci/nmc help
ctest --test-dir build-ci --output-on-failure --verbose --no-tests=error -R nmc_smoke_benchmark
```

If you touch orbital core modules:

```bash
cmake --preset orbital-core-only
cmake --build --preset build-orbital
ctest --test-dir build-orbital-core --output-on-failure
```

## Code Conventions

- C++20, RAII, const-correctness, and explicit ownership.
- Keep `src/` layering intact:
  - `domain` for RL/environment logic
  - `application` for orchestration
  - `infrastructure` for persistence/artifacts/reporting
  - `interfaces` for CLI
- Keep names descriptive and avoid throwaway identifiers.
- Add comments only when clarifying intent or design constraints.

## Pull Request Guidelines

- Keep PR scope focused.
- Include a short problem statement and validation steps.
- Update docs when behavior/commands/paths change.
- If functionality is roadmap-only, mark it clearly (do not present as shipped).

## Reporting Issues

Please include:

- OS and compiler versions
- exact configure/build/run commands
- whether MuJoCo is enabled
- logs or stack traces
- generated artifact paths when relevant
