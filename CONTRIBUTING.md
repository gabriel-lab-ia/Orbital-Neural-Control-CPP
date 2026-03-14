# Contributing

Thank you for considering a contribution to `Neuro Motor CPP`.

## Scope

Contributions are welcome in these areas:

- PPO training stability and code quality
- MuJoCo environment adapters
- visualization and export tooling
- documentation and reproducibility
- testing and CI improvements

## Development Workflow

1. Fork the repository.
2. Create a feature branch.
3. Build locally with CMake.
4. Run the baseline training flow.
5. Submit a pull request with a focused change.

## Local Build

```bash
cmake --preset dev
cmake --build --preset build
./build/motor
```

## Code Style

- Prefer modern C++20.
- Keep interfaces small and explicit.
- Avoid unnecessary abstraction in performance-critical paths.
- Preserve the current modular layout under `src/env`, `src/model`, `src/train`, and `src/utils`.

## Pull Request Guidelines

- Keep changes narrowly scoped.
- Describe behavioral impact clearly.
- Include validation steps.
- Add or update documentation when behavior changes.

## Issues

If you open an issue, please include:

- operating system
- compiler version
- whether MuJoCo is enabled
- exact command used
- relevant logs or screenshots
