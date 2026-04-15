# core/

`core/` is the C++20 orbital control systems library layer.

## Scope

- orbital state and dynamics primitives
- control command and mission reward modeling
- PPO-oriented control objective helpers
- mission rollout API for telemetry generation

## Build

```bash
cmake -S core -B build/core
cmake --build build/core
ctest --test-dir build/core --output-on-failure
```

## Namespaces

- `orbital::simulation`
- `orbital::control`
- `orbital::rl`
