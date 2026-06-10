# PPO Training Flow

```mermaid
sequenceDiagram
    participant Env as CPU Environments
    participant Trainer as PPOTrainer
    participant Device as LibTorch CPU/CUDA
    participant Store as Artifacts + SQLite

    Env->>Trainer: CPU observations
    Trainer->>Device: batched observations.to(device)
    Device-->>Trainer: actions, log probs, values
    Trainer->>Env: actions.to(CPU)
    Env-->>Trainer: rewards, terminated, truncated, next observations
    Trainer->>Device: rollout tensors, GAE, minibatches, optimizer
    Device-->>Trainer: metrics and updated model
    Trainer->>Store: CPU serialization and structured metadata
```

The environment boundary is intentionally explicit. CUDA currently accelerates policy/value work, not environment propagation. This avoids claiming a CUDA-first simulator while preserving a path toward vectorized GPU environments.

Truncation and termination semantics remain distinct in the rollout buffer so time-limit truncation can bootstrap value estimates.

