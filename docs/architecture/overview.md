# Architecture Overview

Orbital Neural Control CPP ships a reproducible C++20 baseline and keeps optional platform modules outside the critical CLI path.

```mermaid
flowchart LR
    CLI[nmc train / eval / benchmark] --> Resolver[Device Resolver]
    Resolver --> CPU[LibTorch CPU]
    Resolver --> CUDA[LibTorch CUDA]
    CLI --> Env[CPU Environment Simulation]
    Env --> PPO[PPO Trainer]
    PPO --> CPU
    PPO --> CUDA
    CPU --> Artifacts[Artifacts + SQLite]
    CUDA --> Artifacts
    CLI --> Infer[Inference Backend Factory]
    Infer --> CPU
    Infer --> CUDA
    Infer --> TRT[TensorRT Native or Explicit Fallback]
    Artifacts --> Backend[Optional C++ API]
    Backend --> UI[Optional Mission Control UI]
    FastAPI[Optional FastAPI Stub] -. future bridge .-> Infer
```

## Shipped Boundaries

- The default CI and reproducibility path is LibTorch CPU.
- CUDA-aware LibTorch training and inference are selected at runtime when the installed LibTorch build and host support CUDA.
- Environments remain CPU simulated. Observations cross to the selected LibTorch device and actions cross back to CPU deliberately.
- TensorRT reports whether native runtime or LibTorch fallback was used.
- Backend, frontend, FastAPI, Docker, and Kubernetes remain optional modules.

