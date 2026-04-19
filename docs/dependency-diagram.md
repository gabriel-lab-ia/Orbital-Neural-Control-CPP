# Dependency Diagram

```mermaid
graph TD
  subgraph Toolchain
    CMake["CMake >= 3.26"]
    Ninja["Ninja"]
    GCC["g++ (C++20)"]
    Vcpkg["external/vcpkg (manifest mode)"]
  end

  subgraph ThirdParty
    LibTorch["third_party/libtorch (prebuilt)"]
    Eigen["eigen3 (vcpkg)"]
    SQLite["sqlite3 (vcpkg)"]
    Boost["boost-json/system (optional backend feature)"]
  end

  subgraph Runtime
    NMC["nmc runtime"]
    PPO["PPO trainer/policy/value"]
    Sim["orbital + point-mass simulation"]
    Infer["inference backends"]
    Artifacts["artifact pipeline + SQLite telemetry"]
  end

  CMake --> NMC
  Ninja --> NMC
  GCC --> NMC
  Vcpkg --> Eigen
  Vcpkg --> SQLite
  Vcpkg --> Boost
  LibTorch --> Infer
  Eigen --> PPO
  SQLite --> Artifacts
  PPO --> NMC
  Sim --> NMC
  Infer --> NMC
  Artifacts --> NMC
```

## Deterministic constraints

- fixed triplet: `x64-linux`
- vcpkg pinned by `builtin-baseline`
- LibTorch loaded from pinned prebuilt package path (`third_party/libtorch`)
- TensorRT path optional with non-fatal fallback to LibTorch
