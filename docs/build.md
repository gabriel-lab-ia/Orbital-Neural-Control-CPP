# Deterministic Build Guide

All commands must run from repository root.

## Build requirements

- Linux (primary target)
- CMake >= 3.26
- Ninja
- GCC/G++ (C++20)
- `libsqlite3-dev`
- `curl`, `unzip`

## One-command release build

```bash
./tools/build_release.sh
```

This script performs:

1. vcpkg bootstrap in `external/vcpkg` (manifest mode)
2. LibTorch prebuilt download in `third_party/libtorch`
3. Release configure with Ninja
4. Build

## Manual deterministic build

```bash
./tools/setup_vcpkg.sh
./tools/setup_libtorch.sh

cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-linux \
  -DVCPKG_HOST_TRIPLET=x64-linux \
  -DVCPKG_FEATURE_FLAGS=manifests,binarycaching

cmake --build build
```

## Smoke validation

```bash
ctest --test-dir build --output-on-failure --verbose -R "nmc_smoke_benchmark|nmc_inference_parity"
python3 scripts/validate_artifacts.py --root artifacts --strict
```

## TensorRT behavior

- `ENABLE_TENSORRT=ON` by default
- if TensorRT SDK is present, native runtime is enabled
- if TensorRT SDK is missing, build continues with LibTorch backend fallback

Force CPU-first baseline:

```bash
cmake -S . -B build -G Ninja -DENABLE_TENSORRT=OFF \
  -DCMAKE_TOOLCHAIN_FILE=external/vcpkg/scripts/buildsystems/vcpkg.cmake
```

## Notes on reproducibility

- vcpkg baseline is pinned in `vcpkg.json`
- triplet is fixed to `x64-linux`
- LibTorch is consumed from `third_party/libtorch` only
- CI uses Release + Ninja + smoke tests + artifact integrity validation

## Dependency policy

The default dependency graph intentionally excludes heavyweight/transitive stacks that are not required by the baseline runtime.

Removed from default path:

- OpenCV
- LevelDB
- MPI
- OpenCL
- Vulkan
- LLVM toolchain dependencies

Baseline manifest dependencies are limited to:

- `eigen3`
- `sqlite3`
