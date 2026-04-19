# third_party

This directory is intentionally reserved for prebuilt binary dependencies required for deterministic builds.

Current expected layout:

```text
third_party/
  libtorch/
    share/cmake/Torch/TorchConfig.cmake
```

Populate it with:

```bash
./tools/setup_libtorch.sh
```

Notes:

- `third_party/libtorch` is ignored by git.
- CI downloads and caches the same prebuilt archive to avoid source builds.
