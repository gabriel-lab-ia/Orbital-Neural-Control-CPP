# Runtime Device Selection

The shared resolver lives in `src/domain/runtime/device_resolver.*`.

```mermaid
flowchart TD
    Start[Read --device and --cuda-device] --> CPUReq{device = cpu?}
    CPUReq -->|yes| CPU[Resolve cpu]
    CPUReq -->|no| Available{CUDA available and index valid?}
    Available -->|yes| CUDA[Resolve cuda:index]
    Available -->|no| Strict{device = cuda and --no-cuda-fallback?}
    Strict -->|yes| Fail[Fail before run starts]
    Strict -->|no| Fallback[Resolve cpu and record fallback]
```

Commands:

```bash
./build/nmc train --device cpu
./build/nmc train --device auto
./build/nmc train --device cuda --cuda-device 0 --no-cuda-fallback
./build/nmc eval --device auto --checkpoint artifacts/latest/checkpoint.pt
./build/nmc benchmark --quick --device auto
```

`cpu` is the default. `auto` and non-strict `cuda` requests record any fallback in manifests and summaries. The portable C++ API reports CUDA availability and device count; the metadata name remains `cuda:<index>` unless a future CUDA SDK-specific inspector is added.

