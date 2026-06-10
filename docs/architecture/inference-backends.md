# Inference Backends

```mermaid
flowchart TD
    Eval[nmc eval] --> Resolver[Resolve LibTorch fallback device]
    Eval --> Factory[Inference Backend Factory]
    Factory --> LT[LibTorch]
    LT --> CPU[libtorch_cpu]
    LT --> CUDA[libtorch_cuda]
    Factory --> TRT[TensorRT Controller]
    TRT --> Native{Native artifact and runtime available?}
    Native -->|yes| Engine[tensorrt_native]
    Native -->|no| Fallback[LibTorch fallback on resolved device]
```

TensorRT native execution requires an ONNX or engine artifact and a compatible TensorRT/CUDA build. A `.pt` evaluation through a TensorRT CLI name may use LibTorch fallback. Summaries expose `runtime`, `uses_cuda`, and `is_emulated`; fallback latency must not be described as TensorRT acceleration.

