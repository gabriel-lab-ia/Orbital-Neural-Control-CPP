# Artifact Contract

The existing layout is preserved:

```text
artifacts/
  runs/<run_id>/
    manifest.json
    training_metrics.csv
    training_summary.json
    evaluation_summary.json
    live_rollout.csv
    checkpoints/policy_last.pt
    checkpoints/policy_last.meta
  latest/
  benchmarks/
  experiments.sqlite
```

Schema `1.1` manifests and summaries add a `runtime` object:

```json
{
  "compute_backend_requested": "auto",
  "compute_backend_resolved": "cpu",
  "torch_device": "cpu",
  "cuda_available": false,
  "cuda_device_count": 0,
  "cuda_device_index": 0,
  "cuda_device_name": "",
  "cuda_fallback_used": true
}
```

Older schema `1.0` artifacts remain valid. Strict validation requires runtime metadata for new schema `1.1` outputs.

