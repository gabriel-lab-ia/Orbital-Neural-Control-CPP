# Frontend Mission Control

```mermaid
flowchart LR
    API[C++ REST API] --> Client[Typed OpenAPI Client]
    WS[WebSocket Replay / Telemetry] --> Runtime[Mission Runtime Store]
    Client --> Runtime
    Demo[Clearly Labeled Demo Dataset] --> Runtime
    Runtime --> Orbit[3D Orbit View]
    Runtime --> Telemetry[Telemetry + Device Cards]
    Runtime --> Replay[Timeline + Events]
    Runtime --> Runs[Run Explorer]
```

The React/TypeScript UI exposes backend runtime, resolved Torch device, CUDA fallback status, latency, replay, and artifacts where the API provides them. When the backend is unavailable, the UI labels the source as mock/demo fallback. It must not present demo values as live mission telemetry.

PPO time-series charts remain a roadmap item until the backend exposes structured training metrics rather than requiring the browser to parse CSV artifacts.

