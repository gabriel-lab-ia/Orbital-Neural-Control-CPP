# Backend API Flow

```mermaid
sequenceDiagram
    participant UI as Mission Control
    participant API as C++ Backend
    participant DB as SQLite/PostgreSQL
    participant Files as Artifact Files

    UI->>API: GET /runs
    API->>DB: query run config + summary
    DB-->>API: runtime/backend metadata
    API-->>UI: typed run envelope
    UI->>API: GET /runs/{id}/replay
    API->>Files: read bounded telemetry CSV
    API->>DB: query events
    API-->>UI: replay payload
    UI->>API: WebSocket /ws/runs/{id}/stream
    API-->>UI: bounded replay chunks
```

The backend is optional and currently has no authentication layer. Job execution is disabled by default. Do not expose it publicly without authentication, authorization, request limits, TLS termination, and network policy.

