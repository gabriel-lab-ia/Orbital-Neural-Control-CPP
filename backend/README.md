# backend/

Mission telemetry backend with REST + WebSocket endpoints.

## Endpoints

- `GET /health`
- `GET /api/telemetry/snapshot`
- `WS /ws/telemetry`

## Build

```bash
cmake -S backend -B build/backend
cmake --build build/backend
./build/backend/orbital_backend
```
