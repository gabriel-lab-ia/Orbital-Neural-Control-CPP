# Frontend Mission Console

Technical mission-control client built with **React + TypeScript + Vite**.

This frontend is an operational console for:
- run selection
- replay and live telemetry streaming
- synchronized 3D orbital visualization
- mission timeline + event markers
- benchmark and telemetry inspection

## Runtime Architecture

The frontend is organized around one shared mission state and one backend contract layer:

```text
src/
  api/                  # runtime orchestration (REST snapshot + WS live merge)
  store/                # global mission store (run/mode/playback/stream state)
  shared/api/           # generated OpenAPI types + HTTP client
  entities/             # stable UI domain models and adapters
  features/             # mission features (orbital view, replay controls, telemetry)
  widgets/              # composed technical panels
  pages/                # route-level pages
```

## What Is Implemented (Current)

- React 18 + TypeScript strict mode + Vite
- OpenAPI-typed backend integration (REST + WebSocket flows)
- Global mission store for replay/live synchronization
- 3D Earth viewport with:
  - day/night textures
  - cloud layer
  - normal/specular maps
  - atmosphere glow
  - orbit path + spacecraft marker
  - camera modes (`earth_lock`, `spacecraft_follow`, `mission_replay`, `free_inspect`)
- Replay controls:
  - play/pause
  - step forward/back
  - scrubber
  - speed multiplier
  - keyboard shortcuts (`Space`, `←`, `→`)
- Mission HUD panels for telemetry and benchmark summaries
- Run Explorer page (`/runs`)

## Backend Contract Integration

- Source of truth: OpenAPI-generated types under `src/shared/api/generated/`.
- REST used for initial snapshot/load:
  - runs
  - replay payloads
  - summaries
- WebSocket used for live stream updates:
  - telemetry sample
  - replay chunk
- Fallback behavior:
  - if backend endpoints are unavailable, frontend enters degraded mode and uses local replay dataset for demo continuity.

## Local Run

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Earth Texture Setup (Recommended)

For realistic Earth material maps, place files in:

```text
frontend/public/textures/earth/
  earth_day_4k.jpg
  earth_night_4k.jpg
  earth_clouds_4k.png
  earth_normal_4k.jpg
  earth_specular_4k.jpg
```

If these files are missing, the frontend automatically falls back to procedural textures (no crash).

## Suggested Texture Sources

- NASA Visible Earth / Blue Marble
- NASA Earth Observatory public imagery
- Solar System Scope texture packs (license-check before commercial use)

Always verify license and attribution terms before shipping.

## Backend Wiring

Set Vite environment variables when using the backend:

```bash
VITE_BACKEND_HTTP=http://localhost:8080
VITE_BACKEND_WS=ws://localhost:8080
```

## Engineering Notes

- Desktop-first layout; still responsive for medium/small screens.
- 3D renderer is lazy-loaded to avoid inflating initial route bundle.
- `orbital-scene-canvas` remains the largest chunk by design; heavy rendering is isolated from initial app boot.
