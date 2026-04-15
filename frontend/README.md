# Frontend Mission Dashboard

Next.js 15 mission-control frontend for live orbital telemetry visualization.

## Stack

- Next.js 15 + TypeScript
- Tailwind CSS
- shadcn/ui-style component primitives
- React Three Fiber + Drei
- Recharts

## Local Run

```bash
cd frontend
npm install
npm run dev
```

Set backend endpoints when needed:

```bash
NEXT_PUBLIC_BACKEND_HTTP=http://localhost:8080
NEXT_PUBLIC_BACKEND_WS=ws://localhost:8080
```
