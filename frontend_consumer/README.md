# Shovs Consumer Frontend

Consumer-facing plane for Shovs LLM OS.

## Stack

- **React 19** + **TypeScript**
- **Framer Motion** — FSM-driven input bar transitions, blur overlay, message materialization, options panel slide-in
- **Three.js** — Real-time GLSL shader orb with phase-driven fluid simulation (idle/thinking/working/finalizing)
- **Vite** with dev proxy to `http://127.0.0.1:8000`

## Key design decisions

### Input FSM

The input bar has 5 states: `idle → focused → submitted → streaming → done`
Framer Motion animates the bar width, border-radius, and padding between states. On submit, the bar compresses to a centered pill (480px max). On done, it expands back.

### Blur overlay

A `position: fixed` overlay with `backdrop-filter: blur(22px)` fades in during `submitted`/`streaming` states, dimming the thread so the agent status pill is the focus.

### Three.js orb (`AgentOrb.tsx`)

- Custom vertex shader: FBM (fractal Brownian motion) noise displaces sphere vertices, turbulence strength lerps between phases
- Custom fragment shader: Fresnel rim lighting, phase-driven color palette (cool blue idle → electric blue thinking → purple working → teal finalizing)
- Single canvas per orb instance, RAF loop with dt-clamped lerping for smooth phase transitions
- Used in 3 places: empty state (80px), message avatars (28px), status pill (32px)

### Agent status

- Appears above the composer when running
- `AnimatePresence` cycles through `activityShort` text with slide transitions
- After 6s of streaming, an expandable steps log fades in
- Cancel button appears in the topbar during active streams

### Options panel

- Slides in from the right (spring physics), backdrop dims the app
- Model selector, session clear, storage status, backup/reset with inline confirm

## Setup

```bash
npm install
npm run dev
```

Backend must be running on `http://127.0.0.1:8000`. The Vite dev server proxies `/api` to it.

## Product Role

This frontend is intentionally narrower than Shovs Platform:

- consumer chat experience
- lightweight settings and storage controls
- fewer operator diagnostics
- same backend kernel underneath

## API endpoints consumed

| Method | Path                           | Purpose                       |
| ------ | ------------------------------ | ----------------------------- |
| GET    | `/api/consumer/models`         | Fetch grouped model list      |
| GET    | `/api/consumer/options`        | Load saved model preference   |
| POST   | `/api/consumer/options`        | Save model preference         |
| POST   | `/api/consumer/chat/stream`    | SSE streaming chat (FormData) |
| GET    | `/api/consumer/storage/status` | Storage sizes                 |
| POST   | `/api/consumer/storage/backup` | Trigger backup                |
| POST   | `/api/consumer/storage/reset`  | Reset with backup             |

## SSE event protocol

```
data: {"type": "session", "session_id": "..."}
data: {"type": "phase", "phase": "thinking"|"working"|"finalizing"}
data: {"type": "activity_short", "text": "..."}
data: {"type": "activity_detail", "text": "...", "detail": "..."}
data: {"type": "token", "content": "..."}
data: {"type": "done", "session_id": "..."}
data: {"type": "error", "message": "..."}
```
