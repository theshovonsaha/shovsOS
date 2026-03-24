# Frontend Nova

Operator workspace for Shovs LLM OS.

## Goals

- Replicate existing core capabilities (chat, sessions, tools, monitor, logs, voice, guardrails).
- Improve scalability for additional backend surfaces.
- Keep the interface minimal by default while exposing deep diagnostics on demand.
- Let users build stronger agents through profile defaults, bootstrap docs, and runtime controls.

## Run

```bash
cd frontend_nova
npm install
npm run dev
```

The app uses Vite proxy settings to target backend routes at `http://127.0.0.1:8000`.

## Current Responsibilities

- operator chat and sessions
- trace monitor and readable timeline
- loop, planner, and reasoning controls
- storage admin
- agent builder presets and bootstrap composition summary

## Build

```bash
npm run build
```
