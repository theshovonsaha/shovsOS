# Architecture

Shovs is organized as a human control center over an autonomous agent body.

## System Model

1. Human control center:

- `frontend_shovs`: operator-facing workspace for visibility, control, and debugging.
- `frontend_consumer`: simplified user-facing interaction plane.

2. Autonomous agent body:

- runtime execution (`run_engine`)
- orchestration, tools, memory, traces, checkpoints, and evals

3. Shared substrate:

- session state
- run store
- semantic/vector memory
- tool registry
- provider adapters

## Current Runtime Reality

Canonical execution center:

- managed runtime: `run_engine/engine.py`

Convergence objective (active):

- one canonical runtime contract
- one pass ledger model
- one tool-calling contract
- one memory-commit policy
- one frontend-visible execution story

## Runtime Flow (Operator View)

```text
{Human Control Center}
  frontend_shovs || frontend_consumer
    --> {/chat/stream || /consumer/chat/stream}
      --> {RunEngine}
        --> plan -> act -> observe -> verify -> commit
        --> ToolRegistry
        --> SessionManager
        --> RunStore + TraceStore
        --> Memory lanes {deterministic facts || candidate signals}
        --> Provider Adapter {local/cloud}
```

## Control Planes

1. API entrypoints:

- `/chat/stream`
- `/consumer/chat/stream`
- `/logs/*`
- `/rag/*`

2. Explainability surfaces:

- trace events
- run checkpoints
- pass records
- memory inspector payloads

## Design Principles

1. Truth lanes over transcript guessing.
2. Explicit phases over opaque loops.
3. Deterministic state transitions where possible.
4. Human-readable logs for all critical decisions.
5. Backward compatibility during convergence, then controlled deprecation.

## Key References

- [README.md](README.md)
- [documentation/public/VISION.md](documentation/public/VISION.md)
- [documentation/public/DEVELOPER_GUIDE.md](documentation/public/DEVELOPER_GUIDE.md)
- [ROADMAP.md](ROADMAP.md)
