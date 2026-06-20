# Architecture

Shovs is organized as a human control center over an autonomous agent body.

## System Model

1. Human control center:

- `frontend_shovs`: operator-facing workspace for visibility, control, and debugging.
- `frontend_consumer`: simplified user-facing interaction plane.

2. Autonomous agent body:

- managed runtime execution (`run_engine`)
- run ledger, orchestration, tools, memory, traces, checkpoints, artifacts, and evals

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
- one run ledger model
- one tool-calling contract
- one memory-commit policy
- one frontend-visible execution story

## Runtime Flow (Operator View)

```text
{Human Control Center}
  frontend_shovs || frontend_consumer
    --> {/chat/stream || /consumer/chat/stream}
      --> {RunEngine}
        --> run ledger
        --> plan -> act -> observe -> verify -> memory_commit
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
- artifacts
- evals
- memory inspector payloads

3. Correctness surfaces:

- tool call/result linking
- side-effect verification
- response verification
- scenario-state evals for workflows where the path matters

## Design Principles

1. Truth lanes over transcript guessing.
2. Explicit phases over opaque loops.
3. Deterministic state transitions where possible.
4. Human-readable logs for all critical decisions.
5. Scenario evals over final-answer vibes.
6. Backward compatibility during convergence, then controlled deprecation.

## Key References

- [README.md](README.md)
- [documentation/public/VISION.md](documentation/public/VISION.md)
- [documentation/public/DEVELOPER_GUIDE.md](documentation/public/DEVELOPER_GUIDE.md)
- [ROADMAP.md](ROADMAP.md)
