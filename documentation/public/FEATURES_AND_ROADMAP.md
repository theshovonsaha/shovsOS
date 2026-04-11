# Shovs LLM OS Features and Roadmap

This page reflects the current project as implemented, not the older marketing framing.

## Core Features Implemented

### 1. Language OS Runtime

- canonical managed runtime in [run_engine/engine.py](../../run_engine/engine.py)
- compatibility-native runtime in [engine/core.py](../../engine/core.py)
- phase-aware context compilation
- selectable execution modes:
  - `single`
  - `managed`
  - `auto`

### 2. Managed Loop

Managed mode supports:
- `plan`
- `act`
- `observe`
- `verify`
- `commit`

This happens inside one run, not as a default multi-agent swarm.

### 3. Run Model

- first-class `run_id`
- persisted loop checkpoints
- run artifacts
- run evals
- trace events tied to runs

### 4. Memory and State

- deterministic facts
- candidate signal lane
- semantic graph memory
- vector memory
- session RAG
- task tracking

### 5. State Integrity

- grounded fact filtering
- failed-turn preservation
- task-bootstrap suppression inside a run
- follow-up context sanitation after tool execution
- prompt overflow retry for local models
- cumulative evidence retention across a run so verified exact-domain fetches survive later noisy searches

### 6. Provider Layer

Supported providers:
- Ollama
- LM Studio
- llama.cpp
- local OpenAI-compatible servers
- OpenAI
- Groq
- Anthropic
- Gemini
- Nvidia

### 7. Model-Aware Runtime Shaping

- execution profiles for small local, local standard, and frontier-class models
- adaptive prompt budgets
- adaptive evidence packet sizing
- smaller acting surfaces for weaker local models

### 8. Memory and Embedding Compatibility

- runtime embed-model propagation into memory tools
- Ollama embedding compatibility across `/api/embed` and legacy `/api/embeddings`
- OpenAI-compatible embedding transport for LM Studio, llama.cpp, and local OpenAI servers

### 9. Frontend Planes

- Nova workspace
- consumer frontend

Nova already includes:
- model/provider controls
- loop controls
- planner toggle
- reasoning visibility
- readable monitor
- storage admin
- agent builder with presets
- bootstrap-doc and prompt contribution summary

## Why This Matters

The value of the system is not only "many tools" or "many providers."

The differentiator is:
- explicit execution structure
- truthful state transitions
- smaller, cleaner prompt payloads
- better coherence for smaller local models

## What Is Still In Progress

These are the remaining meaningful gaps.

### 1. Checkpoint-Native Prompting

The runtime already sanitizes follow-up context, but more of the prompt can still be compiled directly from checkpoint/evidence state instead of conversational carryover.

### 2. External Adapter Parity In Practice

The contract exists and managed runtime is default, but provider-specific behavior still needs broader long-run validation in heterogeneous deployments.

### 3. Runtime Decomposition

The behavior is right, but too much control logic still lives in [engine/core.py](../../engine/core.py).

### 4. Stronger Small-Model Tool Obedience

This is much better than before, but still one of the main practical gaps for local small models.

### 5. Richer Monitor Lanes

The monitor is much more readable now, but reasoning and observer activity can still be surfaced more clearly.

### 6. Active Workspace Builder Controls

The builder is functional in Nova Dashboard, but the same controls should also be easier to reach from the active workspace.

## Near-Term Roadmap

### Stability and Release

- continue hardening small-model tool execution
- improve degraded-mode behavior for unstable providers
- add more release-facing examples and walkthroughs
- document recommended production storage and retention patterns

### Runtime

- push more context compilation to checkpoint-derived packets
- keep shrinking raw message carryover
- continue separating task/admin state from evidence state

### Adapters

- validate the managed-loop contract on more real non-native runtimes
- improve model listing, health reporting, and capability discovery

### Product Planes

- further tighten consumer behavior on top of the kernel
- keep Nova focused, mobile-friendly, operator-readable, and builder-capable

## Longer-Term Direction

The long-term direction remains:
- one kernel
- multiple planes
- plural context
- explicit runs and phases
- local-first and enterprise-ready deployment options

That is the actual project direction for Shovs LLM OS.
