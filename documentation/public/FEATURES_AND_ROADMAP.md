# Shovs LLM OS Features and Roadmap

This page describes the current project in plain language. It avoids older marketing framing and focuses on what the runtime actually does.

## Core Features Implemented

### 1. Language OS Runtime

- canonical managed runtime in [run_engine/engine.py](../../run_engine/engine.py)
- compatibility-native runtime in [engine/core.py](../../engine/core.py)
- phase-aware context compilation
- managed-first execution model
- compatibility shims for older request shapes

### 2. Managed Loop

Managed mode supports:

- `plan`
- `act`
- `observe`
- `verify`
- `memory_commit`

This happens inside one run. It is not a default multi-agent swarm. The runtime can delegate through tools or templates, but the normal path is one managed run with observable phases.

### 3. Run Ledger and Run Model

- first-class `run_id`
- canonical run ledger
- linked tool calls and tool results
- evidence records
- verification records
- continuation state for unfinished work
- persisted loop checkpoints
- run artifacts
- run evals
- trace events tied to runs

### 4. Workflow Contracts

- deterministic task-shape classification for simple chat, source collection, coding, research, and memory-correction-style turns
- source-collection contracts with entity quotas, per-entity source requirements, fetch requirements, tool policy, and completion gates
- contract snapshots persisted through the run ledger
- contract state visible in phase packets and runtime attention

### 5. Agent Pass Framework

- workflow contracts map to typed specialist passes rather than one large prompt
- pass roles include retrieval, reasoning, summary, scoring, evaluation, and orchestration
- context strategies distinguish direct response, local retrieval, chunk-wise analysis, global reasoning, deterministic ranking, scenario-state evals, and completion gates
- pass graphs are stored in the run ledger so the UI and tests can inspect the intended workflow structure
- pass graph execution state tracks node status, dependencies, attempts, outputs, and graph completion

### 6. Runtime Attention

- deterministic phase-weighted scoring over run ledger records
- separate heads for objective, pending work, tool state, evidence, verification, memory, continuation, and risk
- workflow contract and pass graph records included as high-signal state
- attention snapshots included in phase packets and traces
- designed to make context selection inspectable without modifying model internals

### 7. Memory and State

- deterministic facts
- candidate signal lane
- temporal voiding for corrected facts
- conflict and dispute visibility
- semantic graph memory
- vector memory
- session RAG
- task tracking

### 8. State Integrity

- grounded fact filtering
- failed-turn preservation
- task-bootstrap suppression inside a run
- follow-up context sanitation after tool execution
- prompt overflow retry for local models
- cumulative evidence retention across a run so verified exact-domain fetches survive later noisy searches
- source-contract workflow guards for entity/source collection tasks
- ledger-enforced source gates for opt-in runs: wrong next tools, unlocked-entity searches, and off-contract fetches are detected before execution
- completion gates that mark source workflows incomplete until required evidence is present
- scenario-state evals that catch wrong tool paths even when final text sounds plausible

### 9. Tool and Evidence Discipline

- hidden tool-call draft parsing before UI emission
- unknown tool rejection before execution
- duplicate tool-call suppression
- ledger-authority source selection in managed runs, so deterministic source tools use successful prior tool results instead of model-supplied payloads
- deterministic pivoting when a workflow contract knows the next required action
- policy violation and recovery events for drift, duplicate loops, missing evidence, failed fetches, and unsupported response claims
- side-effect guard for unsupported file/write claims
- `HARD_FAILURE` status when write tools cannot verify expected outputs

### 10. Provider Layer

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

### 11. Model-Aware Runtime Shaping

- execution profiles for small local, local standard, and frontier-class models
- adaptive prompt budgets
- adaptive evidence packet sizing
- smaller acting surfaces for weaker local models

### 12. Memory and Embedding Compatibility

- runtime embed-model propagation into memory tools
- Ollama embedding compatibility across `/api/embed` and legacy `/api/embeddings`
- OpenAI-compatible embedding transport for LM Studio, llama.cpp, and local OpenAI servers

### 13. Frontend Planes

- Shovs Platform workspace
- consumer frontend

Shovs Platform already includes:

- model/provider controls
- loop controls
- planner toggle
- reasoning visibility
- Harness Lab for comparing plain model behavior against Shovs runtime wedges
- Harness Lab policy comparison across plain model, model + tools, Shovs ReAct, Shovs Plan-Execute, and Shovs Graph Harness
- Trace Monitor lanes for Policy, Graph, and Recovery events
- readable monitor
- trace replay
- run eval display
- storage admin
- agent builder with presets
- bootstrap-doc and prompt contribution summary

## Why This Matters

The value of the system is not only "many tools" or "many providers."

The research bet is:

- explicit execution structure
- truthful state transitions
- smaller, cleaner prompt payloads
- more coherent behavior for smaller local models in tested paths
- failures that can be replayed and understood

The public credibility path is:

- [HARNESS.md](../../HARNESS.md)
- [BENCHMARKS.md](../../BENCHMARKS.md)
- [EVALS.md](../../EVALS.md)
- [CLAIMS.md](../../CLAIMS.md)
- [RESULTS.md](../../RESULTS.md)

## What Is Still In Progress

These are the remaining meaningful gaps.

### 1. Ledger-Enforced Prompting

The runtime already carries a run ledger and phase packets. More prompt inputs should be derived directly from ledger requirements instead of local lists or conversational carryover.

### 2. Broader Scenario Evals

The source-collection scenario eval exists. More workflow templates should have scenario evals: shopping advice, coding changes, research reports, memory correction, and multi-turn continuation.

The first public benchmark surface is [Agent Harness Core](../../benchmarks/agent_harness_core/README.md). It should stay small, deterministic, and easy to reproduce.

### 3. External Adapter Parity In Practice

The contract exists and managed runtime is default, but provider-specific behavior still needs broader long-run testing across heterogeneous deployments.

### 4. Runtime Decomposition

The behavior is converging, but too much control logic still lives in [engine/core.py](../../engine/core.py).

### 5. Stronger Small-Model Tool Obedience

This is much better than before, but still one of the main practical gaps for local small models.

### 6. Richer Monitor Lanes

The monitor is much more readable now, but source contracts, missing slots, continuation state, workflow plugin decisions, and eval failures can still be surfaced more clearly.

### Workflow Plugins

ShovsOS now routes domain-specific source-collection behavior through workflow plugins instead of adding one-off branches directly inside the run loop.

Current plugins:

- `stock_movers_source_collection`: locks stock tickers from a mover source, then enforces separate searches and fetched-source quotas.
- `local_place_source_collection`: locks places from search results, then enforces separate searches and fetched-source quotas.
- `comparison_source_collection`: locks compared items/products/options, then enforces per-entity evidence collection.

This is the pattern for future niches: shopping comparison, local-store checks, restaurant research, paper triage, or enterprise evidence collection can each become a plugin while the engine keeps the same contract, ledger, trace, and verification rules.

### Control Policies

The managed runtime now resolves an explicit control policy per run.

- `react` for lightweight reason-act-observe tasks.
- `plan_observe` for the existing planner/actor/observer compatibility loop.
- `plan_execute` for web/source workflows where untrusted page content should not rewrite the task.
- `graph_harness` for coding, higher-risk, or long-horizon workflows with typed pass graphs.

This lets ShovsOS compare and compose agent architectures instead of betting the whole system on one loop style.

Harness Lab now exposes those policy differences directly. It returns each mode's control policy, ledger mode, trace summary, state-eval score, policy-trace score, and issues, so the UI can show why one run drifted and another passed.

### Context Ladder

Phase packets now include a compact context ladder:

- keyword hint
- compact memory signal
- relevant block
- evidence reference
- raw payload reference

This keeps actor context smaller and more inspectable. Raw payloads stay available for verifier and UI inspection instead of being dumped into every model turn.

### 7. Active Workspace Builder Controls

The builder is functional in Shovs Platform Dashboard, but the same controls should also be easier to reach from the active workspace.

## Near-Term Roadmap

### Stability and Release

- continue hardening small-model tool execution
- improve degraded-mode behavior for unstable providers
- add more release-facing examples and walkthroughs
- document recommended production storage and retention patterns

### Runtime

- push more context compilation to ledger-derived packets
- keep shrinking raw message carryover
- continue separating task/admin state from evidence state
- make workflow templates declare eval scenarios and evidence requirements
- promote selected shadow-mode ledger checks into enforcement where tests prove stability

### Adapters

- validate the managed-loop contract on more real non-native runtimes
- improve model listing, health reporting, and capability discovery

### UI Planes

- further tighten consumer behavior on top of the runtime
- keep Shovs Platform focused, mobile-friendly, operator-readable, and builder-capable

## Longer-Term Direction

The long-term direction remains:

- one kernel
- multiple planes
- plural context
- explicit runs and phases
- local-first deployment and enterprise-readiness research

That is the actual project direction for Shovs LLM OS.
