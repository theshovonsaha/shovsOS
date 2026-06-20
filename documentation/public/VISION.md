# Shovs Vision

## The Thesis

Shovs is a Language OS.

The model is still probabilistic. It can still make mistakes. The thesis is not that a runtime can make an LLM perfectly deterministic.

The thesis is that a runtime can make agent behavior more stable at the task level by keeping important state outside the model and making every step inspectable.

Shovs does this through:

- controlled context assembly
- explicit execution phases
- a canonical run ledger
- verified tool grounding
- structured fact and candidate lanes
- checkpoints, artifacts, traces, and evals
- replayable failures

The goal is to make LLM-powered systems:

- more coherent
- easier to inspect
- easier to compose
- more reliable across tools, sessions, and models

## Core Claim

Language is the interface.

Context is the operating surface.

The runtime is the stabilizer.

Shovs uses language to talk to users and models, but it does not treat the chat transcript as the only state. The runtime turns language into structured execution state, runs phases over that state, and then gives the model only the context it needs for the current job.

In plain terms:

- the user owns the goal
- the runtime owns the state
- the model helps plan, act, observe, and explain
- tools produce evidence
- verification checks whether the answer is allowed
- memory stores only what is eligible to persist

## What Makes Shovs Different

Shovs is not just:

- a prompt wrapper
- a chatbot shell
- a generic tool loop
- a thin UI over a model provider

Shovs is a runtime with:

- a managed `plan -> act -> observe -> verify -> memory_commit` loop
- a run ledger for plan steps, tool calls, tool results, evidence, memory writes, verification, and continuation state
- phase-aware context compilation
- deterministic fact vs candidate-signal lanes
- side-effect guards for tool claims
- scenario-state evals that inspect the path taken, not only the final answer
- provider portability across local and cloud models
- agent construction on top of the same kernel

## The Platform Idea

The platform is the kernel.

Agents are compositions built on top of it.

The product shape is:

- a human control center for supervising and steering execution
- an autonomous agent body that performs work across tools, memory, and model runtimes

In practical terms:

- control center: Shovs Platform and operator-facing observability surfaces
- autonomous body: runtime phases, tool execution, memory updates, and verification flows

A robust agent on Shovs is defined by a combination of:

- system prompt
- tool set
- workspace bootstrap docs
- model and embedding model
- planner defaults
- context engine mode
- memory behavior
- risk and evidence policy
- workflow template

That means developers should be able to build:

- native Shovs agents
- research agents
- coding agents
- operator agents
- OpenClaw-like workspace agents
- LangChain-style or LangGraph-style integrations

without changing the kernel itself.

## Why Small Models Matter

One of the strongest tests of the platform is whether it can make smaller local models behave coherently.

If the runtime can improve:

- prompt discipline
- tool obedience
- context quality
- failure recovery
- traceability

for small models, then larger frontier models benefit even more.

Shovs aims to:

- raise the floor for small models
- raise the ceiling for large models

This matters because good agent behavior should not depend only on one frontier model being smart enough to recover from messy state. The runtime should reduce the mess.

## The Reliability Wedge

The smallest useful proof is not a giant autonomous assistant.

The smallest proof is a run where the system can show:

1. what the user asked for
2. what entities or targets were selected
3. which tools were allowed
4. which tool calls actually happened
5. which results succeeded
6. what evidence was gathered
7. what memory was written or rejected
8. why the final answer passed or failed verification

That is why Shovs invests in run ledgers, trace replay, memory inspection, and scenario evals.

If an agent gives a polished answer after searching the wrong things, Shovs should be able to mark the run as wrong.

That is the practical standard.

The public proof path is intentionally small:

1. Define the harness: [HARNESS.md](../../HARNESS.md)
2. Run deterministic checks: [BENCHMARKS.md](../../BENCHMARKS.md)
3. Inspect scenario-state evals: [EVALS.md](../../EVALS.md)
4. Check claim boundaries: [CLAIMS.md](../../CLAIMS.md)
5. Compare latest local results: [RESULTS.md](../../RESULTS.md)

## Frontier Direction

The long-term ambition is to make Shovs a next-generation open-source runtime for serious AI systems:

- local-first when useful
- cloud-capable when needed
- deeply observable
- easy to debug
- easy to extend
- structurally more reliable than prompt-heavy agent wrappers

The project should feel futuristic not because it adds more knobs, but because it makes intelligence:

- cleaner
- more controllable
- more interoperable
- more testable

## Public Standard

For Shovs to earn its claim, it must continue proving:

1. Better coherence through runtime discipline, not only bigger models.
2. Better observability through logs, traces, checkpoints, and evals.
3. Better extensibility through profiles, adapters, tools, and workspace context.
4. Better developer ergonomics for building robust agents on the same kernel.

## Practical Reading

If you are evaluating the project, the important question is not “does it chat?”

It is:

“Does this runtime make LLM behavior more reliable, inspectable, and composable across real tasks?”

That is the real Shovs vision.
