# Shovs LLM OS Developer Guide

This guide is for engineers extending the Shovs runtime, not just prompting it.

## Mental Model

Shovs is a runtime with explicit execution semantics.

The core structure is:
- session
- run
- phase
- checkpoint
- tool result
- artifact
- eval

The runtime already supports:
- `single` loop execution
- `managed` execution with `plan -> act -> observe -> verify -> commit`
- phase-aware context compilation
- model-aware execution profiles and prompt budgets
- truth-vs-candidate memory lanes
- run-scoped tracing and persistence

## Core Modules

### Runtime

- [run_engine/engine.py](/Users/theshovonsaha/Developer/Github/agent/run_engine/engine.py)
  - managed runtime execution path
  - phase packet flow, pass ledger writes, and verification-gated memory commit

- [engine/core.py](/Users/theshovonsaha/Developer/Github/agent/engine/core.py)
  - legacy/native execution kernel
  - prompt construction
  - loop control
  - tool execution
  - follow-up context sanitation
  - model execution profile shaping

Current state:
- managed runtime is canonical and default (`runtime_kind=managed`)
- legacy/native code remains only as low-level compatibility/test coverage, not as a live route selection path
- publish-direction is one canonical runtime contract with observable state, evidence, and memory policies converged on the managed runtime

- [engine/context_schema.py](/Users/theshovonsaha/Developer/Github/agent/engine/context_schema.py)
  - typed context item model

- [engine/context_compiler.py](/Users/theshovonsaha/Developer/Github/agent/engine/context_compiler.py)
  - phase-specific context compilation

- [engine/fact_guard.py](/Users/theshovonsaha/Developer/Github/agent/engine/fact_guard.py)
  - grounded fact filtering

### Orchestration

- [orchestration/orchestrator.py](/Users/theshovonsaha/Developer/Github/agent/orchestration/orchestrator.py)
  - planner, observer, verifier prompts

- [orchestration/agent_manager.py](/Users/theshovonsaha/Developer/Github/agent/orchestration/agent_manager.py)
  - runtime management and adapter parity wiring

- [orchestration/run_store.py](/Users/theshovonsaha/Developer/Github/agent/orchestration/run_store.py)
  - run records
  - loop checkpoints
  - artifacts
  - evals

### Memory

- [orchestration/session_manager.py](/Users/theshovonsaha/Developer/Github/agent/orchestration/session_manager.py)
- [memory/semantic_graph.py](/Users/theshovonsaha/Developer/Github/agent/memory/semantic_graph.py)
- [memory/vector_engine.py](/Users/theshovonsaha/Developer/Github/agent/memory/vector_engine.py)
- [memory/session_rag.py](/Users/theshovonsaha/Developer/Github/agent/memory/session_rag.py)
- [memory/task_tracker.py](/Users/theshovonsaha/Developer/Github/agent/memory/task_tracker.py)
- [engine/deterministic_facts.py](/Users/theshovonsaha/Developer/Github/agent/engine/deterministic_facts.py)
- [engine/direct_fact_policy.py](/Users/theshovonsaha/Developer/Github/agent/engine/direct_fact_policy.py)
- [engine/compression_fact_policy.py](/Users/theshovonsaha/Developer/Github/agent/engine/compression_fact_policy.py)
- [shovs_memory/memory.py](/Users/theshovonsaha/Developer/Github/agent/shovs_memory/memory.py)

### Tools

- [plugins/tool_registry.py](/Users/theshovonsaha/Developer/Github/agent/plugins/tool_registry.py)
- [plugins/tools.py](/Users/theshovonsaha/Developer/Github/agent/plugins/tools.py)
- [plugins/tools_web.py](/Users/theshovonsaha/Developer/Github/agent/plugins/tools_web.py)

### Providers

- [llm/adapter_factory.py](/Users/theshovonsaha/Developer/Github/agent/llm/adapter_factory.py)
- [llm/openai_adapter.py](/Users/theshovonsaha/Developer/Github/agent/llm/openai_adapter.py)
- other adapters in `/llm`

## Execution Model

### Single Loop

Use when:
- model is small or local
- task is bounded
- you want minimal orchestration overhead

Characteristics:
- direct actor loop
- fewer passes
- better default for local OpenAI-compatible servers in `auto`
- more aggressive prompt compression for smaller local profiles

### Managed Loop

Use when:
- task is multi-step
- tool strategy matters
- you want explicit controller behavior

Characteristics:
- planner phase
- actor execution
- observation over tool results
- verification before memory commit
- persisted loop checkpoints

## Model Execution Profiles

The runtime now classifies the active adapter/model into a prompt-shaping profile before building the acting surface.

Current profiles include:
- `small_local`
- `tool_native_local`
- `local_standard`
- `frontier_native`
- `frontier_standard`

These profiles change:
- system/context budget
- history budget per message
- follow-up evidence packet size
- how aggressively compact evidence is enforced

This is one of the main reasons small local models perform better than they would under a single universal prompt template.

## State Integrity Rules

These are central to the project.

### Deterministic Facts

Only verified facts should harden into deterministic memory.

Do not write code that allows:
- assistant guesses
- inferred ticker swaps
- imagined file creation
- speculative claims

to become hard truth without grounding.

Current deterministic coverage includes explicit user statements for:
- preferred name
- location
- timezone
- preferred editor
- package manager
- primary language

Direct-fact queries over these fields can now answer from trusted memory without unnecessary tool use when the fact is already present.

### Candidate Signals

Weak or unverified signals should be downgraded, not promoted.

That means:
- use candidate context
- keep it visible for planning or verification
- do not treat it as truth

Compression-side alias noise should also be blocked. Example:
- keep `User location = Vancouver`
- block `Shovon lives in Vancouver` from becoming a second hard fact when it is only a paraphrase of the trusted user lane

## Memory and Embedding Plumbing

Memory tools should use the active runtime embedding model, not an unrelated default.

Current behavior:
- runtime embed model is propagated into memory tools
- Ollama embedding transport supports both `/api/embed` and legacy `/api/embeddings`
- LM Studio, llama.cpp, and other OpenAI-compatible servers use `/v1/embeddings`

When debugging memory failures, check:
- selected provider
- selected `EMBED_MODEL`
- actual embedding endpoint exposed by the local runner

### Task State

`todo_write` should initialize the workflow, not dominate it.

Current runtime behavior:
- bootstrap tasks once
- prefer `todo_update` after that
- sanitize follow-up prompts so task admin does not crowd out evidence or synthesis

## Adding a Tool

Register tools through the registry.

Typical pattern:

```python
from plugins.tool_registry import Tool

async def _my_tool(query: str, **kwargs) -> str:
    return f"processed: {query}"

MY_TOOL = Tool(
    name="my_tool",
    description="Do one bounded thing.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
    handler=_my_tool,
)
```

Guidelines:
- keep arguments explicit
- keep results structured when possible
- make failure paths deterministic
- return machine-usable results, not only prose

## Adding an Agent Profile

Profiles live in [orchestration/agent_profiles.py](/Users/theshovonsaha/Developer/Github/agent/orchestration/agent_profiles.py).

Keep them narrow:
- model
- tools
- system prompt
- runtime behavior (`managed` default; `native` only for compatibility cases)
- workspace bootstrap docs
- default loop/context/planner behavior

Do not overload profiles with runtime logic that belongs in the kernel.

## Adding a Provider

Provider wiring should go through:
- [llm/adapter_factory.py](/Users/theshovonsaha/Developer/Github/agent/llm/adapter_factory.py)
- the specific adapter in `/llm`

Expectations:
- list models if possible
- support health checks
- stream tokens consistently
- support local vs cloud semantics cleanly

If the provider is OpenAI-compatible, prefer using the shared adapter path unless there is a strong reason for a dedicated adapter.

## Frontend Expectations

### Nova

Nova is the main operator workspace.

It should expose:
- provider/model selection
- loop controls
- planner toggle
- reasoning visibility
- readable monitor timeline
- storage admin
- agent builder defaults and bootstrap composition

It should not dump everything at once, especially on mobile.

### Consumer

Consumer should stay narrower and product-facing. Kernel complexity should not automatically surface there.

## Testing Guidance

Useful test areas:
- tool call parsing
- forced-tool retry
- context overflow retry
- model-profile budget selection
- run/checkpoint persistence
- fact guard behavior
- deterministic fact extraction and correction handling
- direct-fact no-tool answers from trusted memory
- sandbox file tool path compatibility
- managed loop traces
- follow-up sanitation after tool execution
- memory provider and embedding transport compatibility

Representative test files:
- [tests/test_tool_loop_guards.py](/Users/theshovonsaha/Developer/Github/agent/tests/test_tool_loop_guards.py)
- [tests/test_layer_data_flow.py](/Users/theshovonsaha/Developer/Github/agent/tests/test_layer_data_flow.py)
- [tests/test_state_integrity.py](/Users/theshovonsaha/Developer/Github/agent/tests/test_state_integrity.py)
- [tests/test_context_compiler.py](/Users/theshovonsaha/Developer/Github/agent/tests/test_context_compiler.py)
- [tests/test_vector_engine.py](/Users/theshovonsaha/Developer/Github/agent/tests/test_vector_engine.py)

## Design Principles

### 1. Language Is Interface, Not All State

Text comes in and out, but the runtime should increasingly operate on structured state.

### 2. Truthfulness Beats Fluency

Never let the system appear to have done work it did not actually do.

### 3. Small Models Matter

If the runtime can keep small models coherent, larger models become even more reliable.

### 4. One Run Beats Fake Swarms By Default

Prefer one explicit managed loop over many loosely coordinated pseudo-agents unless there is a clear reason to split work.

## Where The Architecture Is Still Open

The kernel is real, but still evolving.

Likely next refinement areas:
- tighter checkpoint-native prompt compilation
- stronger external adapter parity in production use
- more first-class phase objects outside the large `engine/core.py`
- richer reasoning and observation monitor lanes

## API Docs

Run the backend and open:

[http://localhost:8000/docs](http://localhost:8000/docs)
