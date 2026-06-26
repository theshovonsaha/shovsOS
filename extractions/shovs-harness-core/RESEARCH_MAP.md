# Research Map

The extraction is positioned as a small harness layer, not a frontier model.

## What The Field Is Validating

1. Tool-heavy agents need orchestration, not just larger models.
   - ToolOrchestra reports that lightweight orchestrators coordinating tools/models can improve accuracy and cost tradeoffs.
   - Small-agent collaboration work reports that orchestration quality can matter more than raw model scale on tool-heavy tasks.

2. Agent/tool systems need protocol normalization and state feedback.
   - Agent-as-Tool / ParaManager frames tools and agents as a standardized action space with explicit state feedback.
   - This maps directly to the ledger: one action format, linked results, visible state.

3. Agent evaluation is moving from final-text judging to state-based judging.
   - Proxy State-Based Evaluation checks full traces against scenario facts, expected final state, and hallucination constraints.
   - This extraction uses a simpler version: source contracts plus trace state.

4. Tool traces are structured enough to optimize.
   - ToolSpec argues that tool-calling traces have recurring schema patterns.
   - This extraction keeps the schema small enough to test directly.

5. Observability must include context and cognitive state, not just API logs.
   - AgentTrace describes operational, cognitive, and contextual trace surfaces.
   - This extraction records contract, ledger, attention, and trace eval as separate inspectable surfaces.

## The Actual ShovsOS Uniqueness

The unique part is not any single module. It is the combination:

```text
contract -> ledger authority -> phase attention -> continuation gate -> state eval
```

Most agent systems have some of these. The useful wedge is making them small, composable, and testable enough to drop into another agent stack.

## Claims We Can Make Now

- The extraction rejects orphaned tool results.
- It rejects claims against failed tool results.
- It detects entity drift in deterministic source-collection traces.
- It keeps acting until source quota is satisfied.
- It compiles equivalent source-collection requests across unrelated topics.

## Claims We Cannot Make Yet

- It does not prove higher real-world task success.
- It does not prove lower token cost on live models.
- It does not prove superiority over LangGraph, Claude Code, Codex, OpenAI Agents SDK, or other frameworks.
- It does not yet include broad live web or Ollama benchmarks.

Those claims require repeated runs, fixed scenarios, raw trace export, and plain-loop baselines.
