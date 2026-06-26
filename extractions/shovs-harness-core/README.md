# Shovs Harness Core

A small extraction from ShovsOS for testing agent reliability wedges without the full app.

This is not another chat UI. It is a tiny harness kernel:

1. Turn a user request into a source contract.
2. Let models propose actions.
3. Let the ledger decide what actually happened.
4. Keep phase attention small and weighted.
5. Evaluate the trace against state, not vibes.

The code is deliberately boring. The value is in the invariant:

```text
model text is not authority
tool results are not authority unless linked to a tool call
final answers cannot cite failed or missing tool results
workflow completion is checked against a contract
```

## Why This Exists

ShovsOS grew into a large research system: memory, context lanes, evidence lanes, phase packets, traces, tool guards, and runtime tests. The useful wedge is smaller:

> A plug-in harness that makes agent runs inspectable, topic-agnostic, and harder to fake.

That matters because the same failure appears across domains:

```text
User: find top 3 X, search each, fetch 3 URLs each
Agent: finds X correctly
Agent: drifts to unrelated Y
Agent: fetches generic pages
Agent: answers confidently anyway
```

The harness catches that as a state error.

## What Is Unique Here

| Wedge | What it does | Why it is useful |
| --- | --- | --- |
| Source contract | Compiles plain language into counts and required tools | Keeps "3 each" from becoming random browsing |
| Run ledger | Records tool calls and results as linked facts | Stops "I used a tool" hallucinations |
| Runtime attention | Scores context by phase | Avoids flooding the model with stale context |
| Continuation gate | Refuses final response until quota/state is satisfied | Keeps multi-step work alive |
| Trace eval | Checks the run against locked entities and required steps | Produces a concrete pass/fail report |

## Research Fit

This extraction lines up with current agent research without copying any system:

- [ToolOrchestra](https://arxiv.org/abs/2511.21689) argues for efficient orchestration of models and tools.
- [Small Model as Master Orchestrator](https://arxiv.org/abs/2604.17009) points at protocol normalization and state feedback for extensible agent/tool systems.
- [Can Small Agents Collaborate to Beat a Single Large Language Model?](https://arxiv.org/abs/2601.11327) reports that orchestration quality can matter more than model scale for tool-heavy work.
- [Proxy State-Based Evaluation](https://arxiv.org/abs/2602.16246) evaluates tool/user hallucinations against scenario state.
- [AgentTrace](https://arxiv.org/abs/2602.10133) frames structured traces as operational, cognitive, and contextual observability.
- [ToolSpec](https://arxiv.org/abs/2604.13519) notes that tool traces are structured enough for schema-aware execution.

The extraction thesis is simple:

> Most agent failures are not fixed by louder prompts. They are fixed by typed state, linked tool truth, small context, and state-based evals.

## Run

From this folder:

```bash
python -m pytest -q
```

From the repo root:

```bash
python -m pytest extractions/shovs-harness-core/tests -q
```

## Frontend Demo

There is a small browser demo in `frontend/`. It has no backend dependency.

From the repo root:

```bash
cd extractions/shovs-harness-core/frontend
npm run dev
```

Then open the local Vite URL. The page lets you:

- paste a source-collection task
- see the inferred contract
- compare a drifted plain loop against a harness loop
- inspect the trace timeline
- inspect raw contract/eval/decision JSON

Build check:

```bash
cd extractions/shovs-harness-core/frontend
npm run build
```

## Example

```python
from shovs_harness_core import HarnessKernel

kernel = HarnessKernel("Find top 3 sushi places, search each, fetch 3 URLs each.")

decision = kernel.decide()
assert decision.next_tool == "web_search"

kernel.add_tool_result("web_search", {"query": "sushi"}, True, {"results": []})

for i in range(9):
    kernel.add_tool_result("web_fetch", {"url": f"https://source.test/{i}"}, True, {"body": "ok"})

assert kernel.decide().state == "respond"
```

## What This Is Not

- It is not a full agent runtime.
- It is not a benchmark claim.
- It is not proof that ShovsOS beats another system.
- It is a small falsifiable kernel for testing whether specific reliability wedges work.

## Next Useful Tests

1. Add live Ollama probes where the model proposes actions and the harness accepts or rejects them.
2. Add real web fixtures for stocks, restaurants, products, papers, and local stores.
3. Compare a plain model loop against this harness on the same traces.
4. Export trace reports as JSON for a frontend inspector.
