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
| Action gate (`enforce_proposed_actions`) | The model proposes actions; the gate decides which become state | Drift, over-fetch, off-contract URLs, and premature responses become *violations*, not tool truth |
| Kernel control plane (`run_source_collection`) | A deterministic loop drives the whole multi-step run; the model is a slot-filler | One "top N, M each" task = **2 LLM calls** (lock entities + synthesize), not 2T+3 |

## Kernel-Driven Control Plane

The newest piece (`run_source_collection`) inverts the usual agent loop. Instead
of the model deciding *plan → act → observe → verify* (one LLM call each, every
turn), a **deterministic kernel owns control flow** and the model is called at
exactly two points:

```text
contract (the todo)                              ← deterministic
  → discovery search           (deterministic query)
  → LLM: lock the N entities    ← model slot #1
  → for each entity:           (deterministic structure)
       entity search           (topic-aware query, no LLM)
       fetch M urls             ← contract decides the count
  → LLM: synthesize the answer ← model slot #2
```

Everything else — which tool, when to search, how many to fetch, when to stop,
which entity a fetch belongs to — is deterministic and contract-enforced, so the
run **cannot drift, under-fetch, or stop early** by construction.

A real run of the messy stock task (`scripts/kernel_demo.py`, Gemini
`gemini-2.5-flash`, live `web_search`/`web_fetch`):

```text
task   : top 3 stocks with major jumps, search each, fetch 3 urls each, tldr table
locked : ['MSTR', 'MRNA', 'AAPL']        fetched: 9/9     contract eval: score=1.0 ok=True
LLM    : 2 calls   (vs the orchestrator loop's ~2*9+3 = 21)
output : a grounded TLDR table, every row backed by a fetched, ledger-linked source
```

It is topic-agnostic: `discovery_query` and `entity_search_query` compile the
probe from plain language (stocks → "<E> stock news", restaurants → "<E>
reviews sources", products → "<E> reviews price sources", papers → "<E>
research paper sources"). Tools and the model are injected, so the core stays
free of any web/LLM import:

```python
from shovs_harness_core import run_source_collection

result = await run_source_collection(
    "Search top 3 stocks today, search each, fetch 3 urls each, write a tldr table.",
    search_fn=...,            # query -> (urls, text)        e.g. real web_search
    fetch_fn=...,             # url   -> (ok, content)       e.g. real web_fetch
    extract_entities_fn=...,  # model slot #1 (lock N)
    synth_fn=...,             # model slot #2 (final answer)
)
assert result.llm_calls == 2 and result.eval.ok
```

The offline test (`tests/test_source_runner.py`) proves the structure
deterministically: 3 entities × 3 URLs = 9 fetches, exactly 2 LLM calls, drift
structurally impossible, eval score 1.0.

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
../../venv/bin/python -m pytest -q
```

From the repo root:

```bash
venv/bin/python -m pytest extractions/shovs-harness-core/tests -q
```

## New Developer Setup

From the repo root:

```bash
# Create/use the main repo venv first if needed.
python -m venv venv
venv/bin/python -m pip install -r requirements.txt

# Install JS dependencies used by the small frontend demo.
npm install
cd frontend_consumer && npm install && cd ..

# Check the extracted harness environment.
venv/bin/python extractions/shovs-harness-core/scripts/check_setup.py
```

The setup checker reports:

- Python version and `pytest`
- Node/npm availability
- whether the Vite/TypeScript binaries used by the frontend demo exist
- whether `llama-server` / `llama-cli` are on `PATH`
- whether a local llama.cpp OpenAI-compatible server is reachable

The checker treats llama.cpp as optional. The harness tests and static demo work
without it.

## Run The Harness As A Live App

The extraction now has a tiny dependency-free JSON backend and a Vite frontend.
The frontend still works as a static demo, but when the backend is running it can
call the real Python extension through `POST /api/run`.

From the repo root:

```bash
# Terminal 1
extractions/shovs-harness-core/scripts/run_backend.sh

# Terminal 2
extractions/shovs-harness-core/scripts/run_frontend.sh
```

Open:

```text
http://127.0.0.1:5177
```

Or run both together:

```bash
extractions/shovs-harness-core/scripts/run_live_app.sh
```

Default local endpoints:

| Surface | URL |
| --- | --- |
| Frontend | `http://127.0.0.1:5177` |
| Backend health | `http://127.0.0.1:8091/health` |
| Manifest | `http://127.0.0.1:8091/api/manifest` |
| Run extension | `POST http://127.0.0.1:8091/api/run` |
| llama.cpp health | `http://127.0.0.1:8091/api/llamacpp/health` |
| llama.cpp probe | `POST http://127.0.0.1:8091/api/llamacpp/probe` |

Use custom ports if needed:

```bash
HARNESS_BACKEND_PORT=8092 HARNESS_FRONTEND_PORT=5178 \
  extractions/shovs-harness-core/scripts/run_live_app.sh
```

Quick backend smoke:

```bash
curl -s http://127.0.0.1:8091/health
curl -s http://127.0.0.1:8091/api/manifest
curl -s -X POST http://127.0.0.1:8091/api/run \
  -H 'Content-Type: application/json' \
  -d '{"objective":"Search top 3 stocks today, search each, fetch 3 URLs each.","include_traces":false}'
```

### Live kernel run (real tools + a real model)

`scripts/kernel_demo.py` wires the kernel to the main project's real
`web_search`/`web_fetch` and a Gemini model, and runs the full multi-entity
source-collection task through `run_source_collection`. It prints the locked
entities, fetch coverage, contract eval, and the **LLM call count vs the
orchestrator loop**.

```bash
cd extractions/shovs-harness-core
../../venv/bin/python scripts/kernel_demo.py gemini-2.5-flash   # needs GEMINI_API_KEY in the repo .env
```

### Modules

| Module | Surface |
| --- | --- |
| `contract` | `infer_source_contract`, `discovery_query`, `entity_search_query` |
| `ledger` | `Ledger` — linked tool-call/result truth; rejects orphan results and failed-result claims |
| `kernel` | `HarnessKernel.decide()` — the deterministic next-step decision |
| `evals` | `evaluate_trace` — state-based pass/fail (drift, missing quota) |
| `proposers` | `ProposedAction`, `ScriptedProposer`, `LLMProposer` (inject any adapter) |
| `action_runner` | `enforce_proposed_actions` — model proposes, harness disposes |
| `source_runner` | `run_source_collection` — the deterministic control plane |
| `extension` | `run_extension_payload` / `HarnessExtension` — embeddable JSON surface, no heavy deps |
| `llamacpp` | `LlamaCppClient` — local, OpenAI-compatible, sovereign inference |

## Use As An Extension

This folder should be treated as the small ShovsOS engine extension, not just a
demo. It has no dependency on the main app, FastAPI, React, Ollama, or any cloud
model SDK.

```python
from shovs_harness_core import run_extension_payload

report = run_extension_payload({
    "objective": "Search top 3 stocks today, search each, fetch 3 URLs each.",
    "entities": ["ROKU", "TBN", "SENEA"],
})

assert report["reports"]["harness_loop"]["ok"] is True
assert report["reports"]["plain_loop"]["ok"] is False
```

The extension output is plain JSON:

- `contract`: what the request requires
- `decision`: what the kernel would do next
- `reports`: state-based evals for traces
- `traces`: optional example traces for UI/debug
- `verdict`: small comparison summary

A host can mount this behind an API route, CLI command, plugin, or frontend
without importing the full ShovsOS platform.

## Use With llama.cpp

`shovs-harness-core` includes a tiny dependency-free llama.cpp client for
OpenAI-compatible local servers. The model is used as a proposer; the harness
still decides what is valid.

### Install llama.cpp

Option A: Homebrew.

Homebrew should be writable by your normal macOS user. If `brew install` says
`/opt/homebrew is not writable`, that is a Homebrew prefix ownership problem,
not a llama.cpp problem. Do not use `sudo brew install`.

```bash
brew doctor
brew --prefix
ls -ld "$(brew --prefix)" "$(brew --prefix)/var/homebrew/locks"
brew install llama.cpp
command -v llama-server
command -v llama-cli
```

If Homebrew reports that specific directories are not writable, repair only the
Homebrew prefix/directories it reports, then retry. On a normal Apple Silicon
single-user Homebrew install, the prefix is usually `/opt/homebrew`.

Example repair when Homebrew explicitly reports these paths:

```bash
sudo chown -R "$USER" /opt/homebrew /opt/homebrew/share/zsh /opt/homebrew/share/zsh/site-functions /opt/homebrew/var/homebrew/locks
chmod u+w /opt/homebrew /opt/homebrew/share/zsh /opt/homebrew/share/zsh/site-functions /opt/homebrew/var/homebrew/locks
brew doctor
brew install llama.cpp
```

If you do not want to repair Homebrew, use the source build below. The harness
does not require Homebrew.

Option B: build llama.cpp from source:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
./build/bin/llama-server --help
```

On Apple Silicon, current llama.cpp CMake builds use Metal acceleration by
default on macOS. If your local build does not, rebuild with Metal explicitly:

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

Download a trusted `.gguf` instruct model separately. For an M4 Max with 36 GB
RAM, start with a 7B/8B/14B instruct model in `Q4_K_M` or `Q5_K_M`. Larger
models can work, but the harness should first prove behavior on fast local
models.

### Start llama.cpp Safely

Use localhost only. Do not bind to `0.0.0.0` unless you intentionally want LAN
access.

This repo uses `8081` in examples because `8080` is commonly used by Docker or
other local services:

```text
http://127.0.0.1:8081/v1
```

Start the server:

```bash
llama-server \
  -m /absolute/path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8081
```

Then run:

```bash
cd extractions/shovs-harness-core
LLAMACPP_BASE_URL=http://127.0.0.1:8081/v1 \
LLAMACPP_DEFAULT_MODEL=your-local-model \
../../venv/bin/python scripts/llamacpp_probe.py
```

If you started the live backend, you can also check:

```bash
curl -s http://127.0.0.1:8091/api/llamacpp/health
curl -s -X POST http://127.0.0.1:8091/api/llamacpp/probe \
  -H 'Content-Type: application/json' \
  -d '{"objective":"Search top 3 stocks today, search each, fetch 3 URLs each.","model":"your-local-model"}'
```

The probe asks the local model to propose a locked-entity/fetch/respond action
list, then runs that list through `enforce_proposed_actions(...)`.

This is the intended control boundary:

```text
llama.cpp model -> proposed JSON actions -> harness gate -> accepted trace/report
```

If the model drifts to an off-list entity, fetches a URL outside the contract,
duplicates a fetch, or responds early, the action is reported as a violation
instead of becoming tool truth.

### ShovsOS Env For llama.cpp

For the full ShovsOS app:

```env
LLM_PROVIDER=llamacpp
LLAMACPP_BASE_URL=http://127.0.0.1:8081/v1
LLAMACPP_API_KEY=llama.cpp
LLAMACPP_DEFAULT_MODEL=your-local-model
DEFAULT_MODEL=llamacpp:your-local-model
```

Keep cloud fallbacks separate. The local harness should be able to fail cleanly
when llama.cpp is down instead of silently switching to a different provider.

## Extension Vs Full App

Use this extraction to prove the wedge. Use the full ShovsOS app to show the
complete platform.

| Surface | Use it for | Avoid using it for |
| --- | --- | --- |
| `shovs-harness-core` | public proof, tests, package, extension, evaluator | broad chat UX claims |
| full ShovsOS | integrated runtime, memory, UI, tools, traces | explaining the core idea first |

The main app should consume this kind of kernel as a module. It should not be
the only way to understand the architecture.

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

## Status & Next

Done: source contract + ledger + trace eval + runtime attention; `enforce_proposed_actions`
(model proposes, harness disposes); the `run_source_collection` control plane (2 LLM calls,
verified live on a 9-URL task); local proposers via `llamacpp`; the embeddable `extension`
surface; plain-vs-harness comparison and JSON reports. 22 tests pass.

Genuinely next:

1. Promote this kernel to drive the main ShovsOS engine loop (the "one move" at full scale), replacing the orchestrator's per-stage LLM calls.
2. Same-provider model fallback in the demo so a daily-quota 429 routes to a model with headroom.
3. An optional third model slot for per-source analysis (today the run uses one synthesis call).
4. Golden fixtures drawn from real failure transcripts, gated in CI.
