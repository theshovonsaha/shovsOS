# Shovs LLM OS

A local-first **thinking runtime** for autonomous agents.

Not a chat wrapper. Not a RAG pipeline. A runtime that compiles **typed, phase-aware context** for every model call, separates **truth from candidates**, executes tools with **side-effect honesty**, and remembers across sessions through a **unified convergent memory engine**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)
![Frontend](https://img.shields.io/badge/frontend-React-61dafb.svg)

---

## Why this exists

Most "agent platforms" hand the LLM a chat transcript and a tool list, then hope. Shovs treats language as the I/O medium, not the runtime state. Each LLM call receives a **compiled phase packet** — a structured assembly of typed `ContextItem`s drawn from explicit lanes (deterministic facts, candidate signals, working evidence, conversation tension, skill instructions, historical anchors, and more). The packet shape changes with the phase: planning sees durable anchors; acting sees tool evidence; verification sees fact records.

Five things make this different from a prompt wrapper:

|                         | What it means                                                                                                                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Voids & Updates**     | Corrections write a void over the stale claim. "Actually, I moved to Berlin" invalidates the prior location — the engine doesn't keep both.                                                                                             |
| **Side-Effect Honesty** | Tool results carry structured contracts. `bash` and `file_create` emit mandatory verification metadata; the runtime returns `HARD_FAILURE` if expected paths don't exist post-execution. The planner sees real consequences, not prose. |
| **Sticky Skills**       | Skills declare triggers in `SKILL.md` frontmatter. The loader keeps a registry; activation is per-turn, scoped to PLANNING + ACTING phases only — never bleeding into RESPONSE.                                                         |
| **Phase-Aware Context** | `ContextItem`s declare `phase_visibility`. Skill instructions appear in PLANNING/ACTING, never in RESPONSE or MEMORY_COMMIT. Working evidence is acting-time. Deterministic facts are visible everywhere.                               |
| **Resonance**           | A second-pass scoring step lifts modules that share goals with confidently-relevant ones. The packet emerges as a **coherent theme**, not a top-N grab bag.                                                                             |

---

## Architecture (real, not aspirational)

```mermaid
flowchart TD
  classDef magic fill:#f9f0ff,stroke:#d0bdf4,stroke-width:2px,color:#333
  classDef guard fill:#ffebee,stroke:#ffcdd2,stroke-width:2px,color:#333
  classDef storage fill:#fff3e0,stroke:#ffcc80,stroke-width:2px,color:#333
  classDef tech fill:#e3f2fd,stroke:#90caf9,stroke-width:2px,color:#333

  subgraph FRONTEND["🖥️ Product Surfaces (The Hook)"]
    direction LR
    A1["🛠️ Shovs OS (Operator)<br/>Workspace & Builder"]:::tech
    A2["✨ Consumer UI<br/>Viral 'Magic' Chat Surface"]:::magic
  end

  subgraph FastAPI_ENTRYPOINTS["🚪 API Gateways"]
    B1["/chat/stream"]
    B2["/consumer/chat/stream"]
    B3["/sessions/* · /agents/* · /memory/* · /rag/*"]
  end

  subgraph RunEngine["⚙️ Autonomous Loop (The Engine)"]
    direction TB
    C1{"1. PLAN"} --> C2("2. ACT") --> C3{"3. OBSERVE"} --> C4("4. VERIFY") --> C5[/"5. COMMIT"/]:::magic
    
    C2 -.-> C6["Tool Sandbox"]
    C4 -.-> C8["🛡️ Side-Effect Guard<br/>(Trust & Correctness)"]:::guard
    C5 -.-> C9["🧠 Convergent Memory Pipeline<br/>(Retention & Wow-Factor)"]:::magic
  end

  subgraph CONTEXT_GOVERNOR["🧠 Context Governor (The Secret Sauce)"]
    D1["ContextEngineV3 — Unified Convergent Memory"]:::tech
    D2["Phase-Aware Packet Compiler"]:::tech
    D1 --> D2
  end

  subgraph STORAGE_TOPOLOGY["🗄️ Storage & Knowledge Graph"]
    direction LR
    F1[("SQLite<br/>Ledgers")]:::storage
    F6[("Memory Graph<br/>(Facts & Voids)")]:::storage
    F8[("Vector DB<br/>(Chroma)")]:::storage
  end

  subgraph PROVIDERS["☁️ Model Agnostic"]
    G1["Local: Ollama, MLX, llama.cpp"]
    G2["Cloud: OpenAI, Anthropic, Gemini"]
    G3["Unified embedding transport"]
    G3 --> G1 & G2
  end

  %% Flow
  A1 -->|Admin| B1
  A2 -->|End-User| B2
  B1 & B2 & B3 --> RunEngine
  
  RunEngine -->|Drives| PROVIDERS
  RunEngine -->|Persists| STORAGE_TOPOLOGY
  
  CONTEXT_GOVERNOR -->|Reads| STORAGE_TOPOLOGY
  RunEngine -->|Assembles Packet via| CONTEXT_GOVERNOR
```

### Phase packet flow (one turn, end-to-end)

```
user message
     │
     ▼
session_manager.create() ──── HOOK: session_started ────▶ subscribers
     │
     ▼
RunEngine.stream()
     │
     ├─ Load session + current facts
     ├─ Analyze conversation tension
     ├─ Discover available skills          (run_engine/skill_loader.py)
     ├─ Classify code intent               (run_engine/code_intent.py)
     │
     ├─ PLANNING phase
     │     PacketBuildInputs → build_phase_packet()
     │     orchestrator.plan_with_context()
     │     ──── HOOK: plan_generated ──▶ subscribers
     │
     ├─ Inject active skill into PacketBuildInputs.active_skill_context
     │
     ├─ ACTING phase (loop)
     │     For each tool turn:
     │       compile packet → actor selects tool
     │         ──── HOOK: tool_selected ──▶ subscribers
     │       ToolRegistry executes (Docker sandbox for bash)
     │       side_effect_guard verifies expected paths
     │         ──── HOOK: tool_completed ──▶ subscribers
     │         ──── HOOK: hard_failure   ──▶ if status=HARD_FAILURE
     │       ToolLoopGuard circuit-breaks repeat failures
     │
     ├─ OBSERVATION phase
     │     orchestrator decides continue / finalize
     │
     ├─ RESPONSE phase
     │     actor generates final answer (skill context REMOVED)
     │
     ├─ VERIFICATION phase
     │     check response against tool evidence
     │     reject if claims unsupported
     │
     └─ MEMORY_COMMIT phase
           deterministic extraction → fact_guard → semantic graph
           compression-side facts → candidate signals if ungrounded
             ──── HOOK: memory_stored ──▶ per accepted fact
           ──── HOOK: run_complete   ──▶ subscribers
```

### Lifecycle hooks (`plugins/hook_registry.py`)

Subscribe to runtime events from any plugin:

```python
from plugins.hook_registry import hooks, HookEvent

@hooks.on("memory_stored")
async def log_fact(event: HookEvent) -> None:
    print(f"Stored {event.data['subject']} {event.data['predicate']} {event.data['object']}")
```

| Event             | Fires when                         | Payload keys                                 |
| ----------------- | ---------------------------------- | -------------------------------------------- |
| `session_started` | `session_manager.create()` returns | `agent_id, model, owner_id, [plane]`         |
| `plan_generated`  | planner returns structured plan    | `route, skill, tools, confidence, strategy`  |
| `tool_selected`   | actor chose a tool                 | `tool_name, arguments_preview`               |
| `tool_completed`  | tool execution finished            | `tool_name, success, turn`                   |
| `hard_failure`    | tool returned HARD_FAILURE         | `tool_name, turn, preview`                   |
| `memory_stored`   | fact accepted into semantic graph  | `subject, predicate, object, turn, owner_id` |
| `run_complete`    | run finished                       | `run_id, route, tool_count, success`         |

Handlers run concurrently via `asyncio.gather`; exceptions are logged, never raised into the engine loop.

---

## Quick start

```bash
# One-shot install (idempotent — safe to re-run)
./scripts/install.sh

# Verify the install
python3 scripts/doctor.py

# Run Shovs Platform (operator workspace)
npm run dev:shovs
# → http://localhost:5174

# Run Consumer plane
npm run dev:consumer

# Backend only
npm run dev:backend
# → http://localhost:8000  ·  /docs for OpenAPI
```

`scripts/doctor.py` checks: Python ≥3.10, provider API keys, Ollama reachability, writable DB/chroma/logs paths, all 5 LLM adapters import, skill loader works (≥1 skill), unified context engine has its budget knobs.

The full interactive setup (Docker services, embed model selection) is in [setup-linux-mac.sh](setup-linux-mac.sh) and [setup-windows.ps1](setup-windows.ps1).

---

## Configure providers

Pick one (or several — the runtime supports per-agent provider selection):

```env
# Local (no API key required)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2

LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
DEFAULT_MODEL=qwen2.5-coder-3b-instruct-mlx

LLM_PROVIDER=llamacpp
LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1

# Cloud
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
NVIDIA_API_KEY=nvapi-...
```

Embedding transport auto-detects `/api/embed` (current Ollama) and `/api/embeddings` (legacy); LM Studio / llama.cpp / OpenAI-compatible servers use `/v1/embeddings`.

---

## Repository map

```
shovsOS/
├── api/                    FastAPI routes  (main.py, consumer_routes.py, owner.py, ...)
├── engine/                 Context, schema, compiler, governor, fact guard, side-effect guard
│   ├── context_engine_v3.py    Unified convergent memory engine
│   ├── context_engine_v2.py    Convergent ranking + resonance
│   ├── context_engine.py       Linear bullet compression (V1, internal)
│   ├── context_governor.py     Always returns V3 — single surface
│   ├── context_schema.py       ContextItem, ContextKind, ContextPhase
│   ├── context_compiler.py     compile_context_items() — phase-aware assembly
│   ├── side_effect_guard.py    HARD_FAILURE / unsupported-claim detection
│   ├── conversation_tension.py Cross-turn contradiction detection
│   └── deterministic_facts.py  User-stated fact extraction
│
├── run_engine/             Managed runtime
│   ├── engine.py               Main loop (planning → acting → observe → verify → commit)
│   ├── context_packets.py      PacketBuildInputs → build_phase_packet()
│   ├── memory_pipeline.py      Grounding, compression normalization, fact commit
│   ├── skill_loader.py         SKILL.md discovery + loading
│   ├── code_intent.py          Pre-planning code intent classifier
│   └── tool_selection.py       Actor request shaping, tool-call parsing
│
├── orchestration/          Orchestrator, run store, session manager, agent profiles
├── memory/                 Semantic graph (SQLite), vector engine (Chroma), session RAG
├── plugins/                Tool registry + tools + hook registry
│   ├── tool_registry.py        Tool dataclass, brace-counting tool-call detector
│   ├── tools.py                Built-in tools (web_search, file_create, bash, query_memory, ...)
│   ├── tools_web.py            Canonical web search/fetch
│   ├── shovs_meta_gateway.py   External-agent gateway (memory palace tools)
│   └── hook_registry.py        Lifecycle event pub/sub (7 events)
│
├── llm/                    Provider adapters (5: Ollama, OpenAI, Anthropic, Groq, Gemini)
├── shovs_memory/           Installable memory wedge (uses orchestration + memory)
├── frontend_shovs/          Operator workspace (React + Vite)
├── frontend_consumer/      Consumer plane
├── .agent/skills/          9 platform skills (agent_platform_backend, debugging, frontend_design, ...)
├── scripts/                install.sh, doctor.py
└── documentation/public/   VISION, DEVELOPER_GUIDE, FEATURES_AND_ROADMAP, SETUP, SHOVS_MEMORY
```

---

## What `shovs-memory` is

The smallest adoptable surface of this repo. Use it when you want deterministic fact writes, correction-aware temporal memory, and inspectable memory state — without the full runtime loop.

```python
from orchestration.session_manager import SessionManager
from shovs_memory import ShovsMemory

sessions = SessionManager()
memory = ShovsMemory(session_id="user-123", owner_id="owner-123", session_manager=sessions)

memory.apply_user_message("My name is Shovon. I use Cursor.", turn=1)
memory.apply_user_message("Actually, I moved to Berlin.", turn=2)

print(memory.current_facts())   # latest fact for each predicate
print(memory.fact_timeline())   # history with voids
print(memory.inspect())         # full memory state snapshot
```

Use the full runtime (`RunEngine`) when you also want loop orchestration, tool execution, traces, checkpoints, and artifacts.

---

## Skills

Drop a `SKILL.md` under `.agent/skills/{name}/`:

```markdown
---
name: debugging
description: Use when the user reports a bug, crash, traceback, or regression.
triggers: bug, crash, traceback, regression, broken, exception, error, fix, debug
eligibility: auto
---

# Debugging skill

Approach:

1. Reproduce the failure deterministically.
2. Bisect: change one thing at a time.
3. ...
```

The loader parses the simple frontmatter (single-line `triggers:` is comma-separated). Skills are surfaced into PLANNING and ACTING packets only — never RESPONSE or MEMORY_COMMIT — and carry `trace_id="skill_loader:{name}"` for full provenance.

Built-in skills: `agent_platform_backend`, `code_review`, `debugging`, `frontend_design`, `memory_palace`, `pdf`, `shell_workflow`, `testing`, `web_research`.

---

## Tool contract & side-effect honesty

Every tool returns a JSON payload with `success`, `status`, and (when applicable) `verification`:

```json
{
  "type": "bash_result",
  "success": true,
  "status": "SUCCESS",
  "command": "...",
  "output": "...",
  "verification": {
    "expected_paths": ["/sandbox/report.html"],
    "missing_paths": []
  }
}
```

When `bash` or `file_create` is called and the expected write target does not exist post-execution, the tool returns `status: "HARD_FAILURE"`. The `side_effect_guard` then blocks the response from claiming success, and the runtime emits the `hard_failure` lifecycle hook.

Adding a tool:

```python
from plugins.tool_registry import Tool, registry

async def _my_tool(query: str) -> str:
    return f"result: {query}"

registry.register(Tool(
    name="my_tool",
    description="One sentence: what it does and when to use it.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    handler=_my_tool,
    tags=["category"],
))
```

Add it to `ALL_TOOLS` in [plugins/tools.py](plugins/tools.py); registration runs at API startup.

---

## Why small models matter here

Runtime discipline raises the floor for small local models — and the ceiling for frontier models. Phase-specific context, model-aware budget shaping (`small_local`, `tool_native_local`, `local_standard`, `frontier_native`, `frontier_standard`), candidate-vs-truth separation, and prompt sanitation after evidence gathering all reduce the noise that small models can't recover from.

If a 3B model can stay coherent across multi-tool runs, an Opus / GPT-5 class model becomes nearly bulletproof.

---

## Public docs

- [SETUP](documentation/public/SETUP.md) — environment, providers, troubleshooting
- [DEVELOPER_GUIDE](documentation/public/DEVELOPER_GUIDE.md) — engineering reference
- [VISION](documentation/public/VISION.md) — thesis, claims, what makes Shovs different
- [FEATURES_AND_ROADMAP](documentation/public/FEATURES_AND_ROADMAP.md) — what ships, what's next
- [SHOVS_MEMORY](documentation/public/SHOVS_MEMORY.md) — memory wedge reference
- [ARCHITECTURE](ARCHITECTURE.md) — runtime + storage topology
- [ROADMAP](ROADMAP.md) — convergence phases
- [CONTRIBUTING](CONTRIBUTING.md) · [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) · [SECURITY](SECURITY.md) · [GOVERNANCE](GOVERNANCE.md) · [SUPPORT](SUPPORT.md)

---

## License

MIT
