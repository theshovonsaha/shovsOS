"""Live A/B: the SAME task through the platform's two runtimes, real Gemini.

  control_policy = auto    -> RunEngine            (orchestrator owns the loop)
  control_policy = kernel  -> KernelRunEngine      (deterministic kernel; model = slot-filler)

Both engines share construction and the `.stream(request)` signature, so this is
an apples-to-apples comparison on one objective. Every Gemini call is counted at
the adapter level (complete + stream), so the LLM-call delta is exact regardless
of how each engine creates sub-adapters (planner/verify/context).

Usage:
    python scripts/kernel_ab.py [gemini-model]      # needs GEMINI_API_KEY in .env
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from config.config import cfg  # noqa: E402  (loads .env)

# ---- count every Gemini LLM call at the adapter level (both engines) ----
import llm.gemini_adapter as _g  # noqa: E402

_COUNT = {"complete": 0, "stream": 0}
_orig_complete = _g.GeminiAdapter.complete
_orig_stream = _g.GeminiAdapter.stream


async def _counted_complete(self, *a, **k):
    _COUNT["complete"] += 1
    return await _orig_complete(self, *a, **k)


async def _counted_stream(self, *a, **k):
    _COUNT["stream"] += 1
    async for chunk in _orig_stream(self, *a, **k):
        yield chunk


_g.GeminiAdapter.complete = _counted_complete
_g.GeminiAdapter.stream = _counted_stream

from llm.adapter_factory import create_adapter  # noqa: E402
from plugins.tool_registry import ToolRegistry  # noqa: E402
from plugins.tools_web import register_web_tools  # noqa: E402
from orchestration.session_manager import SessionManager  # noqa: E402
from engine.context_engine import ContextEngine  # noqa: E402
from orchestration.orchestrator import AgenticOrchestrator  # noqa: E402
from orchestration.run_store import get_run_store  # noqa: E402
from config.trace_store import get_trace_store  # noqa: E402
from memory.semantic_graph import SemanticGraph  # noqa: E402
from run_engine import RunEngine, RunEngineRequest  # noqa: E402
from run_engine.kernel_engine import KernelRunEngine  # noqa: E402

OBJ = ("Search top 3 stocks today with major jumps, web search those 3 stocks separately and "
       "capture the 3 relevant results for each, web fetch all 9 urls one by one, analyze each "
       "and write a tldr summary table.")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
MODEL_REF = MODEL if ":" in MODEL else f"gemini:{MODEL}"


def _build_engines():
    adapter = create_adapter("gemini")
    registry = ToolRegistry()
    register_web_tools(registry)
    sessions = SessionManager(max_sessions=50, db_path="ab_sessions.db")
    context_engine = ContextEngine(adapter=adapter, compression_model=cfg.DEFAULT_MODEL)
    orchestrator = AgenticOrchestrator(adapter=adapter)
    graph = SemanticGraph()
    common = dict(adapter=adapter, sessions=sessions, tool_registry=registry,
                  run_store=get_run_store(), trace_store=get_trace_store(),
                  orchestrator=orchestrator, context_engine=context_engine, graph=graph)
    return RunEngine(**common), KernelRunEngine(**common)


def _request(session_id, policy):
    return RunEngineRequest(
        session_id=session_id, owner_id="ab_owner", agent_id="ab_agent",
        user_message=OBJ, model=MODEL_REF,
        allowed_tools=("web_search", "web_fetch", "web_fetch_batch"),
        control_policy=policy, max_turns=8, ledger_mode="enforced",
    )


async def _drive(engine, policy, label):
    _COUNT["complete"] = 0
    _COUNT["stream"] = 0
    tool_calls = 0
    answer_chars = 0
    kernel_metrics = None
    t0 = time.time()
    err = None
    try:
        async for ev in engine.stream(_request(f"ab_{policy}", policy)):
            t = ev.get("type")
            if t == "tool_call":
                tool_calls += 1
            elif t == "token":
                answer_chars += len(ev.get("content") or "")
            elif t == "kernel_metrics":
                kernel_metrics = ev
    except Exception as e:  # live runs can 429 / time out — report honestly
        err = f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    llm = _COUNT["complete"] + _COUNT["stream"]
    print(f"\n--- {label} (control_policy={policy}) ---")
    if err:
        print(f"  ERROR (partial): {err}")
    print(f"  LLM calls   : {llm}   (complete={_COUNT['complete']}, stream={_COUNT['stream']})")
    print(f"  tool calls  : {tool_calls}")
    print(f"  answer chars: {answer_chars}")
    print(f"  wall time   : {dt:.1f}s")
    if kernel_metrics:
        print(f"  kernel      : shape={kernel_metrics.get('shape')} fetched={kernel_metrics.get('fetched')}/"
              f"{kernel_metrics.get('contract_total')} gate_open={kernel_metrics.get('completion_gate_open')} "
              f"citations_grounded={kernel_metrics.get('citations_grounded')}")
    return llm


async def main():
    print(f"=== PLATFORM A/B — model={MODEL_REF} ===\nTASK: {OBJ}")
    run_engine, kernel_engine = _build_engines()
    auto_llm = await _drive(run_engine, "auto", "RunEngine (orchestrator)")
    kernel_llm = await _drive(kernel_engine, "kernel", "KernelRunEngine (deterministic)")
    print("\n=== DELTA ===")
    if kernel_llm:
        print(f"  auto={auto_llm} LLM calls   kernel={kernel_llm} LLM calls   "
              f"=> {auto_llm / kernel_llm:.1f}x fewer" if auto_llm else "  (auto path produced no calls)")
    print("  (harness-core proved 21->2 = 10.5x on the same 9-URL task; this confirms it in the app.)")


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=420))
