#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, AsyncIterator

from dotenv import load_dotenv


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _clip(text: str, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    keep = max(80, limit // 2)
    return f"{value[:keep]} ... {value[-keep:]}"


def _message_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    total_chars = 0
    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        total_chars += len(content)
        rows.append({
            "index": index,
            "role": message.get("role"),
            "chars": len(content),
            "estimated_tokens": _estimate_tokens(content),
            "preview": _clip(content, 500),
        })
    return {
        "message_count": len(messages),
        "roles": [row["role"] for row in rows],
        "total_chars": total_chars,
        "estimated_tokens": _estimate_tokens("\n\n".join(str(m.get("content") or "") for m in messages)),
        "messages": rows,
    }


def _section_sizes(text: str) -> list[dict[str, Any]]:
    lines = str(text or "").splitlines()
    sections: list[dict[str, Any]] = []
    current = {"label": "preamble", "chars": 0, "line_count": 0}
    for line in lines:
        stripped = line.strip()
        is_header = (
            stripped.endswith(":")
            or stripped.startswith("Context ")
            or stripped.startswith("Canonical ")
            or stripped.startswith("--- ")
            or stripped.startswith("Workflow ")
            or stripped.startswith("Turn ")
        )
        if is_header and current["chars"]:
            sections.append(current)
            current = {"label": stripped[:90], "chars": 0, "line_count": 0}
        current["chars"] += len(line) + 1
        current["line_count"] += 1
    if current["chars"]:
        sections.append(current)
    return sorted(sections, key=lambda item: item["chars"], reverse=True)[:12]


class InstrumentedGroqAdapter:
    def __init__(self):
        from llm.groq_adapter import GroqLLMAdapter

        self.inner = GroqLLMAdapter()
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, model: str, messages: list[dict], **kwargs) -> str:
        started = time.perf_counter()
        response = await self.inner.complete(model=model, messages=messages, **kwargs)
        self.calls.append({
            "kind": "complete",
            "model": model,
            "kwargs": {k: v for k, v in kwargs.items() if k not in {"images", "tools"}},
            "tools_count": len(kwargs.get("tools") or []),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "input": _message_stats(messages),
            "output_chars": len(response),
            "output_estimated_tokens": _estimate_tokens(response),
            "output_preview": _clip(response, 1200),
            "largest_input_sections": _section_sizes("\n\n".join(str(m.get("content") or "") for m in messages)),
        })
        return response

    async def stream(self, *, model: str, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        started = time.perf_counter()
        chunks: list[str] = []
        async for chunk in self.inner.stream(model=model, messages=messages, **kwargs):
            chunks.append(str(chunk))
            yield chunk
        response = "".join(chunks)
        self.calls.append({
            "kind": "stream",
            "model": model,
            "kwargs": {k: v for k, v in kwargs.items() if k not in {"images", "tools"}},
            "tools_count": len(kwargs.get("tools") or []),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "input": _message_stats(messages),
            "output_chars": len(response),
            "output_estimated_tokens": _estimate_tokens(response),
            "output_preview": _clip(response, 1200),
            "largest_input_sections": _section_sizes("\n\n".join(str(m.get("content") or "") for m in messages)),
        })

    async def health(self) -> bool:
        return await self.inner.health()

    async def list_models(self) -> list[str]:
        return await self.inner.list_models()


def _load_trace_index(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "trace_index.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _summarize_pass(pass_record) -> dict[str, Any]:
    payload = asdict(pass_record)
    compiled = payload.get("compiled_context") if isinstance(payload.get("compiled_context"), dict) else {}
    context_items = compiled.get("items") if isinstance(compiled.get("items"), list) else []
    payload["compiled_context_summary"] = {
        "keys": list(compiled.keys())[:20],
        "item_count": len(context_items),
        "phase": compiled.get("phase"),
        "context_chars": compiled.get("content_chars") or compiled.get("chars") or None,
        "top_context_items": sorted(
            [
                {
                    "id": item.get("id") or item.get("item_id") or item.get("title") or "unknown",
                    "kind": item.get("kind"),
                    "chars": item.get("chars") or len(str(item.get("content") or "")),
                    "priority": item.get("priority"),
                }
                for item in context_items
                if isinstance(item, dict)
            ],
            key=lambda item: int(item.get("chars") or 0),
            reverse=True,
        )[:12],
    }
    payload.pop("compiled_context", None)
    return payload


async def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from config.trace_store import TraceStore
    from engine.context_engine import ContextEngine
    from memory.semantic_graph import SemanticGraph
    from orchestration.run_store import RunStore
    from orchestration.session_manager import SessionManager
    from plugins.tool_registry import ToolRegistry
    from plugins.tools import register_all_tools
    from run_engine import RunEngine, RunEngineRequest

    model = args.model
    adapter = InstrumentedGroqAdapter()
    registry = ToolRegistry()
    register_all_tools(registry)
    sessions = SessionManager(max_sessions=20, db_path=args.session_db)
    runs = RunStore(db_path=args.runs_db)
    trace_store = TraceStore()
    context_engine = ContextEngine(adapter=adapter, compression_model=model)
    graph = SemanticGraph(db_path=args.memory_db)

    class ProbeRunEngine(RunEngine):
        def _resolve_adapter(self, model_name: str):
            if str(model_name or "").startswith("groq:"):
                return adapter
            return adapter

    engine = ProbeRunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=trace_store,
        orchestrator=None,
        context_engine=context_engine,
        graph=graph,
    )

    request = RunEngineRequest(
        session_id=args.session_id,
        owner_id="probe-owner",
        agent_id="probe-agent",
        user_message=args.prompt,
        model=model,
        system_prompt=(
            "You are a concise runtime probe assistant. Answer plainly. "
            "Do not invent tool calls; only use successful tool evidence."
        ),
        context_mode="v3",
        allowed_tools=tuple(args.allowed_tools),
        forced_tools=tuple(args.forced_tools),
        use_planner=False,
        max_tool_calls=args.max_tool_calls,
        max_turns=args.max_turns,
        ledger_mode=args.ledger_mode,
        control_policy=args.control_policy,
        memory_commit_mode=args.memory_commit_mode,
        prompt_version="probe_trace_v1",
    )

    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    async for event in engine.stream(request):
        events.append(event)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    run = runs.latest_for_session(args.session_id, owner_id="probe-owner")
    run_id = run.run_id if run else ""
    passes = runs.list_passes(run_id) if run_id else []
    traces = _load_trace_index(Path(args.trace_dir))
    trace_counts = Counter(str(item.get("event_type") or "") for item in traces)
    token_events = [item.get("content", "") for item in events if item.get("type") == "token"]
    tool_events = [item for item in events if str(item.get("type") or "").startswith("tool")]

    return {
        "probe": {
            "model": model,
            "prompt": args.prompt,
            "session_id": args.session_id,
            "run_id": run_id,
            "duration_ms": duration_ms,
            "allowed_tools": list(args.allowed_tools),
            "forced_tools": list(args.forced_tools),
            "memory_commit_mode": args.memory_commit_mode,
        },
        "final_text": "".join(str(item) for item in token_events).strip(),
        "stream_event_counts": dict(Counter(str(item.get("type") or "") for item in events)),
        "tool_events": tool_events,
        "model_calls": adapter.calls,
        "passes": [_summarize_pass(item) for item in passes],
        "trace": {
            "event_count": len(traces),
            "event_counts": dict(trace_counts),
            "largest_events": sorted(
                [
                    {
                        "event_type": item.get("event_type"),
                        "size_bytes": item.get("size_bytes"),
                        "preview": item.get("preview"),
                        "payload_ref": item.get("payload_ref"),
                    }
                    for item in traces
                ],
                key=lambda item: int(item.get("size_bytes") or 0),
                reverse=True,
            )[:15],
        },
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Groq-backed ShovsOS runtime trace probe.")
    parser.add_argument("--model", default="groq:llama-3.3-70b-versatile")
    parser.add_argument("--prompt", default="Say hello in one sentence and explain whether you used tools.")
    parser.add_argument("--allowed-tool", dest="allowed_tools", action="append", default=[])
    parser.add_argument("--forced-tool", dest="forced_tools", action="append", default=[])
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--ledger-mode", default="shadow")
    parser.add_argument("--control-policy", default="auto")
    parser.add_argument("--memory-commit-mode", default="skip", choices=["sync", "async", "skip"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    if not os.getenv("GROQ_API_KEY"):
        print(json.dumps({"success": False, "error": "GROQ_API_KEY is not configured."}, indent=2))
        return 2

    out_dir = Path(args.out or "logs/probes").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    with TemporaryDirectory(prefix="shovs-groq-probe-") as tmp:
        tmp_path = Path(tmp)
        args.session_id = f"groq-probe-{stamp}"
        args.session_db = str(tmp_path / "sessions.db")
        args.runs_db = str(tmp_path / "runs.db")
        args.memory_db = str(tmp_path / "memory.db")
        args.trace_dir = str(tmp_path / "traces")
        os.environ["TRACE_STORE_DIR"] = args.trace_dir
        os.environ["TRACE_INLINE_MAX_BYTES"] = "1200"
        report = await _run_probe(args)

    report_path = out_dir / f"groq-runtime-trace-{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "success": True,
        "report_path": str(report_path),
        "run_id": report["probe"]["run_id"],
        "duration_ms": report["probe"]["duration_ms"],
        "model_call_count": len(report["model_calls"]),
        "model_input_tokens": [call["input"]["estimated_tokens"] for call in report["model_calls"]],
        "stream_event_counts": report["stream_event_counts"],
        "trace_event_count": report["trace"]["event_count"],
        "pass_count": len(report["passes"]),
        "final_preview": _clip(report["final_text"], 500),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
