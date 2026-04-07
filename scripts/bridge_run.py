#!/usr/bin/env python3
"""
bridge_run.py — Run the engine with the BridgeAdapter
------------------------------------------------------
This script starts a full managed-loop run using the bridge provider.
The BridgeAdapter writes each prompt to a handoff file in agent_sandbox/bridge/.
An external agent (you, Copilot, another process) reads the handoff, writes a
response file, and the engine continues.

Usage:
    python scripts/bridge_run.py "What are the top 3 AI stocks to watch?"
    python scripts/bridge_run.py --task "Research AAPL earnings" --tools web_search
    python scripts/bridge_run.py --task "Summarize latest ML papers" --no-planner

The handoff files appear in agent_sandbox/bridge/ as:
    handoff_<id>.json   — the prompt (messages, model, tools, temperature)
    response_<id>.json  — you write this: {"content": "your response here"}

Environment:
    BRIDGE_DIR           — override handoff directory (default: agent_sandbox/bridge)
    BRIDGE_TIMEOUT       — seconds to wait for response (default: 300)
    BRIDGE_POLL_INTERVAL — poll frequency in seconds (default: 1.0)
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.bridge_adapter import BridgeAdapter
from plugins.tool_registry import ToolRegistry
from orchestration.session_manager import SessionManager
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.run_store import get_run_store
from config.trace_store import get_trace_store
from engine.context_engine import ContextEngine
from run_engine import RunEngine, RunEngineRequest


def build_engine(bridge_dir: str | None = None) -> RunEngine:
    """Wire a RunEngine with BridgeAdapter as the model backend."""
    adapter = BridgeAdapter(bridge_dir=bridge_dir)
    tool_registry = ToolRegistry()
    session_manager = SessionManager(max_sessions=50, db_path="bridge_sessions.db")
    context_engine = ContextEngine(adapter=adapter, compression_model="bridge")
    orchestrator = AgenticOrchestrator(adapter=adapter)

    return RunEngine(
        adapter=adapter,
        sessions=session_manager,
        tool_registry=tool_registry,
        run_store=get_run_store(),
        trace_store=get_trace_store(),
        orchestrator=orchestrator,
        context_engine=context_engine,
    )


async def run(
    task: str,
    tools: list[str] | None = None,
    use_planner: bool = True,
    bridge_dir: str | None = None,
    max_turns: int = 3,
) -> None:
    engine = build_engine(bridge_dir)
    request = RunEngineRequest(
        session_id="bridge-cli",
        owner_id="bridge",
        agent_id="bridge-agent",
        user_message=task,
        model="bridge",
        system_prompt="You are a helpful assistant. Answer thoroughly using available tools.",
        use_planner=use_planner,
        max_turns=max_turns,
        allowed_tools=tuple(tools) if tools else (),
        forced_tools=tuple(tools) if tools else (),
    )

    print(f"\n{'='*60}")
    print(f"BRIDGE RUN")
    print(f"Task: {task}")
    print(f"Planner: {use_planner} | Max turns: {max_turns}")
    print(f"Bridge dir: {engine.adapter.bridge_dir}")
    print(f"{'='*60}\n")

    events = []
    async for event in engine.stream(request):
        event_type = event.get("type", "")

        if event_type == "session":
            print(f"[session] {event.get('session_id')} | run {event.get('run_id')}")
        elif event_type == "plan":
            print(f"[plan] strategy: {event.get('strategy')}")
            print(f"       tools: {event.get('tools')}")
        elif event_type == "tool_call":
            print(f"[tool_call] {event.get('tool_name')}({json.dumps(event.get('arguments', {}), default=str)[:200]})")
        elif event_type == "tool_result":
            status = "OK" if event.get("success") else "FAIL"
            print(f"[tool_result] {event.get('tool_name')} [{status}] {str(event.get('content_preview', ''))[:200]}")
        elif event_type == "token":
            sys.stdout.write(event.get("content", ""))
            sys.stdout.flush()
        elif event_type == "verification_warning":
            print(f"\n[verification_warning] {event.get('issues')}")
        elif event_type == "done":
            print(f"\n\n[done] run_id={event.get('run_id')}")
        elif event_type == "conversation_tension":
            if event.get("summary"):
                print(f"[tension] {event.get('summary')}")
        else:
            pass  # swallow internal events

        events.append(event)

    # Write run summary
    summary_path = os.path.join(
        engine.adapter.bridge_dir, "last_run_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(
            {
                "task": task,
                "event_count": len(events),
                "events": [
                    {k: v for k, v in e.items() if k != "content" or e.get("type") != "token"}
                    for e in events
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n[summary] Written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the engine with BridgeAdapter")
    parser.add_argument("task", nargs="?", help="Task / user message")
    parser.add_argument("--task", "-t", dest="task_flag", help="Task (alternative flag)")
    parser.add_argument("--tools", nargs="*", help="Force specific tools")
    parser.add_argument("--no-planner", action="store_true", help="Disable planner")
    parser.add_argument("--bridge-dir", help="Override bridge directory")
    parser.add_argument("--max-turns", type=int, default=3, help="Max tool turns")
    args = parser.parse_args()

    task = args.task or args.task_flag
    if not task:
        parser.error("Provide a task: bridge_run.py 'your task here'")

    asyncio.run(run(
        task=task,
        tools=args.tools,
        use_planner=not args.no_planner,
        bridge_dir=args.bridge_dir,
        max_turns=args.max_turns,
    ))


if __name__ == "__main__":
    main()
