"""
Shovs Convergence Eval Harness
===============================
Turns the platform from "trust me it works" into a number anyone can check.

It runs a fixed battery of tasks against one or more models and records, per
run, whether each phase boundary held:

  - planner_ok        : planner produced an actionable plan (not silent fallback)
  - tools_called      : at least one tool actually executed
  - completed         : the run reached a final answer without stalling
  - hallucination_flag: side-effect guard or verification flagged an unsupported claim
  - latency_s         : wall-clock

Output is a CSV + a printed summary table. The single most important row for a
funding conversation is planner success rate by model — it shows the architecture
is sound and the bottleneck is model quality (i.e. compute), not design.

USAGE
-----
    python -m evals.convergence_eval \
        --models "ollama:gemma4,ollama:llama3.3-70b,anthropic:claude-haiku-4-5,anthropic:claude-sonnet-4-6" \
        --runs 10 \
        --out evals/results.csv

The harness talks to the runtime through a thin RunnerProtocol so it has no hard
dependency on engine internals. Provide a runner that yields the runtime's event
dicts; a reference adapter for the managed RunEngine is included below and can be
swapped for a stub in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol


# ── Task battery ─────────────────────────────────────────────────────────────
# Deliberately graded from trivial to compound so the harness shows exactly
# where each model's planner starts to break.

@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    tier: str                  # "single" | "two_step" | "compound"
    expects_tools: bool
    success_substrings: tuple[str, ...] = ()   # any present in final answer = content win


TASK_BATTERY: tuple[EvalTask, ...] = (
    EvalTask(
        task_id="single_search_ticker",
        prompt="Web search the single most active stock by volume today. Reply with only its ticker symbol.",
        tier="single",
        expects_tools=True,
    ),
    EvalTask(
        task_id="single_fact_fetch",
        prompt="Fetch https://example.com and tell me the main heading text.",
        tier="single",
        expects_tools=True,
        success_substrings=("example",),
    ),
    EvalTask(
        task_id="two_step_search_then_summarize",
        prompt="Search for the latest news about a major tech company today, then summarize the single most important development in two sentences.",
        tier="two_step",
        expects_tools=True,
    ),
    EvalTask(
        task_id="compound_three_stocks",
        prompt=(
            "1. Web search the top 3 stocks by volume today. "
            "2. For each, search one recent news item. "
            "3. Write a 3-bullet summary with one source link per stock."
        ),
        tier="compound",
        expects_tools=True,
    ),
    EvalTask(
        task_id="direct_no_tool",
        prompt="What is 17 multiplied by 23? Answer with just the number.",
        tier="single",
        expects_tools=False,
        success_substrings=("391",),
    ),
)


# ── Runner protocol ──────────────────────────────────────────────────────────
# The harness consumes runtime event dicts. Anything that yields events with a
# "type" key works. This keeps the eval decoupled from engine internals.

class RunnerProtocol(Protocol):
    async def run(self, *, model: str, prompt: str) -> AsyncIterator[dict[str, Any]]:
        ...


@dataclass
class RunObservation:
    planner_ok: bool = False
    planner_failed: bool = False
    tools_called: int = 0
    completed: bool = False
    hallucination_flag: bool = False
    final_text: str = ""
    error: str = ""
    latency_s: float = 0.0


async def observe_run(
    runner: RunnerProtocol,
    *,
    model: str,
    prompt: str,
    timeout_s: float = 120.0,
) -> RunObservation:
    """Drive one run and fold its event stream into a flat observation.

    Event contract (loose, matches the runtime's emitted dicts):
      type == "planner_failure"          → planner_failed
      type == "plan" / "strategy" w/tools → planner_ok
      type == "tool_completed"            → tools_called += 1
      type == "token"                     → accumulate final_text
      type in {"hard_failure","verification_failed","side_effect_violation"}
                                          → hallucination_flag
      type == "run_complete" / "done"     → completed
      type == "error"                     → error
    """
    obs = RunObservation()
    start = time.perf_counter()
    try:
        async def _drive() -> None:
            async for event in runner.run(model=model, prompt=prompt):
                etype = str(event.get("type") or "")
                if etype == "planner_failure":
                    obs.planner_failed = True
                elif etype in {"plan", "plan_generated"}:
                    if event.get("tools") or event.get("route") == "direct_answer":
                        obs.planner_ok = True
                elif etype == "strategy" and not obs.planner_failed:
                    # legacy signal: a strategy with real content implies a plan
                    if str(event.get("content") or "").strip():
                        obs.planner_ok = True
                elif etype == "tool_completed":
                    obs.tools_called += 1
                elif etype in {"hard_failure", "verification_failed", "side_effect_violation"}:
                    obs.hallucination_flag = True
                elif etype == "token":
                    obs.final_text += str(event.get("content") or "")
                elif etype in {"run_complete", "done"}:
                    obs.completed = True
                elif etype == "error":
                    obs.error = str(event.get("message") or "error")

        await asyncio.wait_for(_drive(), timeout=timeout_s)
    except asyncio.TimeoutError:
        obs.error = "timeout"
    except Exception as exc:  # pragma: no cover - defensive
        obs.error = f"{type(exc).__name__}: {exc}"
    finally:
        obs.latency_s = round(time.perf_counter() - start, 3)
    return obs


# ── Scoring ──────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    model: str
    task_id: str
    tier: str
    planner_ok: bool
    tools_called: int
    completed: bool
    content_ok: bool
    hallucination_flag: bool
    error: str
    latency_s: float


def score_run(task: EvalTask, obs: RunObservation) -> RunResult:
    content_ok = True
    if task.success_substrings:
        text = obs.final_text.lower()
        content_ok = any(s.lower() in text for s in task.success_substrings)
    # planner_ok is true if it explicitly succeeded and never explicitly failed
    planner_ok = obs.planner_ok and not obs.planner_failed
    return RunResult(
        model="",  # filled by caller
        task_id=task.task_id,
        tier=task.tier,
        planner_ok=planner_ok,
        tools_called=obs.tools_called,
        completed=obs.completed,
        content_ok=content_ok,
        hallucination_flag=obs.hallucination_flag,
        error=obs.error,
        latency_s=obs.latency_s,
    )


# ── Orchestration ────────────────────────────────────────────────────────────
async def run_battery(
    runner: RunnerProtocol,
    *,
    models: list[str],
    runs: int,
    tasks: tuple[EvalTask, ...] = TASK_BATTERY,
) -> list[RunResult]:
    results: list[RunResult] = []
    for model in models:
        for task in tasks:
            for _ in range(runs):
                obs = await observe_run(runner, model=model, prompt=task.prompt)
                result = score_run(task, obs)
                result.model = model
                results.append(result)
                _print_tick(result)
    return results


def _print_tick(r: RunResult) -> None:
    flag = "✓" if r.planner_ok else "✗"
    extra = f" err={r.error}" if r.error else ""
    print(f"  [{flag}] {r.model:<32} {r.task_id:<28} tools={r.tools_called} done={int(r.completed)}{extra}")


# ── Reporting ────────────────────────────────────────────────────────────────
def summarize(results: list[RunResult]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[RunResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for model, rows in by_model.items():
        n = len(rows)
        planner_rate = sum(1 for r in rows if r.planner_ok) / n if n else 0.0
        complete_rate = sum(1 for r in rows if r.completed) / n if n else 0.0
        content_rate = sum(1 for r in rows if r.content_ok) / n if n else 0.0
        halluc_rate = sum(1 for r in rows if r.hallucination_flag) / n if n else 0.0
        latencies = [r.latency_s for r in rows if r.latency_s > 0]
        summary[model] = {
            "n": n,
            "planner_success_rate": round(planner_rate, 3),
            "completion_rate": round(complete_rate, 3),
            "content_accuracy": round(content_rate, 3),
            "hallucination_rate": round(halluc_rate, 3),
            "median_latency_s": round(statistics.median(latencies), 2) if latencies else 0.0,
        }
    return summary


def write_csv(results: list[RunResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [
            "model", "task_id", "tier", "planner_ok", "tools_called",
            "completed", "content_ok", "hallucination_flag", "error", "latency_s",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("CONVERGENCE SUMMARY  (the row that matters for funding: planner_success_rate)")
    print("=" * 78)
    header = f"{'model':<34}{'planner':>9}{'complete':>10}{'content':>9}{'halluc':>8}{'p50 s':>8}"
    print(header)
    print("-" * 78)
    for model, s in summary.items():
        print(
            f"{model:<34}"
            f"{s['planner_success_rate']:>9.0%}"
            f"{s['completion_rate']:>10.0%}"
            f"{s['content_accuracy']:>9.0%}"
            f"{s['hallucination_rate']:>8.0%}"
            f"{s['median_latency_s']:>8.1f}"
        )
    print("=" * 78)


# ── Reference runner (managed RunEngine) ─────────────────────────────────────
# Swap this for a stub in CI. Kept import-light so the harness loads without a
# full runtime present.
class ManagedRunEngineRunner:
    def __init__(self, *, agent_id: str = "default", owner_id: str = "eval"):
        self.agent_id = agent_id
        self.owner_id = owner_id

    async def run(self, *, model: str, prompt: str) -> AsyncIterator[dict[str, Any]]:
        # Local import so the module loads even when the runtime isn't installed.
        from orchestration.agent_manager import AgentManager  # type: ignore
        # The caller is expected to wire a real AgentManager; this reference
        # implementation assumes a module-level factory `get_agent_manager()`.
        from orchestration.bootstrap import get_agent_manager  # type: ignore

        manager: AgentManager = get_agent_manager()
        instance = manager.get_agent_instance(
            agent_id=self.agent_id, model_override=model, owner_id=self.owner_id
        )
        async for event in instance.chat_stream(
            user_message=prompt, model=model, owner_id=self.owner_id
        ):
            yield event


# ── Stub runner for smoke-testing the harness itself (no runtime needed) ─────
class StubRunner:
    """Deterministic fake runner so the harness can be tested in CI without
    a live model. Emulates a 'small model fails planner 30% of the time'
    distribution so the summary math is exercised."""

    def __init__(self, planner_fail_rate: float = 0.3):
        self._fail_rate = planner_fail_rate
        self._counter = 0

    async def run(self, *, model: str, prompt: str) -> AsyncIterator[dict[str, Any]]:
        self._counter += 1
        fails = (self._counter % 10) < (self._fail_rate * 10)
        if fails and "claude" not in model.lower():
            yield {"type": "planner_failure", "code": "PLANNER_FAIL"}
            yield {"type": "error", "message": "planner failed"}
            return
        yield {"type": "plan", "tools": ["web_search"], "route": "tool_loop"}
        yield {"type": "tool_completed", "tool_name": "web_search", "success": True}
        yield {"type": "token", "content": "AAPL 391 example"}
        yield {"type": "run_complete"}


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shovs convergence eval harness")
    p.add_argument("--models", default="stub:small,stub:claude",
                   help="comma-separated model strings")
    p.add_argument("--runs", type=int, default=10, help="runs per (model, task)")
    p.add_argument("--out", default="evals/results.csv", help="CSV output path")
    p.add_argument("--stub", action="store_true",
                   help="use the built-in StubRunner (no runtime required)")
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    runner: RunnerProtocol = StubRunner() if args.stub or all(
        m.startswith("stub:") for m in models
    ) else ManagedRunEngineRunner()

    print(f"Running {args.runs} runs/task across {len(models)} model(s), "
          f"{len(TASK_BATTERY)} tasks each...\n")
    results = await run_battery(runner, models=models, runs=args.runs)

    if results:
        write_csv(results, args.out)
        print(f"\nWrote {len(results)} rows → {args.out}")
    summary = summarize(results)
    print_summary(summary)


def main() -> None:
    asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    main()