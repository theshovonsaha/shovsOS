from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import SourceContract


@dataclass(frozen=True)
class TraceEval:
    ok: bool
    score: float
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_trace(contract: SourceContract, events: list[dict[str, Any]]) -> TraceEval:
    """State-based trace check for one common agent failure: drift after planning."""

    failures: list[str] = []
    searches = [e for e in events if e.get("tool") == "web_search" and e.get("ok", True)]
    fetches = [e for e in events if e.get("tool") == "web_fetch" and e.get("ok", True)]
    if "web_search" in contract.required_tools and not searches:
        failures.append("missing_search")
    if contract.total_urls and len(fetches) < contract.total_urls:
        failures.append("missing_fetch_quota")

    locked = {str(e.get("entity")).upper() for e in events if e.get("kind") == "entity_locked" and e.get("entity")}
    drifted = [
        str(e.get("entity")).upper()
        for e in searches + fetches
        if e.get("entity") and locked and str(e.get("entity")).upper() not in locked
    ]
    if drifted:
        failures.append("entity_drift:" + ",".join(sorted(set(drifted))))

    target = max(1, len(contract.required_tools) + (1 if contract.total_urls else 0))
    passed = target - len(failures)
    score = max(0.0, round(passed / target, 3))
    return TraceEval(
        ok=not failures,
        score=score,
        failures=failures,
        metrics={
            "search_count": len(searches),
            "fetch_count": len(fetches),
            "required_fetch_count": contract.total_urls,
            "locked_entities": sorted(locked),
        },
    )
