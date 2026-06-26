from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionItem:
    id: str
    kind: str
    text: str
    status: str = "info"


_WEIGHTS = {
    "plan": {"objective": 1.0, "contract": 0.9, "pending": 0.8, "evidence": 0.2, "risk": 0.9},
    "act": {"objective": 0.5, "contract": 1.0, "pending": 1.0, "evidence": 0.4, "risk": 1.0},
    "verify": {"objective": 0.4, "contract": 1.0, "pending": 0.9, "evidence": 1.0, "risk": 1.0},
    "respond": {"objective": 0.6, "contract": 0.8, "pending": 0.9, "evidence": 1.0, "risk": 1.0},
}


def select_attention(items: list[AttentionItem], phase: str, limit: int = 6) -> list[tuple[float, AttentionItem]]:
    weights = _WEIGHTS.get(phase, _WEIGHTS["plan"])
    scored: list[tuple[float, AttentionItem]] = []
    for item in items:
        score = weights.get(item.kind, 0.1)
        if item.status in {"error", "blocked", "missing"}:
            score += weights.get("risk", 0.0)
        if item.status in {"done", "stale"}:
            score *= 0.35
        scored.append((round(score, 3), item))
    scored.sort(key=lambda row: (-row[0], row[1].kind, row[1].id))
    return scored[:limit]
