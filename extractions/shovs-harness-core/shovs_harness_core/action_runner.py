from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contract import SourceContract
from .evals import TraceEval, evaluate_trace
from .proposers import ProposedAction


@dataclass(frozen=True)
class ActionViolation:
    code: str
    message: str
    action_index: int
    action: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionRunReport:
    accepted_events: list[dict[str, Any]]
    violations: list[ActionViolation]
    eval: TraceEval
    can_respond: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_events": self.accepted_events,
            "violations": [asdict(item) for item in self.violations],
            "eval": asdict(self.eval),
            "can_respond": self.can_respond,
        }


def enforce_proposed_actions(
    contract: SourceContract,
    actions: list[ProposedAction],
    *,
    candidate_urls: dict[str, list[str]] | None = None,
) -> ActionRunReport:
    """Validate model-proposed actions against the source contract.

    The model may propose. This function decides what becomes state. Invalid
    fetches, drifted entities, duplicate URLs, and premature responses are
    recorded as violations instead of becoming successful trace events.
    """

    candidates = {str(k).upper(): list(v) for k, v in (candidate_urls or {}).items()}
    target_entities = contract.entity_count or 0
    per_entity = contract.urls_per_entity or 0
    locked: list[str] = []
    fetched_urls: set[str] = set()
    fetch_counts: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    violations: list[ActionViolation] = []

    for index, action in enumerate(actions):
        kind = str(action.action or "").lower()
        if kind == "lock":
            if locked:
                violations.append(_violation("duplicate_lock", "entities were already locked", index, action))
                continue
            proposed = _unique_upper(action.entities)
            if target_entities and len(proposed) < target_entities:
                violations.append(
                    _violation("lock_underfilled", f"expected {target_entities} entities, got {len(proposed)}", index, action)
                )
            if target_entities and len(proposed) > target_entities:
                violations.append(
                    _violation("lock_overfilled", f"expected {target_entities} entities, got {len(proposed)}; clamped", index, action)
                )
            locked = proposed[: target_entities or len(proposed)]
            for entity in locked:
                events.append({"kind": "entity_locked", "entity": entity, "summary": "accepted lock"})
                if "web_search" in contract.required_tools:
                    events.append({"tool": "web_search", "entity": entity, "ok": True, "summary": "deterministic entity search"})
            continue

        if kind == "fetch":
            entity = str(action.entity or "").strip().upper()
            url = str(action.url or "").strip()
            if not locked:
                violations.append(_violation("fetch_before_lock", "fetch proposed before entity lock", index, action))
                continue
            if not entity or entity not in locked:
                violations.append(_violation("entity_drift", f"fetch entity is not locked: {entity or 'missing'}", index, action))
                continue
            if not url:
                violations.append(_violation("missing_url", "fetch action has no URL", index, action))
                continue
            if candidates and url not in candidates.get(entity, []):
                violations.append(_violation("off_contract_url", f"URL is not in candidates for {entity}", index, action))
                continue
            if url in fetched_urls:
                violations.append(_violation("duplicate_fetch", "URL was already fetched", index, action))
                continue
            if per_entity and fetch_counts.get(entity, 0) >= per_entity:
                violations.append(_violation("over_fetch", f"{entity} already met fetch quota", index, action))
                continue
            fetched_urls.add(url)
            fetch_counts[entity] = fetch_counts.get(entity, 0) + 1
            events.append({"tool": "web_fetch", "entity": entity, "url": url, "ok": True, "summary": "accepted fetch"})
            continue

        if kind == "respond":
            current_eval = evaluate_trace(contract, events)
            if not current_eval.ok:
                violations.append(
                    _violation("premature_respond", f"response blocked; failures={current_eval.failures}", index, action)
                )
            continue

        violations.append(_violation("unknown_action", f"unknown action: {kind or 'missing'}", index, action))

    final_eval = evaluate_trace(contract, events)
    return ActionRunReport(
        accepted_events=events,
        violations=violations,
        eval=final_eval,
        can_respond=final_eval.ok,
    )


def _unique_upper(values: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for value in values:
        item = str(value or "").strip().upper()
        if item and item not in out:
            out.append(item)
    return out


def _violation(code: str, message: str, index: int, action: ProposedAction) -> ActionViolation:
    return ActionViolation(
        code=code,
        message=message,
        action_index=index,
        action={
            "action": action.action,
            "entity": action.entity,
            "url": action.url,
            "entities": list(action.entities),
        },
    )
