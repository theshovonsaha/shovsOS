from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SourceCollectionScenario:
    """Expected state for multi-entity source collection workflows.

    The evaluator is intentionally domain-neutral: the caller supplies the
    locked entities and expected URL quotas. It then checks whether the runtime
    searched and fetched against that state instead of drifting to noisy
    entities discovered later.
    """

    objective: str
    entities: list[str]
    urls_per_entity: int
    total_urls: int
    discovery_url: str = ""
    query_template: str = "{entity} stock news June 13 2026"
    forbidden_query_terms: list[str] = field(default_factory=list)
    allowed_fetch_urls_by_entity: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SourceCollectionEval:
    passed: bool
    score: float
    detail: str
    issues: list[str]
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_entities(entities: list[str]) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for item in entities:
        entity = str(item or "").strip().upper()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        clean.append(entity)
    return clean


def _query_for_entity(template: str, entity: str) -> str:
    try:
        return str(template).format(entity=entity)
    except Exception:
        return f"{entity} {template}".strip()


def evaluate_source_collection_trace(
    *,
    scenario: SourceCollectionScenario,
    tool_calls: list[dict[str, Any]],
    source_contract_events: Optional[list[dict[str, Any]]] = None,
) -> SourceCollectionEval:
    """Evaluate a run trace against a source-collection scenario state.

    This is a lightweight local analogue of proxy state-based evaluation:
    infer the observable state from tool calls and compare it against the
    expected scenario, rather than trusting the final answer text.
    """

    entities = _clean_entities(scenario.entities)
    expected_queries = {
        entity: _query_for_entity(scenario.query_template, entity)
        for entity in entities
    }
    searches = [
        str((item.get("query") if isinstance(item, dict) else "") or "").strip()
        for item in tool_calls
        if isinstance(item, dict) and str(item.get("tool") or item.get("tool_name") or "") == "web_search"
    ]
    fetches = [
        str((item.get("url") if isinstance(item, dict) else "") or "").strip()
        for item in tool_calls
        if isinstance(item, dict) and str(item.get("tool") or item.get("tool_name") or "") == "web_fetch"
    ]

    issues: list[str] = []
    searches_by_entity: dict[str, list[str]] = {}
    for entity, expected_query in expected_queries.items():
        matching = [query for query in searches if query == expected_query]
        searches_by_entity[entity] = matching
        if not matching:
            issues.append(f"missing_targeted_search:{entity}")

    forbidden_hits = [
        query
        for query in searches
        for term in scenario.forbidden_query_terms
        if term and term.upper() in query.upper()
    ]
    if forbidden_hits:
        issues.append("forbidden_query_drift")

    if scenario.discovery_url and scenario.discovery_url not in fetches:
        issues.append("missing_discovery_fetch")

    fetched_by_entity: dict[str, list[str]] = {}
    allowed_by_entity = {
        str(entity).upper(): [str(url) for url in urls]
        for entity, urls in (scenario.allowed_fetch_urls_by_entity or {}).items()
    }
    allowed_all = {url for urls in allowed_by_entity.values() for url in urls}
    for entity in entities:
        allowed_urls = allowed_by_entity.get(entity, [])
        fetched = [url for url in fetches if url in allowed_urls] if allowed_urls else [
            url for url in fetches if entity.lower() in url.lower() or entity.upper() in url.upper()
        ]
        fetched_by_entity[entity] = fetched
        if len(fetched) < scenario.urls_per_entity:
            issues.append(f"missing_fetch_quota:{entity}:{len(fetched)}/{scenario.urls_per_entity}")

    entity_fetches = [url for urls in fetched_by_entity.values() for url in urls]
    if len(entity_fetches) < scenario.total_urls:
        issues.append(f"missing_total_fetch_quota:{len(entity_fetches)}/{scenario.total_urls}")

    if allowed_all:
        off_contract_fetches = [
            url
            for url in fetches
            if url
            and url != scenario.discovery_url
            and url not in allowed_all
        ]
        if off_contract_fetches:
            issues.append("off_contract_fetch")
    else:
        off_contract_fetches = []

    event_entities = []
    event_contracts = []
    for event in source_contract_events or []:
        data = event.get("data") if isinstance(event, dict) else event
        if not isinstance(data, dict):
            continue
        if data.get("entities"):
            event_entities.append([str(item).upper() for item in data.get("entities") or []])
        if data.get("source_contract"):
            event_contracts.append(dict(data.get("source_contract") or {}))
    if entities and event_entities and entities not in event_entities:
        issues.append("source_contract_entities_mismatch")
    if event_contracts and not any(int(contract.get("total_urls") or 0) == scenario.total_urls for contract in event_contracts):
        issues.append("source_contract_total_mismatch")

    checks = [
        all(searches_by_entity.get(entity) for entity in entities),
        not forbidden_hits,
        not scenario.discovery_url or scenario.discovery_url in fetches,
        all(len(fetched_by_entity.get(entity, [])) >= scenario.urls_per_entity for entity in entities),
        len(entity_fetches) >= scenario.total_urls,
        not off_contract_fetches,
    ]
    score = sum(1 for check in checks if check) / len(checks)
    passed = not issues
    detail = "source collection matched scenario state" if passed else "; ".join(issues)
    return SourceCollectionEval(
        passed=passed,
        score=round(score, 4),
        detail=detail,
        issues=issues,
        state={
            "objective": scenario.objective,
            "entities": entities,
            "expected_queries": expected_queries,
            "searches": searches,
            "fetches": fetches,
            "searches_by_entity": searches_by_entity,
            "fetched_by_entity": fetched_by_entity,
            "entity_fetch_count": len(entity_fetches),
            "total_url_requirement": scenario.total_urls,
            "forbidden_hits": forbidden_hits,
            "off_contract_fetches": off_contract_fetches,
            "source_contract_event_count": len(source_contract_events or []),
        },
    )
