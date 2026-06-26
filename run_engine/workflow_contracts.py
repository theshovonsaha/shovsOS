from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


WORKFLOW_CONTRACT_VERSION = "workflow-contract-v1"

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class EntityLock:
    value: str
    entity_type: str = "entity"
    source: str = "runtime"
    status: str = "pending"


@dataclass(frozen=True)
class EvidenceRequirement:
    id: str
    description: str
    status: str = "pending"
    required_count: int = 0
    observed_count: int = 0


@dataclass(frozen=True)
class CompletionGate:
    final_answer_allowed: bool
    missing_slots: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class WorkflowContract:
    id: str
    version: str
    workflow_shape: str
    objective: str
    allowed_tools: list[str] = field(default_factory=list)
    entity_locks: list[EntityLock] = field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = field(default_factory=list)
    completion_gate: CompletionGate = field(default_factory=lambda: CompletionGate(final_answer_allowed=True))
    tool_policy: dict[str, Any] = field(default_factory=dict)
    continuation_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "Workflow Contract:",
            f"- shape: {self.workflow_shape}",
            f"- final answer allowed: {self.completion_gate.final_answer_allowed}",
        ]
        if self.allowed_tools:
            lines.append("- allowed tools: " + ", ".join(self.allowed_tools))
        if self.entity_locks:
            lines.append("Entity locks:")
            for lock in self.entity_locks[:8]:
                lines.append(f"- {lock.value} ({lock.status}, {lock.entity_type})")
        if self.evidence_requirements:
            lines.append("Evidence requirements:")
            for req in self.evidence_requirements[:8]:
                count = ""
                if req.required_count:
                    count = f" [{req.observed_count}/{req.required_count}]"
                lines.append(f"- {req.id} ({req.status}){count}: {req.description}")
        if self.completion_gate.missing_slots:
            lines.append("Missing slots:")
            lines.extend(f"- {slot}" for slot in self.completion_gate.missing_slots[:8])
        if self.completion_gate.reason:
            lines.append(f"Gate reason: {self.completion_gate.reason}")
        return "\n".join(lines)


def infer_workflow_contract(
    objective: str,
    *,
    allowed_tools: list[str] | tuple[str, ...] | None = None,
) -> WorkflowContract:
    text = str(objective or "").strip()
    tools = [str(tool).strip() for tool in (allowed_tools or []) if str(tool).strip()]
    shape = classify_workflow_shape(text)
    if shape == "source_collection":
        return _source_collection_contract(text, tools)
    if shape == "simple_chat":
        return WorkflowContract(
            id="contract_simple_chat",
            version=WORKFLOW_CONTRACT_VERSION,
            workflow_shape=shape,
            objective=text,
            allowed_tools=[],
            completion_gate=CompletionGate(final_answer_allowed=True, reason="simple chat does not need tools"),
        )
    return WorkflowContract(
        id=f"contract_{shape}",
        version=WORKFLOW_CONTRACT_VERSION,
        workflow_shape=shape,
        objective=text,
        allowed_tools=tools,
        completion_gate=CompletionGate(final_answer_allowed=True, reason="no deterministic completion quota inferred"),
        metadata={"inference": "shape_only"},
    )


def classify_workflow_shape(objective: str) -> str:
    text = str(objective or "").strip()
    lowered = text.lower()
    if _looks_like_simple_chat(lowered):
        return "simple_chat"
    if _looks_like_source_collection(lowered):
        return "source_collection"
    if any(term in lowered for term in ("fix", "implement", "test", "bug", "code", "repo", "frontend", "backend")):
        return "coding_change"
    if any(term in lowered for term in ("remember", "my name", "i live", "actually", "call me")):
        return "memory_correction"
    if any(term in lowered for term in ("research", "analyze", "compare", "report", "summarize")):
        return "research_report"
    return "open_ended_chat"


def _looks_like_simple_chat(lowered: str) -> bool:
    text = str(lowered or "").strip()
    if not text:
        return True
    normalized = re.sub(r"[!.?,]+", " ", text)
    tokens = [token for token in normalized.split() if token]
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you|ok|okay)[!. ]*", text):
        return True
    if len(tokens) > 6:
        return False
    action_terms = {
        "search", "find", "fetch", "research", "analyze", "compare", "summarize",
        "write", "draft", "make", "create", "implement", "fix", "remember",
        "what", "why", "how", "when", "where", "who",
    }
    if any(token in action_terms for token in tokens):
        return False
    greeting_terms = {"hi", "hello", "hey", "yo"}
    ack_terms = {"thanks", "thank", "ok", "okay", "cool", "great", "nice", "got", "it", "again", "there", "helps"}
    token_set = set(tokens)
    return bool(token_set & greeting_terms) or bool(token_set & ack_terms and len(tokens) <= 4)


def update_contract_from_tool_results(
    contract: WorkflowContract,
    tool_results: list[dict[str, Any]],
    *,
    tool_turn: int = 0,
    max_tool_turns: int = 0,
) -> WorkflowContract:
    if contract.workflow_shape != "source_collection":
        return contract
    metadata = dict(contract.metadata or {})
    entity_count = int(metadata.get("entity_count") or 0)
    per_entity = int(metadata.get("results_per_entity") or 0)
    total_fetches = int(metadata.get("total_fetches") or 0)
    entity_values = [lock.value for lock in contract.entity_locks if lock.status == "locked"]
    searched_entities = set()
    fetched_urls = set()
    # Track the order of search queries so fetched URLs can be attributed to the
    # entity whose search most recently preceded them, even when the fetched URL
    # host does not literally contain the ticker (e.g. finance.yahoo.com/quote/X).
    last_searched_entity: str | None = None
    selected_urls_by_entity: dict[str, set[str]] = {entity: set() for entity in entity_values}
    fetched_urls_by_entity: dict[str, set[str]] = {entity: set() for entity in entity_values}

    for result in tool_results or []:
        if not isinstance(result, dict) or not result.get("success"):
            continue
        tool_name = str(result.get("tool_name") or result.get("tool") or "")
        args = result.get("arguments") if isinstance(result.get("arguments"), dict) else {}
        if tool_name == "web_search":
            query = str(args.get("query") or "").upper()
            for entity in entity_values:
                if re.search(rf"\b{re.escape(entity.upper())}\b", query):
                    searched_entities.add(entity)
                    last_searched_entity = entity
                    for url in _extract_urls_from_result(result):
                        selected_urls_by_entity.setdefault(entity, set()).add(url)
        elif tool_name in {"web_fetch", "web_fetch_batch"}:
            result_urls = set(_extract_urls_from_result(result))
            url = str(args.get("url") or "").strip()
            if url:
                result_urls.add(url)
            fetched_urls.update(result_urls)
            # Broadened attribution: a fetched URL counts for an entity if the
            # entity name appears in the URL, else for the most recently searched
            # entity. This stops the gate locking when a generic-host source
            # (Yahoo/Fool) is fetched after an entity-targeted search.
            for fetched in result_urls:
                attributed = False
                for entity in entity_values:
                    if entity.lower() in fetched.lower():
                        fetched_urls_by_entity.setdefault(entity, set()).add(fetched)
                        attributed = True
                if not attributed and last_searched_entity is not None:
                    fetched_urls_by_entity.setdefault(last_searched_entity, set()).add(fetched)

    missing_slots: list[str] = []
    requirements: list[EvidenceRequirement] = []
    if entity_count and len(entity_values) < entity_count:
        missing_slots.append("locked_entities")
    for entity in entity_values:
        searched = entity in searched_entities
        if not searched:
            missing_slots.append(f"{entity}_search_results")
        selected_count = len(selected_urls_by_entity.get(entity, set()))
        fetched_for_entity = len(fetched_urls_by_entity.get(entity, set()))
        # Coverage counts both search-selected URLs and fetched-and-attributed
        # URLs, so an entity backed by a real fetched source is never reported
        # as missing just because the selection quota was computed from search.
        entity_coverage = max(selected_count, fetched_for_entity)
        if per_entity and entity_coverage < per_entity:
            missing_slots.append(f"{entity}_selected_urls")
        requirements.append(
            EvidenceRequirement(
                id=f"{entity}_sources",
                description=f"Search and select {per_entity} source URL(s) for {entity}",
                status="complete" if searched and entity_coverage >= per_entity else "pending",
                required_count=per_entity,
                observed_count=entity_coverage,
            )
        )
    if total_fetches and len(fetched_urls) < total_fetches:
        missing_slots.append("total_fetched_urls")
    requirements.append(
        EvidenceRequirement(
            id="fetched_sources",
            description=f"Fetch {total_fetches} selected source URL(s)",
            status="complete" if not total_fetches or len(fetched_urls) >= total_fetches else "pending",
            required_count=total_fetches,
            observed_count=len(fetched_urls),
        )
    )
    # Hard escape: the gate must never lock forever. Once the overall fetch
    # target is met, or the loop has spent its turn budget, allow the answer and
    # let the response phase be honest about any slot that stayed open.
    min_fetches_met = bool(total_fetches) and len(fetched_urls) >= total_fetches
    hard_escape = bool(max_tool_turns) and tool_turn >= max_tool_turns
    complete = (not missing_slots) or min_fetches_met or hard_escape
    if complete and missing_slots:
        gate_reason = (
            "source collection complete (hard escape: turn budget reached)"
            if hard_escape and not min_fetches_met
            else "source collection complete (fetch target met)"
        )
    else:
        gate_reason = "source collection complete" if complete else "source collection missing required evidence"
    return WorkflowContract(
        id=contract.id,
        version=contract.version,
        workflow_shape=contract.workflow_shape,
        objective=contract.objective,
        allowed_tools=list(contract.allowed_tools),
        entity_locks=list(contract.entity_locks),
        evidence_requirements=requirements,
        completion_gate=CompletionGate(
            final_answer_allowed=complete,
            missing_slots=list(dict.fromkeys(missing_slots)),
            reason=gate_reason,
        ),
        tool_policy=dict(contract.tool_policy),
        continuation_policy=dict(contract.continuation_policy),
        metadata={
            **metadata,
            "searched_entities": sorted(searched_entities),
            "fetched_url_count": len(fetched_urls),
            "gate_hard_escape": bool(hard_escape),
        },
    )


def _source_collection_contract(objective: str, allowed_tools: list[str]) -> WorkflowContract:
    lowered = objective.lower()
    entity_count = _extract_number_after(r"\btop\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", lowered)
    entity_match = _extract_number_after(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:stocks|companies|products|items|tools|options|entities|sources)\b",
        lowered,
    )
    if entity_match:
        entity_count = entity_match
    per_entity = _extract_number_after(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:relevant\s+)?(?:results?|sources?|urls?|links?|articles?|pages?)\s+(?:for\s+)?(?:each|per)\b",
        lowered,
    )
    total_fetches = _extract_number_after(r"\b(?:fetch|read|open)\s+(?:all\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+urls?\b", lowered)
    if not total_fetches:
        total_fetches = _extract_number_after(r"\ball\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+urls?\b", lowered)
    if not per_entity and total_fetches and entity_count:
        per_entity = max(1, total_fetches // entity_count)
    if entity_count and per_entity and (
        not total_fetches
        or (
            total_fetches == per_entity
            and bool(re.search(r"\b(each|per|separately)\b", lowered))
            and not re.search(r"\ball\s+\d+\s+urls?\b", lowered)
        )
    ):
        total_fetches = entity_count * per_entity
    entity_count = max(1, min(entity_count or 1, 20))
    per_entity = max(1, min(per_entity or 1, 20))
    total_fetches = max(1, min(total_fetches or entity_count * per_entity, 100))
    missing = ["locked_entities"]
    if total_fetches:
        missing.append("total_fetched_urls")
    return WorkflowContract(
        id="contract_source_collection",
        version=WORKFLOW_CONTRACT_VERSION,
        workflow_shape="source_collection",
        objective=objective,
        allowed_tools=[tool for tool in allowed_tools if tool in {"web_search", "web_fetch", "web_fetch_batch", "source_collect", "source_contract", "source_select", "source_next_action", "source_coverage"}],
        entity_locks=[],
        evidence_requirements=[
            EvidenceRequirement(
                id="entity_lock",
                description=f"Discover and lock {entity_count} entity/entities before targeted source gathering",
                required_count=entity_count,
                observed_count=0,
            ),
            EvidenceRequirement(
                id="fetched_sources",
                description=f"Fetch {total_fetches} selected source URL(s)",
                required_count=total_fetches,
                observed_count=0,
            ),
        ],
        completion_gate=CompletionGate(
            final_answer_allowed=False,
            missing_slots=missing,
            reason="source collection requires locked entities and fetched source coverage",
        ),
        tool_policy={
            "needs_separate_queries": bool(re.search(r"\b(each|separately|per)\b", lowered)),
            "forbid_unlocked_entity_drift": True,
            "answer_from_fetched_sources_only": True,
        },
        continuation_policy={
            "persist_when_incomplete": True,
            "resume_from_missing_slots": True,
        },
        metadata={
            "entity_count": entity_count,
            "results_per_entity": per_entity,
            "total_fetches": total_fetches,
        },
    )


def _looks_like_source_collection(lowered: str) -> bool:
    has_collection = any(term in lowered for term in ("top ", "each", "separately", "per "))
    has_sources = any(term in lowered for term in ("url", "source", "result", "article", "link", "fetch", "web fetch", "read"))
    has_action = any(term in lowered for term in ("search", "find", "compare", "capture", "collect", "fetch"))
    return has_collection and has_sources and has_action


def _extract_number_after(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        return 0
    raw = str(match.group(1) or "").lower()
    if raw.isdigit():
        return int(raw)
    return int(_NUMBER_WORDS.get(raw, 0) or 0)


def _extract_urls_from_result(result: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    args = result.get("arguments") if isinstance(result.get("arguments"), dict) else {}
    if args.get("url"):
        urls.append(str(args.get("url")).strip())
    for key in ("extracted_urls", "fetched_urls", "selected_urls"):
        value = result.get(key)
        if isinstance(value, list):
            urls.extend(str(item).strip() for item in value if str(item).strip())
    content = str(result.get("content") or "")
    urls.extend(re.findall(r"https?://[^\s)\"'>,]+", content))
    return list(dict.fromkeys(url for url in urls if url.startswith("http")))
