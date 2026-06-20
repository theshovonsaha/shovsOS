"""Snippet composer — the unified retrieval surface.

The 80% move from the convergence vision: pull typed snippets from every
store, rank by ``relevance × state_affinity × source_authority × recency``,
drop below threshold, pack to budget.

Every existing store (facts, loci/drawers, candidate_signals, task_tracker)
becomes a snippet source via a kind handler. Every new feature (workflows,
documents, agent-built artifacts) registers as a new snippet kind.

The composer never summarizes — it selects. Snippets that survive the
budget are rendered verbatim from their source. Compression happens by
dropping low-score snippets, not by rewording high-score ones.

This file is the foundation. Subsequent passes will:
- Wire ``compose_snippets`` into ``run_engine/context_packets.py`` so packet
  building uses one composer call instead of 4-5 separate retrieval calls.
- Register ``workflow`` and ``document`` snippet kinds.
- Promote ``session_rag`` to owner-scope so its chunks become snippets.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from memory.semantic_graph import SemanticGraph
from memory.retrieval import _tokenize_set, detect_locus_by_overlap


# ── Snippet contract ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Snippet:
    """Universal container for a piece of text the runtime knows about.

    All snippet kinds — facts, drawers, candidates, workflows, documents,
    tools, tasks — fit this shape. The packet builder reads the composer's
    ranked list and emits one ContextItem per kind.

    Score is computed by the composer's ranking pipeline; sources are not
    expected to provide it. Confidence comes from the source store
    (deterministic facts = 1.0; candidate signals < 1.0).

    state_affinity is the set of dynamic-state names this snippet is most
    useful for (e.g. ``("recall",)`` for fact_history snippets). Empty
    means "all states." See ``project_dynamic_state_machine.md`` in memory
    for the state vocabulary.
    """

    kind: str
    text: str
    score: float
    source: str
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    state_affinity: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.text)


# ── Ranking primitives ──────────────────────────────────────────────────────


_ARCHAEOLOGY_RE = re.compile(
    r"\b(originally|first|before|earlier|did i (?:say|tell|mention)|"
    r"what did i (?:say|tell|originally)|when did i|history|previously|"
    r"used to|at first)\b",
    re.IGNORECASE,
)


def _query_signals_recall(query: str) -> bool:
    """True when the query has archaeology signals — past-tense lookups,
    'originally', 'when did I correct'. Triggers fact_history retrieval."""
    return bool(_ARCHAEOLOGY_RE.search(query or ""))


def _lexical_relevance(query: str, text: str) -> float:
    """Token-overlap relevance score in [0, 1]. Cheap, deterministic, no
    embedding required. Matches the BM25 spirit without the index cost."""
    q = _tokenize_set(query)
    if not q:
        return 0.0
    t = _tokenize_set(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / math.sqrt(len(q) * len(t))


def _state_affinity_boost(snippet_kind: str, state: str, affinity: tuple[str, ...]) -> float:
    """Multiplier in [0.3, 1.5] based on whether the snippet's kind matches
    the active state's expected lane subset.

    - 1.5 when the snippet's affinity explicitly includes the active state
    - 1.0 when the snippet has no affinity declared (works in any state)
    - 0.5 when the active state usually doesn't pull this kind
    - 0.3 when the kind is explicitly mismatched

    The state→preferred-kinds map encodes the dynamic-state design.
    """
    state_kinds: dict[str, set[str]] = {
        "gather": {"fact", "drawer", "document", "candidate", "tool"},
        "build": {"fact", "drawer", "workflow", "tool", "task"},
        "verify": {"fact", "task"},
        "respond": {"fact", "identity", "task"},
        "clarify": {"candidate", "fact", "identity"},
        "recall": {"fact", "fact_history", "drawer", "candidate"},
    }
    preferred = state_kinds.get(state, set())
    if affinity and state in affinity:
        return 1.5
    if not affinity and snippet_kind in preferred:
        return 1.0
    if snippet_kind in preferred:
        return 1.0
    if affinity and state not in affinity:
        return 0.3
    return 0.5


def _recency_decay(valid_from: Optional[Any]) -> float:
    """Multiplier in [0.5, 1.0] favoring more recent valid_from values.

    valid_from is the turn number the fact became valid. We don't have
    wall-clock timestamps for facts (only for created_at), but turn-level
    recency is enough for ranking within a session. None → 1.0 (no decay).
    """
    if valid_from is None:
        return 1.0
    try:
        n = int(valid_from)
    except (TypeError, ValueError):
        return 1.0
    # Logarithmic decay: turn 1 = 0.5, turn 100 = ~1.0. Old facts still
    # rankable but recent ones win on equal relevance.
    return 0.5 + 0.5 * (n / (n + 50.0))


_SOURCE_AUTHORITY: dict[str, float] = {
    # Deterministic stores rank highest.
    "semantic_graph.facts.current": 1.0,
    "semantic_graph.loci.compiled_drawer": 0.9,
    "semantic_graph.facts.history": 0.85,
    # Weak signals rank lower.
    "session_manager.candidate_signals": 0.55,
    # Future kinds — placeholders so the composer doesn't error on them.
    "workflow_registry": 0.95,
    "document_store": 0.7,
    "task_tracker": 0.85,
    "tool_registry": 0.95,
}


def _source_authority(source: str) -> float:
    return _SOURCE_AUTHORITY.get(source, 0.5)


# ── Kind handlers (each yields candidate Snippets, score=0 for now) ────────
#
# Handlers are pure functions: query + context → list[Snippet] with score=0.
# The composer applies the ranking pipeline after collection.


def _fact_snippets(
    query: str,
    *,
    graph: SemanticGraph,
    owner_id: Optional[str],
    limit: int = 40,
) -> list[Snippet]:
    """Current facts across all sessions for the owner (the L0 identity slice
    + everything that hasn't been voided). Each fact becomes one snippet."""
    try:
        facts = graph.get_owner_current_facts(owner_id, limit=limit) or []
    except Exception:
        return []
    out: list[Snippet] = []
    for subject, predicate, obj in facts:
        text = f"- {subject} — {predicate}: {obj}"
        out.append(Snippet(
            kind="fact",
            text=text,
            score=0.0,
            source="semantic_graph.facts.current",
            provenance={"subject": subject, "predicate": predicate, "object": obj},
            confidence=1.0,
            state_affinity=("gather", "respond", "verify", "clarify"),
        ))
    return out


def _fact_history_snippets(
    query: str,
    *,
    graph: SemanticGraph,
    owner_id: Optional[str],
    limit: int = 8,
) -> list[Snippet]:
    """Predicate-history snippets — only fired when the query has archaeology
    signals. Pulls the timeline for predicates named in the query.

    This closes the 50-turn shovs scenario's archaeology gap: turn 36
    ("did I say I moved to Berlin?"), turn 41 ("what was budget first vs now?"),
    turn 50 ("when did I correct Berlin?")."""
    if not _query_signals_recall(query):
        return []
    # Cheap heuristic for which predicates the question targets.
    from engine.direct_fact_policy import direct_fact_predicates
    predicates = direct_fact_predicates(query)
    if not predicates:
        return []
    out: list[Snippet] = []
    for predicate in predicates:
        try:
            history = graph.get_predicate_history(
                owner_id=owner_id,
                subject="User",
                predicate=predicate,
                limit=limit,
            )
        except Exception:
            history = []
        for row in history:
            status = row.get("status", "current")
            obj = row.get("object", "")
            valid_from = row.get("valid_from")
            valid_to = row.get("valid_to")
            if status == "current":
                text = f"- {predicate} (current, since turn {valid_from}): {obj}"
            else:
                text = (
                    f"- {predicate} (superseded, valid turns {valid_from}–{valid_to}): {obj}"
                )
            out.append(Snippet(
                kind="fact_history",
                text=text,
                score=0.0,
                source="semantic_graph.facts.history",
                provenance={
                    "subject": "User",
                    "predicate": predicate,
                    "object": obj,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "status": status,
                },
                confidence=1.0,
                state_affinity=("recall",),
            ))
    return out


def _drawer_snippets(
    query: str,
    *,
    graph: SemanticGraph,
    owner_id: Optional[str],
    available_loci: Optional[list[dict]] = None,
    neighbor_fanout: int = 3,
) -> list[Snippet]:
    """Compiled-drawer snippets for the locus the query targets, plus 1-hop
    neighbors. Identical content to Slice 5's spatial_drawers ContextItem,
    but exposed as snippets so the composer can rank/budget them with
    everything else."""
    if not available_loci:
        return []
    detected = detect_locus_by_overlap(query, list(available_loci))
    if not detected:
        return []
    out: list[Snippet] = []
    try:
        primary = graph.get_compiled_drawer(detected)
    except Exception:
        primary = None
    if primary:
        out.append(Snippet(
            kind="drawer",
            text=primary,
            score=0.0,
            source="semantic_graph.loci.compiled_drawer",
            provenance={"locus_id": detected, "hop": 0},
            confidence=1.0,
            state_affinity=("gather", "build", "respond"),
        ))
    try:
        neighbors = graph.get_locus_neighbors(detected, max_depth=1, max_fanout=neighbor_fanout)
    except Exception:
        neighbors = []
    for nbr_id, _depth, nbr_score in neighbors:
        try:
            nbr_drawer = graph.get_compiled_drawer(nbr_id)
        except Exception:
            nbr_drawer = None
        if not nbr_drawer:
            continue
        out.append(Snippet(
            kind="drawer",
            text=nbr_drawer,
            score=0.0,  # composer will weight by edge score in ranking
            source="semantic_graph.loci.compiled_drawer",
            provenance={"locus_id": nbr_id, "hop": 1, "edge_score": float(nbr_score)},
            confidence=0.85,
            state_affinity=("gather",),
        ))
    return out


def _candidate_snippets(
    query: str,
    *,
    candidate_signals: Optional[Iterable[dict]] = None,
) -> list[Snippet]:
    """Candidate signals as queryable snippets.

    Closes the 50-turn turn 49 gap ("what candidate signals are you tracking?")
    by making them composable with everything else. Today they're rendered
    into the prompt but invisible to retrieval."""
    if not candidate_signals:
        return []
    out: list[Snippet] = []
    for signal in candidate_signals:
        if not isinstance(signal, dict):
            continue
        fact = str(
            signal.get("fact")
            or " ".join(
                part for part in (
                    signal.get("subject"),
                    signal.get("predicate"),
                    signal.get("object"),
                ) if part
            )
        ).strip()
        if not fact:
            continue
        reason = str(signal.get("grounding_reason") or signal.get("reason") or "candidate").strip()
        try:
            confidence = float(signal.get("confidence", 0.4))
        except (TypeError, ValueError):
            confidence = 0.4
        out.append(Snippet(
            kind="candidate",
            text=f"- (candidate, {reason}): {fact}",
            score=0.0,
            source="session_manager.candidate_signals",
            provenance=dict(signal),
            confidence=confidence,
            state_affinity=("gather", "clarify"),
        ))
    return out


# ── The composer ────────────────────────────────────────────────────────────


_DEFAULT_PER_KIND_BUDGET_FRACTION = 0.4  # no kind gets more than 40% of budget


def compose_snippets(
    query: str,
    *,
    owner_id: Optional[str],
    graph: SemanticGraph,
    state: str = "gather",
    available_loci: Optional[list[dict]] = None,
    candidate_signals: Optional[Iterable[dict]] = None,
    budget_chars: int = 4000,
    kinds: Optional[set[str]] = None,
    score_threshold: float = 0.05,
) -> list[Snippet]:
    """Pull, rank, and budget snippets from every registered source.

    Args:
        query: the user query (or composed objective)
        owner_id: tenant scope; required for owner-scoped reads
        graph: SemanticGraph instance for fact/drawer/history reads
        state: dynamic-state name (gather/build/verify/respond/clarify/recall)
        available_loci: list of {id, name, description} dicts; required for
            drawer snippets and locus detection
        candidate_signals: session_manager.candidate_signals (already loaded
            by the caller; the composer doesn't fetch session state itself)
        budget_chars: hard cap on total snippet text length
        kinds: optional whitelist (e.g., ``{'fact', 'candidate'}``) to scope
            retrieval. None = all known kinds.
        score_threshold: drop snippets below this combined score

    Returns:
        list[Snippet] sorted by descending score, packed to budget,
        with per-kind caps applied.
    """
    # 1. Gather candidate snippets from each enabled kind handler.
    enabled = kinds or {"fact", "fact_history", "drawer", "candidate"}
    candidates: list[Snippet] = []
    if "fact" in enabled:
        candidates.extend(_fact_snippets(query, graph=graph, owner_id=owner_id))
    if "fact_history" in enabled:
        candidates.extend(_fact_history_snippets(query, graph=graph, owner_id=owner_id))
    if "drawer" in enabled:
        candidates.extend(_drawer_snippets(
            query, graph=graph, owner_id=owner_id, available_loci=available_loci,
        ))
    if "candidate" in enabled:
        candidates.extend(_candidate_snippets(query, candidate_signals=candidate_signals))

    if not candidates:
        return []

    # 2. Score each snippet via the ranking pipeline.
    scored: list[Snippet] = []
    for snip in candidates:
        relevance = _lexical_relevance(query, snip.text)
        # Identity-brief style snippets (every owner fact) don't have to
        # match the query lexically — they're context, not retrieval hits.
        # Floor relevance at 0.1 for owner-scoped facts so they survive
        # the threshold even on novel queries.
        if snip.kind == "fact" and snip.source == "semantic_graph.facts.current":
            relevance = max(relevance, 0.15)
        # Drawer snippets surface because the locus matched, not because
        # every drawer line overlaps the query — floor accordingly.
        if snip.kind == "drawer":
            relevance = max(relevance, 0.4)
        # fact_history is targeted retrieval — already filtered by predicate
        # match in the handler, so floor it high.
        if snip.kind == "fact_history":
            relevance = max(relevance, 0.7)

        affinity_mult = _state_affinity_boost(snip.kind, state, snip.state_affinity)
        authority = _source_authority(snip.source)
        recency = _recency_decay(snip.provenance.get("valid_from"))
        confidence = snip.confidence

        score = relevance * affinity_mult * authority * recency * confidence
        if score < score_threshold:
            continue
        scored.append(Snippet(
            kind=snip.kind,
            text=snip.text,
            score=round(score, 4),
            source=snip.source,
            provenance=snip.provenance,
            confidence=snip.confidence,
            state_affinity=snip.state_affinity,
        ))

    # 3. Sort by score descending, then pack to budget with per-kind caps.
    scored.sort(key=lambda s: (-s.score, s.kind, s.source))

    per_kind_cap = int(budget_chars * _DEFAULT_PER_KIND_BUDGET_FRACTION)
    per_kind_used: dict[str, int] = {}
    total_used = 0
    out: list[Snippet] = []
    for snip in scored:
        slen = len(snip.text)
        used_for_kind = per_kind_used.get(snip.kind, 0)
        if used_for_kind + slen > per_kind_cap:
            continue
        if total_used + slen > budget_chars:
            continue
        out.append(snip)
        per_kind_used[snip.kind] = used_for_kind + slen
        total_used += slen

    return out


# ── Trace summary helper ────────────────────────────────────────────────────


def summarize_composition(snippets: list[Snippet]) -> dict[str, Any]:
    """Compact summary for trace events. Caller passes this to the trace
    store so operators can see what the composer chose without having to
    reconstruct from raw snippet text."""
    by_kind: dict[str, int] = {}
    for snip in snippets:
        by_kind[snip.kind] = by_kind.get(snip.kind, 0) + 1
    return {
        "snippet_count": len(snippets),
        "total_chars": sum(len(s) for s in snippets),
        "by_kind": by_kind,
        "top_score": max((s.score for s in snippets), default=0.0),
        "top_provenance": [
            {"kind": s.kind, "score": s.score, "source": s.source}
            for s in snippets[:5]
        ],
    }
