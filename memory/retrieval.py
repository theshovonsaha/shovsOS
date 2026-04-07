from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional
import re

from memory.semantic_graph import SemanticGraph
from memory.vector_engine import VectorEngine


_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "it",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "not",
    "with",
    "by",
    "from",
    "as",
    "this",
    "that",
    "what",
    "which",
    "who",
    "my",
    "your",
}


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return {token for token in cleaned.split() if len(token) > 2 and token not in _STOPWORDS}


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize(text)
    if not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(1, len(q_tokens))


@dataclass
class _Hit:
    dedupe_key: str
    score: float
    source: str
    payload: dict


def _upsert_hit(
    table: dict[str, _Hit],
    *,
    dedupe_key: str,
    score: float,
    source: str,
    payload: dict,
) -> None:
    existing = table.get(dedupe_key)
    if existing is None:
        table[dedupe_key] = _Hit(
            dedupe_key=dedupe_key,
            score=score,
            source=source,
            payload={**payload, "sources": [source]},
        )
        return

    if score > existing.score:
        existing.score = score
    sources = existing.payload.get("sources") or []
    if source not in sources:
        sources.append(source)
        existing.payload["sources"] = sources
    for key, value in payload.items():
        if key not in existing.payload or not existing.payload.get(key):
            existing.payload[key] = value


async def unified_memory_search(
    query: str,
    *,
    owner_id: Optional[str],
    session_id: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.5,
    graph: Optional[SemanticGraph] = None,
) -> dict:
    """
    Canonical retrieval lane used by tools and APIs.
    Merges semantic graph hits, deterministic session facts, and vector-session hits.
    """
    started = perf_counter()
    safe_top_k = max(1, min(int(top_k or 5), 50))
    search_limit = max(safe_top_k * 3, 12)
    graph = graph or SemanticGraph()
    owner_scope = owner_id if owner_id is not None else None

    merged: dict[str, _Hit] = {}
    source_counts = {"semantic_graph": 0, "deterministic_fact": 0, "vector_engine": 0}

    semantic_hits = await graph.traverse(
        query,
        top_k=search_limit,
        threshold=max(0.0, float(threshold)),
        owner_id=owner_scope,
    )
    for item in semantic_hits:
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        object_ = str(item.get("object") or "").strip()
        signature = f"{subject}|{predicate}|{object_}".lower()
        score = float(item.get("similarity") or 0.0)
        _upsert_hit(
            merged,
            dedupe_key=signature,
            score=score,
            source="semantic_graph",
            payload={
                "kind": "triplet",
                "id": item.get("id"),
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "created_at": item.get("created_at"),
            },
        )
        source_counts["semantic_graph"] += 1

    if session_id:
        current_facts = graph.get_current_facts(session_id, owner_id=owner_scope)
        for subject, predicate, object_ in current_facts:
            fact_text = f"{subject} {predicate} {object_}".strip()
            lexical = _lexical_overlap_score(query, fact_text)
            # Keep deterministic lane high-priority, but still rank by relevance.
            score = 0.62 + min(0.35, lexical * 0.35)
            signature = f"{subject}|{predicate}|{object_}".lower()
            _upsert_hit(
                merged,
                dedupe_key=signature,
                score=score,
                source="deterministic_fact",
                payload={
                    "kind": "fact",
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "created_at": None,
                },
            )
            source_counts["deterministic_fact"] += 1

        try:
            vector_hits = await VectorEngine(
                session_id=session_id,
                agent_id="default",
                owner_id=owner_scope,
            ).query(query, limit=search_limit)
        except Exception:
            vector_hits = []

        for index, item in enumerate(vector_hits):
            key = str(item.get("key") or "").strip()
            anchor = str(item.get("anchor") or "").strip()
            if not key and not anchor:
                continue
            signature = f"{key}|{anchor[:120]}".lower()
            rank_score = max(0.05, 0.45 - (index * 0.03))
            lexical = _lexical_overlap_score(query, f"{key} {anchor}")
            score = rank_score + min(0.25, lexical * 0.25)
            _upsert_hit(
                merged,
                dedupe_key=signature,
                score=score,
                source="vector_engine",
                payload={
                    "kind": "anchor",
                    "id": item.get("id"),
                    "key": key,
                    "anchor": anchor,
                    "created_at": str((item.get("metadata") or {}).get("created_at") or ""),
                },
            )
            source_counts["vector_engine"] += 1

    ranked = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:safe_top_k]
    duration_ms = round((perf_counter() - started) * 1000.0, 2)
    results = []
    for hit in ranked:
        payload = dict(hit.payload)
        payload["score"] = round(float(hit.score), 4)
        payload["sources"] = list(payload.get("sources") or [])
        results.append(payload)

    return {
        "query": query,
        "results": results,
        "stats": {
            "total_hits": len(results),
            "candidate_pool": len(merged),
            "duration_ms": duration_ms,
            "source_counts": source_counts,
            "session_scoped": bool(session_id),
        },
    }

