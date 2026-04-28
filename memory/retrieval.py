from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Optional
import re

from memory.semantic_graph import SemanticGraph
from memory.vector_engine import VectorEngine
from config.logger import log
from engine.direct_fact_policy import direct_fact_predicates, normalize_memory_predicate


_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "by", "from", "as",
    "this", "that", "what", "which", "who", "my", "your",
}

# ── Query sanitizer (adapted from mempalace/query_sanitizer.py) ────────────
# Problem: agents sometimes prepend system prompts (2000+ chars) to memory
# queries, causing near-total retrieval failure because the embedding vector
# is dominated by the system prompt rather than the actual question.
# Recovery: extract the actual question from the tail of the contaminated query.

_SAFE_QUERY_LENGTH = 200   # Below this: almost certainly clean
_MAX_QUERY_LENGTH = 250    # Hard cap on extracted query length
_MIN_QUERY_LENGTH = 10     # Extracted result shorter than this = failed
_QUESTION_MARK_RE = re.compile(r"[?？]\s*[\"']?\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？\n]+")
_QUOTE_CHARS = {"'", '"'}


def _sanitize_query(raw_query: str) -> str:
    """
    Extract the actual search intent from a potentially prompt-contaminated query.
    Uses tail-extraction: system prompts are prepended, so the real query is last.
    """
    if not raw_query or not raw_query.strip():
        return raw_query or ""
    raw_query = raw_query.strip()
    if len(raw_query) <= _SAFE_QUERY_LENGTH:
        return raw_query

    def _strip_quotes(s: str) -> str:
        s = s.strip()
        while len(s) >= 2 and s[:1] in _QUOTE_CHARS and s[:1] == s[-1:]:
            s = s[1:-1].strip()
        return s

    def _trim(candidate: str) -> str:
        candidate = _strip_quotes(candidate)
        if len(candidate) <= _MAX_QUERY_LENGTH:
            return candidate
        frags = [_strip_quotes(f) for f in _SENTENCE_SPLIT_RE.split(candidate) if f.strip()]
        for frag in reversed(frags):
            if _MIN_QUERY_LENGTH <= len(frag) <= _MAX_QUERY_LENGTH:
                return frag
        return candidate[-_MAX_QUERY_LENGTH:].strip()

    segments = [s.strip() for s in raw_query.split("\n") if s.strip()]

    # Step 1: look for a question-mark sentence (most reliable signal)
    for seg in reversed(segments):
        if _QUESTION_MARK_RE.search(seg) and len(seg) >= _MIN_QUERY_LENGTH:
            candidate = _trim(seg)
            if len(candidate) >= _MIN_QUERY_LENGTH:
                log("memory", "retrieval", f"Query sanitized via question extraction: {len(raw_query)}→{len(candidate)} chars")
                return candidate

    # Step 2: last meaningful line (system prompts are prepended, query is last)
    for seg in reversed(segments):
        if len(seg) >= _MIN_QUERY_LENGTH:
            candidate = _trim(seg)
            if len(candidate) >= _MIN_QUERY_LENGTH:
                log("memory", "retrieval", f"Query sanitized via tail extraction: {len(raw_query)}→{len(candidate)} chars")
                return candidate

    # Step 3: hard tail truncation
    return raw_query[-_MAX_QUERY_LENGTH:].strip()


# ── BM25 hybrid re-ranker (adapted from mempalace/searcher.py) ─────────────
# After vector retrieval produces a candidate set, BM25 re-ranks it using
# term-frequency overlap. This recovers exact-phrase queries that embeddings
# can miss (e.g. "what did I say about Stripe" where "Stripe" is a keyword).

def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [token for token in cleaned.split() if len(token) > 2 and token not in _STOPWORDS]


def _tokenize_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    Okapi-BM25 with Lucene-smoothed IDF. Safe for small candidate sets (re-ranking only).
    IDF computed over the provided corpus so rare-in-candidates terms score higher.
    """
    n_docs = len(documents)
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    avgdl = sum(doc_lens) / n_docs or 1.0

    df: dict[str, int] = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {
        term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
        for term in query_terms
    }

    scores: list[float] = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _hybrid_rerank(
    hits: list["_Hit"],
    query: str,
    *,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list["_Hit"]:
    """
    Re-rank a vector-retrieved candidate set using BM25 as a secondary signal.
    Convex combination: final_score = 0.6 * vector_sim + 0.4 * bm25_norm.
    Deterministic-fact hits (score ≥ 0.97) are pinned at the top and bypass re-ranking.
    """
    if len(hits) < 2:
        return hits

    # Split: pinned (deterministic facts) stay at top; rest get re-ranked.
    pinned = [h for h in hits if h.score >= 0.97]
    candidates = [h for h in hits if h.score < 0.97]
    if not candidates:
        return hits

    def _hit_text(h: "_Hit") -> str:
        p = h.payload
        parts = [
            str(p.get("subject") or ""),
            str(p.get("predicate") or ""),
            str(p.get("object") or ""),
            str(p.get("anchor") or ""),
            str(p.get("content") or ""),
        ]
        return " ".join(filter(None, parts))

    docs = [_hit_text(h) for h in candidates]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    for hit, raw, norm in zip(candidates, bm25_raw, bm25_norm):
        vec_sim = min(hit.score, 1.0)  # already normalised in [0,1]
        hit.score = round(vector_weight * vec_sim + bm25_weight * norm, 4)
        hit.payload["bm25_score"] = round(raw, 3)

    reranked = sorted(candidates, key=lambda h: h.score, reverse=True)
    return pinned + reranked


def detect_locus_by_overlap(query: str, loci: list[dict]) -> Optional[str]:
    """Token-overlap locus detector against an in-memory list of locus dicts.

    Used by both ``_detect_locus_from_query`` (which loads loci from a graph)
    and the orchestrator's plan path (which already has the list). Centralizing
    the algorithm here keeps the planner and the retrieval lane in sync — they
    used to disagree (the planner did a brittle substring match, retrieval did
    token overlap), causing facts to land in one locus and queries to find
    nothing.
    """
    if not loci:
        return None
    q_tokens = _tokenize_set(query)
    if not q_tokens:
        return None

    best_id: Optional[str] = None
    best_score = 0
    for locus in loci:
        lid = str(locus.get("id", "")).lower()
        lname = str(locus.get("name", "")).lower()
        id_tokens = _tokenize_set(lid.replace("_", " "))
        name_tokens = _tokenize_set(lname)
        locus_tokens = id_tokens | name_tokens
        if not locus_tokens:
            continue
        shared = q_tokens & locus_tokens
        if not shared:
            continue
        if len(shared) > best_score:
            best_score = len(shared)
            best_id = locus.get("id")
    return best_id


def _detect_locus_from_query(query: str, graph: SemanticGraph, owner_id: Optional[str] = None) -> Optional[str]:
    """Perform a 'Spatial Scan' on the query to see if it targets a Locus.

    Loads the locus list from ``graph`` then delegates to
    :func:`detect_locus_by_overlap`. Kept as a thin wrapper because the
    retrieval lane has the graph in scope, while the planner does not.
    """
    return detect_locus_by_overlap(query, graph.list_loci(owner_id=owner_id))


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = _tokenize_set(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize_set(text)
    if not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(1, len(q_tokens))


def _normalize_subject(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _normalize_object(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _build_direct_fact_index(
    current_facts: list[tuple[str, str, str]],
) -> dict[str, tuple[str, str, str]]:
    indexed: dict[str, tuple[str, str, str]] = {}
    for subject, predicate, object_ in current_facts or []:
        canonical = normalize_memory_predicate(predicate)
        if canonical:
            indexed[canonical] = (subject, canonical, object_)
    return indexed


def _build_current_fact_index(
    current_facts: list[tuple[str, str, str]],
) -> dict[tuple[str, str], tuple[str, str, str]]:
    indexed: dict[tuple[str, str], tuple[str, str, str]] = {}
    for subject, predicate, object_ in current_facts or []:
        normalized_subject = _normalize_subject(subject)
        canonical = normalize_memory_predicate(predicate)
        if not normalized_subject or not canonical:
            continue
        indexed[(normalized_subject, canonical)] = (subject, canonical, object_)
    return indexed


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
    locus_id: Optional[str] = None,
    top_k: int = 5,
    threshold: float = 0.3,
    graph: Optional[SemanticGraph] = None,
) -> dict:
    """
    Canonical retrieval lane used by tools and APIs.
    Merges compiled drawers, semantic graph hits, deterministic facts, and vector hits.
    """
    started = perf_counter()
    safe_top_k = max(1, min(int(top_k or 5), 50))
    search_limit = max(safe_top_k * 3, 12)
    graph = graph or SemanticGraph()
    owner_scope = owner_id if owner_id is not None else None

    # Guard: sanitize the query before embedding — prevents system-prompt
    # contamination from silently destroying retrieval quality.
    query = _sanitize_query(query)

    # Step 0: Spatial Scan (Auto-detect Locus if not explicitly provided)
    if not locus_id:
        locus_id = _detect_locus_from_query(query, graph, owner_id=owner_scope)
        if locus_id:
            log("memory", "retrieval", f"Spatial scan detected Locus target: {locus_id}")

    merged: dict[str, _Hit] = {}
    source_counts = {"compiled_drawer": 0, "semantic_graph": 0, "deterministic_fact": 0, "vector_engine": 0}
    suppressed_conflicts = 0
    target_predicates = direct_fact_predicates(query)
    current_facts: list[tuple[str, str, str]] = []
    direct_fact_index: dict[str, tuple[str, str, str]] = {}
    current_fact_index: dict[tuple[str, str], tuple[str, str, str]] = {}

    if session_id:
        current_facts = list(graph.get_current_facts(session_id, owner_id=owner_scope) or [])
        direct_fact_index = _build_direct_fact_index(current_facts)
        current_fact_index = _build_current_fact_index(current_facts)

    # Step 1: Karpathy Pattern - Prioritize Compiled Drawers (High-Density Context)
    if locus_id:
        drawer_content = graph.get_compiled_drawer(locus_id)
        if drawer_content:
            _upsert_hit(
                merged,
                dedupe_key=f"drawer|{locus_id}",
                score=1.0, # Absolute priority
                source="compiled_drawer",
                payload={
                    "kind": "compiled_drawer",
                    "locus_id": locus_id,
                    "content": drawer_content,
                },
            )
            source_counts["compiled_drawer"] += 1

        # Slice 4: expand to neighbor loci so an answer that lives one hop
        # away (e.g. detected "european_travel_plan" → linked "passports")
        # still surfaces. Capped depth ≤2, fanout ≤5, score decays per hop.
        if hasattr(graph, "get_locus_neighbors"):
            try:
                neighbors = graph.get_locus_neighbors(locus_id, max_depth=2, max_fanout=5)
            except Exception:
                neighbors = []
            for neighbor_id, _depth, neighbor_score in neighbors:
                neighbor_drawer = graph.get_compiled_drawer(neighbor_id)
                if not neighbor_drawer:
                    continue
                _upsert_hit(
                    merged,
                    dedupe_key=f"drawer|{neighbor_id}",
                    score=min(0.95, max(0.0, float(neighbor_score))),
                    source="compiled_drawer",
                    payload={
                        "kind": "compiled_drawer",
                        "locus_id": neighbor_id,
                        "content": neighbor_drawer,
                        "neighbor_of": locus_id,
                    },
                )
                source_counts["compiled_drawer"] += 1

    semantic_hits = await graph.traverse(
        query,
        top_k=search_limit,
        threshold=max(0.0, float(threshold)),
        owner_id=owner_scope,
        locus_id=locus_id,
    )
    for item in semantic_hits:
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        object_ = str(item.get("object") or "").strip()
        canonical_predicate = normalize_memory_predicate(predicate)
        active_fact_for_key = current_fact_index.get((_normalize_subject(subject), canonical_predicate))
        if active_fact_for_key is not None:
            active_subject, _, active_object = active_fact_for_key
            if (
                _normalize_subject(active_subject) == _normalize_subject(subject)
                and _normalize_object(active_object) != _normalize_object(object_)
            ):
                suppressed_conflicts += 1
                continue
        if target_predicates and canonical_predicate in target_predicates:
            active_fact = direct_fact_index.get(canonical_predicate)
            if active_fact is not None:
                active_subject, _, active_object = active_fact
                if (
                    _normalize_subject(active_subject) != _normalize_subject(subject)
                    or _normalize_object(active_object) != _normalize_object(object_)
                ):
                    suppressed_conflicts += 1
                    continue
        signature = f"{subject}|{canonical_predicate or predicate}|{object_}".lower()
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
                "predicate": canonical_predicate or predicate,
                "raw_predicate": predicate,
                "object": object_,
                "created_at": item.get("created_at"),
            },
        )
        source_counts["semantic_graph"] += 1

    if session_id:
        for subject, predicate, object_ in current_facts:
            canonical_predicate = normalize_memory_predicate(predicate)
            if target_predicates and canonical_predicate not in target_predicates:
                continue
            fact_text = f"{subject} {predicate} {object_}".strip()
            lexical = _lexical_overlap_score(query, fact_text)
            # Keep deterministic lane high-priority, but still rank by relevance.
            score = 0.62 + min(0.35, lexical * 0.35)
            if target_predicates and canonical_predicate in target_predicates:
                score = max(score, 0.97)
            signature = f"{subject}|{canonical_predicate}|{object_}".lower()
            _upsert_hit(
                merged,
                dedupe_key=signature,
                score=score,
                source="deterministic_fact",
                payload={
                    "kind": "fact",
                    "subject": subject,
                    "predicate": canonical_predicate,
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
            metadata_fact = str((item.get("metadata") or {}).get("fact") or "").strip()
            if not key and not anchor:
                continue
            if metadata_fact:
                lowered_parts = metadata_fact.split()
                if len(lowered_parts) >= 3:
                    subject = lowered_parts[0]
                    predicate = normalize_memory_predicate(lowered_parts[1])
                    object_ = " ".join(lowered_parts[2:])
                    active_fact_for_key = current_fact_index.get((_normalize_subject(subject), predicate))
                    if active_fact_for_key is not None:
                        _, _, active_object = active_fact_for_key
                        if _normalize_object(active_object) != _normalize_object(object_):
                            suppressed_conflicts += 1
                            continue
            vector_predicates = direct_fact_predicates(f"{key} {anchor}")
            if target_predicates:
                overlapping = vector_predicates & target_predicates
                if overlapping and any(predicate in direct_fact_index for predicate in overlapping):
                    suppressed_conflicts += 1
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

    # BM25 hybrid re-rank: vector retrieval provides the candidate pool;
    # BM25 corrects ordering for exact-phrase queries that embeddings can miss.
    # Deterministic facts (score ≥ 0.97) are pinned at the top and bypass re-ranking.
    all_candidates = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
    reranked = _hybrid_rerank(all_candidates, query)
    ranked = reranked[:safe_top_k]

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
            "suppressed_conflicts": suppressed_conflicts,
            "session_scoped": bool(session_id),
            "bm25_applied": len(all_candidates) >= 2,
        },
    }
