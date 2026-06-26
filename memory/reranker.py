"""
BGE Reranker — post-retrieval cross-encoder scoring.
Lazy-loaded. Falls back to original order if unavailable.
"""
from __future__ import annotations

_reranker = None


def _get():
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            _reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
        except ImportError:
            _reranker = False
    return _reranker


def rerank(query: str, results: list[dict], text_key: str = "content", top_n: int = 5) -> list[dict]:
    """
    Rerank results by cross-encoder relevance score.
    Falls back gracefully if model is unavailable.
    """
    r = _get()
    if len(results) <= 1:
        return results[:top_n]
    if not r:
        return _lexical_rerank(query, results, text_key=text_key, top_n=top_n)
    try:
        pairs = [[query, item.get(text_key, "")] for item in results]
        scores = r.compute_score(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:top_n]]
    except Exception:
        return _lexical_rerank(query, results, text_key=text_key, top_n=top_n)


def _lexical_rerank(query: str, results: list[dict], text_key: str = "content", top_n: int = 5) -> list[dict]:
    terms = _terms(query)
    if not terms:
        return results[:top_n]
    ranked = []
    for index, item in enumerate(results):
        text = " ".join([
            str(item.get(text_key) or ""),
            str(item.get("source") or ""),
            str(item.get("tool_name") or ""),
            str(item.get("filename") or ""),
        ]).lower()
        exact = sum(1 for term in terms if f" {term} " in f" {text} ")
        partial = sum(1 for term in terms if term in text)
        base_score = float(item.get("score") or 0.0)
        score = (exact * 3.0) + partial + base_score
        ranked.append((-score, index, item))
    return [item for _score, _index, item in sorted(ranked)[:top_n]]


def _terms(query: str) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "when", "where", "how"}
    terms = []
    for raw in str(query or "").lower().replace("-", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        if len(token) < 2 or token in stop:
            continue
        terms.append(token)
    return list(dict.fromkeys(terms))[:16]
