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
    if not r or len(results) <= 1:
        return results[:top_n]
    try:
        pairs = [[query, item.get(text_key, "")] for item in results]
        scores = r.compute_score(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:top_n]]
    except Exception:
        return results[:top_n]
