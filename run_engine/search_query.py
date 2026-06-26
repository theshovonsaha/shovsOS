from __future__ import annotations

import re


_WORKFLOW_MARKERS = (
    "web fetch",
    "fetch all",
    "fetch the",
    "fetch those",
    "capture",
    "analyze",
    "analyse",
    "report",
    "tldr",
    "tl;dr",
    "summary table",
    "one by one",
    "separately",
    "for each",
)
_TRAILING_WORKFLOW_RE = re.compile(
    r"\s+(?:"
    r"(?:then|and then|after that)\b.*|"
    r"(?:web\s+search|search)\s+(?:those|these|each|all|the\s+\d+)\b.*|"
    r"(?:web\s+fetch|fetch)\s+(?:all|those|these|the|\d+|urls?)\b.*|"
    r"(?:capture|analy[sz]e|write|draft|produce|create|summari[sz]e|make)\b.*"
    r")$",
    re.IGNORECASE,
)
_LEADING_SEARCH_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:web\s+search|search\s+for|search|look\s+up|google)\s+",
    re.IGNORECASE,
)
_FILLER_RE = re.compile(
    r"\b(?:please|for\s+me|right\s+now|as\s+a\s+table|in\s+a\s+table)\b",
    re.IGNORECASE,
)


def is_workflow_like_search_query(query: str) -> bool:
    text = re.sub(r"\s+", " ", str(query or "")).strip().lower()
    if not text:
        return False
    marker_count = sum(1 for marker in _WORKFLOW_MARKERS if marker in text)
    if marker_count >= 1 and len(text) >= 70:
        return True
    if marker_count >= 2:
        return True
    if len(text) > 140:
        return True
    return bool(re.search(r"\bweb\s+search\b.*\bweb\s+fetch\b", text))


def compile_web_search_query(query: str) -> str:
    """Compile a user/workflow sentence into the probe a search engine needs.

    The full user text may contain plan steps, quotas, output-format requests,
    and tool instructions. Those belong in the run ledger, not in
    ``web_search.query``. This helper is intentionally conservative for short
    ordinary searches and only cuts aggressively when workflow markers appear.
    """

    original = re.sub(r"\s+", " ", str(query or "")).strip()
    if not original:
        return ""

    compiled = original
    should_compact = is_workflow_like_search_query(compiled)
    if should_compact:
        compiled = _TRAILING_WORKFLOW_RE.sub("", compiled).strip()
        compiled = _LEADING_SEARCH_RE.sub("", compiled).strip()
        compiled = _FILLER_RE.sub(" ", compiled).strip()
    else:
        compiled = re.sub(r"^\s*(?:please\s+)?(?:web\s+search|search\s+for|search)\s+", "", compiled, flags=re.IGNORECASE).strip()

    compiled = re.sub(r"\s+", " ", compiled).strip(" .,:;")
    if not compiled:
        return original[:120].strip()
    return compiled[:180].rstrip(" .,:;")
