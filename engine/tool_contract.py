from __future__ import annotations

import json
import re
from typing import Any


TOOL_ARG_PRIORITY = [
    "query",
    "url",
    "path",
    "title",
    "filename",
    "language",
    "command",
    "prompt",
]
CONTENT_SIZE_SUMMARY_KEYS = ["code", "html", "content", "svg", "script", "css", "markup"]


def clip_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def format_tool_argument_value(key: str, value: Any) -> str:
    if isinstance(value, str):
        normalized = " ".join(value.split()).strip()
        if not normalized:
            return "empty"
        lowered_key = key.lower()
        if any(token in lowered_key for token in CONTENT_SIZE_SUMMARY_KEYS):
            return f"{len(normalized)} chars"
        return clip_text(normalized, 72)
    if isinstance(value, list):
        return f"{len(value)} item{'' if len(value) == 1 else 's'}"
    if isinstance(value, dict):
        return f"{len(value)} field{'' if len(value) == 1 else 's'}"
    return str(value)


def summarize_arguments(arguments: dict[str, Any] | None = None, *, max_items: int = 3) -> str:
    entries = list((arguments or {}).items())
    if not entries:
        return "starting..."
    sorted_entries = sorted(
        entries,
        key=lambda item: (
            TOOL_ARG_PRIORITY.index(item[0]) if item[0] in TOOL_ARG_PRIORITY else len(TOOL_ARG_PRIORITY),
            item[0],
        ),
    )
    displayed = [
        f"{key}: {format_tool_argument_value(key, value)}"
        for key, value in sorted_entries[:max_items]
    ]
    remaining = len(sorted_entries) - len(displayed)
    if remaining > 0:
        return f"{' · '.join(displayed)} · +{remaining} more"
    return " · ".join(displayed)


def canonical_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    max_items: int = 3,
) -> dict[str, Any]:
    args = dict(arguments or {})
    normalized_name = str(tool_name or "unknown")
    return {
        "tool": normalized_name,
        "tool_name": normalized_name,
        "arguments": args,
        "arguments_summary": summarize_arguments(args, max_items=max_items),
    }


def tool_call_signature(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    try:
        return json.dumps(
            {
                "tool": str(tool_name or ""),
                "arguments": arguments or {},
            },
            sort_keys=True,
            default=str,
        )
    except Exception:
        return f"{tool_name}:{repr(arguments)}"


def is_retry_sensitive_tool(tool_name: str) -> bool:
    return str(tool_name or "").lower() in {"web_search", "web_fetch"}


# ── AI-readable tool result signals ─────────────────────────────────────────
#
# These signals are appended to tool result content so the actor LLM knows
# what to do next without having to infer it from raw content alone.
#
# Signal format (always at end of content, one per line):
#   [READ_MORE: <url>]       — fetch this URL next to get more detail
#   [KEY_FACT: <text>]       — a fact worth hardening to memory
#   [NEXT_PROBE: <query>]    — a better follow-up search query
#   [TRUNCATED: <n> chars]   — content was cut; fetch URL for full version
#   [NO_RESULTS]             — search returned nothing useful
#   [AUTH_REQUIRED]          — page requires login; try a different source
#
# The actor prompt instructs the LLM to read these signals.


_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_PRICING_RE = re.compile(r"\b(pricing|plans?|cost|tiers?|subscribe)\b", re.IGNORECASE)
_LOGIN_RE = re.compile(r"\b(sign in|log in|login|register|subscribe to read|paywall|members only)\b", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "for", "in", "on", "at", "to",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "what", "which", "who", "how", "when", "where", "why", "this", "that",
    "with", "from", "by", "as", "it", "its", "not", "no", "so", "if", "then",
})


def _relevance_score(query: str, title: str, snippet: str, url: str) -> float:
    """
    Score a search result by keyword overlap with the query.

    Approach (inspired by OpenClaw link-understanding/detect.ts):
    - Tokenize query into meaningful keywords (strip stopwords)
    - Weighted hit: title hit = 3pts, snippet hit = 1pt, url-path hit = 2pts
    - Normalize by keyword count so short queries aren't disadvantaged
    - Returns 0.0–1.0 (higher = more relevant)
    """
    if not query:
        return 0.0

    def _tokens(text: str) -> set[str]:
        raw = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in raw if w not in _STOPWORDS and len(w) > 1}

    kw = _tokens(query)
    if not kw:
        return 0.0

    title_hits = len(kw & _tokens(title))
    snippet_hits = len(kw & _tokens(snippet))

    # URL path keywords (strip scheme + domain for path scoring)
    try:
        path = re.sub(r"^https?://[^/]+", "", url).lower()
        url_hits = len(kw & _tokens(path))
    except Exception:
        url_hits = 0

    raw_score = title_hits * 3 + snippet_hits * 1 + url_hits * 2
    max_score = len(kw) * 3  # best case: all kw in title
    return raw_score / max_score if max_score > 0 else 0.0


def _pick_best_url(results: list[dict], query: str) -> str:
    """
    Pick the most relevant URL from a list of search result dicts.

    Scores each result by keyword overlap with the query.
    Falls back to first URL with a snippet, then first URL overall.
    """
    if not results:
        return ""

    scored: list[tuple[float, str]] = []
    for r in results[:10]:
        u = str(r.get("url") or r.get("link") or "").strip()
        if not u.startswith("http"):
            continue
        title = str(r.get("title") or "").strip()
        snippet = str(r.get("snippet") or r.get("description") or "").strip()
        score = _relevance_score(query, title, snippet, u)
        scored.append((score, u))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_url = scored[0]
        if best_score > 0:
            return best_url

    # Fallback: first result with snippet, then first result
    for r in results[:5]:
        u = str(r.get("url") or r.get("link") or "").strip()
        snippet = str(r.get("snippet") or r.get("description") or "").strip()
        if u.startswith("http") and snippet:
            return u
    if results:
        fallback = str(results[0].get("url") or results[0].get("link") or "").strip()
        if fallback.startswith("http"):
            return fallback
    return ""


def generate_tool_signals(
    tool_name: str,
    content: str,
    *,
    arguments: dict[str, Any] | None = None,
    is_truncated: bool = False,
    truncated_chars: int = 0,
) -> list[str]:
    """
    Inspect a tool result and emit AI-readable signals.
    Returns a list of signal strings to append to the content.
    """
    signals: list[str] = []
    tool = str(tool_name or "").lower()
    text = str(content or "")

    if tool == "web_search":
        query = str((arguments or {}).get("query", ""))
        # Parse JSON search results to extract best candidate URL for follow-up fetch.
        try:
            payload = json.loads(text)
            results = payload.get("results") or payload.get("organic_results") or []
            if isinstance(results, list) and results:
                best_url = _pick_best_url(results, query)
                if best_url:
                    signals.append(f"[READ_MORE: {best_url}]")

                # Suggest a pricing-focused follow-up if the query looks commercial
                if _PRICING_RE.search(query) and best_url:
                    host = re.match(r"https?://[^/]+", best_url)
                    if host:
                        signals.append(f"[NEXT_PROBE: {host.group(0)}/pricing]")

                if not results:
                    signals.append("[NO_RESULTS]")
        except (json.JSONDecodeError, AttributeError):
            # Plain-text search result — extract first URL found
            urls = _URL_RE.findall(text)
            if urls:
                signals.append(f"[READ_MORE: {urls[0]}]")
            else:
                signals.append("[NO_RESULTS]")

    elif tool == "web_fetch":
        if _LOGIN_RE.search(text[:500]):
            signals.append("[AUTH_REQUIRED]")
            fetched_url = str((arguments or {}).get("url", ""))
            if fetched_url:
                signals.append(f"[NEXT_PROBE: site:{fetched_url.split('/')[2]} pricing OR plans]")
        if is_truncated and truncated_chars > 0:
            fetched_url = str((arguments or {}).get("url", ""))
            signals.append(f"[TRUNCATED: {truncated_chars} chars remaining]")
            if fetched_url:
                signals.append(f"[READ_MORE: {fetched_url}]")

    elif tool == "query_memory":
        if not text.strip() or text.strip() in {"[]", "{}", "No results", "none"}:
            signals.append("[NO_RESULTS]")
        else:
            # Try to extract a key fact
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                signals.append(f"[KEY_FACT: {lines[0][:120]}]")

    return signals


def enrich_tool_result_content(
    tool_name: str,
    content: str,
    *,
    arguments: dict[str, Any] | None = None,
    is_truncated: bool = False,
    truncated_chars: int = 0,
) -> str:
    """
    Append AI-readable signals to a tool result's content string.
    Safe — returns original content if signal generation fails.
    """
    try:
        signals = generate_tool_signals(
            tool_name,
            content,
            arguments=arguments,
            is_truncated=is_truncated,
            truncated_chars=truncated_chars,
        )
        if signals:
            return content.rstrip() + "\n\n" + "\n".join(signals)
    except Exception:
        pass
    return content


def diagnose_tool_failure(
    tool_name: str,
    content: str,
    arguments: dict[str, Any] | None = None,
) -> list[str]:
    """Classify a failed tool result and emit recovery hints.

    Failed tools used to drop into the actor's view as raw error text, leaving
    the model to guess the recovery path (often by inventing tools or repeating
    the same call). Surfacing structured `[SIGNAL: detail]` hints lets the
    actor's existing signal-reading rules drive an intel-gathering retry
    instead of a blind one.
    """
    text = str(content or "").strip()
    lowered = text.lower()
    hints: list[str] = []

    if (
        "argument error" in lowered
        or "unexpected keyword argument" in lowered
        or "missing 1 required" in lowered
        or "missing required argument" in lowered
        or "got multiple values for" in lowered
    ):
        hints.append(
            f"[ARG_ERROR: {tool_name} signature mismatch — call list_tools "
            f"to confirm the tool's parameters before retrying]"
        )
    elif "unknown tool" in lowered or "tool not found" in lowered or "no such tool" in lowered:
        hints.append(
            f"[UNKNOWN_TOOL: '{tool_name}' is not registered — call list_tools "
            f"to see what is actually available; do not invent tool names]"
        )
    elif "embedding request failed" in lowered or "failed to generate embedding" in lowered:
        hints.append(
            "[EMBED_DOWN: embedding service unreachable — vector indexing skipped; "
            "deterministic fact storage still works, retry later for semantic recall]"
        )
    elif (
        " 401" in text or " 403" in text
        or "unauthorized" in lowered or "permission denied" in lowered
        or ("auth" in lowered and ("required" in lowered or "fail" in lowered))
    ):
        hints.append("[AUTH_REQUIRED: try a different source or skip this fetch]")
    elif " 503" in text or " 502" in text or " 504" in text or "service unavailable" in lowered:
        hints.append("[PROVIDER_DOWN: upstream provider unavailable — wait or switch provider]")
    elif " 404" in text or "not found" in lowered:
        hints.append("[NOT_FOUND: target does not exist — verify path/url before retry]")
    elif "connection" in lowered and ("fail" in lowered or "refused" in lowered or "reset" in lowered):
        hints.append("[NETWORK: connection failed — retry once, then try a different source]")
    elif "invalid json" in lowered or "decode error" in lowered or "expecting value" in lowered:
        hints.append("[BAD_FORMAT: tool input malformed — re-encode arguments as valid JSON]")
    elif "timeout" in lowered or "timed out" in lowered:
        hints.append("[TIMEOUT: operation took too long — retry with narrower scope or different tool]")

    if not hints:
        hints.append(
            f"[FAILURE: {tool_name} did not succeed — gather more context "
            f"(list_tools, query_memory, or read related state) before retrying]"
        )

    return hints


def canonical_tool_result(
    item: dict[str, Any],
    *,
    preview_chars: int = 220,
) -> dict[str, Any]:
    tool_name = str(item.get("tool_name") or item.get("tool") or "unknown")
    success = bool(item.get("success"))
    content = str(item.get("content") or item.get("preview") or "")
    arguments = item.get("arguments")
    normalized: dict[str, Any] = {
        "tool": tool_name,
        "tool_name": tool_name,
        "success": success,
        "status": "ok" if success else "failed",
        "preview": clip_text(content.strip(), preview_chars),
    }
    if isinstance(arguments, dict) and arguments:
        normalized["arguments"] = dict(arguments)
        normalized["arguments_summary"] = summarize_arguments(arguments)
    return normalized


def format_tool_result_line(
    item: dict[str, Any],
    *,
    preview_chars: int = 220,
    include_status_label: bool = False,
) -> str:
    summary = canonical_tool_result(item, preview_chars=preview_chars)
    if include_status_label:
        return f"- {summary['tool_name']} [{summary['status']}]: {summary['preview']}"
    return (
        f"- {summary['tool_name']}: success={str(summary['success']).lower()} "
        f"preview={summary['preview']}"
    )


def summarize_tool_results(
    tool_results: list[dict[str, Any]],
    *,
    limit: int = 4,
    preview_chars: int = 220,
) -> list[dict[str, Any]]:
    items = list(tool_results or [])[-max(limit, 0) :]
    return [canonical_tool_result(item, preview_chars=preview_chars) for item in items]