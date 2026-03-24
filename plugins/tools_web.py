"""
Web Search & Fetch — Production Implementations
------------------------------------------------
Replaces the fragile DuckDuckGo scrape in tools.py.

Search backends (priority order, auto-fallback):
  1. SearXNG       — self-hosted, no key, no limits, multi-engine aggregator
  2. Brave Search  — free tier 2000/month, independent index, clean JSON API
  3. Tavily        — agent-optimized, includes page content, free tier
  4. Exa           — neural/semantic search, free tier
  5. Groq compound — uses compound-beta model with built-in web search

Web fetch backends:
  1. Jina Reader   — r.jina.ai prefix, returns clean markdown, free, no key
  2. httpx fallback — raw HTML stripped (original approach)

Environment variables:
  SEARXNG_URL          — your SearXNG instance  e.g. http://localhost:8080
                         (if not set, SearXNG backend is skipped)
  BRAVE_SEARCH_KEY     — api.search.brave.com key (free tier: 2000 req/month)
                         https://api.search.brave.com/app/keys
  TAVILY_API_KEY       — tavily.com key (free tier: 1000 req/month)
                         https://app.tavily.com
  EXA_API_KEY          — exa.ai key (free tier available)
                         https://dashboard.exa.ai
  GROQ_API_KEY         — groq.com key (free tier available)
                         https://console.groq.com

Running SearXNG locally (recommended — Docker one-liner):
  docker run -d -p 8080:8080 \\
    -e SEARXNG_SECRET_KEY=$(openssl rand -hex 32) \\
    --name searxng searxng/searxng:latest

  Then set: export SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import ipaddress
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Optional

import httpx
from groq import AsyncGroq

from plugins.tool_registry import Tool, ToolRegistry


# ─── Config ───────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


SEARXNG_URL  = os.getenv("SEARXNG_URL", "").rstrip("/")
BRAVE_KEY    = os.getenv("BRAVE_SEARCH_KEY", "")
TAVILY_KEY   = os.getenv("TAVILY_API_KEY", "")
EXA_KEY      = os.getenv("EXA_API_KEY", "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")
EXA_SEARCH_TYPE = (os.getenv("EXA_SEARCH_TYPE", "auto") or "auto").strip().lower()
EXA_TEXT_MAX_CHARACTERS = max(500, _env_int("EXA_TEXT_MAX_CHARACTERS", 20000))

HTTP_TIMEOUT = 20.0
MAX_WEB_FETCH_CHARS = 50000
MIN_WEB_FETCH_CHARS = 500
WEB_FETCH_ALLOW_PRIVATE = (os.getenv("WEB_FETCH_ALLOW_PRIVATE", "false") or "false").strip().lower() == "true"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/120 Safari/537.36"
    )
}

# ── Groq compound-beta model name ─────────────────────────────────────────────
# BUG FIX: was "groq/compound" — that is not a valid model name and causes 413.
# The correct model strings are "compound-beta" and "compound-beta-mini".
GROQ_COMPOUND_MODEL = "compound-beta"
TRACKING_PARAMS_TO_REMOVE = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_eid",
    "msclkid",
    "twclid",
}
MAX_UNIQUE_DOMAINS_IN_SUMMARY = 8
# 1 bonus point per ~120 chars rewards richer snippets without over-weighting verbosity.
SNIPPET_SIGNAL_CHARS = 120
SNIPPET_SIGNAL_MAX_BONUS = 3
# Over-fetch to create a larger pool for dedupe/curation, then trim to requested size.
FETCH_MULTIPLIER = 2
FETCH_BONUS = 4
MAX_FETCH_LIMIT = 30
# Tool contract currently caps requested web results to keep payloads compact.
MAX_REQUESTED_RESULTS = 20
TITLE_SIGNAL_WEIGHT = 3
SNIPPET_SIGNAL_WEIGHT = 3
URL_SIGNAL_WEIGHT = 2
SOURCE_SIGNAL_WEIGHT = 1


def _json_error_payload(tool: str, error: str, **extra) -> str:
    payload = {
        "type": f"{tool}_error",
        "tool": tool,
        "error": str(error),
    }
    payload.update(extra)
    return json.dumps(payload)


def _clean_text(value: str) -> str:
    """Normalize text fields for cleaner context curation."""
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def _normalize_url(url: str) -> str:
    """Normalize URL for dedupe (remove tracking params, fragments, extra slashes)."""
    url = str(url or "").strip()
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except Exception:
        return url

    query_params = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        lowercase_key = k.lower()
        if lowercase_key.startswith("utm_") or lowercase_key in TRACKING_PARAMS_TO_REMOVE:
            continue
        query_params.append((k, v))

    clean_path = re.sub(r"/{2,}", "/", parts.path or "")
    clean_path = clean_path.rstrip("/") or "/"
    clean_query = urlencode(query_params, doseq=True)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), clean_path, clean_query, ""))


def _extract_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").strip().lower()
    except Exception:
        return ""


def _is_private_or_local_host(host: str) -> bool:
    host = (host or "").strip().lower()
    if not host:
        return True
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        ip = ipaddress.ip_address(host)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        if host.endswith(".local") or host.endswith(".internal") or host.endswith(".localhost"):
            return True
        return False


def _validate_fetch_url(url: str) -> tuple[bool, str, str, str]:
    raw = str(url or "").strip()
    if not raw:
        return False, "empty URL", "", ""

    normalized = _normalize_url(raw)
    try:
        parts = urlsplit(normalized)
    except Exception:
        return False, f"invalid URL '{raw}'", "", ""

    if parts.scheme not in {"http", "https"}:
        return False, f"invalid URL '{raw}' — only http/https are allowed", "", ""

    host = _extract_host(normalized)
    if not host:
        return False, f"invalid URL '{raw}' — missing hostname", "", ""

    if not WEB_FETCH_ALLOW_PRIVATE and _is_private_or_local_host(host):
        return False, f"blocked URL host '{host}' (private/local not allowed)", normalized, host

    return True, "", normalized, host


def _extract_readable_text_from_html(html: str) -> str:
    cleaned = re.sub(
        r"<(script|style|noscript|nav|footer|header|aside|form|svg|canvas)[^>]*>.*?</\1>",
        "",
        html or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"</(p|div|section|article|li|h[1-6]|br)>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _generate_dedupe_key(title: str, snippet: str, url: str) -> str:
    normalized_url = _normalize_url(url)
    if normalized_url:
        return normalized_url
    hash_input = f"title:{title.strip().lower()}|snippet:{snippet.strip().lower()}"
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def _estimate_unique_candidates(raw_results: list[dict]) -> int:
    """Fast estimate used in auto-backend loop to avoid repeated full curation."""
    seen: set[str] = set()
    for item in raw_results or []:
        title = str(item.get("title", ""))
        snippet = str(item.get("snippet", ""))
        url = str(item.get("url", ""))
        dedupe_key = _generate_dedupe_key(title, snippet, url)
        if dedupe_key:
            seen.add(dedupe_key)
    return len(seen)


def _curate_results(raw_results: list[dict], num_results: int) -> tuple[list[dict], dict]:
    """Clean, deduplicate, and score results for better LLM context quality."""
    cleaned: list[dict] = []
    seen: set[str] = set()
    domain_counts: dict[str, int] = {}

    for item in raw_results or []:
        title = _clean_text(item.get("title", ""))
        snippet = _clean_text(item.get("snippet", ""))
        url = _normalize_url(item.get("url", ""))
        source = _clean_text(item.get("source", ""))
        if not (title or snippet or url):
            continue

        dedupe_key = _generate_dedupe_key(title, snippet, url)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if url:
            try:
                host = urlsplit(url).netloc
                if host:
                    domain_counts[host] = domain_counts.get(host, 0) + 1
            except Exception:
                pass

        signal_score = 0
        if title:
            signal_score += TITLE_SIGNAL_WEIGHT
        if snippet:
            signal_score += SNIPPET_SIGNAL_WEIGHT
            # Award up to +3 signal for richer snippets (1 point per ~120 chars).
            signal_score += min(len(snippet) // SNIPPET_SIGNAL_CHARS, SNIPPET_SIGNAL_MAX_BONUS)
        if url:
            signal_score += URL_SIGNAL_WEIGHT
        if source and not source.endswith("-error"):
            signal_score += SOURCE_SIGNAL_WEIGHT

        curated = {"title": title, "url": url, "snippet": snippet}
        if source:
            curated["source"] = source
        if item.get("published"):
            curated["published"] = item.get("published")
        if "score" in item:
            curated["score"] = item.get("score")
        curated["_signal_score"] = signal_score
        cleaned.append(curated)

    cleaned.sort(key=lambda r: r.get("_signal_score", 0), reverse=True)
    final_results = []
    for entry in cleaned[:num_results]:
        item = dict(entry)
        item.pop("_signal_score", None)
        final_results.append(item)

    return final_results, {
        "requested_results": int(num_results),
        "raw_results_considered": len(raw_results or []),
        "curated_results": len(final_results),
        "unique_domains": sorted(domain_counts, key=lambda x: domain_counts.get(x, 0), reverse=True)[:MAX_UNIQUE_DOMAINS_IN_SUMMARY],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_executed_tool_results(tool) -> list[dict]:
    """
    Parse search results from a Groq executed tool object.

    BUG FIX: tool.search_results is an ExecutedToolSearchResults *object*, not a
    plain dict.  Calling .get("results", []) on it raises:
        AttributeError: 'ExecutedToolSearchResults' object has no attribute 'get'

    We handle the object form (attribute access) with a fallback to dict form
    so this works regardless of future SDK changes.
    """
    sr = getattr(tool, "search_results", None)
    if sr is None:
        return []

    # Case 1: plain dict (future-proof / older SDK versions)
    if isinstance(sr, dict):
        raw = sr.get("results", [])
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            }
            for r in raw
        ]

    # Case 2: ExecutedToolSearchResults object — access .results attribute
    raw = getattr(sr, "results", None) or []
    results = []
    for r in raw:
        if isinstance(r, dict):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            })
        else:
            # SDK result object with attributes
            results.append({
                "title":   getattr(r, "title", ""),
                "url":     getattr(r, "url", ""),
                "snippet": getattr(r, "content", getattr(r, "snippet", "")),
            })
    return results


async def _search_groq(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Groq compound-beta built-in web search.
    The compound-beta model has native web-search capability baked in.
    """
    if not GROQ_KEY:
        return None
    try:
        client = AsyncGroq(api_key=GROQ_KEY)

        # SANITIZATION: Force strict search mode via system prompt
        print(f"[INTERNAL DETECTION] Sanitizing Groq search input for: {query[:50]}...")
        messages = [
            {"role": "system", "content": "You are a robotic search utility. 1. Perform a search. 2. Output ONLY the tool results. 3. If tools fail, output a list of URLs and titles in JSON format: [{\"title\":\"...\",\"url\":\"...\",\"snippet\":\"...\"}]. Do not provide prose."},
            {"role": "user", "content": f"Search for: {query[:500]}"}
        ]

        response = await client.chat.completions.create(
            model=GROQ_COMPOUND_MODEL,
            messages=messages,
            temperature=0.0, # minimal creativity
        )

        msg = response.choices[0].message
        executed = getattr(msg, "executed_tools", None) or []

        results = []
        for tool in executed:
            parsed = _parse_executed_tool_results(tool)
            results.extend(parsed)
            # Remove the 'break' to capture results from multiple tools if present

        if results:
            print(f"[INTERNAL DETECTION] Groq structured extraction success. Extracted {len(results)} items.")
            return [
                {**r, "source": "groq"}
                for r in results[:num_results]
            ]
        
        # FAILSAFE: If no tools, try to parse JSON or URLs from msg.content
        content = getattr(msg, "content", "") or ""
        if not results and content:
            print(f"[INTERNAL DETECTION] No tool results. Attempting fallback extraction from prose ({len(content)} chars)...")
            # 1. Try to find a JSON block
            json_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if json_match:
                try:
                    results = json.loads(json_match.group(0))
                    print("[INTERNAL DETECTION] Failsafe: Successfully parsed JSON from prose.")
                except:
                    pass
            
            # 2. Try to find Markdown links [title](url)
            if not results:
                links = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", content)
                for title, url in links:
                    results.append({"title": title, "url": url, "snippet": ""})
                if results:
                    print(f"[INTERNAL DETECTION] Failsafe: Extracted {len(results)} links from prose.")

        if results:
            print(f"[INTERNAL DETECTION] Groq structured/failsafe extraction success. Items: {len(results)}")
            return [
                {**r, "source": "groq"}
                for r in results[:num_results]
            ]
        
        print(f"[INTERNAL DETECTION] Groq returned unusable content. Discarding for sanitation.")
        return None

    except Exception as e:
        msg = f"Groq Search failed: {e}"
        print(f"[web_search] {msg}")
        # Return error marker so caller knows Groq was tried but failed
        return [{"title": "Error", "url": "", "snippet": msg, "source": "groq-error"}]


async def _search_searxng(query: str, num_results: int) -> Optional[list[dict]]:
    """
    SearXNG JSON API.
    Aggregates Google, Bing, DuckDuckGo, Brave, Wikipedia, etc. simultaneously.
    Run locally — completely private, no external API key.
    """
    if not SEARXNG_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q":         query,
                    "format":    "json",
                    "engines":   "google,bing,duckduckgo,brave,wikipedia",
                    "language":  "en",
                    "safesearch": "0",
                },
                headers=HTTP_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("results", [])[:num_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", ""),
                "source":  "searxng",
            })
        return results if results else None

    except Exception as e:
        msg = f"SearXNG failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "searxng-error"}]


async def _search_brave(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Brave Search API.
    Independent index (not Bing-resold). Free tier: 2000 req/month.
    Get key: https://api.search.brave.com/app/keys
    """
    if not BRAVE_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={
                    "q":     query,
                    "count": min(num_results, 20),
                },
                headers={
                    "Accept":               "application/json",
                    "Accept-Encoding":      "gzip",
                    "X-Subscription-Token": BRAVE_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        if data.get("query", {}).get("answer"):
            results.append({
                "title":   "Featured Answer",
                "url":     "",
                "snippet": data["query"]["answer"],
                "source":  "brave-answer",
            })

        for r in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("description", ""),
                "source":  "brave",
            })

        return results if results else None

    except Exception as e:
        msg = f"Brave failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "brave-error"}]


async def _search_tavily(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Tavily Search API.
    Purpose-built for AI agents — returns results with full page content.
    Free tier: 1000 req/month. https://app.tavily.com
    """
    if not TAVILY_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":              TAVILY_KEY,
                    "query":                query,
                    "search_depth":         "basic",
                    "max_results":          min(num_results, 10),
                    "include_answer":       True,
                    "include_raw_content":  False,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        if data.get("answer"):
            results.append({
                "title":   "Tavily Answer",
                "url":     "",
                "snippet": data["answer"],
                "source":  "tavily-answer",
            })
        for r in data.get("results", [])[:num_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", ""),
                "score":   r.get("score", 0),
                "source":  "tavily",
            })

        return results if results else None

    except Exception as e:
        msg = f"Tavily failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "tavily-error"}]


async def _search_exa(query: str, num_results: int) -> Optional[list[dict]]:
    """
    Exa (formerly Metaphor) — neural/semantic search.
    Free tier available. https://dashboard.exa.ai
    """
    if not EXA_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "type": EXA_SEARCH_TYPE,
                    "num_results": min(num_results, 10),
                    "contents": {
                        "text": {
                            "max_characters": EXA_TEXT_MAX_CHARACTERS,
                        }
                    },
                },
                headers={
                    "x-api-key":    EXA_KEY,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("results", [])[:num_results]:
            text_body = r.get("text")
            if isinstance(text_body, str):
                text_snippet = text_body[:450]
            else:
                text_snippet = ""

            highlights = r.get("highlights") or []
            highlight_snippet = ""
            if isinstance(highlights, list) and highlights:
                first_highlight = highlights[0]
                if isinstance(first_highlight, str):
                    highlight_snippet = first_highlight

            snippet = (
                r.get("summary")
                or highlight_snippet
                or text_snippet
                or ""
            )

            published = r.get("publishedDate") or r.get("published_date") or ""
            results.append({
                "title":     r.get("title", ""),
                "url":       r.get("url", ""),
                "snippet":   snippet,
                "published": published,
                "source":    "exa",
            })
        return results if results else None

    except Exception as e:
        msg = f"Exa failed: {e}"
        print(f"[web_search] {msg}")
        return [{"title": "Error", "url": "", "snippet": msg, "source": "exa-error"}]


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED SEARCH TOOL  —  auto-fallback chain
# ══════════════════════════════════════════════════════════════════════════════

async def _web_search(query: str, num_results: int = 8, backend: str = "auto", search_engine: str = None) -> str:
    """
    Web search with automatic fallback chain.
    backend: "auto" | "searxng" | "brave" | "tavily" | "exa" | "groq"
    
    Auto priority: SearXNG → Brave → Tavily → Exa → Groq
    (SearXNG first because it's local and unlimited; Groq last to save tokens)
    """
    from plugins.tools import _search_duckduckgo
    
    if search_engine and backend == "auto":
        backend = search_engine.lower()
        
    print(f"[INTERNAL DETECTION] Starting search for: '{query}' (backend: {backend})")
    num_results = max(1, min(int(num_results), MAX_REQUESTED_RESULTS))
    multiplied_fetch = num_results * FETCH_MULTIPLIER
    buffered_fetch = num_results + FETCH_BONUS
    fetch_target = min(max(multiplied_fetch, buffered_fetch), MAX_FETCH_LIMIT)
    collected_results: list[dict] = []
    used_backends: list[str] = []

    if backend == "auto":
        backends = [
            ("searxng", _search_searxng),
            ("brave",   _search_brave),
            ("tavily",  _search_tavily),
            ("exa",     _search_exa),
            ("groq",    _search_groq),
        ]
        for name, fn in backends:
            print(f"[INTERNAL DETECTION] Trying search backend: {name}")
            candidate = await fn(query, fetch_target)
            # Skip error-only results and try next backend
            if candidate and not all(r.get("source", "").endswith("-error") for r in candidate):
                collected_results.extend(candidate)
                used_backends.append(name)
                print(f"[INTERNAL DETECTION] Search success via {name}. Found {len(candidate)} results.")
                if _estimate_unique_candidates(collected_results) >= num_results:
                    break
            if candidate:
                print(f"[INTERNAL DETECTION] Backend {name} returned error or no results.")
    else:
        fn_map = {
            "groq":       _search_groq,
            "searxng":    _search_searxng,
            "brave":      _search_brave,
            "tavily":     _search_tavily,
            "exa":        _search_exa,
            "duckduckgo": _search_duckduckgo,
        }
        if backend not in fn_map:
            return _json_error_payload(
                "web_search",
                f"unknown backend '{backend}'",
                query=query,
                valid_backends=["auto", "groq", "searxng", "brave", "tavily", "exa", "duckduckgo"],
            )
        single_backend_results = await fn_map[backend](query, fetch_target)
        if single_backend_results:
            collected_results.extend(single_backend_results)
            used_backends.append(backend)

    if not collected_results:
        if backend != "auto":
            return _json_error_payload(
                "web_search",
                f"backend '{backend}' returned no results",
                query=query,
                backend=backend,
            )
        configured = []
        if SEARXNG_URL: configured.append("searxng")
        if BRAVE_KEY:   configured.append("brave")
        if TAVILY_KEY:  configured.append("tavily")
        if EXA_KEY:     configured.append("exa")
        if GROQ_KEY:    configured.append("groq")

        if not configured:
            return _json_error_payload(
                "web_search",
                "no backends configured",
                query=query,
                configured_backends=configured,
                help=(
                    "Set one of: SEARXNG_URL, BRAVE_SEARCH_KEY, TAVILY_API_KEY, "
                    "EXA_API_KEY, GROQ_API_KEY"
                ),
            )
        return _json_error_payload(
            "web_search",
            f"all configured backends ({', '.join(configured)}) returned no results",
            query=query,
            configured_backends=configured,
        )

    curated_results, context_summary = _curate_results(collected_results, num_results)
    if not curated_results:
        return _json_error_payload(
            "web_search",
            "all configured backends returned unusable results",
            query=query,
            backend=backend,
        )

    backend_name = "+".join(used_backends) if used_backends else "none"

    # Format output as JSON for frontend premium cards and cleaner LLM context
    return json.dumps({
        "type": "web_search_results",
        "query": query,
        "results": curated_results,
        "backend": backend_name,
        "engine": backend_name,
        "context_summary": context_summary,
    })


WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Search the web for current information. "
        "Tries SearXNG (self-hosted) → Brave → Tavily → Exa → Groq in order. "
        "Returns titles, URLs, and content snippets. "
        "Use for recent news, current facts, or anything beyond training data."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific for better results.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 8, max 20)",
                "default": 8,
            },
            "backend": {
                "type": "string",
                "description": "Force a specific backend: auto (default), searxng, brave, tavily, exa, groq",
                "default": "auto",
            },
        },
        "required": ["query"],
    },
    handler=_web_search,
    tags=["web", "search"],
    response_format="json",
)


# ══════════════════════════════════════════════════════════════════════════════
#  WEB FETCH  —  Jina Reader + httpx fallback
# ══════════════════════════════════════════════════════════════════════════════

async def _fetch_groq(url: str, max_chars: int) -> Optional[str]:
    """
    Groq compound-beta built-in URL visit.
    BUG FIX: was "groq/compound" → correct name is "compound-beta"
    """
    if not GROQ_KEY:
        return None
    try:
        client = AsyncGroq(api_key=GROQ_KEY)
        response = await client.chat.completions.create(
            model=GROQ_COMPOUND_MODEL,
            messages=[{"role": "user", "content": f"Visit and extract the main content of: {url}"}],
        )

        msg = response.choices[0].message
        executed = getattr(msg, "executed_tools", None) or []

        for tool in executed:
            tool_type = getattr(tool, "type", "")
            if "visit" in tool_type or "fetch" in tool_type or "browse" in tool_type:
                content = getattr(tool, "output", "") or getattr(tool, "content", "")
                if content:
                    return str(content)[:max_chars]

        # If no visit tool found, the model's text response may contain the content
        text = getattr(msg, "content", "") or ""
        if text and len(text) > 200:
            return text[:max_chars]

        return None
    except Exception as e:
        print(f"[web_fetch] Groq Visit failed: {e}")
        return None


async def _web_fetch(url: str, max_chars: int = 8000, use_jina: bool = True) -> str:
    """
    Fetch clean readable content from a URL.

    Primary:  Jina Reader (r.jina.ai prefix) — free, no key, clean markdown
    Secondary: Groq compound-beta URL visit (if key configured)
    Fallback: httpx + regex HTML strip
    """
    print(f"[INTERNAL DETECTION] Starting fetch for URL: {url} (use_jina: {use_jina})")
    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 8000
    max_chars = max(MIN_WEB_FETCH_CHARS, min(max_chars, MAX_WEB_FETCH_CHARS))
    if isinstance(use_jina, str):
        use_jina = use_jina.lower() != "false"

    ok, error_message, normalized_url, host = _validate_fetch_url(url)
    if not ok:
        return _json_error_payload(
            "web_fetch",
            error_message,
            url=str(url or ""),
        )

    def _format_result(
        text_content: str,
        *,
        backend: str,
        final_url: Optional[str] = None,
        content_type: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> str:
        text_content = str(text_content)
        is_truncated = len(text_content) > max_chars
        final_content = text_content[:max_chars]
        if is_truncated:
            final_content += f"\n\n[truncated — {len(text_content) - max_chars} more chars]"
        return json.dumps({
            "type": "web_fetch_result",
            "url": normalized_url,
            "final_url": final_url or normalized_url,
            "host": host,
            "backend": backend,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content": final_content,
            "truncated": is_truncated,
            "total_length": len(text_content),
            "title": normalized_url,
            "content_type": content_type or "",
            "status_code": status_code if status_code is not None else 200,
        })

    # ── Jina Reader (Primary — free, best quality) ───────────────────────────
    if use_jina:
        jina_url = f"https://r.jina.ai/{normalized_url}"
        try:
            async with httpx.AsyncClient(
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
                headers={
                    **HTTP_HEADERS,
                    "X-Return-Format": "markdown",
                    "X-Timeout":       "15",
                },
            ) as client:
                resp = await client.get(jina_url)
                resp.raise_for_status()
                text = resp.text.strip()

            if text and len(text) > 100:
                return _format_result(
                    text,
                    backend="jina",
                    final_url=normalized_url,
                    content_type=resp.headers.get("content-type"),
                    status_code=resp.status_code,
                )
        except Exception as e:
            print(f"[web_fetch] Jina Reader failed ({e}), falling back")

    # ── Groq Visit (Secondary) ────────────────────────────────────────────────
    if GROQ_KEY:
        content = await _fetch_groq(normalized_url, max_chars)
        if content:
            return _format_result(content, backend="groq", final_url=normalized_url)

    # ── httpx + HTML strip (Fallback) ─────────────────────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers=HTTP_HEADERS,
        ) as client:
            resp = await client.get(normalized_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "json" in content_type:
                return _format_result(
                    json.dumps(resp.json(), indent=2),
                    backend="httpx-json",
                    final_url=str(resp.url),
                    content_type=content_type,
                    status_code=resp.status_code,
                )

            html = resp.text

        text = _extract_readable_text_from_html(html)
        return _format_result(
            text,
            backend="httpx-html",
            final_url=str(resp.url),
            content_type=content_type,
            status_code=resp.status_code,
        )

    except httpx.HTTPStatusError as e:
        return _json_error_payload(
            "web_fetch",
            f"HTTP {e.response.status_code}",
            url=normalized_url,
            status_code=e.response.status_code,
        )
    except Exception as e:
        return _json_error_payload(
            "web_fetch",
            str(e),
            url=normalized_url,
        )


WEB_FETCH_TOOL = Tool(
    name="web_fetch",
    description=(
        "Fetch the full readable content of a URL as clean text or markdown. "
        "Primary: Jina Reader (r.jina.ai) — free, no key, returns clean markdown. "
        "Secondary: Groq compound-beta URL visit (if key set). "
        "Fallback: direct HTTP + HTML stripping. "
        "Use when you need to read an entire page referenced in search results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (must start with http:// or https://)",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 8000)",
                "default": 8000,
            },
            "use_jina": {
                "type": "boolean",
                "description": "Use Jina Reader for clean markdown (default true)",
                "default": True,
            },
        },
        "required": ["url"],
    },
    handler=_web_fetch,
    tags=["web", "fetch"],
    response_format="json",
)


# ══════════════════════════════════════════════════════════════════════════════
#  Registration helper
# ══════════════════════════════════════════════════════════════════════════════

def register_web_tools(registry: ToolRegistry) -> None:
    """
    Register the production web search and fetch tools.
    Call AFTER register_all_tools() to override the default DDG implementations.
    """
    registry.register(WEB_SEARCH_TOOL)
    registry.register(WEB_FETCH_TOOL)
    print("[web_tools] Registered: web_search, web_fetch")
    _print_backend_status()


def _print_backend_status():
    """Print which backends are configured on startup."""
    print("[web_tools] Backend status:")
    print(f"  Groq     : {'✓  key configured (compound-beta)' if GROQ_KEY else '✗  not configured (set GROQ_API_KEY)'}")
    print(f"  SearXNG  : {'✓ ' + SEARXNG_URL if SEARXNG_URL else '✗  not configured (set SEARXNG_URL)'}")
    print(f"  Brave    : {'✓  key configured' if BRAVE_KEY else '✗  not configured (set BRAVE_SEARCH_KEY)'}")
    print(f"  Tavily   : {'✓  key configured' if TAVILY_KEY else '✗  not configured (set TAVILY_API_KEY)'}")
    print(f"  Exa      : {'✓  key configured' if EXA_KEY else '✗  not configured (set EXA_API_KEY)'}")
    print(f"  Jina     : ✓  always available (no key required)")
