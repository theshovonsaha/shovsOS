"""
Built-in Tools
--------------
Real implementations for:
  - web_search       canonical web search from tools_web.py
  - web_fetch        canonical web fetch from tools_web.py
  - image_search     DuckDuckGo images (no API key required)
    - bash             sandboxed Docker command execution
  - file_create      write file inside sandbox dir
  - file_view        read file / list directory
  - file_str_replace in-place string replacement
  - weather_fetch    open-meteo.com (free, no API key)
  - places_search    Google Places API  (needs GOOGLE_PLACES_API_KEY env var)
  - places_map       static HTML map via Google Maps Embed

Registration
------------
  from agent.tools import register_all_tools
  register_all_tools(tool_registry)          # in main.py after tool_registry is created

Environment variables (only places_search / places_map need them):
  GOOGLE_PLACES_API_KEY   — Google Cloud project with Places API enabled
  SANDBOX_DIR             — root for file tools  (default: ./agent_sandbox)
  BASH_TIMEOUT            — seconds before bash kill (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlsplit

import httpx

from engine.direct_fact_policy import normalize_memory_predicate
from plugins.tool_registry import Tool, ToolRegistry
from plugins.tools_web import (
    WEB_FETCH_TOOL as CANONICAL_WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL as CANONICAL_WEB_SEARCH_TOOL,
    _web_fetch as _canonical_web_fetch,
    _web_search as _canonical_web_search,
)
from config.logger import log
from memory.task_tracker import get_session_task_tracker
from orchestration.session_manager import SessionManager

if TYPE_CHECKING:
    from orchestration.agent_manager import AgentManager


# ─── Config ───────────────────────────────────────────────────────────────────

GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
SANDBOX_DIR       = Path(os.getenv("SANDBOX_DIR", "./agent_sandbox")).resolve()
BASH_TIMEOUT      = int(os.getenv("BASH_TIMEOUT", "30"))
HTTP_TIMEOUT      = 20.0

SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

_agent_manager: Optional[AgentManager] = None  # Injected during registration

WEB_SEARCH_TOOL = CANONICAL_WEB_SEARCH_TOOL
WEB_FETCH_TOOL = CANONICAL_WEB_FETCH_TOOL
_web_search = _canonical_web_search
_web_fetch = _canonical_web_fetch


def _normalize_search_results(results: list[dict], max_results: int = 8) -> list[dict]:
    """Compatibility helper kept for tests and fallback search formatting."""
    normalized: list[dict] = []
    seen_urls: set[str] = set()

    for item in results or []:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        snippet = re.sub(r"\s+", " ", str(item.get("snippet", "")).strip()).strip()
        source = str(item.get("source", "")).strip()

        if not title and not snippet:
            continue
        if not snippet:
            snippet = title or "Untitled"
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."

        url_key = url.lower().rstrip("/")
        if url_key and url_key in seen_urls:
            continue
        if url_key:
            seen_urls.add(url_key)

        entry = {"title": title or "Untitled", "url": url, "snippet": snippet}
        if source:
            entry["source"] = source
        normalized.append(entry)
        if len(normalized) >= max(1, max_results):
            break

    return normalized


def _format_search_results(
    query: str,
    results: list[dict],
    engine: str = "unknown",
    max_results: int = 8,
) -> str:
    cleaned = _normalize_search_results(results, max_results=max_results)
    context_summary = "\n".join(f"- {item['snippet']}" for item in cleaned[:3])
    return json.dumps(
        {
            "type": "web_search_results",
            "query": query,
            "engine": engine,
            "context_summary": context_summary,
            "results": cleaned,
        }
    )


def _json_loads_maybe(value: str) -> dict:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _money_values(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?:[$€£]\s?\d[\d,]*(?:\.\d{2})?|\d[\d,]*(?:\.\d{2})?\s?(?:USD|CAD|EUR|GBP))", text or "", flags=re.IGNORECASE)))


def _rating_values(text: str) -> list[str]:
    patterns = [
        r"\b\d(?:\.\d)?\s?/\s?5\b",
        r"(?<!/)\b\d(?:\.\d)?\s?stars?\b",
        r"\b\d{2,3}%\s?(?:positive|recommended|recommend)\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text or "", flags=re.IGNORECASE))
    return list(dict.fromkeys(found))


def _extract_buying_signals(text: str, query: str) -> dict:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    lowered = cleaned.lower()
    pros = []
    cons = []
    for word in ("return", "warranty", "shipping", "battery", "weight", "noise", "privacy", "subscription", "compatibility"):
        idx = lowered.find(word)
        if idx >= 0:
            snippet = cleaned[max(0, idx - 90): idx + 160].strip()
            if any(term in snippet.lower() for term in ("not", "limited", "issue", "complaint", "problem", "expensive", "short")):
                cons.append(snippet)
            else:
                pros.append(snippet)
    return {
        "query": query,
        "prices": _money_values(cleaned)[:6],
        "ratings": _rating_values(cleaned)[:4],
        "pros": pros[:3],
        "cons": cons[:3],
    }


CANADIAN_STORE_PROFILES: dict[str, dict[str, str]] = {
    "costco": {"label": "Costco", "domain": "costco.ca", "best_for": "bulk, household, groceries, electronics deals"},
    "canadian_tire": {"label": "Canadian Tire", "domain": "canadiantire.ca", "best_for": "tools, automotive, home, seasonal, outdoor"},
    "shoppers": {"label": "Shoppers Drug Mart", "domain": "shoppersdrugmart.ca", "best_for": "pharmacy, beauty, toiletries, convenience"},
    "metro": {"label": "Metro", "domain": "metro.ca", "best_for": "groceries, fresh food, weekly flyer items"},
    "dollarama": {"label": "Dollarama", "domain": "dollarama.com", "best_for": "cheap household, party, school, small essentials"},
    "walmart": {"label": "Walmart Canada", "domain": "walmart.ca", "best_for": "general retail, groceries, baby, home, electronics"},
    "bestbuy": {"label": "Best Buy Canada", "domain": "bestbuy.ca", "best_for": "electronics, appliances, computers, accessories"},
}


def _normalize_store_keys(stores: Optional[list[str]]) -> list[str]:
    if not stores:
        return ["walmart", "canadian_tire", "costco", "bestbuy", "shoppers", "metro", "dollarama"]
    normalized: list[str] = []
    aliases = {
        "canadian tire": "canadian_tire",
        "ct": "canadian_tire",
        "shopper": "shoppers",
        "shoppers drug mart": "shoppers",
        "best buy": "bestbuy",
        "best buy canada": "bestbuy",
        "walmart canada": "walmart",
    }
    for store in stores:
        raw = str(store or "").strip().lower().replace("-", " ").replace("_", " ")
        key = aliases.get(raw, raw.replace(" ", "_"))
        if key in CANADIAN_STORE_PROFILES and key not in normalized:
            normalized.append(key)
    return normalized or ["walmart", "canadian_tire", "costco", "bestbuy"]


async def _shopping_advice(
    query: str,
    budget: str = "",
    priorities: Optional[list[str]] = None,
    region: str = "US",
    location: str = "",
    stores: Optional[list[str]] = None,
    max_candidates: int = 4,
) -> str:
    """Deterministic buyer workflow: search, fetch top pages, extract facts, return a final-answer patch."""
    priorities = priorities or []
    max_candidates = max(1, min(int(max_candidates or 4), 6))
    store_keys = _normalize_store_keys(stores)
    region_hint = f" {region}" if region else ""
    location_hint = f" near {location}" if location else ""
    budget_hint = f" under {budget}" if budget else ""
    priority_hint = f" best for {', '.join(priorities[:4])}" if priorities else ""
    broad_query = f"{query}{budget_hint}{priority_hint}{location_hint}{region_hint} price review official store"
    search_queries = [broad_query]
    for key in store_keys[:6]:
        profile = CANADIAN_STORE_PROFILES[key]
        search_queries.append(f"site:{profile['domain']} {query}{budget_hint}{location_hint}{region_hint}")

    candidates: list[dict] = []
    verified_urls: list[str] = []
    warnings: list[str] = []
    seen_urls: set[str] = set()
    store_coverage = {CANADIAN_STORE_PROFILES[key]["label"]: 0 for key in store_keys}
    for search_query in search_queries:
        if len(candidates) >= max_candidates:
            break
        search_payload = _json_loads_maybe(await _web_search(search_query, num_results=max_candidates + 2))
        results = search_payload.get("results") if isinstance(search_payload.get("results"), list) else []
        for item in results:
            if len(candidates) >= max_candidates:
                break
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            url_key = url.lower().rstrip("/")
            if not url or url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            matched_store = ""
            for key in store_keys:
                profile = CANADIAN_STORE_PROFILES[key]
                if profile["domain"] in url.lower() or profile["label"].lower().split()[0] in str(item.get("title") or "").lower():
                    matched_store = profile["label"]
                    store_coverage[matched_store] = store_coverage.get(matched_store, 0) + 1
                    break
            fetch_payload = _json_loads_maybe(await _web_fetch(url, max_chars=5000))
            content = str(fetch_payload.get("content") or item.get("snippet") or "")
            success = fetch_payload.get("type") == "web_fetch_result" and not fetch_payload.get("error")
            if success:
                verified_urls.append(url)
            else:
                warnings.append(f"Could not fully verify {url}")
            signals = _extract_buying_signals(content, query)
            title = str(fetch_payload.get("title") or item.get("title") or url).strip()
            candidate = {
                "title": title[:180],
                "store": matched_store or "Unknown store",
                "url": url,
                "verified": bool(success),
                "prices": signals["prices"],
                "ratings": signals["ratings"],
                "pros": signals["pros"],
                "cons": signals["cons"],
                "snippet": re.sub(r"\s+", " ", str(item.get("snippet") or ""))[:280],
            }
            candidates.append(candidate)

    verified_candidates = [item for item in candidates if item.get("verified")]
    best = verified_candidates[0] if verified_candidates else (candidates[0] if candidates else {})
    patch_lines = []
    if best:
        patch_lines.append(f"Best verified lead: {best.get('title')}")
        if best.get("prices"):
            patch_lines.append(f"Observed price signal: {', '.join(best['prices'][:3])}")
        if best.get("ratings"):
            patch_lines.append(f"Rating signal: {', '.join(best['ratings'][:2])}")
        patch_lines.append(f"Verified URL: {best.get('url')}")
    if warnings:
        patch_lines.append(f"Limits: {'; '.join(warnings[:2])}")

    return json.dumps({
        "type": "shopping_advice_result",
        "success": bool(candidates),
        "query": query,
        "budget": budget,
        "priorities": priorities,
        "region": region,
        "location": location,
        "stores_requested": [CANADIAN_STORE_PROFILES[key]["label"] for key in store_keys],
        "store_coverage": store_coverage,
        "search_queries": search_queries,
        "candidates": candidates,
        "verified_urls": verified_urls,
        "warnings": warnings,
        "answer_patch": {
            "format": "concise_buyer_advice_v1",
            "recommendation": best,
            "comparison_table": [
                {
                    "store": item.get("store"),
                    "item": item.get("title"),
                    "price": (item.get("prices") or ["not found"])[0],
                    "rating": (item.get("ratings") or ["not found"])[0],
                    "verified": item.get("verified"),
                    "url": item.get("url"),
                }
                for item in candidates[:4]
            ],
            "must_say": patch_lines,
            "do_not_claim": [
                "Do not claim a product was purchased, reserved, or added to cart.",
                "Do not state a precise current price unless it appears in candidate.prices.",
                "Do not cite URLs outside verified_urls.",
                "Do not claim local in-store availability unless fetched content explicitly says it.",
            ],
            "needs_user_choice": len(verified_candidates) < 2,
        },
    })


SHOPPING_ADVICE_TOOL = Tool(
    name="shopping_advice",
    description=(
        "Buyer workflow for consumer shopping questions. Searches, fetches candidate product/review pages, "
        "extracts price/rating/pro/con signals, and returns a compact verified answer_patch. "
        "Use this before broad web_search for product recommendations, buying decisions, and deal checks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What the user wants to buy or compare."},
            "budget": {"type": "string", "description": "Budget constraint such as '$900' or 'under 1000 CAD'."},
            "priorities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "User priorities such as battery life, durability, privacy, return policy.",
            },
            "region": {"type": "string", "description": "Shopping region/country, default US.", "default": "US"},
            "location": {"type": "string", "description": "City/neighbourhood/postal hint such as Toronto, ON or M5V."},
            "stores": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Preferred stores, e.g. Costco, Canadian Tire, Shoppers, Metro, Dollarama, Walmart, Best Buy.",
            },
            "max_candidates": {"type": "integer", "description": "Candidate pages to verify, default 4.", "default": 4},
        },
        "required": ["query"],
    },
    handler=_shopping_advice,
    tags=["shopping", "web", "buyer"],
    response_format="json",
)


NUMBER_WORDS = {
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


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return _json_loads_maybe(value)
    return {}


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = _json_loads_maybe(value)
        if isinstance(parsed.get("results"), list):
            return parsed["results"]
    return []


def _host_from_url(url: str) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower()
    except Exception:
        return ""


def _result_url(item: dict) -> str:
    return str(item.get("normalized_url") or item.get("url") or item.get("link") or "").strip().rstrip(".,")


def _content_excerpt(text: str, max_chars: int = 1200) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _fetch_success_payload(value) -> dict:
    payload = _as_dict(value)
    if not payload:
        return {}
    if payload.get("type") == "web_fetch_result" and not payload.get("error"):
        return payload
    return {}


def _normalize_entity(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _extract_number_after(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    raw = str(match.group(1)).lower()
    if raw.isdigit():
        return int(raw)
    return NUMBER_WORDS.get(raw, default)


def _infer_source_contract(objective: str, entities: Optional[list[str]] = None) -> dict:
    text = str(objective or "")
    lowered = text.lower()
    entity_count = len([e for e in (entities or []) if str(e).strip()])
    top_count = _extract_number_after(r"\btop\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", lowered)
    if top_count:
        entity_count = top_count
    if not entity_count:
        entity_count = _extract_number_after(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:stocks|companies|products|items|tools|options|entities|sources)\b",
            lowered,
        )

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
    if not total_fetches and entity_count and per_entity:
        total_fetches = entity_count * per_entity

    needs_separate_queries = bool(re.search(r"\b(each|separately|per)\b", lowered))
    needs_fetch = bool(re.search(r"\b(fetch|read|open|visit)\b", lowered))
    return {
        "objective": text.strip(),
        "entity_count": max(0, min(entity_count, 20)),
        "results_per_entity": max(0, min(per_entity, 20)),
        "total_fetches": max(0, min(total_fetches, 100)),
        "needs_separate_queries": needs_separate_queries,
        "needs_fetch": needs_fetch,
        "workflow": [
            "discover_entities",
            "lock_entities",
            "search_each_entity" if needs_separate_queries else "search_sources",
            "select_urls",
            "fetch_selected_urls" if needs_fetch else "summarize_sources",
            "verify_coverage",
            "answer_from_fetched_sources",
        ],
    }


async def _source_contract(objective: str, entities: Optional[list[str]] = None, topic: str = "") -> str:
    """Infer a deterministic source-gathering contract from a user request."""
    clean_entities = [_normalize_entity(item) for item in (entities or []) if _normalize_entity(item)]
    contract = _infer_source_contract(objective, clean_entities)
    query_templates = []
    if clean_entities:
        query_templates = [f"{entity} {topic}".strip() for entity in clean_entities]
    elif topic:
        query_templates = [topic.strip()]
    return json.dumps({
        "type": "source_contract",
        "success": True,
        "contract": contract,
        "locked_entities": clean_entities,
        "query_templates": query_templates,
        "rules": [
            "Do not introduce new entities after lock_entities unless the user changes the objective.",
            "Fetch only URLs selected from prior search results.",
            "A final answer may cite only fetched URLs or successful deterministic facts.",
            "If required slots are missing, call source_next_action instead of guessing.",
        ],
    })


async def _source_select(
    search_payloads: Optional[list] = None,
    entities: Optional[list[str]] = None,
    per_entity: int = 3,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
) -> str:
    """Select deterministic fetch URLs from one or more web_search payloads."""
    per_entity = max(1, min(int(per_entity or 3), 20))
    clean_entities = [_normalize_entity(item) for item in (entities or []) if _normalize_entity(item)]
    allowed = {str(domain or "").lower().lstrip(".") for domain in (allowed_domains or []) if str(domain or "").strip()}
    blocked = {str(domain or "").lower().lstrip(".") for domain in (blocked_domains or []) if str(domain or "").strip()}
    selected: dict[str, list[dict]] = {entity: [] for entity in clean_entities} if clean_entities else {"general": []}
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}

    for payload in search_payloads or []:
        parsed = _as_dict(payload)
        query = str(parsed.get("query") or "")
        results = parsed.get("results") if isinstance(parsed.get("results"), list) else _as_list(payload)
        query_target = ""
        for entity in clean_entities:
            if re.search(rf"\b{re.escape(entity)}\b", query, flags=re.IGNORECASE):
                query_target = entity
                break
        target_keys = [query_target] if query_target else (clean_entities or ["general"])
        for raw in results or []:
            if not isinstance(raw, dict):
                continue
            url = _result_url(raw)
            host = str(raw.get("host") or _host_from_url(url))
            if not url.startswith("http") or url in seen_urls:
                continue
            if allowed and not any(host == domain or host.endswith(f".{domain}") for domain in allowed):
                continue
            if blocked and any(host == domain or host.endswith(f".{domain}") for domain in blocked):
                continue
            for key in target_keys:
                if len(selected.setdefault(key, [])) >= per_entity:
                    continue
                selected[key].append({
                    "url": url,
                    "host": host,
                    "title": str(raw.get("title") or "")[:180],
                    "snippet": re.sub(r"\s+", " ", str(raw.get("snippet") or ""))[:280],
                    "rank": raw.get("rank"),
                    "source_query": query,
                })
                seen_urls.add(url)
                domain_counts[host] = domain_counts.get(host, 0) + 1
                break

    missing = [key for key, urls in selected.items() if len(urls) < per_entity]
    flattened = [item for urls in selected.values() for item in urls]
    return json.dumps({
        "type": "source_selection",
        "success": bool(flattened),
        "selected_by_entity": selected,
        "selected_urls": [item["url"] for item in flattened],
        "coverage": {
            "required_per_entity": per_entity,
            "missing_entities": missing,
            "domain_counts": domain_counts,
            "selected_count": len(flattened),
        },
        "next_hint": "fetch_selected_urls" if flattened else "run_more_targeted_searches",
    })


async def _source_next_action(
    objective: str,
    contract: Optional[dict] = None,
    entities: Optional[list[str]] = None,
    searched_queries: Optional[list[str]] = None,
    selected_urls: Optional[list[str]] = None,
    fetched_urls: Optional[list[str]] = None,
    topic: str = "",
) -> str:
    """Compute the next deterministic source-gathering action from state."""
    clean_entities = [_normalize_entity(item) for item in (entities or []) if _normalize_entity(item)]
    resolved_contract = contract if isinstance(contract, dict) else _infer_source_contract(objective, clean_entities)
    searched = {str(item or "").lower().strip() for item in (searched_queries or []) if str(item or "").strip()}
    selected = [str(item or "").strip() for item in (selected_urls or []) if str(item or "").strip()]
    fetched = {str(item or "").strip() for item in (fetched_urls or []) if str(item or "").strip()}
    per_entity = max(1, int(resolved_contract.get("results_per_entity") or 1))
    entity_count = int(resolved_contract.get("entity_count") or len(clean_entities) or 0)
    total_fetches = int(resolved_contract.get("total_fetches") or (entity_count * per_entity if entity_count else len(selected)))

    if entity_count and len(clean_entities) < entity_count:
        return json.dumps({
            "type": "source_next_action",
            "status": "missing_input",
            "next_tool": "web_search",
            "arguments": {"query": topic or objective},
            "reason": f"Need {entity_count} locked entities before targeted source collection.",
            "missing_slots": ["locked_entities"],
        })

    for entity in clean_entities:
        query = f"{entity} {topic}".strip() or entity
        if query.lower() not in searched:
            return json.dumps({
                "type": "source_next_action",
                "status": "continue",
                "next_tool": "web_search",
                "arguments": {"query": query},
                "reason": f"Entity '{entity}' has not been searched with the locked query.",
                "missing_slots": [f"{entity}_search_results"],
            })

    unfetched = [url for url in selected if url and url not in fetched]
    if unfetched and len(fetched) < total_fetches:
        return json.dumps({
            "type": "source_next_action",
            "status": "continue",
            "next_tool": "web_fetch",
            "arguments": {"url": unfetched[0]},
            "reason": f"Fetch selected URL {len(fetched) + 1} of {total_fetches}.",
            "missing_slots": [f"{max(0, total_fetches - len(fetched))}_remaining_fetches"],
        })

    if len(fetched) < total_fetches:
        return json.dumps({
            "type": "source_next_action",
            "status": "missing_input",
            "next_tool": "source_select",
            "arguments": {"entities": clean_entities, "per_entity": per_entity},
            "reason": "More selected URLs are required before fetch can complete.",
            "missing_slots": ["selected_urls"],
        })

    return json.dumps({
        "type": "source_next_action",
        "status": "finalize",
        "next_tool": "",
        "arguments": {},
        "reason": "Required source coverage is complete.",
        "missing_slots": [],
    })


async def _web_fetch_batch(
    urls: list[str],
    max_chars_per_url: int = 4000,
    use_jina: bool = True,
    max_urls: int = 12,
) -> str:
    """Fetch multiple selected URLs sequentially with compact structured output."""
    max_urls = max(1, min(int(max_urls or 12), 20))
    max_chars_per_url = max(500, min(int(max_chars_per_url or 4000), 12000))
    seen: set[str] = set()
    selected_urls: list[str] = []
    for raw_url in urls or []:
        url = str(raw_url or "").strip().rstrip(".,")
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        selected_urls.append(url)
        if len(selected_urls) >= max_urls:
            break

    fetched_sources: list[dict] = []
    failures: list[dict] = []
    for url in selected_urls:
        raw = await _web_fetch(url, max_chars=max_chars_per_url, use_jina=use_jina)
        payload = _as_dict(raw)
        success_payload = _fetch_success_payload(payload)
        if success_payload:
            content = str(success_payload.get("content") or "")
            fetched_sources.append({
                "url": str(success_payload.get("url") or url),
                "final_url": str(success_payload.get("final_url") or success_payload.get("url") or url),
                "host": str(success_payload.get("host") or _host_from_url(url)),
                "title": str(success_payload.get("title") or url)[:180],
                "status_code": success_payload.get("status_code"),
                "backend": success_payload.get("backend"),
                "content_excerpt": _content_excerpt(content, max_chars=1600),
                "total_length": success_payload.get("total_length"),
                "truncated": bool(success_payload.get("truncated")),
            })
        else:
            failures.append({
                "url": url,
                "error": str(payload.get("error") or "fetch failed"),
                "type": str(payload.get("type") or "web_fetch_error"),
            })

    return json.dumps({
        "type": "web_fetch_batch_result",
        "success": bool(fetched_sources),
        "requested_count": len(urls or []),
        "attempted_count": len(selected_urls),
        "fetched_count": len(fetched_sources),
        "failed_count": len(failures),
        "fetched_urls": [item["url"] for item in fetched_sources],
        "fetched_sources": fetched_sources,
        "failures": failures,
        "answer_rules": [
            "Cite only fetched_sources.url or fetched_sources.final_url.",
            "Do not claim details that are not present in content_excerpt.",
            "If failed_count is nonzero, disclose the fetch gap briefly.",
        ],
    })


async def _source_coverage(
    contract: Optional[dict] = None,
    entities: Optional[list[str]] = None,
    selected_payload: Optional[dict] = None,
    fetch_payloads: Optional[list] = None,
    fetched_urls: Optional[list[str]] = None,
) -> str:
    """Verify source-gathering coverage against a deterministic contract."""
    clean_entities = [_normalize_entity(item) for item in (entities or []) if _normalize_entity(item)]
    resolved_contract = contract if isinstance(contract, dict) else {}
    per_entity = max(1, int(resolved_contract.get("results_per_entity") or 1))
    required_entities = int(resolved_contract.get("entity_count") or len(clean_entities) or 0)
    total_required = int(resolved_contract.get("total_fetches") or (required_entities * per_entity if required_entities else 0))

    selection = selected_payload if isinstance(selected_payload, dict) else _as_dict(selected_payload)
    selected_by_entity = selection.get("selected_by_entity") if isinstance(selection.get("selected_by_entity"), dict) else {}
    selected_urls = [
        str(url or "").strip()
        for url in (selection.get("selected_urls") or [])
        if str(url or "").strip()
    ]
    fetched: set[str] = {str(url or "").strip() for url in (fetched_urls or []) if str(url or "").strip()}
    fetched_sources: list[dict] = []
    for payload in fetch_payloads or []:
        parsed = _as_dict(payload)
        if parsed.get("type") == "web_fetch_batch_result":
            for source in parsed.get("fetched_sources") or []:
                if isinstance(source, dict):
                    fetched_sources.append(source)
                    if source.get("url"):
                        fetched.add(str(source.get("url")))
                    if source.get("final_url"):
                        fetched.add(str(source.get("final_url")))
            continue
        success_payload = _fetch_success_payload(parsed)
        if success_payload:
            fetched_sources.append(success_payload)
            if success_payload.get("url"):
                fetched.add(str(success_payload.get("url")))
            if success_payload.get("final_url"):
                fetched.add(str(success_payload.get("final_url")))

    entity_status: dict[str, dict] = {}
    missing_slots: list[str] = []
    for entity in clean_entities:
        selected_items = selected_by_entity.get(entity) if isinstance(selected_by_entity.get(entity), list) else []
        entity_selected_urls = [
            str(item.get("url") or "").strip()
            for item in selected_items
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        ]
        entity_fetched = [url for url in entity_selected_urls if url in fetched]
        if len(entity_selected_urls) < per_entity:
            missing_slots.append(f"{entity}_selected_urls")
        if len(entity_fetched) < min(per_entity, len(entity_selected_urls) or per_entity):
            missing_slots.append(f"{entity}_fetched_urls")
        entity_status[entity] = {
            "selected": len(entity_selected_urls),
            "fetched": len(entity_fetched),
            "required": per_entity,
            "complete": len(entity_selected_urls) >= per_entity and len(entity_fetched) >= per_entity,
        }

    if required_entities and len(clean_entities) < required_entities:
        missing_slots.append("locked_entities")
    if total_required and len(fetched) < total_required:
        missing_slots.append("total_fetched_urls")

    complete = not missing_slots
    next_tool = ""
    if not complete:
        if "total_fetched_urls" in missing_slots and selected_urls:
            unfetched = [url for url in selected_urls if url not in fetched]
            next_tool = "web_fetch_batch" if unfetched else "source_select"
        elif any(slot.endswith("_selected_urls") for slot in missing_slots):
            next_tool = "source_select"
        else:
            next_tool = "source_next_action"

    return json.dumps({
        "type": "source_coverage",
        "success": True,
        "complete": complete,
        "status": "complete" if complete else "incomplete",
        "missing_slots": list(dict.fromkeys(missing_slots)),
        "entity_status": entity_status,
        "required": {
            "entity_count": required_entities,
            "results_per_entity": per_entity,
            "total_fetches": total_required,
        },
        "observed": {
            "locked_entities": len(clean_entities),
            "selected_urls": len(selected_urls),
            "fetched_urls": len(fetched),
            "fetched_sources": len(fetched_sources),
        },
        "next_tool": next_tool,
        "answer_allowed": complete,
    })


SOURCE_CONTRACT_TOOL = Tool(
    name="source_contract",
    description=(
        "Deterministically convert a research/shopping/news request into a source-gathering contract: "
        "how many entities, how many URLs per entity, whether separate searches are required, and what must be fetched. "
        "Use before web_search for multi-source workflows."
    ),
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The user's exact objective."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Already locked entities, if known."},
            "topic": {"type": "string", "description": "Optional query topic, e.g. 'stock news June 13 2026'."},
        },
        "required": ["objective"],
    },
    handler=_source_contract,
    tags=["kernel", "source", "web"],
    response_format="json",
)


SOURCE_SELECT_TOOL = Tool(
    name="source_select",
    description=(
        "Deterministically select fetch URLs from web_search JSON payloads. Dedupes URLs, tracks host coverage, "
        "groups by locked entity, and reports missing coverage. Use before web_fetch."
    ),
    parameters={
        "type": "object",
        "properties": {
            "search_payloads": {"type": "array", "items": {}, "description": "One or more web_search JSON payloads."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Locked entities to group URLs under."},
            "per_entity": {"type": "integer", "description": "URLs required per entity.", "default": 3},
            "allowed_domains": {"type": "array", "items": {"type": "string"}, "description": "Optional allowed domains."},
            "blocked_domains": {"type": "array", "items": {"type": "string"}, "description": "Optional blocked domains."},
        },
        "required": ["search_payloads"],
    },
    handler=_source_select,
    tags=["kernel", "source", "web"],
    response_format="json",
)


SOURCE_NEXT_ACTION_TOOL = Tool(
    name="source_next_action",
    description=(
        "Given the source contract and observed state, deterministically return the next tool call needed "
        "or finalize. Use when the model is unsure whether to search, select URLs, fetch, or answer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "The user's exact objective."},
            "contract": {"type": "object", "description": "A source_contract.contract object, if available."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Locked entities."},
            "searched_queries": {"type": "array", "items": {"type": "string"}, "description": "Queries already searched."},
            "selected_urls": {"type": "array", "items": {"type": "string"}, "description": "URLs selected for fetch."},
            "fetched_urls": {"type": "array", "items": {"type": "string"}, "description": "URLs already fetched successfully."},
            "topic": {"type": "string", "description": "Query suffix to use for entity searches."},
        },
        "required": ["objective"],
    },
    handler=_source_next_action,
    tags=["kernel", "source", "web"],
    response_format="json",
)


async def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """Compatibility fallback used by tools_web when DuckDuckGo is requested directly."""
    loop = asyncio.get_running_loop()
    try:
        def sync_search():
            from ddgs import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=num_results))
        raw_results = await loop.run_in_executor(None, sync_search)
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "duckduckgo",
            })
        return results
    except Exception as e:
        print(f"[_search_duckduckgo] Error: {e}")
        return []



# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE SEARCH  —  DuckDuckGo images (no API key)
# ══════════════════════════════════════════════════════════════════════════════

async def _image_search(query: str, num_results: int = 5) -> str:
    """Search for images via DuckDuckGo and return URLs + titles."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        ),
        "Referer": "https://duckduckgo.com/",
    }
    try:
        # Step 1: get vqd token (required by DDG image API)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            token_resp = await client.get(
                "https://duckduckgo.com/",
                params={"q": query},
                headers=headers,
            )
            vqd_match = re.search(r'vqd=(["\'])([^"\']+)\1', token_resp.text)
            if not vqd_match:
                return f"image_search: could not retrieve search token for '{query}'"
            vqd = vqd_match.group(2)

            # Step 2: image results
            img_resp = await client.get(
                "https://duckduckgo.com/i.js",
                params={"q": query, "vqd": vqd, "f": ",,,,,", "p": "1"},
                headers=headers,
            )
            data = img_resp.json()

        results = data.get("results", [])[:num_results]
        if not results:
            return f"No images found for: {query}"

        lines = [f"Image results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', 'Untitled')}")
            lines.append(f"    Image URL:    {r.get('image', '')}")
            lines.append(f"    Source URL:   {r.get('url', '')}")
            lines.append(f"    Dimensions:   {r.get('width', '?')}x{r.get('height', '?')}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"image_search error: {e}"


IMAGE_SEARCH_TOOL = Tool(
    name="image_search",
    description=(
        "Search for images using DuckDuckGo. "
        "Returns image URLs, source pages, and dimensions. "
        "No API key required."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query":       {"type": "string",  "description": "Image search query"},
            "num_results": {"type": "integer", "description": "Number of images (default 5)", "default": 5},
        },
        "required": ["query"],
    },
    handler=_image_search,
    tags=["web", "images"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  BASH  —  Docker sandbox execution
# ══════════════════════════════════════════════════════════════════════════════

# Commands that are blocked outright regardless of context
_BLOCKED_COMMANDS = re.compile(
    r"\b(rm\s+-rf\s+/|mkfs|dd\s+if=|:\(\)\{\s*:\|:&\s*\};:|shutdown|reboot|"
    r"chmod\s+777\s+/|curl\s+.*\|\s*sh|wget\s+.*\|\s*sh)",
    re.IGNORECASE
)


def _verification_payload(
    *,
    mode: str,
    expected_paths: Optional[list[str]] = None,
    existing_paths: Optional[list[str]] = None,
    missing_paths: Optional[list[str]] = None,
    bytes_written: Optional[int] = None,
) -> dict:
    return {
        "mode": mode,
        "expected_paths": list(expected_paths or []),
        "existing_paths": list(existing_paths or []),
        "missing_paths": list(missing_paths or []),
        "exists_after": not bool(missing_paths),
        "bytes_written": bytes_written,
    }


def _resolve_sandbox_target(path_str: str, *, workdir: Optional[str] = None) -> Optional[Path]:
    raw = str(path_str or "").strip().strip("'\"")
    if not raw:
        return None
    if raw.startswith("/sandbox/") or raw == "/sandbox":
        return _safe_path(raw)
    if raw.startswith("/workspace/"):
        raw = raw[len("/workspace/"):]
    elif raw == "/workspace":
        raw = "."
    if raw.startswith("/"):
        return None
    try:
        base = _safe_path(workdir or ".")
    except ValueError:
        base = _safe_path(".")
    target = (base / raw).resolve()
    if not str(target).startswith(str(SANDBOX_DIR)):
        return None
    return target


def _extract_write_targets(command: str, *, workdir: Optional[str] = None) -> list[Path]:
    text = str(command or "").strip()
    if not text:
        return []

    targets: list[Path] = []
    seen: set[str] = set()

    def add_target(raw_path: str) -> None:
        target = _resolve_sandbox_target(raw_path, workdir=workdir)
        if target is None:
            return
        key = str(target)
        if key in seen:
            return
        seen.add(key)
        targets.append(target)

    try:
        parts = shlex.split(text)
    except ValueError:
        parts = []

    for idx, token in enumerate(parts):
        if token == "touch":
            for maybe_path in parts[idx + 1:]:
                if maybe_path.startswith("-"):
                    continue
                if maybe_path in {"&&", "||", ";"}:
                    break
                add_target(maybe_path)
            break
        if token == "mkdir":
            for maybe_path in parts[idx + 1:]:
                if maybe_path.startswith("-"):
                    continue
                if maybe_path in {"&&", "||", ";"}:
                    break
                add_target(maybe_path)
            break
        if token in {"cp", "mv"} and idx + 2 < len(parts):
            add_target(parts[idx + 2])
            break
        if token == "tee" and idx + 1 < len(parts):
            add_target(parts[idx + 1])
            break

    for match in re.finditer(r"(?:^|[^\w])>>?\s*(?P<path>[^\s;&|]+)", text):
        add_target(match.group("path"))

    return targets


def _verify_expected_paths(paths: list[Path]) -> dict:
    if not paths:
        return _verification_payload(mode="not_applicable")

    existing: list[str] = []
    missing: list[str] = []
    bytes_written = 0
    for target in paths:
        rel = str(target.relative_to(SANDBOX_DIR))
        if target.exists():
            existing.append(rel)
            if target.is_file():
                bytes_written += int(target.stat().st_size)
        else:
            missing.append(rel)
    return _verification_payload(
        mode="filesystem_check",
        expected_paths=[str(target.relative_to(SANDBOX_DIR)) for target in paths],
        existing_paths=existing,
        missing_paths=missing,
        bytes_written=bytes_written,
    )


async def _bash(command: str, timeout: int = BASH_TIMEOUT, workdir: Optional[str] = None) -> str:
    """
    Execute a bash command in a throwaway Docker container.
    STRICT SECURITY: If Docker is unavailable, no execution is possible.
    """
    # STRICT SAFETY: Block dangerous commands before they hit Docker
    if _BLOCKED_COMMANDS.search(command):
        # Even on a denial, surface verification so the planner sees
        # missing write targets and cannot claim success on the next turn.
        denied_verification = _verify_expected_paths(_extract_write_targets(command, workdir=workdir))
        denied_hard_failure = bool(denied_verification.get("missing_paths"))
        return json.dumps(
            {
                "type": "bash_result",
                "success": False,
                "status": "HARD_FAILURE" if denied_hard_failure else "DENIED",
                "message": (
                    "Bash command blocked for safety; expected write targets are missing."
                    if denied_hard_failure
                    else "Bash command blocked for safety (contains destructive patterns)."
                ),
                "command": command,
                "verification": denied_verification,
                "output": "[denied] Bash command blocked for safety (contains destructive patterns).",
            }
        )

    from plugins.docker_sandbox import run_in_docker
    output = await run_in_docker(command, timeout=timeout, workdir=workdir)
    try:
        effective_workdir = str(_safe_path(workdir or ".").relative_to(SANDBOX_DIR))
    except ValueError:
        effective_workdir = "."
    verification = _verify_expected_paths(_extract_write_targets(command, workdir=workdir))
    hard_failure = bool(verification.get("missing_paths"))
    denied = str(output or "").lower().startswith("[denied]")
    execution_error = str(output or "").lower().startswith("[error]") or str(output or "").lower().startswith("[timeout]")
    success = not hard_failure and not denied and not execution_error
    status = (
        "HARD_FAILURE"
        if hard_failure
        else "DENIED"
        if denied
        else "FAILED"
        if execution_error
        else "SUCCESS"
    )
    message = (
        "Expected write targets were not present after bash execution."
        if hard_failure
        else "Bash command executed."
        if success
        else "Bash command did not complete successfully."
    )
    return json.dumps(
        {
            "type": "bash_result",
            "success": success,
            "status": status,
            "message": message,
            "command": command,
            "workdir": effective_workdir,
            "verification": verification,
            "output": output,
        }
    )



BASH_TOOL = Tool(
    name="bash",
    description=(
        "Execute a bash command in a sandboxed Linux environment. "
        "Use for running scripts, installing packages, processing files, "
        "or any shell task. Output is captured and returned. "
        f"Working directory: {SANDBOX_DIR}. Timeout: {BASH_TIMEOUT}s."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string",  "description": "Bash command to run"},
            "timeout": {"type": "integer", "description": f"Timeout seconds (default {BASH_TIMEOUT})", "default": BASH_TIMEOUT},
            "workdir": {"type": "string",  "description": "Working directory (must be inside sandbox)"},
        },
        "required": ["command"],
    },
    handler=_bash,
    tags=["code", "shell"],
    response_format="json",
)


# ══════════════════════════════════════════════════════════════════════════════
#  FILE TOOLS  —  create / view / str_replace
# ══════════════════════════════════════════════════════════════════════════════

def _safe_path(path_str: str) -> Path:
    """Resolve path and ensure it stays inside SANDBOX_DIR."""
    normalized = str(path_str or "").strip()
    if normalized.startswith("/sandbox/"):
        normalized = normalized[len("/sandbox/"):]
    elif normalized == "/sandbox":
        normalized = "."
    p = (SANDBOX_DIR / normalized).resolve()
    if not str(p).startswith(str(SANDBOX_DIR)):
        raise ValueError(f"Path '{path_str}' escapes sandbox — refused")
    return p


async def _file_create(path: Optional[str] = None, content: str = "", filename: Optional[str] = None, encoding: str = "utf-8") -> str:
    """Create or overwrite a file inside the sandbox."""
    try:
        target_path = path or filename
        if not target_path:
            return json.dumps(
                {
                    "type": "file_create_result",
                    "success": False,
                    "status": "FAILED",
                    "message": "either 'path' or 'filename' is required",
                    "verification": _verification_payload(mode="not_applicable"),
                }
            )
        target = _safe_path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)

        rel_path = target.relative_to(SANDBOX_DIR)
        verification = _verify_expected_paths([target])
        hard_failure = bool(verification.get("missing_paths"))
        payload = {
            "type": "file_create_result",
            "success": not hard_failure,
            "status": "HARD_FAILURE" if hard_failure else "SUCCESS",
            "message": (
                "Expected file was not present after write."
                if hard_failure
                else f"Created: {rel_path} ({len(content)} chars)"
            ),
            "path": str(rel_path),
            "encoding": encoding,
            "verification": verification,
        }

        if str(rel_path).lower().endswith(".html"):
            payload["preview"] = {
                "status": "success",
                "type": "app_view",
                "title": str(rel_path),
                "filename": str(rel_path),
                "path": f"/sandbox/{rel_path}",
            }

        return json.dumps(payload)
    except ValueError as e:
        return json.dumps(
            {
                "type": "file_create_result",
                "success": False,
                "status": "FAILED",
                "message": f"file_create error: {e}",
                "verification": _verification_payload(mode="not_applicable"),
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "type": "file_create_result",
                "success": False,
                "status": "FAILED",
                "message": f"file_create error: {e}",
                "verification": _verification_payload(mode="not_applicable"),
            }
        )


FILE_CREATE_TOOL = Tool(
    name="file_create",
    description=(
        "Create a new file (or overwrite) at the given path inside the sandbox. "
        "Parent directories are created automatically. "
        "CRITICAL: If creating .html dashboards, you MUST follow the 'V8 Platinum Standard': "
        "1. AESTHETICS: Use 'bg-black' (#000) with glassmorphism and glowing neon accents. "
        "2. INTERACTIVITY: Implement SPA-style views via vanilla JS (toggle .hidden or style.display). "
        "3. ASSETS: Use Lucide icons (cdn) and Unsplash imagery. No generic place-holders."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path":     {"type": "string", "description": "File path relative to sandbox root (Alias: filename)"},
            "filename": {"type": "string", "description": "Alias for path"},
            "content":  {"type": "string", "description": "File content to write (MUST escape double quotes!)"},
            "encoding": {"type": "string", "description": "Encoding (default utf-8)", "default": "utf-8"},
        },
        "required": ["content"],
    },
    handler=_file_create,
    tags=["files"],
    response_format="json",
)


async def _file_view(
    path: str,
    start_line: Optional[int] = None,
    end_line:   Optional[int] = None,
) -> str:
    """
    Read a file or list a directory.
    If path is a file: returns its contents (or a line range).
    If path is a directory: returns a recursive listing.
    """
    try:
        target = _safe_path(path)
    except ValueError as e:
        return f"file_view error: {e}"

    if not target.exists():
        return f"file_view: '{path}' not found"

    if target.is_dir():
        lines = []
        for item in sorted(target.rglob("*")):
            rel = item.relative_to(SANDBOX_DIR)
            prefix = "📁 " if item.is_dir() else "📄 "
            lines.append(prefix + str(rel))
        return f"Directory: {path}\n" + "\n".join(lines) if lines else f"Directory '{path}' is empty"

    # File
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"file_view error reading file: {e}"

    all_lines = text.splitlines(keepends=True)
    total     = len(all_lines)

    if start_line is not None or end_line is not None:
        s = max(0, (start_line or 1) - 1)
        e = min(total, end_line or total)
        selected = all_lines[s:e]
        header   = f"File: {path} (lines {s+1}–{e} of {total})\n"
    else:
        selected = all_lines
        header   = f"File: {path} ({total} lines)\n"

    content = "".join(selected)
    if len(content) > 12000:
        content = content[:12000] + f"\n\n[truncated — {len(content)-12000} more chars]"

    numbered = []
    for i, line in enumerate(selected, s + 1 if (start_line is not None) else 1):
        numbered.append(f"{i:>4} | {line.rstrip()}")

    return header + "\n".join(numbered)


FILE_VIEW_TOOL = Tool(
    name="file_view",
    description=(
        "Read the contents of a file or list a directory. "
        "Supports line ranges with start_line / end_line. "
        "All paths are relative to the sandbox root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path":       {"type": "string",  "description": "File or directory path (relative to sandbox)"},
            "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
            "end_line":   {"type": "integer", "description": "Last line to read inclusive (optional)"},
        },
        "required": ["path"],
    },
    handler=_file_view,
    tags=["files"],
)


async def _file_str_replace(path: str, old_str: str, new_str: str) -> str:
    """
    Replace an exact string in a file.
    The old_str must appear exactly once — fails if 0 or 2+ matches.
    """
    try:
        target = _safe_path(path)
    except ValueError as e:
        return f"file_str_replace error: {e}"

    if not target.exists():
        return f"file_str_replace: '{path}' not found"

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"file_str_replace: cannot read file: {e}"

    count = content.count(old_str)
    if count == 0:
        return "file_str_replace: old_str not found in file — no changes made"
    if count > 1:
        return (
            f"file_str_replace: old_str appears {count} times — must be unique. "
            "Add more surrounding context to old_str to make it unambiguous."
        )

    updated = content.replace(old_str, new_str, 1)
    try:
        target.write_text(updated, encoding="utf-8")
    except Exception as e:
        return f"file_str_replace: write failed: {e}"

    lines_changed = len(new_str.splitlines()) - len(old_str.splitlines())
    return (
        f"Replaced in {target.relative_to(SANDBOX_DIR)}: "
        f"{len(old_str)} chars → {len(new_str)} chars "
        f"({'+'if lines_changed>=0 else ''}{lines_changed} lines)"
    )


FILE_STR_REPLACE_TOOL = Tool(
    name="file_str_replace",
    description=(
        "Replace an exact string in a file with new content. "
        "old_str must appear exactly once. Add surrounding context "
        "to disambiguate if needed. All paths relative to sandbox root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "File path (relative to sandbox)"},
            "old_str": {"type": "string", "description": "Exact string to find and replace (must be unique in file)"},
            "new_str": {"type": "string", "description": "Replacement string (can be empty to delete)"},
        },
        "required": ["path", "old_str", "new_str"],
    },
    handler=_file_str_replace,
    tags=["files"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER FETCH  —  open-meteo.com (free, no API key)
# ══════════════════════════════════════════════════════════════════════════════

async def _weather_fetch(location: str, units: str = "metric") -> str:
    """
    Fetch current weather + 3-day forecast.
    Geocodes the location string first, then calls open-meteo.
    Entirely free — no API key required.
    """
    try:
        # Step 1: geocode location string → lat/lon
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            geo_data = geo.json()

        results = geo_data.get("results")
        if not results:
            return f"weather_fetch: could not geocode '{location}'"

        r         = results[0]
        lat       = r["latitude"]
        lon       = r["longitude"]
        place     = f"{r.get('name', location)}, {r.get('country', '')}"

        # Step 2: fetch weather
        temp_unit = "celsius" if units == "metric" else "fahrenheit"
        wind_unit = "kmh"     if units == "metric" else "mph"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":             lat,
                    "longitude":            lon,
                    "current":              "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                    "daily":                "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "temperature_unit":     temp_unit,
                    "wind_speed_unit":      wind_unit,
                    "forecast_days":        4,
                    "timezone":             "auto",
                },
            )
            wx.raise_for_status()
            data = wx.json()

        cur   = data.get("current", {})
        daily = data.get("daily", {})
        deg   = "°C" if units == "metric" else "°F"
        spd   = "km/h" if units == "metric" else "mph"

        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Heavy showers",
            95: "Thunderstorm", 99: "Thunderstorm with hail",
        }

        def wmo(code) -> str:
            return wmo_codes.get(int(code), f"Code {code}")

        lines = [
            f"Weather for {place}  ({lat:.2f}°N, {lon:.2f}°E)",
            "",
            "── Current ─────────────────────────────",
            f"  Condition:    {wmo(cur.get('weather_code', 0))}",
            f"  Temperature:  {cur.get('temperature_2m', '?')}{deg}  (feels like {cur.get('apparent_temperature', '?')}{deg})",
            f"  Humidity:     {cur.get('relative_humidity_2m', '?')}%",
            f"  Wind:         {cur.get('wind_speed_10m', '?')} {spd}",
            f"  Precip:       {cur.get('precipitation', 0)} mm",
            "",
            "── 3-Day Forecast ───────────────────────",
        ]

        dates  = daily.get("time", [])
        hi     = daily.get("temperature_2m_max", [])
        lo     = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes  = daily.get("weather_code", [])

        for i in range(1, min(4, len(dates))):
            lines.append(
                f"  {dates[i]}  {wmo(codes[i]):<20} "
                f"↑{hi[i]}{deg} ↓{lo[i]}{deg}  "
                f"precip {precip[i]}mm"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"weather_fetch error: {e}"


WEATHER_TOOL = Tool(
    name="weather_fetch",
    description=(
        "Get current weather and 3-day forecast for any location. "
        "Uses open-meteo.com — no API key required. "
        "Accepts city names, addresses, or coordinates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or address e.g. 'Toronto' or 'Paris, France'"},
            "units":    {"type": "string", "description": "Units: 'metric' (°C, km/h) or 'imperial' (°F, mph)", "default": "metric"},
        },
        "required": ["location"],
    },
    handler=_weather_fetch,
    tags=["weather", "geo"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  PLACES SEARCH  —  Google Places API
#  Requires: GOOGLE_PLACES_API_KEY env var
# ══════════════════════════════════════════════════════════════════════════════

async def _places_search(
    query:      str,
    location:   Optional[str] = None,
    radius_m:   int = 5000,
    max_results: int = 5,
) -> str:
    """
    Search Google Places API.
    query    — e.g. "ramen restaurants in Tokyo"
    location — optional lat,lon bias e.g. "43.65,-79.38"
    """
    if not GOOGLE_PLACES_KEY:
        return (
            "places_search: GOOGLE_PLACES_API_KEY environment variable not set. "
            "Get a key at https://console.cloud.google.com and enable the Places API."
        )

    try:
        params: dict = {
            "query":  query,
            "key":    GOOGLE_PLACES_KEY,
        }
        if location:
            params["location"] = location
            params["radius"]   = radius_m

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return f"places_search API error: {data.get('status')} — {data.get('error_message', '')}"

        places = data.get("results", [])[:max_results]
        if not places:
            return f"No places found for: {query}"

        lines = [f"Places results for: {query}\n"]
        for i, p in enumerate(places, 1):
            loc = p.get("geometry", {}).get("location", {})
            lines.append(f"[{i}] {p.get('name', 'Unknown')}")
            lines.append(f"    Address:  {p.get('formatted_address', 'N/A')}")
            lines.append(f"    Rating:   {p.get('rating', 'N/A')} ({p.get('user_ratings_total', 0)} reviews)")
            lines.append(f"    Types:    {', '.join(p.get('types', [])[:4])}")
            lines.append(f"    Lat/Lon:  {loc.get('lat', '')}, {loc.get('lng', '')}")
            lines.append(f"    Place ID: {p.get('place_id', 'N/A')}")
            if p.get("opening_hours"):
                open_now = p["opening_hours"].get("open_now")
                lines.append(f"    Open now: {'Yes' if open_now else 'No'}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"places_search error: {e}"


PLACES_SEARCH_TOOL = Tool(
    name="places_search",
    description=(
        "Search for real places, businesses, and attractions using Google Places. "
        "Returns names, addresses, ratings, types, coordinates, and Place IDs. "
        "Requires GOOGLE_PLACES_API_KEY environment variable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query":       {"type": "string",  "description": "Natural language query e.g. 'coffee shops near CN Tower Toronto'"},
            "location":    {"type": "string",  "description": "Optional lat,lon bias e.g. '43.65,-79.38'"},
            "radius_m":    {"type": "integer", "description": "Search radius in metres when location set (default 5000)", "default": 5000},
            "max_results": {"type": "integer", "description": "Max results to return (default 5, max 20)", "default": 5},
        },
        "required": ["query"],
    },
    handler=_places_search,
    tags=["places", "geo", "maps"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  PLACES MAP  —  static HTML embed using Google Maps
# ══════════════════════════════════════════════════════════════════════════════

async def _places_map(
    places:    list[dict],
    title:     str = "Map",
    save_path: str = "map.html",
) -> str:
    """
    Generate an interactive HTML map from a list of places.
    Each place: {name, lat, lon, description?}
    Saves to sandbox/save_path. Returns the file path and a summary.
    Renders with Leaflet.js (open source, no API key required).
    """
    if not places:
        return "places_map: no places provided"

    markers_js = []
    bounds_lats = []
    bounds_lons = []

    for p in places:
        lat  = float(p.get("lat") or p.get("latitude") or 0)
        lon  = float(p.get("lon") or p.get("longitude") or p.get("lng") or 0)
        name = p.get("name", "Location").replace("'", "\\'")
        desc = p.get("description", "").replace("'", "\\'")
        bounds_lats.append(lat)
        bounds_lons.append(lon)
        popup = f"<b>{name}</b><br>{desc}" if desc else f"<b>{name}</b>"
        markers_js.append(
            f"L.marker([{lat}, {lon}])"
            f".bindPopup('{popup}')"
            f".addTo(map);"
        )

    center_lat = sum(bounds_lats) / len(bounds_lats)
    center_lon = sum(bounds_lons) / len(bounds_lons)
    markers_str = "\n    ".join(markers_js)

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <style>
        body {{ margin: 0; font-family: sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        #title {{ position:absolute; top:10px; left:50%; transform:translateX(-50%);
                  z-index:999; background:white; padding:8px 16px;
                  border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.2);
                  font-weight:bold; font-size:15px; }}
      </style>
    </head>
    <body>
      <div id="title">{title}</div>
      <div id="map"></div>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <script>
        const map = L.map('map').setView([{center_lat}, {center_lon}], 14);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        {markers_str}
      </script>
    </body>
    </html>
    """).strip()

    try:
        target = _safe_path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return (
            f"Map saved: {target.relative_to(SANDBOX_DIR)}\n"
            f"Contains {len(places)} markers centred at {center_lat:.4f}, {center_lon:.4f}\n"
            f"Open the HTML file in a browser to view the interactive map.\n"
            f"Places mapped:\n" +
            "\n".join(f"  • {p.get('name','?')} ({p.get('lat') or p.get('latitude')}, {p.get('lon') or p.get('longitude') or p.get('lng')})" for p in places)
        )
    except Exception as e:
        return f"places_map error saving file: {e}"


PLACES_MAP_TOOL = Tool(
    name="places_map",
    description=(
        "Generate a local interactive HTML map from a list of places. "
        "Uses Leaflet.js + OpenStreetMap — no API key required. "
        "Each place needs name + lat + lon. Saves map HTML to sandbox. "
        "Best used after places_search to visualise results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "places": {
                "type": "array",
                "description": "List of places to map",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "lat":         {"type": "number"},
                        "lon":         {"type": "number"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "lat", "lon"],
                },
            },
            "title":     {"type": "string", "description": "Map title shown on page (default 'Map')", "default": "Map"},
            "save_path": {"type": "string", "description": "Output HTML path inside sandbox (default 'map.html')", "default": "map.html"},
        },
        "required": ["places"],
    },
    handler=_places_map,
    tags=["places", "maps", "geo"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  DEEP MEMORY (Semantic Graph)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_runtime_embed_model(kwargs) -> str:
    embed_model = str(kwargs.get("_embed_model") or "").strip()
    return embed_model or "nomic-embed-text"


def _build_runtime_graph(kwargs):
    from memory.semantic_graph import SemanticGraph

    embed_model = _resolve_runtime_embed_model(kwargs)
    try:
        return SemanticGraph(embedding_model=embed_model)
    except TypeError:
        return SemanticGraph()


def _normalize_memory_subject(subject: Optional[str]) -> str:
    value = str(subject or "").strip()
    return value or "User"


def _resolve_runtime_turn(session_id: Optional[str], owner_id: Optional[str]) -> int:
    if not session_id:
        return 1
    try:
        session = SessionManager().get(session_id, owner_id=owner_id)
    except Exception:
        session = None
    return max(1, int(getattr(session, "message_count", 0) or 0))


async def _store_memory(
    subject: Optional[str] = None,
    predicate: str = "",
    object_: str = "",
    locus_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Explicitly store a factual triplet into the semantic graph memory."""
    try:
        graph = _build_runtime_graph(kwargs)
        owner_id = kwargs.get("_owner_id")
        run_id = kwargs.get("_run_id")
        normalized_subject = _normalize_memory_subject(subject)
        normalized_predicate = normalize_memory_predicate(predicate)
        normalized_object = str(object_ or "").strip()
        await graph.add_triplet(
            normalized_subject,
            normalized_predicate,
            normalized_object,
            owner_id=owner_id,
            run_id=run_id,
            locus_id=locus_id
        )
        locus_note = f" (Locus: {locus_id})" if locus_id else ""
        return (
            "Successfully stored memory: "
            f"[{normalized_subject}] --[{normalized_predicate}]--> [{normalized_object}]{locus_note}"
        )
    except Exception as e:
        return f"Failed to store memory: {e}"


async def _update_memory(
    subject: Optional[str] = None,
    predicate: str = "",
    object_: str = "",
    locus_id: Optional[str] = None,
    supersede_existing: bool = True,
    **kwargs,
) -> str:
    """Store a correction-aware memory update for the current session and semantic graph."""
    try:
        graph = _build_runtime_graph(kwargs)
        owner_id = kwargs.get("_owner_id")
        run_id = kwargs.get("_run_id")
        session_id = kwargs.get("_session_id") or kwargs.get("session_id")
        normalized_subject = _normalize_memory_subject(subject)
        normalized_predicate = normalize_memory_predicate(predicate)
        normalized_object = str(object_ or "").strip()

        if session_id:
            turn = _resolve_runtime_turn(session_id, owner_id)
            if supersede_existing:
                graph.replace_temporal_facts(
                    session_id,
                    facts=[
                        {
                            "subject": normalized_subject,
                            "predicate": normalized_predicate,
                            "object": normalized_object,
                            "run_id": run_id,
                            "locus_id": locus_id,
                        }
                    ],
                    voids=[{"subject": normalized_subject, "predicate": normalized_predicate}],
                    turn=turn,
                    owner_id=owner_id,
                )
            else:
                graph.add_temporal_fact(
                    session_id,
                    normalized_subject,
                    normalized_predicate,
                    normalized_object,
                    turn,
                    owner_id=owner_id,
                    run_id=run_id,
                    locus_id=locus_id,
                )

        await graph.add_triplet(
            normalized_subject,
            normalized_predicate,
            normalized_object,
            owner_id=owner_id,
            run_id=run_id,
            locus_id=locus_id,
        )
        locus_note = f" (Locus: {locus_id})" if locus_id else ""
        lane_note = "session + semantic memory" if session_id else "semantic memory only"
        return (
            "Successfully updated memory in "
            f"{lane_note}: [{normalized_subject}] --[{normalized_predicate}]--> "
            f"[{normalized_object}]{locus_note}"
        )
    except Exception as e:
        return f"Failed to update memory: {e}"

STORE_MEMORY_TOOL = Tool(
    name="store_memory",
    description=(
        "Store a single declarative fact or preference about the user or the world into long-term semantic memory. "
        "Use this PROACTIVELY when the user tells you something important to remember for future conversations. "
        "Break the fact into a subject, a predicate (the verb/relationship), and an object. "
        "If the fact is about the user and subject is omitted, default to 'User'. "
        "Use update_memory instead when the new fact corrects or replaces an older one."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity the fact is about (e.g., 'User', 'System', 'John'). Defaults to 'User' when omitted."},
            "predicate": {"type": "string", "description": "The relationship (e.g., 'likes', 'is allergic to', 'works at')"},
            "object_": {"type": "string", "description": "The target of the relationship (e.g., 'spicy food', 'peanuts', 'Google')"},
            "locus_id": {"type": "string", "description": "Optional: Anchor this memory to a specific spatial room (Locus ID)."}
        },
        "required": ["predicate", "object_"]
    },
    handler=_store_memory,
    tags=["memory", "core"]
)

UPDATE_MEMORY_TOOL = Tool(
    name="update_memory",
    description=(
        "Update a remembered fact or preference, especially when the user is correcting earlier context. "
        "This writes to session-scoped deterministic memory and semantic memory so future retrieval prefers the new value. "
        "If the fact is about the user and subject is omitted, default to 'User'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity the fact is about. Defaults to 'User' when omitted."},
            "predicate": {"type": "string", "description": "The relationship being updated (for example 'location' or 'preferred_editor')."},
            "object_": {"type": "string", "description": "The new value that should now be treated as current."},
            "locus_id": {"type": "string", "description": "Optional: Anchor this memory to a specific spatial room (Locus ID)."},
            "supersede_existing": {"type": "boolean", "description": "When true, retire any active fact with the same subject and predicate first. Defaults to true."},
        },
        "required": ["predicate", "object_"]
    },
    handler=_update_memory,
    tags=["memory", "core"]
)

# Meta-recall queries ("what do you remember", "list everything", "*") cannot be
# answered by similarity search because they don't *describe* a topic — they ask
# for a state dump. Embedding "tell me everything you remember" pulls nothing
# because no stored fact is semantically close to that meta-question. Detect
# these and short-circuit to a deterministic dump of current facts and loci.
_META_RECALL_PATTERN = re.compile(
    r"\b(?:everything|all (?:facts?|memor\w*|things|stuff)|"
    r"what do you (?:remember|know)|"
    r"tell me (?:about )?(?:me|yourself|everything|what you know|all)|"
    r"list (?:all )?(?:memor\w*|facts?|everything))\b",
    re.IGNORECASE,
)
_META_RECALL_LITERALS = {"*", "all", "everything", "all memory", "all memories"}


def _is_meta_recall_query(topic: str) -> bool:
    if not topic:
        return False
    stripped = topic.strip().rstrip("?.! ").lower()
    if stripped in _META_RECALL_LITERALS:
        return True
    return bool(_META_RECALL_PATTERN.search(stripped))


def _dump_session_memory(graph, session_id: str, owner_id: Optional[str]) -> str:
    facts = list(graph.get_current_facts(session_id, owner_id=owner_id) or [])
    try:
        loci = list(graph.list_loci(owner_id=owner_id) or [])
    except Exception:
        loci = []

    if not facts and not loci:
        return "No memories stored for this session yet."

    lines = ["Memory dump:"]
    if facts:
        lines.append(f"  Current facts ({len(facts)}):")
        for subject, predicate, object_ in facts:
            lines.append(f"  - [{subject}] --[{predicate}]--> [{object_}]")
    if loci:
        lines.append(f"  Spatial loci ({len(loci)}):")
        for locus in loci:
            lid = locus.get("id") or locus.get("locus_id") or "?"
            name = locus.get("name") or ""
            lines.append(f"  - {lid}: {name}".rstrip(": "))
    return "\n".join(lines)


async def _query_memory(
    topic: str,
    session_id: Optional[str] = None,
    locus_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Traverse the semantic graph memory for facts related to a topic."""
    try:
        owner_id = kwargs.get("_owner_id")
        active_session_id = session_id or kwargs.get("_session_id")
        graph = _build_runtime_graph(kwargs)

        # Intent route: meta-recall questions get a deterministic dump instead
        # of a similarity search that would return nothing useful.
        if _is_meta_recall_query(topic) and active_session_id:
            return _dump_session_memory(graph, active_session_id, owner_id)

        from memory.retrieval import unified_memory_search

        payload = await unified_memory_search(
            topic,
            owner_id=owner_id,
            session_id=active_session_id,
            locus_id=locus_id,
            top_k=5,
            graph=graph,
        )
        results = payload.get("results") or []
        stats = payload.get("stats") or {}

        if not results:
            return f"No memories found related to '{topic}'."

        lines = [
            f"Unified memory results for '{topic}' ({len(results)}):",
            f"  - source counts: {stats.get('source_counts', {})}",
        ]
        for item in results:
            kind = str(item.get("kind") or "")
            score = item.get("score")
            source = ", ".join(item.get("sources") or [])
            if kind in {"fact", "triplet"}:
                lines.append(
                    "  - "
                    f"[{item.get('subject', '')}] --[{item.get('predicate', '')}]--> [{item.get('object', '')}] "
                    f"(score: {score}, source: {source})"
                )
            else:
                anchor = str(item.get("anchor") or "").replace("\n", " ").strip()
                if len(anchor) > 180:
                    anchor = anchor[:177].rstrip() + "..."
                lines.append(
                    "  - "
                    f"[{item.get('key', 'memory_anchor')}] {anchor} "
                    f"(score: {score}, source: {source})"
                )
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to query memory: {e}"

QUERY_MEMORY_TOOL = Tool(
    name="query_memory",
    description=(
        "Search the long-term semantic memory graph for stored facts and preferences. "
        "Use this when you need historical context about a specific topic, like 'dietary restrictions' or 'favorite movies'. "
        "Use this for durable remembered facts, corrections, and preferences rather than session evidence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The broad topic or specific entity to search for (e.g., 'food preferences', 'John')"},
            "locus_id": {"type": "string", "description": "Optional: Limit search to a specific spatial room (Locus ID)."}
        },
        "required": ["topic"]
    },
    handler=_query_memory,
    tags=["memory", "core"]
)


async def _todo_write(
    tasks: list[dict],
    topic: Optional[str] = None,
    _session_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create or replace the in-session task list."""
    session_id = _session_id or kwargs.get("session_id")
    if not session_id:
        return "todo_write failed: missing session context"

    tracker = get_session_task_tracker()
    
    # Case 1: tasks is a string (poor LLM formatting)
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except:
            # Maybe it's a raw string list? 
            # We'll try one more fallback if it looks like JSON but has bad delimiters
            pass

    # Case 2: tasks is a list of strings (should be list of dicts)
    if isinstance(tasks, list):
        normalized = []
        for i, t in enumerate(tasks):
            if isinstance(t, str):
                normalized.append({"id": f"task_{i}", "content": t, "status": "pending"})
            else:
                normalized.append(t)
        tasks = normalized

    return tracker.write(session_id, tasks, topic=topic)


TODO_WRITE_TOOL = Tool(
    name="todo_write",
    description=(
        "Create or replace the current session task list for multi-step work. "
        "Use this at the start of complex workflows before calling other tools. "
        "Provide the full task list and an optional workflow topic. "
        "Output is the canonical task state string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Complete task list for this session.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Stable task id"},
                        "content": {"type": "string", "description": "Action-oriented task text"},
                        "status": {
                            "type": "string",
                            "description": "Task status",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "priority": {
                            "type": "string",
                            "description": "Task priority",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["id", "content"],
                },
            },
            "topic": {
                "type": "string",
                "description": "Short workflow/topic label for the active task set.",
            },
        },
        "required": ["tasks"],
    },
    handler=_todo_write,
    tags=["tasking", "planning"],
)


async def _todo_update(task_id: str, status: str, _session_id: Optional[str] = None, **kwargs) -> str:
    """Update one task status in the current session."""
    session_id = _session_id or kwargs.get("session_id")
    if not session_id:
        return "todo_update failed: missing session context"

    tracker = get_session_task_tracker()
    return tracker.update(session_id, task_id, status)


TODO_UPDATE_TOOL = Tool(
    name="todo_update",
    description=(
        "Update the status of a single task in the current session task list. "
        "Use immediately when a task starts or completes so the workflow state stays accurate. "
        "Output is the updated canonical task state string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id to update"},
            "status": {
                "type": "string",
                "description": "New status value",
                "enum": ["pending", "in_progress", "completed"],
            },
        },
        "required": ["task_id", "status"],
    },
    handler=_todo_update,
    tags=["tasking", "planning"],
)


async def _todo_read(_session_id: Optional[str] = None, **kwargs) -> str:
    """Return the current session task list state."""
    session_id = _session_id or kwargs.get("session_id")
    if not session_id:
        return "todo_read failed: missing session context"
    tracker = get_session_task_tracker()
    if not tracker.has_tasks(session_id):
        return "No active task list for this session."
    return tracker.render(session_id)


TODO_READ_TOOL = Tool(
    name="todo_read",
    description=(
        "Read the current session task list. "
        "Call this to review pending and in-progress tasks before deciding the next step. "
        "Use after todo_write to check what still needs doing, and before todo_update to confirm the task id."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    handler=_todo_read,
    tags=["tasking", "planning"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  HTML RENDERING  —  Isolated Sandboxed Viewer
# ══════════════════════════════════════════════════════════════════════════════

async def _generate_app(html_content: str, title: str = "Standalone App") -> str:
    """
    Saves HTML content to a unique file and injects a premium 'V8 Platinum' theme.
    Returns metadata for the frontend to render an isolated iframe.
    """
    import hashlib
    import time
    
    # Generate a clean filename based on title
    safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title).lower().strip("_")[:30]
    filename = f"{safe_title}.html" if safe_title else "app.html"
    file_path = SANDBOX_DIR / filename
    
    # Inject Premium V8 Platinum Theme if not already present
    if "<!DOCTYPE html>" not in html_content:
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{ --bg: #000; --text: #fff; --primary: #00ff85; }}
        body {{ 
            background: #000; color: var(--text); font-family: 'Inter', sans-serif; margin: 0; min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }}
        .glass {{ background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-thumb {{ background: #222; border-radius: 10px; }}
        .v8-card {{ background: #050505; border: 1px solid #111; border-radius: 12px; transition: all 0.3s ease; }}
        .v8-card:hover {{ border-color: var(--primary); box-shadow: 0 0 20px rgba(0,255,133,0.1); }}
    </style>
</head>
<body class="p-6">
    {html_content}
    <script>lucide.createIcons();</script>
</body>
</html>"""
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return json.dumps({
        "status": "success",
        "type": "app_view",
        "title": title,
        "filename": filename,
        "path": f"/sandbox/{filename}"
    })

GENERATE_APP_TOOL = Tool(
    name="generate_app",
    description=(
        "Generate a standalone, interactive HTML application or dashboard. "
        "MUST follow the 'V8 Platinum Standard' (Production-grade, True Black #000). "
        "Requires modern typography (e.g. Outfit, Syne), glassmorphism, and SPA-style internal routing. "
        "Apps MUST feel alive with micro-animations and staggered reveals."
    ),
    parameters={
        "type": "object",
        "properties": {
            "html_content": {"type": "string", "description": "The HTML body. CRITICAL: Escape all double quotes (\") as \\\" and use 'bg-black' for backgrounds."},
            "title": {"type": "string", "description": "The application title."}
        },
        "required": ["html_content"]
    },
    handler=_generate_app,
    tags=["visual", "app"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  Registration helper
# ══════════════════════════════════════════════════════════════════════════════

async def _pdf_processor(
    action: str,
    path: Optional[str] = None,
    output_path: Optional[str] = None,
    content: Optional[str] = None,
    paths: Optional[list[str]] = None,
    pages: Optional[list[int]] = None,
    rotation: Optional[int] = None,
) -> str:
    """
    Unified PDF processor tool handler.
    Actions: read, create, merge, split, rotate, metadata.
    """
    try:
        def _pdf_preview_payload(pdf_path: str, action_name: str, **extra) -> str:
            rel = str(Path(pdf_path).as_posix()).lstrip("/")
            payload = {
                "status": "success",
                "action": action_name,
                "type": "pdf_preview",
                "file": rel,
                "path": f"/sandbox/{rel}",
                "url": f"/sandbox/{rel}",
                "title": Path(rel).name,
            }
            payload.update(extra)
            return json.dumps(payload)

        if action == "create":
            if not output_path or not content:
                return "Error: 'output_path' and 'content' required for create."
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
            except ImportError:
                return "Error: 'reportlab' library missing. Run 'pip install reportlab'."
            
            target = _safe_path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            c = canvas.Canvas(str(target), pagesize=letter)
            width, height = letter
            text_lines = content.split('\n')
            y = height - 100
            for line in text_lines:
                c.drawString(100, y, line)
                y -= 15
            c.save()
            return _pdf_preview_payload(output_path, "create")

        if action == "read":
            if not path: return "Error: 'path' required for read."
            target = _safe_path(path)
            if not target.exists(): return f"Error: {path} not found."
            
            try:
                import pdfplumber
                with pdfplumber.open(target) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return _pdf_preview_payload(
                        path,
                        "read",
                        content=text,
                        pages=len(pdf.pages),
                    )
            except ImportError:
                # Fallback to pypdf
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(target)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    return _pdf_preview_payload(
                        path,
                        "read",
                        content=text,
                        pages=len(reader.pages),
                        note="using pypdf fallback",
                    )
                except ImportError:
                    return "Error: 'pypdf' or 'pdfplumber' missing."

        if action == "merge":
            if not paths or not output_path: return "Error: 'paths' and 'output_path' required for merge."
            try:
                from pypdf import PdfWriter, PdfReader
                writer = PdfWriter()
                for p in paths:
                    reader = PdfReader(_safe_path(p))
                    for page in reader.pages:
                        writer.add_page(page)
                target = _safe_path(output_path)
                with open(target, "wb") as f:
                    writer.write(f)
                return _pdf_preview_payload(output_path, "merge")
            except ImportError:
                return "Error: 'pypdf' library missing."

        if action == "split":
            if not path: return "Error: 'path' required for split."
            try:
                from pypdf import PdfWriter, PdfReader
                reader = PdfReader(_safe_path(path))
                for i, page in enumerate(reader.pages):
                    writer = PdfWriter()
                    writer.add_page(page)
                    out_name = f"{Path(path).stem}_page_{i+1}.pdf"
                    target = _safe_path(out_name)
                    with open(target, "wb") as f:
                        writer.write(f)
                return json.dumps({"status": "success", "action": "split", "pages": len(reader.pages)})
            except ImportError:
                return "Error: 'pypdf' library missing."

        return f"Error: Unknown action '{action}'."

    except Exception as e:
        return f"Error during PDF operation: {e}"

PDF_PROCESSOR_TOOL = Tool(
    name="pdf_processor",
    description=(
        "Advanced PDF toolkit. Actions: 'read' (extract text), 'create' (basic PDF), "
        "'merge' (multiple files), 'split' (one file per page), 'rotate', 'metadata'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "create", "merge", "split", "rotate", "metadata"]},
            "path": {"type": "string", "description": "Target PDF path in sandbox."},
            "output_path": {"type": "string", "description": "Output path for create/merge/split."},
            "content": {"type": "string", "description": "Text content for 'create'."},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "List of PDF paths to merge."},
            "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to extract/split."},
            "rotation": {"type": "integer", "description": "Degrees to rotate (90, 180, 270)."}
        },
        "required": ["action"]
    },
    handler=_pdf_processor,
    tags=["file", "pdf"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  DELEGATION  —  Agentic Tool for orchestrating other agents
# ══════════════════════════════════════════════════════════════════════════════

async def _delegate_to_agent(target_agent_id: Optional[str] = None, task: Optional[str] = None, **kwargs) -> str:
    """Handler for the delegate_to_agent tool."""
    if not _agent_manager:
        return "Error: Agent Manager not initialized for delegation."
    
    try:
        # Compatibility layer:
        # Some models emit legacy shapes like {"agent": "...", "function": "...", "args": [...]}
        # even when instructed to use {"target_agent_id", "task"}.
        target_agent_id = (
            target_agent_id
            or kwargs.get("agent")
            or kwargs.get("agent_id")
            or kwargs.get("target")
            or "default"
        )

        if not task:
            task = kwargs.get("task")

        legacy_function = kwargs.get("function")
        legacy_args = kwargs.get("args")
        if not task and isinstance(legacy_function, str):
            if legacy_function in {"write_report", "create_report", "create_file", "file_create"} and isinstance(legacy_args, (list, tuple)) and legacy_args:
                out_path = str(legacy_args[0])
                report_body = str(legacy_args[1]) if len(legacy_args) > 1 else ""
                task = (
                    f"Create a file called '{out_path}' in the sandbox. "
                    "Use the file_create tool immediately. "
                    f"Write this content:\n{report_body}"
                )
            elif legacy_args is not None:
                task = f"Execute function '{legacy_function}' with args: {json.dumps(legacy_args)}"
            else:
                task = f"Execute function '{legacy_function}'."

        if not task or not str(task).strip():
            return (
                "Delegation error: missing task. "
                "Provide either {'target_agent_id','task'} or legacy {'agent','function','args'}."
            )

        parent_id = kwargs.get("_session_id")
        log("agent", "system", f"Delegating task to '{target_agent_id}': {str(task)[:50]}...", level="info")
        parent_run_id = kwargs.get("_run_id")
        owner_id = kwargs.get("_owner_id")
        runtime_kind_override = str(kwargs.get("_delegate_runtime_kind") or "").strip().lower() or "managed"
        delegate_kwargs = {
            "parent_id": parent_id,
            "parent_run_id": parent_run_id,
            "owner_id": owner_id,
            "runtime_kind_override": runtime_kind_override,
        }
        result = await _agent_manager.run_agent_task(target_agent_id, task, **delegate_kwargs)
        return result
    except Exception as e:
        return f"Delegation error: {e}"

DELEGATE_TO_AGENT_TOOL = Tool(
    name="delegate_to_agent",
    description=(
        "Delegate a specific task to another specialized agent. "
        "Use this for research, coding, or complex logic that requires a different personality or mindset."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_agent_id": {
                "type": "string", 
                "description": "ID of the agent to delegate to (e.g., 'coder', 'researcher', 'default')."
            },
            "task": {
                "type": "string",
                "description": "Detailed description of what the sub-agent should do."
            },
            "agent": {
                "type": "string",
                "description": "Legacy alias for target_agent_id."
            },
            "function": {
                "type": "string",
                "description": "Legacy function-style delegation request."
            },
            "args": {
                "type": "array",
                "description": "Legacy function arguments for compatibility."
            },
        },
        "required": []
    },
    handler=_delegate_to_agent,
    tags=["agentic", "system"]
)

async def _rag_search(query: str, top_k: int = 5, **kwargs) -> str:
    """Search everything retrieved in this conversation session."""
    session_id = kwargs.get("_session_id")
    if not session_id:
        return "rag_search requires a session context."
    from memory.session_rag import get_session_rag
    rag = get_session_rag(session_id, owner_id=kwargs.get("_owner_id"))
    results = await rag.query(query, top_k=top_k)
    return rag.format_results_for_llm(results, query)

RAG_SEARCH_TOOL = Tool(
    name="rag_search",
    description=(
        "Search everything retrieved earlier in this conversation — web pages fetched, "
        "files created, tool results, uploaded documents. Use this for session evidence recall "
        "before web_search to avoid redundant fetches. Do not use it for durable user preference memory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in session memory",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
    handler=_rag_search,
    tags=["memory", "rag"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  Registration helper
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    WEB_FETCH_TOOL,
    SOURCE_CONTRACT_TOOL,
    SOURCE_SELECT_TOOL,
    SOURCE_NEXT_ACTION_TOOL,
    SHOPPING_ADVICE_TOOL,
    IMAGE_SEARCH_TOOL,
    BASH_TOOL,
    FILE_CREATE_TOOL,
    FILE_VIEW_TOOL,
    FILE_STR_REPLACE_TOOL,
    WEATHER_TOOL,
    PLACES_SEARCH_TOOL,   # BUG FIX: was missing from ALL_TOOLS — caused 'not found in global registry' warning
    PLACES_MAP_TOOL,
    STORE_MEMORY_TOOL,
    UPDATE_MEMORY_TOOL,
    QUERY_MEMORY_TOOL,
    TODO_WRITE_TOOL,
    TODO_UPDATE_TOOL,
    TODO_READ_TOOL,
    GENERATE_APP_TOOL,
    PDF_PROCESSOR_TOOL,
    RAG_SEARCH_TOOL,
    DELEGATE_TO_AGENT_TOOL,
]

def register_all_tools(registry: ToolRegistry, agent_manager: Optional[AgentManager] = None) -> None:
    """Register every built-in tool. Call this in main.py after creating tool_registry."""
    global _agent_manager
    if agent_manager:
        _agent_manager = agent_manager

    for tool in ALL_TOOLS:
        registry.register(tool)
    print(f"[tools] Registered {len(ALL_TOOLS)} built-in tools")


def register_tools(registry: ToolRegistry, *names: str) -> None:
    """Register a specific subset of tools by name."""
    tool_map = {t.name: t for t in ALL_TOOLS}
    for name in names:
        if name in tool_map:
            registry.register(tool_map[name])
        else:
            print(f"[tools] WARNING: unknown tool '{name}' — skipped")
