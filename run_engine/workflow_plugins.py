from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from plugins.tools_web import _validate_fetch_url
from run_engine.search_query import compile_web_search_query


_URL_RE = re.compile(r"https?://[^\s\"'<>),\]]+", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "TODAY", "STOCK",
    "STOCKS", "NYSE", "NASDAQ", "AMEX", "ETF", "ETFS", "CEO", "CFO", "SEC",
    "US", "USA", "USD", "AI", "API", "URL", "JSON", "HTTP", "HTTPS", "TTM",
}


@dataclass(frozen=True)
class WorkflowPluginContract:
    workflow_shape: str
    preferred_policy: str
    entity_lock_rules: dict[str, Any]
    source_requirements: dict[str, Any]
    allowed_tools_by_phase: dict[str, list[str]]
    completion_evaluator: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_shape": self.workflow_shape,
            "preferred_policy": self.preferred_policy,
            "entity_lock_rules": dict(self.entity_lock_rules),
            "source_requirements": dict(self.source_requirements),
            "allowed_tools_by_phase": {key: list(value) for key, value in self.allowed_tools_by_phase.items()},
            "completion_evaluator": self.completion_evaluator,
        }


@dataclass(frozen=True)
class WorkflowPlugin:
    id: str
    label: str
    description: str
    detect: Any
    override: Any
    contract: WorkflowPluginContract


def _safe_json_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def tool_argument_value(item: dict[str, Any], key: str) -> str:
    args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
    return str(args.get(key) or "").strip()


def valid_fetch_url(url: str) -> str:
    ok, _error, normalized, _host = _validate_fetch_url(str(url or "").strip().rstrip(".,"))
    return normalized if ok else ""


def source_collection_contract_from_objective(objective: str) -> dict[str, int]:
    text = str(objective or "").lower()
    if not ("fetch" in text and ("url" in text or "source" in text or "result" in text)):
        return {}
    if not ("separately" in text or "each" in text or "per " in text):
        return {}

    entity_count = 0
    top_match = re.search(r"\btop\s+(\d+)\b", text)
    if top_match:
        entity_count = int(top_match.group(1))
    elif re.search(r"\bthree\b", text):
        entity_count = 3
    entity_match = re.search(r"\b(\d+)\s+(?:stocks|companies|products|tools|items|entities)\b", text)
    if entity_match:
        entity_count = int(entity_match.group(1))

    per_entity = 0
    per_match = re.search(
        r"\b(\d+)\s+(?:relevant\s+)?(?:results?|sources?|urls?|links?|articles?)\s+(?:for\s+)?each\b",
        text,
    )
    if per_match:
        per_entity = int(per_match.group(1))
    elif re.search(r"\bthree\s+(?:relevant\s+)?(?:results?|sources?|urls?|links?|articles?)\s+(?:for\s+)?each\b", text):
        per_entity = 3

    total_urls = 0
    total_match = re.search(r"\b(?:all\s+)?(\d+)\s+urls?\b", text)
    if total_match:
        total_urls = int(total_match.group(1))

    if entity_count <= 0 and total_urls and per_entity:
        entity_count = max(1, total_urls // per_entity)
    if per_entity <= 0 and total_urls and entity_count:
        per_entity = max(1, total_urls // entity_count)
    if entity_count and per_entity and (
        total_urls <= 0
        or (
            total_urls == per_entity
            and bool(re.search(r"\b(each|per|separately)\b", text))
            and not re.search(r"\ball\s+\d+\s+urls?\b", text)
        )
    ):
        total_urls = entity_count * per_entity

    if entity_count <= 0 or per_entity <= 0:
        return {}
    return {
        "entity_count": min(entity_count, 12),
        "urls_per_entity": min(per_entity, 10),
        "total_urls": min(total_urls or entity_count * per_entity, 60),
    }


def default_tool_turn_budget(objective: str, requested_max_turns: Optional[int]) -> int:
    if requested_max_turns is not None:
        return max(1, min(int(requested_max_turns), 24))
    source_contract = source_collection_contract_from_objective(objective)
    if source_contract:
        return min(24, 3 + source_contract["entity_count"] + source_contract["total_urls"])
    return 3


def extract_urls_from_tool_results(tool_results: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for item in tool_results:
        if not item.get("success"):
            continue
        content = str(item.get("content") or "")
        payload = _safe_json_payload(content)
        candidates: list[str] = []
        extracted_urls = item.get("extracted_urls")
        if isinstance(extracted_urls, list):
            candidates.extend(str(url) for url in extracted_urls)
        if payload:
            if payload.get("url"):
                candidates.append(str(payload.get("url")))
            results = payload.get("results") or payload.get("organic_results") or []
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        candidates.append(str(result.get("url") or result.get("link") or ""))
        candidates.extend(match.group(0).rstrip(".,") for match in _URL_RE.finditer(content))
        for url in candidates:
            clean = valid_fetch_url(url)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            urls.append(clean)
            if len(urls) >= limit:
                return urls
    return urls


def explicit_stock_source_workflow_requested(objective: str) -> bool:
    text = str(objective or "").lower()
    return (
        "stock" in text
        and ("top 3" in text or "three" in text or "3 stocks" in text)
        and ("separately" in text or "each" in text)
        and ("web fetch" in text or "fetch" in text)
        and ("9" in text or "3 relevant" in text or "three relevant" in text)
    )


def extract_mover_tickers_from_fetched_pages(tool_results: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    movers_url_markers = (
        "morningstar.com/markets/movers",
        "tradingview.com/markets/stocks-usa/market-movers-gainers",
        "finance.yahoo.com/markets/stocks/gainers",
        "stockanalysis.com/markets/gainers",
    )
    for item in tool_results:
        if str(item.get("tool_name") or "") != "web_fetch" or not item.get("success"):
            continue
        fetched_url = valid_fetch_url(tool_argument_value(item, "url"))
        content = str(item.get("content") or "")
        payload = _safe_json_payload(content)
        if payload and payload.get("content"):
            content = str(payload.get("content") or "")
        looks_like_movers_url = any(marker in fetched_url.lower() for marker in movers_url_markers)
        if not looks_like_movers_url and not re.search(
            r"\b(market movers|top stock market gainers|top gaining|top stock gainers|gainers)\b",
            content,
            re.IGNORECASE,
        ):
            continue
        gainers_block = content
        gainers_match = re.search(r"##\s*Gainers(?P<body>.*?)(?:##\s*Losers|##\s*Actives|$)", content, re.IGNORECASE | re.DOTALL)
        if gainers_match:
            gainers_block = gainers_match.group("body")
        lines = gainers_block.splitlines()
        for index, line in enumerate(lines):
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            line_has_change = "+" in line or "%" in line
            next_has_change = "+" in next_line or "%" in next_line
            if not line_has_change and not next_has_change:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")] if "|" in line else [line]
            stock_cell = cells[1] if len(cells) > 1 else line
            if re.search(r"\b(symbol|chg|price|vol|mkt cap|analyst rating|sector)\b", stock_cell, re.IGNORECASE):
                continue
            candidates = [match.group(0).upper() for match in _TICKER_RE.finditer(stock_cell)]
            candidates = [ticker for ticker in candidates if ticker not in _TICKER_STOPWORDS]
            if not candidates:
                candidates = [match.group(1).upper() for match in re.finditer(r"/stocks/[^/)]+/([a-z]{1,5})/quote", stock_cell, re.IGNORECASE)]
            if not candidates:
                candidates = [
                    match.group(1).upper()
                    for match in re.finditer(r"\b([A-Z]{2,5})(?=[A-Z][A-Za-z][A-Za-z])", stock_cell)
                ]
            for ticker in reversed(candidates):
                if ticker in _TICKER_STOPWORDS or ticker in seen:
                    continue
                seen.add(ticker)
                tickers.append(ticker)
                if len(tickers) >= limit:
                    return tickers
                break
    return tickers


def select_movers_source_url(urls: list[str]) -> str:
    if not urls:
        return ""
    priority_patterns = (
        "morningstar.com/markets/movers",
        "tradingview.com/markets/stocks-usa/market-movers-gainers",
        "finance.yahoo.com/markets/stocks/gainers",
        "stockanalysis.com/markets/gainers",
        "marketwatch.com/tools/stockresearch/marketmap",
        "bankrate.com/investing/best-performing-stocks",
        "nasdaq.com/market-activity/stocks/screener",
    )
    lowered = [(url, url.lower()) for url in urls]
    for pattern in priority_patterns:
        for url, low in lowered:
            if pattern in low:
                return url
    for url, low in lowered:
        if any(term in low for term in ("movers", "gainers", "best-performing-stocks")):
            return url
    return ""


def _remaining_movers_source_url(urls: list[str], fetched_urls: set[str]) -> str:
    return select_movers_source_url([url for url in urls if url and url not in fetched_urls])


def extract_stock_tickers_from_tool_results(tool_results: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    mover_tickers = extract_mover_tickers_from_fetched_pages(tool_results, limit=limit)
    if len(mover_tickers) >= limit:
        return mover_tickers[:limit]

    counts: dict[str, int] = {}
    for item in tool_results:
        if str(item.get("tool_name") or "") != "web_search" or not item.get("success"):
            continue
        content = str(item.get("content") or "")
        payload = _safe_json_payload(content)
        texts: list[str] = []
        results = payload.get("results") or payload.get("organic_results") or []
        if isinstance(results, list):
            for result in results[:10]:
                if not isinstance(result, dict):
                    continue
                texts.extend([
                    str(result.get("title") or ""),
                    str(result.get("snippet") or result.get("description") or ""),
                ])
        texts.append(content[:3000])
        for text in texts:
            for match in _TICKER_RE.finditer(text):
                ticker = match.group(0).upper()
                if ticker in _TICKER_STOPWORDS:
                    continue
                counts[ticker] = counts.get(ticker, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [ticker for ticker, _ in ranked[:limit]]


def urls_by_entity_from_search_results(
    *,
    tool_results: list[dict[str, Any]],
    entities: list[str],
    per_entity: int,
) -> dict[str, list[str]]:
    by_entity: dict[str, list[str]] = {entity: [] for entity in entities}
    seen: set[str] = set()
    for item in tool_results:
        if str(item.get("tool_name") or "") != "web_search" or not item.get("success"):
            continue
        query = tool_argument_value(item, "query").upper()
        entity = next((candidate for candidate in entities if re.search(rf"\b{re.escape(candidate)}\b", query)), "")
        if not entity:
            continue
        extracted_urls = item.get("extracted_urls")
        if isinstance(extracted_urls, list):
            for extracted_url in extracted_urls:
                url = valid_fetch_url(str(extracted_url or "").strip().rstrip(".,"))
                if not url or url in seen:
                    continue
                seen.add(url)
                by_entity[entity].append(url)
                if len(by_entity[entity]) >= per_entity:
                    break
            if len(by_entity[entity]) >= per_entity:
                continue
        payload = _safe_json_payload(str(item.get("content") or ""))
        results = payload.get("results") or payload.get("organic_results") or []
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            url = valid_fetch_url(str(result.get("url") or result.get("link") or "").strip().rstrip(".,"))
            if not url or url in seen:
                continue
            seen.add(url)
            by_entity[entity].append(url)
            if len(by_entity[entity]) >= per_entity:
                break
    return by_entity


def stock_source_workflow_override(
    *,
    objective: str,
    allowed_tools: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not explicit_stock_source_workflow_requested(objective):
        return {}

    contract = source_collection_contract_from_objective(objective) or {
        "entity_count": 3,
        "urls_per_entity": 3,
        "total_urls": 9,
    }
    entity_count = int(contract["entity_count"])
    urls_per_entity = int(contract["urls_per_entity"])
    total_urls = int(contract["total_urls"])

    allowed = {str(item.get("name") or "") for item in allowed_tools if isinstance(item, dict)}
    tickers = extract_stock_tickers_from_tool_results(tool_results, limit=entity_count)
    targeted_queries = [
        tool_argument_value(item, "query").upper()
        for item in tool_results
        if str(item.get("tool_name") or "") == "web_search" and item.get("success")
    ]
    searched_tickers = {
        ticker
        for ticker in tickers
        if any(re.search(rf"\b{re.escape(ticker)}\b", query) for query in targeted_queries)
    }
    fetched_urls = {
        url
        for item in tool_results
        for url in [valid_fetch_url(tool_argument_value(item, "url"))]
        if str(item.get("tool_name") or "") == "web_fetch" and item.get("success")
        if url
    }
    urls_by_entity = urls_by_entity_from_search_results(
        tool_results=tool_results,
        entities=tickers,
        per_entity=urls_per_entity,
    )
    allowed_fetch_urls_by_entity = {
        ticker: urls_by_entity.get(ticker, [])[:urls_per_entity]
        for ticker in tickers
        if urls_by_entity.get(ticker)
    }
    entity_urls = [
        url
        for ticker in tickers
        for url in urls_by_entity.get(ticker, [])[:urls_per_entity]
    ]
    urls = extract_urls_from_tool_results(tool_results, limit=max(12, total_urls + 3))
    unfetched_urls = [url for url in entity_urls if url not in fetched_urls]
    fetched_entity_urls = {url for url in fetched_urls if url in set(entity_urls)}
    mover_source_fetched = len(extract_mover_tickers_from_fetched_pages(tool_results, limit=entity_count)) >= entity_count

    if not mover_source_fetched:
        movers_url = select_movers_source_url(unfetched_urls)
        if not movers_url:
            movers_url = _remaining_movers_source_url(urls, fetched_urls)
        if movers_url and "web_fetch" in allowed:
            return {
                "status": "partial",
                "selected_tools": ["web_fetch"],
                "strategy": "Fetch the next market movers source before locking ticker entities.",
                "notes": "The prior movers source did not yield a locked top-three table, so the workflow must pivot instead of refetching it.",
                "missing_slots": ["verified_top_3_mover_tickers"],
                "argument_clues": {"web_fetch": movers_url},
                "plugin_id": "stock_movers_source_collection",
            }
        return {
            "status": "partial",
            "selected_tools": ["web_search"] if "web_search" in allowed else [],
            "strategy": "Find a market movers source before source collection.",
            "notes": "The explicit stock workflow needs a source-backed top-three gainers table before ticker-specific searches.",
            "missing_slots": ["market_movers_source"],
            "argument_clues": {"web_search": "Morningstar market movers top stock gainers today"},
            "plugin_id": "stock_movers_source_collection",
        }

    for ticker in tickers:
        if ticker not in searched_tickers and "web_search" in allowed:
            return {
                "status": "partial",
                "selected_tools": ["web_search"],
                "strategy": f"Search {ticker} separately before fetching sources.",
                "notes": "The user explicitly asked for separate searches for each stock.",
                "missing_slots": [f"{ticker}_three_relevant_urls"],
                "argument_clues": {"web_search": f"{ticker} stock news June 13 2026"},
                "tickers": tickers,
                "entities": tickers,
                "source_contract": contract,
                "allowed_fetch_urls_by_entity": allowed_fetch_urls_by_entity,
                "plugin_id": "stock_movers_source_collection",
            }

    for ticker in tickers:
        if len(urls_by_entity.get(ticker, [])) < urls_per_entity and "web_search" in allowed:
            return {
                "status": "partial",
                "selected_tools": ["web_search"],
                "strategy": f"Collect source URLs for {ticker}.",
                "notes": "Each locked entity needs its own source quota before unrelated URLs can be fetched.",
                "missing_slots": [f"{ticker}_{urls_per_entity}_urls"],
                "argument_clues": {"web_search": f"{ticker} stock news June 13 2026"},
                "tickers": tickers,
                "entities": tickers,
                "source_contract": contract,
                "allowed_fetch_urls_by_entity": allowed_fetch_urls_by_entity,
                "plugin_id": "stock_movers_source_collection",
            }

    if len(fetched_entity_urls) < total_urls:
        if unfetched_urls and "web_fetch" in allowed:
            next_url = unfetched_urls[0]
            return {
                "status": "partial",
                "selected_tools": ["web_fetch"],
                "strategy": f"Fetch source {len(fetched_entity_urls) + 1} of {total_urls}.",
                "notes": "The user requested all entity-specific URLs to be fetched before the TLDR table.",
                "missing_slots": [f"{total_urls - len(fetched_entity_urls)}_remaining_web_fetches"],
                "argument_clues": {"web_fetch": next_url},
                "tickers": tickers,
                "entities": tickers,
                "fetched_url_count": len(fetched_entity_urls),
                "source_contract": contract,
                "allowed_fetch_urls_by_entity": allowed_fetch_urls_by_entity,
                "plugin_id": "stock_movers_source_collection",
            }
        if "web_search" in allowed and tickers:
            next_ticker = tickers[min(len(searched_tickers), len(tickers) - 1)]
            return {
                "status": "partial",
                "selected_tools": ["web_search"],
                "strategy": f"Collect more URLs for {next_ticker}.",
                "notes": "Need more unique URLs before the source fetches can complete.",
                "missing_slots": ["insufficient_unique_urls"],
                "argument_clues": {"web_search": f"{next_ticker} stock news June 13 2026"},
                "tickers": tickers,
                "entities": tickers,
                "source_contract": contract,
                "allowed_fetch_urls_by_entity": allowed_fetch_urls_by_entity,
                "plugin_id": "stock_movers_source_collection",
            }

    return {}


def local_source_workflow_requested(objective: str) -> bool:
    text = str(objective or "").lower()
    if not source_collection_contract_from_objective(objective):
        return False
    return any(term in text for term in ("place", "places", "restaurant", "restaurants", "sushi", "coffee", "bar", "hotel", "near me", "in toronto"))


def comparison_source_workflow_requested(objective: str) -> bool:
    text = str(objective or "").lower()
    if not source_collection_contract_from_objective(objective):
        return False
    return any(term in text for term in ("compare", "comparison", "prices", "price", "reviews", "ratings", "stores", "products", "shopping"))


def _extract_entities_from_search_titles(tool_results: list[dict[str, Any]], *, limit: int) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for item in tool_results:
        if str(item.get("tool_name") or "") != "web_search" or not item.get("success"):
            continue
        payload = _safe_json_payload(str(item.get("content") or ""))
        results = payload.get("results") or payload.get("organic_results") or []
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "").strip()
            title = re.split(r"\s[-|–—]\s|\s·\s|:", title, maxsplit=1)[0].strip()
            title = re.sub(r"^\d+[.)]\s*", "", title).strip()
            if not title or len(title) > 80:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(title)
            if len(entities) >= limit:
                return entities
    return entities


def generic_source_collection_override(
    *,
    objective: str,
    allowed_tools: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = source_collection_contract_from_objective(objective)
    if not contract:
        return {}
    allowed = {str(item.get("name") or "") for item in allowed_tools if isinstance(item, dict)}
    entity_count = int(contract["entity_count"])
    urls_per_entity = int(contract["urls_per_entity"])
    total_urls = int(contract["total_urls"])
    entities = _extract_entities_from_search_titles(tool_results, limit=entity_count)

    successful_searches = [
        item for item in tool_results
        if str(item.get("tool_name") or "") == "web_search" and item.get("success")
    ]
    if len(entities) < entity_count:
        if "web_search" not in allowed:
            return {}
        return {
            "status": "partial",
            "selected_tools": ["web_search"],
            "strategy": "Search once to identify the entities before per-entity source collection.",
            "notes": "The source workflow needs locked entities before separate URL collection.",
            "missing_slots": ["locked_entities"],
            "argument_clues": {"web_search": compile_web_search_query(objective)},
            "source_contract": contract,
        }

    urls_by_entity = urls_by_entity_from_search_results(
        tool_results=tool_results,
        entities=entities,
        per_entity=urls_per_entity,
    )
    searched_entities = {
        entity
        for entity in entities
        for result in successful_searches
        if re.search(rf"\b{re.escape(entity)}\b", tool_argument_value(result, "query"), re.IGNORECASE)
    }
    for entity in entities:
        if entity not in searched_entities and "web_search" in allowed:
            return {
                "status": "partial",
                "selected_tools": ["web_search"],
                "strategy": f"Search {entity} separately before fetching sources.",
                "notes": "Each locked entity needs its own targeted source search.",
                "missing_slots": [f"{entity}_targeted_search"],
                "argument_clues": {"web_search": f"{entity} reviews sources"},
                "entities": entities,
                "source_contract": contract,
            }
        if len(urls_by_entity.get(entity, [])) < urls_per_entity and "web_search" in allowed:
            return {
                "status": "partial",
                "selected_tools": ["web_search"],
                "strategy": f"Collect more URLs for {entity}.",
                "notes": "Entity-specific URL quota is incomplete.",
                "missing_slots": [f"{entity}_{urls_per_entity}_urls"],
                "argument_clues": {"web_search": f"{entity} reviews sources"},
                "entities": entities,
                "source_contract": contract,
            }

    fetched_urls = {
        valid_fetch_url(tool_argument_value(item, "url"))
        for item in tool_results
        if str(item.get("tool_name") or "") == "web_fetch" and item.get("success")
    }
    entity_urls = [
        url
        for entity in entities
        for url in urls_by_entity.get(entity, [])[:urls_per_entity]
    ]
    unfetched = [url for url in entity_urls if url and url not in fetched_urls]
    if len([url for url in entity_urls if url in fetched_urls]) < total_urls and unfetched and "web_fetch" in allowed:
        return {
            "status": "partial",
            "selected_tools": ["web_fetch"],
            "strategy": f"Fetch source {total_urls - len(unfetched) + 1} of {total_urls}.",
            "notes": "Fetch locked entity URLs before synthesizing.",
            "missing_slots": [f"{len(unfetched)}_remaining_web_fetches"],
            "argument_clues": {"web_fetch": unfetched[0]},
            "entities": entities,
            "source_contract": contract,
            "allowed_fetch_urls_by_entity": urls_by_entity,
        }

    return {}


WORKFLOW_PLUGINS = [
    WorkflowPlugin(
        id="stock_movers_source_collection",
        label="Stock Movers Source Collection",
        description="Lock top stock movers from a movers source, then collect and fetch per-ticker sources.",
        detect=explicit_stock_source_workflow_requested,
        override=stock_source_workflow_override,
        contract=WorkflowPluginContract(
            workflow_shape="source_collection",
            preferred_policy="plan_execute",
            entity_lock_rules={"source": "market_movers_table", "forbid_unlocked_entity_drift": True},
            source_requirements={"separate_queries": True, "fetch_selected_urls": True},
            allowed_tools_by_phase={"discover": ["web_search", "web_fetch"], "collect": ["web_search"], "fetch": ["web_fetch"]},
            completion_evaluator="source_collection_scenario",
        ),
    ),
    WorkflowPlugin(
        id="local_place_source_collection",
        label="Local Place Source Collection",
        description="Lock local places, search each separately, then fetch per-place sources.",
        detect=local_source_workflow_requested,
        override=generic_source_collection_override,
        contract=WorkflowPluginContract(
            workflow_shape="source_collection",
            preferred_policy="plan_execute",
            entity_lock_rules={"source": "search_results", "forbid_unlocked_entity_drift": True},
            source_requirements={"separate_queries": True, "fetch_selected_urls": True},
            allowed_tools_by_phase={"discover": ["web_search"], "collect": ["web_search"], "fetch": ["web_fetch"]},
            completion_evaluator="source_collection_scenario",
        ),
    ),
    WorkflowPlugin(
        id="comparison_source_collection",
        label="Comparison Source Collection",
        description="Lock compared entities or products, gather per-entity source URLs, then fetch evidence for a table.",
        detect=comparison_source_workflow_requested,
        override=generic_source_collection_override,
        contract=WorkflowPluginContract(
            workflow_shape="source_collection",
            preferred_policy="plan_execute",
            entity_lock_rules={"source": "search_results", "forbid_unlocked_entity_drift": True},
            source_requirements={"separate_queries": True, "fetch_selected_urls": True, "answer_from_fetched_sources_only": True},
            allowed_tools_by_phase={"discover": ["web_search"], "collect": ["web_search"], "fetch": ["web_fetch"]},
            completion_evaluator="source_collection_scenario",
        ),
    )
]


def select_workflow_plugin_contract(objective: str) -> dict[str, Any]:
    for plugin in WORKFLOW_PLUGINS:
        if plugin.detect(objective):
            return {"plugin_id": plugin.id, **plugin.contract.to_dict()}
    return {}


def select_workflow_override(
    *,
    objective: str,
    allowed_tools: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    for plugin in WORKFLOW_PLUGINS:
        if not plugin.detect(objective):
            continue
        override = plugin.override(
            objective=objective,
            allowed_tools=allowed_tools,
            tool_results=tool_results,
        )
        if override:
            return {
                "plugin_id": plugin.id,
                "plugin_contract": plugin.contract.to_dict(),
                **override,
            }
    return {}
