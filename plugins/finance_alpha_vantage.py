from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx

from plugins.tool_registry import Tool, ToolRegistry


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_TIMEOUT = 20.0


def _api_key(explicit: str = "") -> str:
    return (
        str(explicit or "").strip()
        or os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
        or os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    )


def _clean_symbol(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", str(symbol or "").strip().upper())
    return cleaned[:16]


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _alpha_error(message: str, *, function: str = "", symbol: str = "", raw: Any = None) -> str:
    payload: dict[str, Any] = {
        "type": "alpha_vantage_error",
        "success": False,
        "provider": "alpha_vantage",
        "function": function,
        "symbol": symbol,
        "error": message,
    }
    if raw is not None:
        payload["raw_preview"] = str(raw)[:500]
    if "api key" in message.lower():
        payload["env_needed"] = "ALPHA_VANTAGE_API_KEY"
    return _json_response(payload)


async def _alpha_get(function: str, *, api_key: str = "", **params: Any) -> dict[str, Any]:
    key = _api_key(api_key)
    if not key:
        return {
            "_success": False,
            "_error": "Alpha Vantage API key not configured. Set ALPHA_VANTAGE_API_KEY.",
        }
    query = {
        "function": function,
        "apikey": key,
        **{k: v for k, v in params.items() if v not in (None, "", [])},
    }
    async with httpx.AsyncClient(timeout=ALPHA_VANTAGE_TIMEOUT) as client:
        response = await client.get(ALPHA_VANTAGE_BASE_URL, params=query)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict):
        return {"_success": False, "_error": "Alpha Vantage returned a non-object payload.", "_raw": data}
    if data.get("Error Message"):
        return {"_success": False, "_error": str(data.get("Error Message")), "_raw": data}
    if data.get("Note") or data.get("Information"):
        return {
            "_success": False,
            "_error": str(data.get("Note") or data.get("Information")),
            "_raw": data,
            "_rate_limited": True,
        }
    return {"_success": True, **data}


def _quote_payload(symbol: str, data: dict[str, Any]) -> dict[str, Any]:
    quote = data.get("Global Quote") if isinstance(data.get("Global Quote"), dict) else {}
    return {
        "type": "alpha_vantage_quote",
        "success": bool(quote),
        "provider": "alpha_vantage",
        "symbol": symbol,
        "price": quote.get("05. price") or quote.get("price") or "",
        "change": quote.get("09. change") or "",
        "change_percent": quote.get("10. change percent") or "",
        "volume": quote.get("06. volume") or "",
        "latest_trading_day": quote.get("07. latest trading day") or "",
        "previous_close": quote.get("08. previous close") or "",
        "open": quote.get("02. open") or "",
        "high": quote.get("03. high") or "",
        "low": quote.get("04. low") or "",
        "raw": quote,
    }


async def alpha_vantage_quote(symbol: str, api_key: str = "") -> str:
    """Get a deterministic current quote from Alpha Vantage GLOBAL_QUOTE."""
    clean = _clean_symbol(symbol)
    if not clean:
        return _alpha_error("Missing stock symbol.", function="GLOBAL_QUOTE")
    data = await _alpha_get("GLOBAL_QUOTE", api_key=api_key, symbol=clean)
    if not data.get("_success"):
        return _alpha_error(str(data.get("_error") or "Alpha Vantage request failed."), function="GLOBAL_QUOTE", symbol=clean, raw=data.get("_raw"))
    payload = _quote_payload(clean, data)
    if not payload["success"]:
        return _alpha_error("No quote found for symbol.", function="GLOBAL_QUOTE", symbol=clean, raw=data)
    return _json_response(payload)


def _normalize_movers(items: Any, limit: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items[: max(1, min(limit, 50))]:
        if not isinstance(item, dict):
            continue
        symbol = _clean_symbol(str(item.get("ticker") or item.get("symbol") or ""))
        if not symbol:
            continue
        normalized.append({
            "symbol": symbol,
            "price": item.get("price") or "",
            "change_amount": item.get("change_amount") or item.get("change") or "",
            "change_percent": item.get("change_percentage") or item.get("change_percent") or "",
            "volume": item.get("volume") or "",
        })
    return normalized


async def alpha_vantage_movers(limit: int = 10, api_key: str = "") -> str:
    """Get Alpha Vantage top gainers, losers, and most-active stocks."""
    limit = max(1, min(int(limit or 10), 50))
    data = await _alpha_get("TOP_GAINERS_LOSERS", api_key=api_key)
    if not data.get("_success"):
        return _alpha_error(str(data.get("_error") or "Alpha Vantage request failed."), function="TOP_GAINERS_LOSERS", raw=data.get("_raw"))
    payload = {
        "type": "alpha_vantage_movers",
        "success": True,
        "provider": "alpha_vantage",
        "last_updated": data.get("last_updated") or "",
        "top_gainers": _normalize_movers(data.get("top_gainers"), limit),
        "top_losers": _normalize_movers(data.get("top_losers"), limit),
        "most_actively_traded": _normalize_movers(data.get("most_actively_traded"), limit),
        "answer_patch": {
            "locked_entities": [_clean_symbol(str(item.get("ticker") or item.get("symbol") or "")) for item in (data.get("top_gainers") or [])[:3] if isinstance(item, dict)],
            "rules": [
                "Use top_gainers as the locked entity list for a top-gainers workflow.",
                "Do not replace these symbols with tickers from generic web search unless a later deterministic source contradicts them.",
            ],
        },
    }
    return _json_response(payload)


def _select_overview_fields(data: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "Symbol", "Name", "Description", "Exchange", "Currency", "Country", "Sector", "Industry",
        "MarketCapitalization", "PERatio", "PEGRatio", "DividendYield", "EPS", "RevenueTTM",
        "GrossProfitTTM", "ProfitMargin", "OperatingMarginTTM", "ReturnOnAssetsTTM",
        "ReturnOnEquityTTM", "QuarterlyRevenueGrowthYOY", "QuarterlyEarningsGrowthYOY",
        "AnalystTargetPrice", "52WeekHigh", "52WeekLow", "Beta",
    ]
    return {field: data.get(field, "") for field in fields if data.get(field) not in (None, "")}


async def alpha_vantage_overview(symbol: str, api_key: str = "") -> str:
    """Get deterministic company fundamentals from Alpha Vantage OVERVIEW."""
    clean = _clean_symbol(symbol)
    if not clean:
        return _alpha_error("Missing stock symbol.", function="OVERVIEW")
    data = await _alpha_get("OVERVIEW", api_key=api_key, symbol=clean)
    if not data.get("_success"):
        return _alpha_error(str(data.get("_error") or "Alpha Vantage request failed."), function="OVERVIEW", symbol=clean, raw=data.get("_raw"))
    overview = _select_overview_fields(data)
    if not overview:
        return _alpha_error("No overview found for symbol.", function="OVERVIEW", symbol=clean, raw=data)
    return _json_response({
        "type": "alpha_vantage_overview",
        "success": True,
        "provider": "alpha_vantage",
        "symbol": clean,
        "overview": overview,
    })


def _normalize_news_feed(feed: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(feed, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in feed[: max(1, min(limit, 50))]:
        if not isinstance(item, dict):
            continue
        rows.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "source": item.get("source") or "",
            "published_at": item.get("time_published") or "",
            "summary": item.get("summary") or "",
            "overall_sentiment_score": item.get("overall_sentiment_score") or "",
            "overall_sentiment_label": item.get("overall_sentiment_label") or "",
        })
    return rows


async def alpha_vantage_news(symbol: str, limit: int = 5, api_key: str = "") -> str:
    """Get market news and sentiment for one ticker from Alpha Vantage NEWS_SENTIMENT."""
    clean = _clean_symbol(symbol)
    if not clean:
        return _alpha_error("Missing stock symbol.", function="NEWS_SENTIMENT")
    limit = max(1, min(int(limit or 5), 50))
    data = await _alpha_get("NEWS_SENTIMENT", api_key=api_key, tickers=clean, limit=limit)
    if not data.get("_success"):
        return _alpha_error(str(data.get("_error") or "Alpha Vantage request failed."), function="NEWS_SENTIMENT", symbol=clean, raw=data.get("_raw"))
    return _json_response({
        "type": "alpha_vantage_news",
        "success": True,
        "provider": "alpha_vantage",
        "symbol": clean,
        "items": _normalize_news_feed(data.get("feed"), limit),
    })


async def finance_snapshot(symbol: str, include_news: bool = True, news_limit: int = 5, api_key: str = "") -> str:
    """Run a compact deterministic finance workflow for one ticker."""
    clean = _clean_symbol(symbol)
    if not clean:
        return _alpha_error("Missing stock symbol.", function="finance_snapshot")

    quote_raw = await _alpha_get("GLOBAL_QUOTE", api_key=api_key, symbol=clean)
    overview_raw = await _alpha_get("OVERVIEW", api_key=api_key, symbol=clean)
    news_raw: dict[str, Any] = {"_success": True, "feed": []}
    if include_news:
        news_raw = await _alpha_get("NEWS_SENTIMENT", api_key=api_key, tickers=clean, limit=max(1, min(int(news_limit or 5), 50)))

    errors = [
        f"{label}: {data.get('_error')}"
        for label, data in (("quote", quote_raw), ("overview", overview_raw), ("news", news_raw))
        if not data.get("_success")
    ]
    quote = _quote_payload(clean, quote_raw) if quote_raw.get("_success") else {}
    overview = _select_overview_fields(overview_raw) if overview_raw.get("_success") else {}
    news = _normalize_news_feed(news_raw.get("feed"), int(news_limit or 5)) if news_raw.get("_success") else []
    success = bool(quote or overview or news) and not (bool(errors) and all("api key" in err.lower() for err in errors))
    must_say = []
    if quote:
        must_say.append(f"{clean} last Alpha Vantage quote: {quote.get('price') or 'not recorded'} ({quote.get('change_percent') or 'change not recorded'}).")
    if overview.get("Name"):
        must_say.append(f"Company: {overview.get('Name')} · {overview.get('Sector', 'sector not recorded')} · {overview.get('Industry', 'industry not recorded')}.")
    if overview.get("MarketCapitalization"):
        must_say.append(f"Market capitalization: {overview.get('MarketCapitalization')}.")
    if news:
        must_say.append(f"News URLs available: {len([item for item in news if item.get('url')])}.")
    payload = {
        "type": "finance_snapshot",
        "success": success,
        "provider": "alpha_vantage",
        "symbol": clean,
        "quote": quote,
        "overview": overview,
        "news": news,
        "errors": errors,
        "answer_patch": {
            "format": "finance_snapshot_v1",
            "must_say": must_say,
            "source_urls": [item["url"] for item in news if item.get("url")],
            "do_not_claim": [
                "Do not provide financial advice or a buy/sell order.",
                "Do not claim real-time data; Alpha Vantage values are as returned by the API.",
                "Do not cite news URLs that are not present in source_urls.",
            ],
            "missing": [
                label
                for label, value in (("quote", quote), ("overview", overview), ("news", news if include_news else ["not_requested"]))
                if not value
            ],
        },
    }
    return _json_response(payload)


ALPHA_VANTAGE_QUOTE_TOOL = Tool(
    name="alpha_vantage_quote",
    description="Get a deterministic stock quote from Alpha Vantage GLOBAL_QUOTE. Requires ALPHA_VANTAGE_API_KEY.",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker such as ABNB, AAPL, MSFT."},
            "api_key": {"type": "string", "description": "Optional override. Prefer ALPHA_VANTAGE_API_KEY env var."},
        },
        "required": ["symbol"],
    },
    handler=alpha_vantage_quote,
    tags=["finance", "stocks", "deterministic"],
    response_format="json",
)

ALPHA_VANTAGE_MOVERS_TOOL = Tool(
    name="alpha_vantage_movers",
    description="Get Alpha Vantage top gainers, losers, and most-active tickers. Use before stock-mover web workflows to lock entities.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Rows per movers section.", "default": 10},
            "api_key": {"type": "string", "description": "Optional override. Prefer ALPHA_VANTAGE_API_KEY env var."},
        },
    },
    handler=alpha_vantage_movers,
    tags=["finance", "stocks", "deterministic"],
    response_format="json",
)

ALPHA_VANTAGE_OVERVIEW_TOOL = Tool(
    name="alpha_vantage_overview",
    description="Get deterministic company fundamentals from Alpha Vantage OVERVIEW.",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker such as ABNB, AAPL, MSFT."},
            "api_key": {"type": "string", "description": "Optional override. Prefer ALPHA_VANTAGE_API_KEY env var."},
        },
        "required": ["symbol"],
    },
    handler=alpha_vantage_overview,
    tags=["finance", "stocks", "fundamentals", "deterministic"],
    response_format="json",
)

ALPHA_VANTAGE_NEWS_TOOL = Tool(
    name="alpha_vantage_news",
    description="Get market news and sentiment for a ticker from Alpha Vantage NEWS_SENTIMENT.",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker such as ABNB, AAPL, MSFT."},
            "limit": {"type": "integer", "description": "Max news items.", "default": 5},
            "api_key": {"type": "string", "description": "Optional override. Prefer ALPHA_VANTAGE_API_KEY env var."},
        },
        "required": ["symbol"],
    },
    handler=alpha_vantage_news,
    tags=["finance", "stocks", "news", "deterministic"],
    response_format="json",
)

FINANCE_SNAPSHOT_TOOL = Tool(
    name="finance_snapshot",
    description=(
        "Deterministic finance super-tool for one ticker. Combines Alpha Vantage quote, overview, and news into "
        "a compact report-ready answer_patch. Use for stock fundamentals, ticker summaries, and analyst-style reports."
    ),
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker such as ABNB, AAPL, MSFT."},
            "include_news": {"type": "boolean", "description": "Whether to include Alpha Vantage news.", "default": True},
            "news_limit": {"type": "integer", "description": "Max news items.", "default": 5},
            "api_key": {"type": "string", "description": "Optional override. Prefer ALPHA_VANTAGE_API_KEY env var."},
        },
        "required": ["symbol"],
    },
    handler=finance_snapshot,
    tags=["finance", "stocks", "deterministic", "report"],
    response_format="json",
)


ALPHA_VANTAGE_TOOLS = [
    ALPHA_VANTAGE_QUOTE_TOOL,
    ALPHA_VANTAGE_MOVERS_TOOL,
    ALPHA_VANTAGE_OVERVIEW_TOOL,
    ALPHA_VANTAGE_NEWS_TOOL,
    FINANCE_SNAPSHOT_TOOL,
]


def register_alpha_vantage_tools(registry: ToolRegistry) -> None:
    for tool in ALPHA_VANTAGE_TOOLS:
        registry.register(tool)
