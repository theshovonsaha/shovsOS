#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from plugins.finance_alpha_vantage import alpha_vantage_movers, finance_snapshot


async def main() -> int:
    load_dotenv(Path.cwd() / ".env")
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "ABNB").strip().upper()
    if not os.getenv("ALPHA_VANTAGE_API_KEY") and not os.getenv("ALPHAVANTAGE_API_KEY"):
        print(json.dumps({
            "success": False,
            "error": "Set ALPHA_VANTAGE_API_KEY before running this live probe.",
            "example": "ALPHA_VANTAGE_API_KEY=... ./venv/bin/python scripts/test_alpha_vantage.py ABNB",
        }, indent=2))
        return 2

    movers = json.loads(await alpha_vantage_movers(limit=3))
    snapshot = json.loads(await finance_snapshot(symbol=symbol, include_news=True, news_limit=3))
    print(json.dumps({
        "success": bool(movers.get("success") and snapshot.get("success")),
        "symbol": symbol,
        "movers_last_updated": movers.get("last_updated"),
        "top_gainers": movers.get("top_gainers", [])[:3],
        "snapshot": {
            "quote": snapshot.get("quote", {}),
            "overview_keys": sorted((snapshot.get("overview") or {}).keys()),
            "news_count": len(snapshot.get("news") or []),
            "errors": snapshot.get("errors", []),
        },
    }, indent=2))
    return 0 if movers.get("success") and snapshot.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
