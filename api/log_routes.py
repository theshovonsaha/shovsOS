"""
Add these routes to main.py to expose the internal log stream.

1. Add to imports at top of main.py:
   from config.logger import get_logger, log

2. Add after app = FastAPI(...):
   from api.log_routes import setup_log_routes
   setup_log_routes(app)

3. In chat_stream(), after agent_instance is created, add:
   log("agent", "system", f"Request: agent={agent_id} model={model or 'default'}")
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional
import json
from config.trace_store import get_trace_store
from api.owner import require_owner_id


def setup_log_routes(app: FastAPI):
    from config.logger import get_logger
    trace_store = get_trace_store()
    _require_owner_id = require_owner_id

    @app.get("/logs/stream")
    async def log_stream(session_id: Optional[str] = None, category: Optional[str] = None):
        """
        SSE stream of internal log entries.
        Optional filters: ?session_id=xxx&category=tool
        """
        logger = get_logger()

        async def generate():
            # Send recent history first so panel isn't blank on connect
            for entry in logger.recent(limit=80):
                if session_id and entry.session != session_id and entry.session != "system":
                    continue
                if category and entry.category != category:
                    continue
                yield entry.to_sse()

            # Then stream live
            q = logger.subscribe()
            try:
                async for chunk in logger.stream(q):
                    # Parse and filter
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        if session_id and data.get("session") not in (session_id, "system"):
                            continue
                        if category and data.get("category") != category:
                            continue
                    except Exception:
                        pass
                    yield chunk
            finally:
                logger.unsubscribe(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.get("/logs/recent")
    async def log_recent(limit: int = 100, session_id: Optional[str] = None, category: Optional[str] = None):
        """REST endpoint for recent logs — useful for initial load."""
        logger = get_logger()
        entries = logger.recent(limit=limit)
        if session_id:
            entries = [e for e in entries if e.session in (session_id, "system")]
        if category:
            entries = [e for e in entries if e.category == category]
        from dataclasses import asdict
        return {"logs": [asdict(e) for e in entries]}

    @app.get("/logs/traces/recent")
    async def trace_recent(
        limit: int = 120,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        event_type: Optional[str] = None,
        before_ts: Optional[float] = None,
    ):
        """
        Paginated trace index for monitor UIs.
        Returns compact summaries with optional payload references.
        """
        owner_id = _require_owner_id(owner_id)
        safe_limit = max(1, min(500, limit))
        normalized_type = None if not event_type or event_type == "all" else event_type
        events = trace_store.list_events(
            limit=safe_limit,
            session_id=session_id,
            run_id=run_id,
            owner_id=owner_id,
            event_type=normalized_type,
            before_ts=before_ts,
        )
        return {
            "events": events,
            "count": len(events),
            "next_before_ts": events[-1].get("ts") if events else None,
        }

    @app.get("/logs/traces/event/{event_id}")
    async def trace_event(event_id: str):
        """Fetch one trace event including full payload when available."""
        event = trace_store.get_event(event_id)
        if not event:
            return {"found": False, "event": None}
        return {"found": True, "event": event}

    @app.get("/logs/traces/stats")
    async def trace_stats(
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        window: int = 400,
    ):
        """High-level trace metrics for dashboard widgets."""
        owner_id = _require_owner_id(owner_id)
        safe_window = max(10, min(2000, window))
        stats = trace_store.stats(session_id=session_id, run_id=run_id, owner_id=owner_id, window=safe_window)
        return stats
