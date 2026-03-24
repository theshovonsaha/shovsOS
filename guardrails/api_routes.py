"""
Guardrails API Routes
---------------------
FastAPI router. Mount this in your main app with one line:

    from guardrails.api_routes import make_guardrail_router
    app.include_router(make_guardrail_router(middleware), prefix="/guardrails")

Endpoints:

  GET  /guardrails/pending              → list all pending confirmations
  GET  /guardrails/pending/{session_id} → list pending for one session
  POST /guardrails/approve/{call_id}    → approve a pending tool call
  POST /guardrails/deny/{call_id}       → deny a pending tool call
  GET  /guardrails/log                  → last 50 audit log entries
  GET  /guardrails/trace/{session_id}   → full trace for a session
  GET  /guardrails/stream/{session_id}  → SSE stream of pending calls (for live UI)
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from guardrails.middleware import GuardrailMiddleware


# ── Request models ─────────────────────────────────────────────────────────────

class DenyRequest(BaseModel):
    reason: Optional[str] = "User denied"


# ── Router factory ─────────────────────────────────────────────────────────────

def make_guardrail_router(middleware: GuardrailMiddleware) -> APIRouter:
    """
    Creates and returns the guardrails router bound to a middleware instance.
    Call this once in main.py and include_router the result.
    """
    router = APIRouter(tags=["guardrails"])

    # ── Pending confirmations ─────────────────────────────────────────────────

    @router.get("/pending")
    async def list_all_pending():
        """List every pending tool confirmation across all sessions."""
        return {"pending": middleware.get_pending()}

    @router.get("/pending/{session_id}")
    async def list_session_pending(session_id: str):
        """List pending confirmations for a specific session."""
        return {"pending": middleware.get_pending(session_id=session_id)}

    # ── Approve / Deny ────────────────────────────────────────────────────────

    @router.post("/approve/{call_id}")
    async def approve_call(call_id: str):
        """Approve a pending tool call. Unblocks the waiting agent."""
        ok = middleware.approve(call_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"call_id '{call_id}' not found or already resolved.")
        return {"status": "approved", "call_id": call_id}

    @router.post("/deny/{call_id}")
    async def deny_call(call_id: str, body: DenyRequest = DenyRequest()):
        """Deny a pending tool call. Agent receives an error result."""
        ok = middleware.deny(call_id, reason=body.reason or "User denied")
        if not ok:
            raise HTTPException(status_code=404, detail=f"call_id '{call_id}' not found or already resolved.")
        return {"status": "denied", "call_id": call_id, "reason": body.reason}

    # ── Audit log ─────────────────────────────────────────────────────────────

    @router.get("/log")
    async def get_audit_log(n: int = 50):
        """Return the last n audit log entries."""
        return {"entries": middleware.get_audit_log(n=n)}

    @router.get("/trace/{session_id}")
    async def get_session_trace(session_id: str):
        """Return all audit log entries for a specific session."""
        return {"session_id": session_id, "trace": middleware.get_session_trace(session_id)}

    # ── SSE stream for live frontend UI ──────────────────────────────────────

    @router.get("/stream/{session_id}")
    async def confirmation_stream(session_id: str):
        """
        Server-Sent Events stream. Frontend subscribes here and receives
        real-time confirmation_required events as they arrive.

        Frontend usage (JS):
            const es = new EventSource('/guardrails/stream/my-session-id');
            es.addEventListener('confirmation_required', e => {
                const data = JSON.parse(e.data);
                showConfirmDialog(data.call_id, data.preview, data.tool);
            });
        """
        async def event_generator():
            seen_call_ids: set[str] = set()
            while True:
                pending = middleware.get_pending(session_id=session_id)
                for item in pending:
                    cid = item["call_id"]
                    if cid not in seen_call_ids:
                        seen_call_ids.add(cid)
                        payload = json.dumps(item)
                        yield f"event: confirmation_required\ndata: {payload}\n\n"

                # Heartbeat every 2 seconds to keep connection alive
                yield ": heartbeat\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
