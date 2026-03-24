"""
Confirmation Gate
-----------------
Async hold mechanism. When a tool call requires confirmation,
execution pauses here until the frontend sends approve/deny.

Flow:
  1. middleware.execute() calls gate.request_confirmation(call_id, ...)
  2. Gate stores a pending asyncio.Event keyed by call_id
  3. Frontend polls GET /guardrails/pending → sees the pending call
  4. User clicks Approve/Deny → POST /guardrails/respond/{call_id}
  5. Gate resolves the event → middleware continues or aborts

Thread-safe for FastAPI's async event loop.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PendingCall:
    call_id:    str
    session_id: str
    tool:       str
    preview:    str
    reason:     str
    arguments:  dict
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved:   bool = False
    approved:   Optional[bool] = None
    deny_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "call_id":    self.call_id,
            "session_id": self.session_id,
            "tool":       self.tool,
            "preview":    self.preview,
            "reason":     self.reason,
            "arguments":  self.arguments,
            "created_at": self.created_at,
            "resolved":   self.resolved,
            "approved":   self.approved,
        }


class ConfirmationGate:
    """
    Manages all pending tool confirmations across sessions.
    One instance shared across the app (instantiate in middleware).
    """

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout = timeout_seconds
        self._pending: dict[str, PendingCall] = {}
        self._events:  dict[str, asyncio.Event] = {}

    # ── Called by middleware — blocks until resolved ───────────────────────────

    async def request_confirmation(
        self,
        session_id: str,
        tool: str,
        preview: str,
        reason: str,
        arguments: dict,
    ) -> tuple[bool, Optional[str]]:
        """
        Register a pending confirmation and wait for a response.

        Returns:
            (True, None)        → approved, proceed
            (False, reason_str) → denied, abort
            (False, "timeout")  → timed out, abort
        """
        call_id = str(uuid.uuid4())[:8]
        event   = asyncio.Event()

        self._pending[call_id] = PendingCall(
            call_id    = call_id,
            session_id = session_id,
            tool       = tool,
            preview    = preview,
            reason     = reason,
            arguments  = arguments,
        )
        self._events[call_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self._cleanup(call_id)
            return False, "timeout"

        pending = self._pending.get(call_id)
        approved = pending.approved if pending else False
        deny_reason = pending.deny_reason if pending else "unknown"
        self._cleanup(call_id)

        return approved, (None if approved else deny_reason)

    # ── Called by API routes — resolves the waiting coroutine ─────────────────

    def approve(self, call_id: str) -> bool:
        """Approve a pending call. Returns False if call_id not found."""
        if call_id not in self._pending:
            return False
        self._pending[call_id].resolved = True
        self._pending[call_id].approved = True
        self._events[call_id].set()
        return True

    def deny(self, call_id: str, reason: str = "User denied") -> bool:
        """Deny a pending call. Returns False if call_id not found."""
        if call_id not in self._pending:
            return False
        self._pending[call_id].resolved  = True
        self._pending[call_id].approved  = False
        self._pending[call_id].deny_reason = reason
        self._events[call_id].set()
        return True

    # ── Query API ──────────────────────────────────────────────────────────────

    def get_pending(self, session_id: Optional[str] = None) -> list[dict]:
        """Return all unresolved pending calls, optionally filtered by session."""
        return [
            p.to_dict()
            for p in self._pending.values()
            if not p.resolved and (session_id is None or p.session_id == session_id)
        ]

    def get_call(self, call_id: str) -> Optional[PendingCall]:
        return self._pending.get(call_id)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def _cleanup(self, call_id: str):
        self._pending.pop(call_id, None)
        self._events.pop(call_id, None)
