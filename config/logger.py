"""
Internal Logger
---------------
Replaces scattered print() calls with a structured log system.
Broadcasts log entries to any connected SSE subscribers (the dev panel).

Usage anywhere in the codebase:
    from config.logger import log
    log("agent", "session_id", "Tool call detected: web_search")
    log("tool",  "session_id", "web_search returned 8 results", level="ok")
    log("rag",   "session_id", "3 anchors retrieved from vector store")
    log("llm",   "session_id", "Turn 0: 412 tokens generated")
    log("ctx",   "session_id", "Compression: 24 lines → 14 lines")

Canonical categories: agent | tool | rag | llm | ctx | system
Aliases normalized automatically: mcp, orch, run_engine, startup, circuit, run, manifest
Levels:     info | ok | warn | error
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import AsyncIterator, Optional


CANONICAL_CATEGORIES = {"agent", "tool", "rag", "llm", "ctx", "system"}
CATEGORY_ALIASES = {
    "mcp": "tool",
    "orch": "agent",
    "run_engine": "agent",
    "startup": "system",
    "circuit": "system",
    "run": "system",
    "manifest": "ctx",
}
CANONICAL_LEVELS = {"info", "ok", "warn", "error"}


def _normalize_category(category: str) -> tuple[str, Optional[str]]:
    raw = str(category or "system").strip().lower() or "system"
    normalized = CATEGORY_ALIASES.get(raw, raw)
    if normalized not in CANONICAL_CATEGORIES:
        return "system", raw
    if normalized != raw:
        return normalized, raw
    return normalized, None


def _normalize_level(level: str) -> tuple[str, Optional[str]]:
    raw = str(level or "info").strip().lower() or "info"
    if raw in CANONICAL_LEVELS:
        return raw, None
    return "info", raw


@dataclass
class LogEntry:
    ts:       float
    category: str   # agent | tool | rag | llm | ctx | system
    session:  str   # session_id or "system"
    message:  str
    level:    str = "info"  # info | ok | warn | error
    owner_id: Optional[str] = None
    meta:     dict = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps(asdict(self))}\n\n"


class InternalLogger:
    """
    Central log bus. Thread-safe enough for asyncio single-process use.
    Keeps a rolling buffer of recent entries for late-connecting clients.
    Broadcasts to all active SSE subscribers.
    """

    MAX_BUFFER = 500

    def __init__(self):
        self._buffer: deque[LogEntry] = deque(maxlen=self.MAX_BUFFER)
        self._subscribers: list[asyncio.Queue] = []

    def log(
        self,
        category: str,
        session:  str,
        message:  str,
        level:    str = "info",
        owner_id: Optional[str] = None,
        **meta,
    ) -> None:
        normalized_category, source_category = _normalize_category(category)
        normalized_level, source_level = _normalize_level(level)
        if owner_id is None:
            owner_id = str(meta.get("owner_id") or "").strip() or None
        if source_category and "source_category" not in meta:
            meta["source_category"] = source_category
        if source_level and "source_level" not in meta:
            meta["source_level"] = source_level
        entry = LogEntry(
            ts=time.time(),
            category=normalized_category,
            session=session,
            message=message,
            level=normalized_level,
            owner_id=owner_id,
            meta=meta,
        )
        self._buffer.append(entry)
        # Also print so existing terminal logging still works
        prefix = {"info": "·", "ok": "✓", "warn": "!", "error": "✗"}.get(normalized_level, "·")
        print(f"[{normalized_category.upper():6}] {prefix} [{session[:8]}] {message}")
        # Broadcast to all connected SSE clients
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    @staticmethod
    def _matches(
        entry: LogEntry,
        *,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> bool:
        if session_id and entry.session not in (session_id, "system"):
            return False
        if category and entry.category != category:
            return False
        if owner_id and entry.owner_id != owner_id:
            return False
        return True

    def recent(
        self,
        limit: int = 100,
        *,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> list[LogEntry]:
        entries = [
            entry
            for entry in self._buffer
            if self._matches(
                entry,
                session_id=session_id,
                category=category,
                owner_id=owner_id,
            )
        ]
        return entries[-limit:]

    async def stream(self, q: asyncio.Queue) -> AsyncIterator[str]:
        """Async generator for SSE streaming."""
        try:
            while True:
                try:
                    entry: LogEntry = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield entry.to_sse()
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass


# ── Singleton ────────────────────────────────────────────────────────────────
_logger = InternalLogger()

def log(
    category: str,
    session: str,
    message: str,
    level: str = "info",
    owner_id: Optional[str] = None,
    **meta,
):
    _logger.log(category, session, message, level, owner_id=owner_id, **meta)

def get_logger() -> InternalLogger:
    return _logger
