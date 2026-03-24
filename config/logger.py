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

Categories: agent | tool | rag | llm | ctx | system
Levels:     info | ok | warn | error
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import AsyncIterator, Optional


@dataclass
class LogEntry:
    ts:       float
    category: str   # agent | tool | rag | llm | ctx | system
    session:  str   # session_id or "system"
    message:  str
    level:    str = "info"  # info | ok | warn | error
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
        **meta,
    ) -> None:
        entry = LogEntry(
            ts=time.time(),
            category=category,
            session=session,
            message=message,
            level=level,
            meta=meta,
        )
        self._buffer.append(entry)
        # Also print so existing terminal logging still works
        prefix = {"info": "·", "ok": "✓", "warn": "!", "error": "✗"}.get(level, "·")
        print(f"[{category.upper():6}] {prefix} [{session[:8]}] {message}")
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

    def recent(self, limit: int = 100) -> list[LogEntry]:
        entries = list(self._buffer)
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

def log(category: str, session: str, message: str, level: str = "info", **meta):
    _logger.log(category, session, message, level, **meta)

def get_logger() -> InternalLogger:
    return _logger