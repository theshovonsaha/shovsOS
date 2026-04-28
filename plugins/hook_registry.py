"""
hook_registry.py — Typed lifecycle event pub/sub for the Shovs runtime.

Inspired by OpenClaw's internal-hooks pattern. Provides a fire-and-forget
async handler registration system for agent lifecycle events.

Usage:
    from plugins.hook_registry import hooks, HookEvent

    # Register a handler (at startup or plugin init)
    @hooks.on("tool_selected")
    async def my_handler(event: HookEvent) -> None:
        print(event.data)

    # Emit an event (from engine.py at key lifecycle points)
    await hooks.emit("tool_selected", {"tool": "web_fetch", "run_id": "..."})

Events fired by the runtime:
    session_started     — new session created; data: {session_id, agent_id}
    plan_generated      — planner returned structured plan; data: {route, skill, confidence, tools}
    tool_selected       — actor chose a tool; data: {tool_name, arguments_preview, run_id}
    tool_completed      — tool execution finished; data: {tool_name, success, duration_ms, run_id}
    memory_stored       — fact written to memory; data: {fact_type, locus_id, owner_id}
    run_complete        — full engine pass finished; data: {run_id, route, tool_count, success}
    hard_failure        — HARD_FAILURE contract triggered; data: {tool_name, reason, run_id}
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("shovs.hooks")

HookHandler = Callable[["HookEvent"], Awaitable[None]]

# Valid lifecycle event names
LIFECYCLE_EVENTS = frozenset({
    "session_started",
    "plan_generated",
    "tool_selected",
    "tool_completed",
    "memory_stored",
    "run_complete",
    "hard_failure",
})


@dataclass
class HookEvent:
    """A single lifecycle event emitted by the runtime."""

    event: str
    data: dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    session_id: Optional[str] = None


class HookRegistry:
    """
    Central registry for lifecycle event handlers.

    Thread-safe for registration. Handlers are called concurrently via
    asyncio.gather — a failing handler is logged but never raises into the
    engine loop (fire-and-forget semantics).
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookHandler]] = defaultdict(list)

    def on(self, event: str) -> Callable[[HookHandler], HookHandler]:
        """Decorator: register an async handler for a lifecycle event."""
        def decorator(fn: HookHandler) -> HookHandler:
            self.register(event, fn)
            return fn
        return decorator

    def register(self, event: str, handler: HookHandler) -> None:
        """Explicitly register a handler for an event."""
        self._handlers[event].append(handler)
        log.debug("Registered hook handler for '%s': %s", event, getattr(handler, "__name__", repr(handler)))

    def unregister(self, event: str, handler: HookHandler) -> bool:
        """Remove a previously registered handler. Returns True if found."""
        handlers = self._handlers.get(event, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    async def emit(
        self,
        event: str,
        data: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Fire all handlers for the given event concurrently.

        Never raises — handler exceptions are caught and logged.
        Returns immediately (fire-and-forget).
        """
        handlers = self._handlers.get(event, [])
        if not handlers:
            return

        hook_event = HookEvent(
            event=event,
            data=data or {},
            run_id=run_id,
            session_id=session_id,
        )

        async def _safe_call(fn: HookHandler) -> None:
            try:
                await fn(hook_event)
            except Exception as exc:
                log.warning("Hook handler '%s' raised for event '%s': %s", getattr(fn, "__name__", repr(fn)), event, exc)

        await asyncio.gather(*(_safe_call(h) for h in handlers), return_exceptions=False)

    def emit_sync(
        self,
        event: str,
        data: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Synchronous fire-and-forget: schedules emit on the running loop
        without blocking. Safe to call from sync code inside an async context.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.emit(event, data, run_id, session_id))
            else:
                loop.run_until_complete(self.emit(event, data, run_id, session_id))
        except RuntimeError:
            pass  # No event loop — silently skip in pure sync contexts

    def list_events(self) -> dict[str, int]:
        """Return a dict of event → handler count for introspection."""
        return {e: len(hs) for e, hs in self._handlers.items() if hs}


# ── Singleton ──────────────────────────────────────────────────────────────
# Import and use this global instance everywhere:
#   from plugins.hook_registry import hooks
hooks = HookRegistry()
