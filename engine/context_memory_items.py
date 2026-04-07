from __future__ import annotations

from typing import Any, Optional

from engine.context_schema import ContextItem, ContextKind, ContextPhase


CONTEXT_ENGINE_PHASE_VISIBILITY = frozenset({
    ContextPhase.PLANNING,
    ContextPhase.ACTING,
    ContextPhase.RESPONSE,
    ContextPhase.VERIFICATION,
})


def build_context_engine_memory_items(
    context_engine: Optional[object],
    current_context: str,
    *,
    fallback_trace_id: str = "memory:context_engine",
    fallback_source: str = "context_engine",
    fallback_provenance: Optional[dict[str, Any]] = None,
) -> list[ContextItem]:
    if not current_context.strip() or context_engine is None:
        return []

    build_items = getattr(context_engine, "build_context_items", None)
    if callable(build_items):
        try:
            maybe_items = build_items(current_context)
            if isinstance(maybe_items, list):
                items = [item for item in maybe_items if isinstance(item, ContextItem)]
                if items:
                    return items
        except Exception:
            pass

    build_block = getattr(context_engine, "build_context_block", None)
    if callable(build_block):
        try:
            block = str(build_block(current_context) or "").strip()
            if block:
                return [
                    ContextItem(
                        item_id="context_engine_memory",
                        kind=ContextKind.MEMORY,
                        title="",
                        content=block,
                        source=fallback_source,
                        priority=60,
                        max_chars=1800,
                        phase_visibility=CONTEXT_ENGINE_PHASE_VISIBILITY,
                        trace_id=fallback_trace_id,
                        provenance=fallback_provenance or {"engine": context_engine.__class__.__name__},
                        formatted=True,
                    )
                ]
        except Exception:
            pass

    return []