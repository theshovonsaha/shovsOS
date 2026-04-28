"""Lazy package init.

Importing ``run_engine.engine`` eagerly here created a cycle: ``engine.core``
→ ``engine.context_governor`` → ``run_engine.memory_pipeline`` triggers
``run_engine/__init__.py``, which used to pull in ``run_engine.engine``,
which re-imports ``engine.context_governor`` mid-initialization. Lazy
attribute loading lets ``run_engine.memory_pipeline`` be imported without
forcing the engine module to load too.
"""

from __future__ import annotations

__all__ = ["RunEngine", "RunEngineRequest"]


def __getattr__(name: str):
    if name == "RunEngine":
        from run_engine.engine import RunEngine
        return RunEngine
    if name == "RunEngineRequest":
        from run_engine.types import RunEngineRequest
        return RunEngineRequest
    raise AttributeError(f"module 'run_engine' has no attribute {name!r}")
