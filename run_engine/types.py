from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from engine.context_schema import ContextPhase


RunEngineRuntimePath = Literal["legacy", "run_engine"]


@dataclass(frozen=True)
class RunEngineRequest:
    session_id: str
    owner_id: str
    agent_id: str
    user_message: str
    model: str
    system_prompt: str = ""
    context_mode: Optional[str] = None
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    use_planner: bool = True
    max_tool_calls: Optional[int] = None
    max_turns: Optional[int] = None
    planner_model: Optional[str] = None
    context_model: Optional[str] = None
    embed_model: Optional[str] = None
    images: Optional[list[str]] = None
    search_backend: Optional[str] = None
    search_engine: Optional[str] = None
    agent_revision: Optional[int] = None
    forced_tools: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompiledPassPacket:
    phase: ContextPhase
    content: str
    trace: dict[str, Any]
