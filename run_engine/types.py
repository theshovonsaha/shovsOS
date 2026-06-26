from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from engine.context_schema import ContextPhase


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
    workflow_template: str = "general_operator_v1"
    prompt_version: str = "role_contracts_v1"
    risk_policy: str = "standard"
    ledger_mode: str = "shadow"
    control_policy: str = "auto"
    forced_tools: tuple[str, ...] = field(default_factory=tuple)
    workspace_path: Optional[str] = None
    # None = let the adapter decide (default for the model). True/False explicitly
    # overrides — mainly meaningful for Ollama thinking models, where False sets
    # think:false in the API payload to suppress reasoning generation entirely.
    reasoning_enabled: Optional[bool] = None
    # sync = wait for compression/indexing before done. async = respond first,
    # then run memory maintenance in a guarded background task. skip = no
    # memory maintenance for this turn.
    memory_commit_mode: str = "sync"


@dataclass(frozen=True)
class CompiledPassPacket:
    phase: ContextPhase
    content: str
    trace: dict[str, Any]
