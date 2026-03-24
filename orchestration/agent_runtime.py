from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Protocol, runtime_checkable

from engine.context_engine import ContextEngine
from guardrails.middleware import GuardrailMiddleware
from llm.base_adapter import BaseLLMAdapter
from orchestration.agent_profiles import AgentProfile
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from config.trace_store import TraceStore
    from orchestration.run_store import RunStore


@runtime_checkable
class AgentRuntimeInstance(Protocol):
    async def chat_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        agent_id: str = "default",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        search_backend: Optional[str] = None,
        search_engine: Optional[str] = None,
        images: Optional[list[str]] = None,
        force_memory: bool = False,
        forced_tools: Optional[list[str]] = None,
        **kw,
        ) -> AsyncIterator[dict]:
        ...


@dataclass(frozen=True)
class AgentRuntimeCapabilities:
    supports_managed_loop: bool = False
    supports_checkpoints: bool = False
    supports_run_artifacts: bool = False
    supports_run_evals: bool = False
    supported_loop_modes: tuple[str, ...] = ("single",)
    preferred_loop_mode: str = "single"
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentRuntimeServices:
    run_store: "RunStore"
    trace_store: "TraceStore"


@dataclass(frozen=True)
class AgentRuntimeContext:
    agent_id: str
    owner_id: Optional[str]
    profile: AgentProfile
    sessions: SessionManager
    context_engine: ContextEngine
    adapter: BaseLLMAdapter
    global_registry: ToolRegistry
    services: AgentRuntimeServices
    orchestrator: Optional[AgenticOrchestrator] = None
    guardrail_middleware: Optional[GuardrailMiddleware] = None
    model_override: Optional[str] = None


class AgentRuntimeAdapter(Protocol):
    runtime_kind: str

    def get_capabilities(
        self,
        context: Optional[AgentRuntimeContext] = None,
    ) -> AgentRuntimeCapabilities:
        ...

    def build_instance(self, context: AgentRuntimeContext) -> AgentRuntimeInstance:
        ...

    async def run_task(
        self,
        context: AgentRuntimeContext,
        task: str,
        *,
        parent_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        loop_mode: Optional[str] = None,
        run_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        ...
