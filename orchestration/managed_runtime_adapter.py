from __future__ import annotations

import uuid
from typing import Any, Optional

from orchestration.agent_runtime import (
    AgentRuntimeAdapter,
    AgentRuntimeCapabilities,
    AgentRuntimeContext,
    AgentRuntimeInstance,
)
from run_engine import RunEngine, RunEngineRequest


class ManagedRuntimeInstance:
    def __init__(self, context: AgentRuntimeContext):
        self._context = context

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
    ):
        del force_memory  # compatibility shim; managed runtime consumes typed request state instead
        context = self._context
        owner_id = kw.get("owner_id", context.owner_id)
        requested_agent_id = agent_id or context.agent_id
        effective_model = model or context.model_override or context.profile.model
        resolved_context_mode = kw.get("context_mode") or getattr(context.profile, "default_context_mode", "v2")
        request = RunEngineRequest(
            session_id=session_id or str(uuid.uuid4()),
            owner_id=str(owner_id or ""),
            agent_id=requested_agent_id,
            user_message=user_message,
            model=effective_model,
            system_prompt=system_prompt or context.profile.system_prompt or "",
            context_mode=resolved_context_mode,
            allowed_tools=tuple(getattr(context.profile, "tools", []) or []),
            use_planner=bool(
                kw.get("use_planner")
                if kw.get("use_planner") is not None
                else getattr(context.profile, "default_use_planner", True)
            ),
            max_tool_calls=kw.get("max_tool_calls"),
            max_turns=kw.get("max_turns"),
            planner_model=kw.get("planner_model"),
            context_model=kw.get("context_model"),
            embed_model=kw.get("embed_model") or getattr(context.profile, "embed_model", None),
            images=images,
            search_backend=search_backend,
            search_engine=search_engine,
            agent_revision=getattr(context.profile, "revision", None),
            forced_tools=tuple(str(item) for item in (forced_tools or []) if isinstance(item, str)),
        )

        engine = RunEngine(
            adapter=context.adapter,
            sessions=context.sessions,
            tool_registry=context.global_registry,
            run_store=context.services.run_store,
            trace_store=context.services.trace_store,
            orchestrator=context.orchestrator,
            context_engine=context.context_engine,
            graph=getattr(context, "graph", None),
        )
        async for event in engine.stream(request):
            yield event


class ManagedAgentRuntimeAdapter:
    runtime_kind = "managed"

    def get_capabilities(
        self,
        context: Optional[AgentRuntimeContext] = None,
    ) -> AgentRuntimeCapabilities:
        del context
        return AgentRuntimeCapabilities(
            supports_managed_loop=True,
            supports_checkpoints=True,
            supports_run_artifacts=True,
            supports_run_evals=True,
            supported_loop_modes=("managed", "auto", "single"),
            preferred_loop_mode="managed",
            notes=("Managed runtime executes via canonical RunEngine.",),
        )

    def build_instance(self, context: AgentRuntimeContext) -> AgentRuntimeInstance:
        return ManagedRuntimeInstance(context)

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
        del loop_mode, parent_run_id, run_metadata
        owner_id = str(context.owner_id or "")
        child_sid = f"delegated_{uuid.uuid4().hex[:8]}"
        effective_model = context.model_override or context.profile.model
        resolved_context_mode = getattr(context.profile, "default_context_mode", "v2")

        if parent_id:
            parent_session = context.sessions.get(parent_id, owner_id=context.owner_id)
            if parent_session and parent_session.model:
                effective_model = parent_session.model
                resolved_context_mode = getattr(parent_session, "context_mode", resolved_context_mode)

        context.sessions.get_or_create(
            session_id=child_sid,
            model=effective_model,
            system_prompt=context.profile.system_prompt,
            agent_id=context.agent_id,
            parent_id=parent_id,
            owner_id=context.owner_id,
        )

        request = RunEngineRequest(
            session_id=child_sid,
            owner_id=owner_id,
            agent_id=context.agent_id,
            user_message=task,
            model=effective_model,
            system_prompt=context.profile.system_prompt or "",
            context_mode=resolved_context_mode,
            allowed_tools=tuple(getattr(context.profile, "tools", []) or []),
            use_planner=bool(getattr(context.profile, "default_use_planner", True)),
            agent_revision=getattr(context.profile, "revision", None),
        )

        engine = RunEngine(
            adapter=context.adapter,
            sessions=context.sessions,
            tool_registry=context.global_registry,
            run_store=context.services.run_store,
            trace_store=context.services.trace_store,
            orchestrator=context.orchestrator,
            context_engine=context.context_engine,
            graph=getattr(context, "graph", None),
        )

        full_response = ""
        async for event in engine.stream(request):
            event_type = event.get("type")
            if event_type == "token":
                full_response += str(event.get("content", ""))
            elif event_type == "error":
                return f"Error from agent '{context.agent_id}': {event.get('message', 'Unknown error')}"
        return full_response.strip()


def create_managed_runtime_adapter() -> AgentRuntimeAdapter:
    return ManagedAgentRuntimeAdapter()
