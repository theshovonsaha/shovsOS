"""
Agent Manager
-------------
Orchestrates the loading of AgentProfiles and instantiation of runtime-backed agents.
"""

import inspect
from typing import Optional

from config.trace_store import get_trace_store
from orchestration.agent_profiles import ProfileManager
from orchestration.run_store import get_run_store
from orchestration.session_manager import SessionManager
from engine.context_engine import ContextEngine
from llm.base_adapter import BaseLLMAdapter
from orchestration.orchestrator import AgenticOrchestrator
from guardrails.middleware import GuardrailMiddleware
from orchestration.agent_runtime import (
    AgentRuntimeAdapter,
    AgentRuntimeCapabilities,
    AgentRuntimeContext,
    AgentRuntimeInstance,
    AgentRuntimeServices,
)
from orchestration.native_runtime_adapter import create_native_runtime_adapter
from plugins.tool_registry import ToolRegistry

class AgentManager:
    def __init__(
        self,
        profiles:        ProfileManager,
        sessions:        SessionManager,
        context_engine:  ContextEngine,
        adapter:         BaseLLMAdapter,
        global_registry: ToolRegistry,
        orchestrator:    Optional[AgenticOrchestrator] = None,
        guardrail_middleware: Optional[GuardrailMiddleware] = None,
    ):
        self.profiles        = profiles
        self.sessions        = sessions
        self.ctx_eng         = context_engine
        self.adapter         = adapter
        self.global_registry = global_registry
        self.orch            = orchestrator
        self.guardrail_middleware = guardrail_middleware
        self._runtime_adapters: dict[str, AgentRuntimeAdapter] = {}
        self.register_runtime_adapter(create_native_runtime_adapter())
        self._agent_cache: dict[tuple[str, str, int], AgentRuntimeInstance] = {}  # owner_id, agent_id, revision

    def _get_profile(self, agent_id: str, owner_id: Optional[str] = None):
        try:
            config = self.profiles.get(agent_id, owner_id=owner_id)
        except TypeError:
            config = self.profiles.get(agent_id)
        if not config:
            try:
                config = self.profiles.get("default", owner_id=owner_id)
            except TypeError:
                config = self.profiles.get("default")
            print(f"[AgentManager] WARNING: Agent '{agent_id}' not found. Falling back to default.")
        return config

    @staticmethod
    def _cache_key(agent_id: str, owner_id: Optional[str], revision: Optional[int]) -> tuple[str, str, int]:
        return (owner_id or "", agent_id, int(revision or 1))

    def register_runtime_adapter(self, adapter: AgentRuntimeAdapter) -> None:
        runtime_kind = (getattr(adapter, "runtime_kind", "") or "").strip().lower()
        if not runtime_kind:
            raise ValueError("runtime adapter must declare runtime_kind")
        self._runtime_adapters[runtime_kind] = adapter

    def get_runtime_adapter(self, runtime_kind: Optional[str] = None) -> AgentRuntimeAdapter:
        normalized = runtime_kind if isinstance(runtime_kind, str) else "native"
        normalized = normalized.strip().lower() or "native"
        adapter = self._runtime_adapters.get(normalized)
        if adapter is None:
            raise RuntimeError(f"Unsupported runtime_kind: {normalized}")
        return adapter

    def _build_runtime_context(
        self,
        *,
        agent_id: str,
        profile,
        owner_id: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> AgentRuntimeContext:
        return AgentRuntimeContext(
            agent_id=agent_id,
            owner_id=owner_id,
            profile=profile,
            sessions=self.sessions,
            context_engine=self.ctx_eng,
            adapter=self.adapter,
            global_registry=self.global_registry,
            services=AgentRuntimeServices(
                run_store=get_run_store(),
                trace_store=get_trace_store(),
            ),
            orchestrator=self.orch,
            guardrail_middleware=self.guardrail_middleware,
            model_override=model_override,
        )

    @staticmethod
    def _supports_kwarg(callable_obj, kwarg_name: str) -> bool:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return kwarg_name in signature.parameters

    def _get_runtime_capabilities(
        self,
        runtime: AgentRuntimeAdapter,
        context: AgentRuntimeContext,
    ) -> AgentRuntimeCapabilities:
        getter = getattr(runtime, "get_capabilities", None)
        if callable(getter):
            try:
                capabilities = getter(context)
                if isinstance(capabilities, AgentRuntimeCapabilities):
                    return capabilities
            except TypeError:
                capabilities = getter()
                if isinstance(capabilities, AgentRuntimeCapabilities):
                    return capabilities
        return AgentRuntimeCapabilities()

    def get_agent_instance(
        self,
        agent_id: str = "default",
        model_override: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> AgentRuntimeInstance:
        """
        Creates (or returns cached) a runtime-backed agent instance based on profile config.
        """
        config = self._get_profile(agent_id, owner_id=owner_id)
        cache_key = self._cache_key(agent_id, owner_id, getattr(config, "revision", 1))
        runtime = self.get_runtime_adapter(getattr(config, "runtime_kind", None))

        # Return cached instance if available - but not if model is overridden
        if not model_override and cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        runtime_context = self._build_runtime_context(
            agent_id=agent_id,
            profile=config,
            owner_id=owner_id,
            model_override=model_override,
        )
        capabilities = self._get_runtime_capabilities(runtime, runtime_context)
        instance = runtime.build_instance(runtime_context)
        try:
            setattr(instance, "_runtime_kind", getattr(runtime, "runtime_kind", "native"))
            setattr(instance, "_runtime_capabilities", capabilities)
        except Exception:
            pass

        self._agent_cache[cache_key] = instance
        return instance

    async def run_agent_task(
        self,
        agent_id: str,
        task: str,
        parent_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        _model_override: Optional[str] = None,
        loop_mode: Optional[str] = None,
    ) -> str:
        """
        Runs an agent to completion for a specific task and returns the final response string.
        Creates a isolated child session if a parent_id is provided.
        Inherits the parent session's active model/provider so cloud adapters work correctly.
        """
        config = self._get_profile(agent_id, owner_id=owner_id)
        runtime = self.get_runtime_adapter(getattr(config, "runtime_kind", None))
        runtime_context = self._build_runtime_context(
            agent_id=agent_id,
            profile=config,
            owner_id=owner_id,
            model_override=_model_override,
        )
        capabilities = self._get_runtime_capabilities(runtime, runtime_context)
        requested_loop_mode = loop_mode or capabilities.preferred_loop_mode
        kwargs = {
            "parent_id": parent_id,
            "parent_run_id": parent_run_id,
        }
        if self._supports_kwarg(runtime.run_task, "loop_mode"):
            kwargs["loop_mode"] = requested_loop_mode
        if self._supports_kwarg(runtime.run_task, "run_metadata"):
            kwargs["run_metadata"] = {
                "runtime_kind": getattr(runtime, "runtime_kind", "native"),
                "runtime_capabilities": {
                    "supports_managed_loop": capabilities.supports_managed_loop,
                    "supports_checkpoints": capabilities.supports_checkpoints,
                    "supports_run_artifacts": capabilities.supports_run_artifacts,
                    "supports_run_evals": capabilities.supports_run_evals,
                    "supported_loop_modes": list(capabilities.supported_loop_modes),
                    "preferred_loop_mode": capabilities.preferred_loop_mode,
                    "notes": list(capabilities.notes),
                },
            }
        return await runtime.run_task(runtime_context, task, **kwargs)


    def invalidate_cache(self, agent_id: str = None, owner_id: Optional[str] = None):
        """Clear agent cache — call when profile is updated."""
        if agent_id is None and owner_id is None:
            self._agent_cache.clear()
            return

        survivors: dict[tuple[str, str, int], AgentCore] = {}
        for key, instance in self._agent_cache.items():
            cached_owner, cached_agent, _ = key
            if agent_id is not None and cached_agent != agent_id:
                survivors[key] = instance
                continue
            if owner_id is not None and cached_owner != (owner_id or ""):
                survivors[key] = instance
                continue
            if agent_id is None and owner_id is not None:
                continue
        self._agent_cache = survivors
