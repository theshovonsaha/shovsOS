from __future__ import annotations

import re
import uuid
from dataclasses import replace
from typing import Optional

from config.logger import log
from engine.context_engine import ContextEngine
from llm.adapter_factory import create_adapter, get_default_model, strip_provider_prefix
from orchestration.agent_runtime import (
    AgentRuntimeAdapter,
    AgentRuntimeCapabilities,
    AgentRuntimeContext,
    AgentRuntimeInstance,
)
from orchestration.orchestrator import AgenticOrchestrator
from plugins.tool_registry import ToolRegistry


class NativeAgentRuntimeAdapter:
    runtime_kind = "native"

    def get_capabilities(
        self,
        context: Optional[AgentRuntimeContext] = None,
    ) -> AgentRuntimeCapabilities:
        return AgentRuntimeCapabilities(
            supports_managed_loop=True,
            supports_checkpoints=True,
            supports_run_artifacts=True,
            supports_run_evals=True,
            supported_loop_modes=("auto", "single", "managed"),
            preferred_loop_mode="auto",
            notes=("Native runtime supports the full managed loop contract.",),
        )

    @staticmethod
    def _provider_from_model(raw_model: Optional[str]) -> Optional[str]:
        if not raw_model:
            return None
        lowered = raw_model.lower()
        known = ("ollama", "openai", "groq", "gemini", "anthropic")
        for sep in (":", "/"):
            if sep in lowered:
                head = lowered.split(sep, 1)[0]
                if head in known:
                    return head
        if lowered in known:
            return lowered
        return None

    def _filtered_registry(self, context: AgentRuntimeContext) -> ToolRegistry:
        filtered_registry = ToolRegistry()
        for tool_name in context.profile.tools:
            tool = context.global_registry.get(tool_name)
            if tool:
                filtered_registry.register(tool)
            else:
                print(
                    f"[NativeAgentRuntimeAdapter] WARNING: Tool '{tool_name}' requested by "
                    f"agent '{context.profile.name}' but not found in global registry."
                )
        return filtered_registry

    def build_instance(self, context: AgentRuntimeContext) -> AgentRuntimeInstance:
        from engine import core as core_module

        filtered_registry = self._filtered_registry(context)
        print(
            f"[AgentManager] Specialized Agent '{context.profile.name}' instantiated with "
            f"{len(filtered_registry.list_tools())} tools."
        )
        return core_module.AgentCore(
            adapter=context.adapter,
            context_engine=context.context_engine,
            session_manager=context.sessions,
            tool_registry=filtered_registry,
            middleware=context.guardrail_middleware,
            orchestrator=context.orchestrator,
            default_model=context.model_override or context.profile.model,
            embed_model=context.profile.embed_model,
            default_system_prompt=context.profile.system_prompt,
            workspace_path=context.profile.workspace_path,
            bootstrap_files=context.profile.bootstrap_files,
            bootstrap_max_chars=context.profile.bootstrap_max_chars,
        )

    async def run_task(
        self,
        context: AgentRuntimeContext,
        task: str,
        *,
        parent_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        loop_mode: Optional[str] = None,
        run_metadata: Optional[dict] = None,
    ) -> str:
        child_sid = f"delegated_{uuid.uuid4().hex[:8]}"
        effective_model = context.model_override or context.profile.model
        delegation_adapter = context.adapter

        if parent_id:
            parent_session = context.sessions.get(parent_id, owner_id=context.owner_id)
            if parent_session and parent_session.model:
                effective_model = parent_session.model
                provider = self._provider_from_model(effective_model)
                if provider:
                    delegation_adapter = create_adapter(provider=effective_model)
        else:
            provider = self._provider_from_model(effective_model)
            if provider:
                delegation_adapter = create_adapter(provider=effective_model)

        clean_model = strip_provider_prefix(effective_model)

        context.sessions.get_or_create(
            session_id=child_sid,
            model=effective_model,
            system_prompt=context.profile.system_prompt,
            agent_id=context.agent_id,
            parent_id=parent_id,
            owner_id=context.owner_id,
        )

        delegate_ctx = ContextEngine(adapter=delegation_adapter, compression_model=clean_model)
        delegate_orch = AgenticOrchestrator(adapter=delegation_adapter)
        delegate_context = replace(
            context,
            adapter=delegation_adapter,
            context_engine=delegate_ctx,
            orchestrator=delegate_orch,
            model_override=effective_model,
        )
        agent = self.build_instance(delegate_context)

        full_response = ""
        child_run_id = uuid.uuid4().hex

        async for event in agent.chat_stream(
            user_message=task,
            session_id=child_sid,
            model=effective_model,
            owner_id=context.owner_id,
            run_id=child_run_id,
            parent_run_id=parent_run_id,
            agent_revision=getattr(context.profile, "revision", None),
            loop_mode=loop_mode or "auto",
            runtime_metadata=run_metadata,
        ):
            if event["type"] == "token":
                full_response += event.get("content", "")
            elif event["type"] == "error":
                err_msg = event.get("message", event.get("text", ""))
                fallback_model = get_default_model(context.adapter)
                is_model_404 = bool(re.search(r"\b404\b", err_msg)) and "model" in err_msg.lower()
                if is_model_404 and effective_model != fallback_model:
                    log(
                        "agent",
                        child_sid,
                        f"Child agent 404 on {effective_model} — retrying with {fallback_model}",
                        level="warn",
                    )
                    return await self.run_task(
                        replace(context, model_override=fallback_model),
                        task,
                        parent_id=parent_id,
                        parent_run_id=parent_run_id,
                    )
                if not err_msg:
                    err_msg = "Unknown error from delegated agent"
                return f"Error from agent '{context.agent_id}': {err_msg}"

        return full_response.strip()


def create_native_runtime_adapter() -> AgentRuntimeAdapter:
    return NativeAgentRuntimeAdapter()
