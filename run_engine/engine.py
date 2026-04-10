from __future__ import annotations

import inspect
import json
import re
from typing import Any, AsyncIterator, Optional

from config.logger import log
from engine.candidate_signals import extract_stance_signals
from engine.conversation_tension import analyze_conversation_tension
from engine.context_schema import ContextPhase
from engine.deterministic_facts import extract_user_stated_fact_updates
from engine.tool_contract import (
    canonical_tool_call,
    clip_text,
    format_tool_result_line,
    is_retry_sensitive_tool,
    summarize_tool_results,
    tool_call_signature,
)
from llm.adapter_factory import create_adapter, strip_provider_prefix
from llm.base_adapter import BaseLLMAdapter
from memory.semantic_graph import SemanticGraph
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolCall, ToolRegistry
from run_engine.context_packets import PacketBuildInputs, build_phase_packet
from run_engine.evidence_lane import (
    build_evidence_focus_lines,
    build_evidence_priority_reminder,
    extract_exact_query_targets,
    is_substantive_tool_result,
    select_working_evidence,
    tool_kind_priority,
    tool_result_matches_exact_target,
)
from run_engine.memory_pipeline import (
    apply_memory_commit,
    build_deterministic_memory_commit,
    build_grounding_text,
    plan_memory_commit,
)
from run_engine.tool_selection import (
    build_actor_request_content,
    extract_tool_call,
    fallback_tool_call,
    summarize_arguments,
)
from run_engine.types import CompiledPassPacket, RunEngineRequest


TOOL_ACTOR_PROMPT = """\
You are the Shovs Run Engine actor.

Your job is to choose exactly one next tool call from the allowed tools.
Use the current user objective, prior tool results, and the execution clue.

Rules:
- The allowed tools are real and available in this runtime right now.
- Prefer the smallest useful next action.
- For current or time-sensitive requests such as latest, current, today, news, prices, or market data, use an allowed web_search tool instead of saying you cannot browse.
- When the user says "search", "find", "look up", "fetch", "gather", or "investigate", always use a tool. Do not ask for clarification.
- Do not ask clarifying questions. If the objective can be advanced by a tool call, make the call.
- Preserve exact user entities, domains, URLs, file names, and keywords.
- If a direct URL already exists, prefer using it rather than inventing a new one.
- Use only the allowed tools.
- Do not create files or code unless the user explicitly asked for a file, script, app, or other artifact.
- Return a tool call, not prose.
"""

FINAL_RESPONSE_PROMPT = """\
You are Shovs, operating inside the canonical Run Engine.

Rules:
- Answer the user directly using the evidence gathered in this run.
- If tools were available earlier in this run, do not claim you lack browsing or tool access.
- Do not mention hidden planning, phase names, checkpoints, or internal protocol.
- Do not fabricate completed actions, tool results, URLs, or files.
- If evidence is missing, say what is missing plainly.
- If the phase context shows drift or contradiction between the user's current turn and earlier user-stated facts, name that tension plainly.
- Do not optimize only for comfort or agreement. If the user's current claim conflicts with earlier facts or evidence, challenge it directly and ask for reconciliation when needed.
"""


class RunEngine:
    def __init__(
        self,
        *,
        adapter: BaseLLMAdapter,
        sessions: SessionManager,
        tool_registry: ToolRegistry,
        run_store: RunStore,
        trace_store,
        orchestrator: Optional[AgenticOrchestrator] = None,
        context_engine: Optional[object] = None,
        graph: Optional[SemanticGraph] = None,
    ):
        self.adapter = adapter
        self.sessions = sessions
        self.tool_registry = tool_registry
        self.run_store = run_store
        self.trace_store = trace_store
        self.orchestrator = orchestrator
        self.context_engine = context_engine
        self.graph = graph
        self._v1_engine = context_engine
        self._v2_engine = None
        self._v3_engine = None

    @staticmethod
    def _is_trivial_acknowledgement(message: str) -> bool:
        lowered = str(message or "").strip().lower()
        return bool(
            re.fullmatch(
                r"(hi|hello|hey|thanks|thank you|ok|okay|cool|nice|yes|yeah|yep|sure|sounds good)[.! ]*",
                lowered,
            )
        )

    @staticmethod
    def _is_ambiguous_followup(message: str) -> bool:
        lowered = str(message or "").strip().lower()
        if not lowered:
            return False
        return bool(
            re.fullmatch(
                r"(yes|yeah|yep|sure|ok|okay|do it|go ahead|please do|run it|use it|save it|again|try again|retry|read full chat|that one|this one|do that)[.! ]*",
                lowered,
            )
        )

    @classmethod
    def _resolve_effective_objective(
        cls,
        user_message: str,
        session_history: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        current = str(user_message or "").strip()
        if not current:
            return ""
        if not cls._is_ambiguous_followup(current):
            return current

        skipped_current = False
        for entry in reversed(list(session_history or [])):
            if str(entry.get("role") or "") != "user":
                continue
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            if not skipped_current and content == current:
                skipped_current = True
                continue
            if cls._is_trivial_acknowledgement(content) or cls._is_ambiguous_followup(content):
                continue
            return content
        return current

    def _resolve_context_engine(
        self,
        mode: Optional[str],
        *,
        compression_model: Optional[str] = None,
    ):
        normalized = str(mode or "v1").strip().lower()
        if normalized not in {"v1", "v2", "v3"}:
            normalized = "v1"

        if normalized == "v2":
            if self._v2_engine is None:
                from engine.context_engine_v2 import ContextEngineV2

                self._v2_engine = ContextEngineV2(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=compression_model or "llama3.2",
                )
            engine = self._v2_engine
        elif normalized == "v3":
            if self._v3_engine is None:
                from engine.context_engine_v3 import ContextEngineV3

                self._v3_engine = ContextEngineV3(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=compression_model or "llama3.2",
                )
            engine = self._v3_engine
        else:
            engine = self._v1_engine

        if engine is not None:
            if hasattr(engine, "set_adapter"):
                engine.set_adapter(self.adapter)
            if compression_model and hasattr(engine, "compression_model"):
                engine.compression_model = compression_model
        return engine

    async def stream(self, request: RunEngineRequest) -> AsyncIterator[dict[str, Any]]:
        session = self.sessions.get_or_create(
            session_id=request.session_id,
            model=request.model,
            system_prompt=request.system_prompt,
            agent_id=request.agent_id,
            owner_id=request.owner_id,
        )
        session_model = request.model or session.model or "llama3.2"
        session_system_prompt = request.system_prompt or session.system_prompt or ""
        requested_context_mode = str(
            request.context_mode or getattr(session, "context_mode", "v1") or "v1"
        ).strip().lower()
        if requested_context_mode not in {"v1", "v2", "v3"}:
            requested_context_mode = str(getattr(session, "context_mode", "v1") or "v1").strip().lower()
        if getattr(session, "context_mode", "v1") != requested_context_mode:
            self.sessions.set_context_mode(session.id, requested_context_mode)
            refreshed_session = self.sessions.get(session.id, owner_id=request.owner_id)
            if refreshed_session is not None:
                session = refreshed_session
        active_context_engine = self._resolve_context_engine(
            requested_context_mode,
            compression_model=request.context_model or session_model,
        )
        current_adapter = self._resolve_adapter(session_model)
        clean_model = strip_provider_prefix(session_model)

        run = self.run_store.start_run(
            session_id=session.id,
            agent_id=request.agent_id,
            model=session_model,
            owner_id=request.owner_id,
            agent_revision=request.agent_revision,
        )

        self._trace(
            request=request,
            run_id=run.run_id,
            event_type="run_engine_start",
            data={"model": session_model, "runtime": "run_engine"},
        )

        is_first_exchange = self.sessions.append_message(session.id, "user", request.user_message)
        yield {"type": "session", "session_id": session.id, "run_id": run.run_id}
        effective_objective = self._resolve_effective_objective(
            request.user_message,
            list(getattr(session, "sliding_window", []) or []),
        )

        try:
            tool_results: list[dict[str, Any]] = []
            selected_tools = list(request.forced_tools)
            planner_enabled = bool(request.use_planner)
            allowed_tools_list = self._list_allowed_tools(request.allowed_tools)
            latest_strategy = ""
            latest_notes = ""
            latest_observation_status = ""
            latest_observation_tools: list[str] = []
            normalized_max_tool_calls = self._normalize_optional_limit(
                request.max_tool_calls,
                minimum=1,
                maximum=24,
            )
            max_turns = max(1, min(int(request.max_turns or 3), 6))
            if normalized_max_tool_calls is not None:
                max_turns = min(max_turns, normalized_max_tool_calls)
            current_facts = self._current_facts(session.id, request.owner_id)
            stance_signals = extract_stance_signals(
                request.user_message,
                turn_index=int(getattr(session, "message_count", 0) or 0),
            )
            if stance_signals:
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="stance_signals_extracted",
                    data={
                        "count": len(stance_signals),
                        "signals": [
                            {
                                "topic": item.get("topic"),
                                "position": item.get("position"),
                                "confidence": item.get("confidence"),
                            }
                            for item in stance_signals[:6]
                        ],
                    },
                )
            deterministic_keyed_facts, deterministic_voids = extract_user_stated_fact_updates(
                request.user_message,
                current_facts=current_facts,
            )
            conversation_tension = analyze_conversation_tension(
                user_message=request.user_message,
                current_facts=current_facts,
                deterministic_keyed_facts=deterministic_keyed_facts,
                session_history=list(getattr(session, "full_history", []) or []),
                candidate_signals=list(getattr(session, "candidate_signals", []) or []),
                current_stance_signals=stance_signals,
            )
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="deterministic_fact_extractor",
                data={
                    "fact_count": len(deterministic_keyed_facts),
                    "void_count": len(deterministic_voids),
                    "facts": [
                        {
                            "subject": item.get("subject"),
                            "predicate": item.get("predicate"),
                            "object": item.get("object"),
                        }
                        for item in deterministic_keyed_facts
                    ],
                    "voids": list(deterministic_voids),
                },
            )
            if conversation_tension.summary or conversation_tension.conflicting_facts:
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="conversation_tension",
                    data={
                        "summary": conversation_tension.summary,
                        "notes": conversation_tension.notes,
                        "challenge_level": conversation_tension.challenge_level,
                        "should_challenge": conversation_tension.should_challenge,
                        "drift_detected": conversation_tension.drift_detected,
                        "conflicting_facts": list(conversation_tension.conflicting_facts),
                    },
                )
                yield {
                    "type": "conversation_tension",
                    "summary": conversation_tension.summary,
                    "notes": conversation_tension.notes,
                    "challenge_level": conversation_tension.challenge_level,
                    "should_challenge": conversation_tension.should_challenge,
                    "drift_detected": conversation_tension.drift_detected,
                    "conflict_count": len(conversation_tension.conflicting_facts),
                }

            if not selected_tools and planner_enabled and self.orchestrator is not None:
                planning_packet = self._compile_phase_packet(
                    context_engine=active_context_engine,
                    run_id=run.run_id,
                    request=request,
                    session=session,
                    phase=ContextPhase.PLANNING,
                    system_prompt=session_system_prompt,
                    effective_objective=effective_objective,
                    current_context=session.compressed_context,
                    allowed_tools=allowed_tools_list,
                    tool_results=tool_results,
                    current_facts=current_facts,
                    conversation_tension=conversation_tension,
                )
                structured_plan = await self.orchestrator.plan_with_context(
                    query=effective_objective,
                    tools_list=allowed_tools_list,
                    model=request.planner_model or session_model,
                    session_has_history=bool(session.full_history),
                    current_fact_count=0,
                    failed_tools=[],
                    compiled_context=planning_packet.content,
                )
                selected_tools = [
                    entry["name"]
                    for entry in structured_plan.get("tools", [])
                    if isinstance(entry, dict) and isinstance(entry.get("name"), str)
                ]
                if structured_plan.get("strategy"):
                    strategy = str(structured_plan.get("strategy"))
                    latest_strategy = strategy
                    latest_notes = str(structured_plan.get("notes") or "")
                    self.run_store.save_checkpoint(
                        run_id=run.run_id,
                        phase="planning",
                        status="planned",
                        strategy=strategy,
                        tools=selected_tools,
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="plan",
                        data={"strategy": strategy, "tools": selected_tools},
                    )
                    yield {
                        "type": "plan",
                        "strategy": strategy,
                        "tools": selected_tools,
                        "confidence": structured_plan.get("confidence"),
                    }
                self._save_pass_record(
                    run_id=run.run_id,
                    phase=ContextPhase.PLANNING,
                    tool_turn=0,
                    status="planned",
                    objective=effective_objective,
                    strategy=str(latest_strategy or structured_plan.get("strategy") or ""),
                    notes=str(latest_notes or structured_plan.get("notes") or ""),
                    selected_tools=selected_tools,
                    tool_results=[],
                    packet=planning_packet,
                )

            if not selected_tools:
                selected_tools = self._bootstrap_tools_for_turn(
                    user_message=request.user_message,
                    allowed_tools=allowed_tools_list,
                    planner_enabled=planner_enabled,
                    session_history=list(getattr(session, "sliding_window", []) or []),
                )
                if selected_tools:
                    bootstrap_strategy = (
                        "Planner returned no actionable tools; using deterministic fallback tool bootstrap."
                    )
                    if not latest_strategy:
                        latest_strategy = bootstrap_strategy
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="plan",
                        data={
                            "strategy": bootstrap_strategy,
                            "tools": selected_tools,
                            "source": "run_engine_fallback",
                        },
                    )
                    yield {
                        "type": "plan",
                        "strategy": bootstrap_strategy,
                        "tools": selected_tools,
                        "confidence": 0.35,
                    }
                    self._save_pass_record(
                        run_id=run.run_id,
                        phase=ContextPhase.PLANNING,
                        tool_turn=0,
                        status="planned_fallback",
                        objective=request.user_message,
                        strategy=latest_strategy,
                        notes="Deterministic fallback tool bootstrap.",
                        selected_tools=selected_tools,
                        tool_results=[],
                        packet=self._compile_phase_packet(
                            context_engine=active_context_engine,
                            run_id=run.run_id,
                            request=request,
                            session=session,
                            phase=ContextPhase.PLANNING,
                            system_prompt=session_system_prompt,
                            effective_objective=effective_objective,
                            current_context=session.compressed_context,
                            allowed_tools=allowed_tools_list,
                            tool_results=tool_results,
                            current_facts=current_facts,
                            conversation_tension=conversation_tension,
                            trace_phase_label="planning_fallback",
                        ),
                    )

            tool_turn = 0
            tool_call_count = 0
            repeated_tool_signatures: dict[str, int] = {}
            while selected_tools and tool_turn < max_turns:
                if (
                    normalized_max_tool_calls is not None
                    and tool_call_count >= normalized_max_tool_calls
                ):
                    limit_message = f"Max tool calls ({normalized_max_tool_calls}) reached for this run."
                    self.run_store.save_checkpoint(
                        run_id=run.run_id,
                        phase="acting",
                        tool_turn=tool_turn,
                        status="max_tool_calls_reached",
                        notes=limit_message,
                        tools=list(selected_tools)[:3],
                        tool_results=summarize_tool_results(tool_results, limit=3, preview_chars=220),
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="tool_limit_reached",
                        data={
                            "max_tool_calls": normalized_max_tool_calls,
                            "tool_call_count": tool_call_count,
                        },
                    )
                    selected_tools = []
                    break
                tool_turn += 1
                acting_packet = self._compile_phase_packet(
                    context_engine=active_context_engine,
                    run_id=run.run_id,
                    request=request,
                    session=session,
                    phase=ContextPhase.ACTING,
                    system_prompt=session_system_prompt,
                    effective_objective=effective_objective,
                    current_context=session.compressed_context,
                    allowed_tools=allowed_tools_list,
                    tool_results=tool_results,
                    tool_turn=tool_turn,
                    strategy=latest_strategy,
                    notes=latest_notes,
                    current_facts=current_facts,
                    conversation_tension=conversation_tension,
                )
                tool_call = await self._select_tool_call(
                    adapter=current_adapter,
                    model=clean_model,
                    request=request,
                    session=session,
                    allowed_tools=selected_tools,
                    tool_results=tool_results,
                    context_block=acting_packet.content,
                )
                if tool_call is None:
                    break

                if not isinstance(tool_call.arguments, dict):
                    tool_call.arguments = {}
                if tool_call.tool_name == "web_search":
                    if request.search_backend and request.search_backend != "auto":
                        tool_call.arguments["backend"] = request.search_backend
                    if request.search_engine:
                        tool_call.arguments["search_engine"] = request.search_engine

                tool_call_count += 1

                validation_error = self.tool_registry.validate_tool_call(tool_call)
                call_payload = canonical_tool_call(tool_call.tool_name, tool_call.arguments or {})
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="tool_call",
                    data={**call_payload, "turn": tool_turn},
                )
                yield {"type": "tool_call", **call_payload}
                if validation_error:
                    result_payload = {
                        "tool_name": tool_call.tool_name,
                        "success": False,
                        "content": f"Tool validation failed: {validation_error}",
                        "arguments": dict(tool_call.arguments or {}),
                    }
                    tool_results.append(result_payload)
                    self.run_store.save_checkpoint(
                        run_id=run.run_id,
                        phase="acting",
                        tool_turn=tool_turn,
                        status="validation_failed",
                        notes=validation_error,
                        tools=[tool_call.tool_name],
                        tool_results=summarize_tool_results([result_payload], limit=1, preview_chars=220),
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="tool_result",
                        data={
                            "tool": tool_call.tool_name,
                            "tool_name": tool_call.tool_name,
                            "success": False,
                            "status": "failed",
                            "content_length": len(result_payload["content"]),
                            "content_preview": clip_text(result_payload["content"], 280),
                            "turn": tool_turn,
                        },
                    )
                    yield {
                        "type": "tool_result",
                        "tool": tool_call.tool_name,
                        "tool_name": tool_call.tool_name,
                        "success": False,
                        "status": "failed",
                        "content_preview": clip_text(result_payload["content"], 280),
                        "content": result_payload["content"],
                    }
                    break

                signature = tool_call_signature(tool_call.tool_name, tool_call.arguments or {})
                if is_retry_sensitive_tool(tool_call.tool_name):
                    repeat_count = repeated_tool_signatures.get(signature, 0)
                    if repeat_count >= 1:
                        duplicate_message = (
                            f"Duplicate {tool_call.tool_name} call suppressed for this run. "
                            "Use the existing result or pivot to a different query/source."
                        )
                        result_payload = {
                            "tool_name": tool_call.tool_name,
                            "success": False,
                            "content": duplicate_message,
                            "arguments": dict(tool_call.arguments or {}),
                        }
                        tool_results.append(result_payload)
                        self.run_store.save_checkpoint(
                            run_id=run.run_id,
                            phase="acting",
                            tool_turn=tool_turn,
                            status="duplicate_suppressed",
                            notes=duplicate_message,
                            tools=[tool_call.tool_name],
                            tool_results=summarize_tool_results([result_payload], limit=1, preview_chars=220),
                        )
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="tool_result",
                            data={
                                "tool": tool_call.tool_name,
                                "tool_name": tool_call.tool_name,
                                "success": False,
                                "status": "failed",
                                "content_length": len(duplicate_message),
                                "content_preview": clip_text(duplicate_message, 280),
                                "turn": tool_turn,
                            },
                        )
                        yield {
                            "type": "tool_result",
                            "tool": tool_call.tool_name,
                            "tool_name": tool_call.tool_name,
                            "success": False,
                            "status": "failed",
                            "content_preview": clip_text(duplicate_message, 280),
                            "content": duplicate_message,
                        }
                        self._save_pass_record(
                            run_id=run.run_id,
                            phase=ContextPhase.ACTING,
                            tool_turn=tool_turn,
                            status="duplicate_suppressed",
                            objective=effective_objective,
                            strategy=latest_strategy,
                            notes=duplicate_message,
                            selected_tools=[tool_call.tool_name],
                            tool_results=tool_results[-3:],
                            packet=acting_packet,
                        )
                        repeated_tool_signatures[signature] = repeat_count + 1
                        if self.orchestrator is None or not planner_enabled:
                            selected_tools = []
                            break
                        observation_packet = self._compile_phase_packet(
                            context_engine=active_context_engine,
                            run_id=run.run_id,
                            request=request,
                            session=session,
                            phase=ContextPhase.VERIFICATION,
                            system_prompt=session_system_prompt,
                            effective_objective=effective_objective,
                            current_context=session.compressed_context,
                            allowed_tools=allowed_tools_list,
                            tool_results=tool_results,
                            tool_turn=tool_turn,
                            strategy=latest_strategy,
                            notes=latest_notes,
                            current_facts=current_facts,
                            conversation_tension=conversation_tension,
                            trace_phase_label="observation",
                        )
                        observation = await self.orchestrator.observe_with_context(
                            query=effective_objective,
                            tools_list=allowed_tools_list,
                            tool_results=tool_results,
                            model=request.planner_model or session_model,
                            compiled_context=observation_packet.content,
                        )
                        normalized_observation = self._normalize_observation_decision(observation)
                        selected_tools = list(normalized_observation["selected_tools"])
                        latest_strategy = str(normalized_observation["strategy"] or latest_strategy)
                        latest_notes = str(normalized_observation["notes"] or latest_notes)
                        latest_observation_status = str(normalized_observation["status"] or "")
                        latest_observation_tools = list(selected_tools)
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="manager_observation",
                            data={
                                "tool_turn": tool_turn,
                                "status": normalized_observation["status"],
                                "raw_status": normalized_observation["raw_status"],
                                "strategy": normalized_observation["strategy"],
                                "notes": normalized_observation["notes"],
                                "confidence": normalized_observation["confidence"],
                                "should_continue": normalized_observation["should_continue"],
                                "tools": selected_tools,
                            },
                        )
                        self.run_store.save_checkpoint(
                            run_id=run.run_id,
                            phase="observation",
                            tool_turn=tool_turn,
                            status=normalized_observation["status"],
                            strategy=normalized_observation["strategy"],
                            notes=normalized_observation["notes"],
                            tools=selected_tools,
                            tool_results=summarize_tool_results(tool_results, limit=3, preview_chars=220),
                        )
                        self._save_pass_record(
                            run_id=run.run_id,
                            phase="observation",
                            tool_turn=tool_turn,
                            status=normalized_observation["status"],
                            objective=effective_objective,
                            strategy=latest_strategy,
                            notes=str(normalized_observation["notes"] or latest_notes),
                            selected_tools=selected_tools,
                            tool_results=tool_results[-3:],
                            packet=observation_packet,
                        )
                        if normalized_observation["should_continue"]:
                            strategy = str(normalized_observation["strategy"] or "Manager selected follow-up tools.")
                            self._trace(
                                request=request,
                                run_id=run.run_id,
                                event_type="plan",
                                data={"strategy": strategy, "tools": selected_tools},
                            )
                            yield {
                                "type": "plan",
                                "strategy": strategy,
                                "tools": selected_tools,
                                "confidence": normalized_observation["confidence"],
                            }
                        if not normalized_observation["should_continue"]:
                            selected_tools = []
                        continue
                repeated_tool_signatures[signature] = repeated_tool_signatures.get(signature, 0) + 1

                tool_result = await self.tool_registry.execute(
                    tool_call,
                    context={
                        "_session_id": session.id,
                        "session_id": session.id,
                        "_owner_id": request.owner_id,
                        "owner_id": request.owner_id,
                        "_run_id": run.run_id,
                        "_agent_id": request.agent_id,
                        "_embed_model": request.embed_model or "nomic-embed-text",
                        "_runtime_path": "run_engine",
                    },
                )
                result_payload = {
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                    "content": tool_result.content,
                    "arguments": dict(tool_call.arguments or {}),
                }
                tool_results.append(result_payload)

                self.run_store.save_checkpoint(
                    run_id=run.run_id,
                    phase="acting",
                    tool_turn=tool_turn,
                    status="tool_executed",
                    tools=[tool_call.tool_name],
                    tool_results=summarize_tool_results([result_payload], limit=1, preview_chars=220),
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="tool_result",
                    data={
                        "tool": tool_result.tool_name,
                        "tool_name": tool_result.tool_name,
                        "success": tool_result.success,
                        "status": "ok" if tool_result.success else "failed",
                        "content_length": len(tool_result.content),
                        "content_preview": clip_text(tool_result.content, 280),
                        "turn": tool_turn,
                    },
                )
                yield {
                    "type": "tool_result",
                    "tool": tool_result.tool_name,
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                    "status": "ok" if tool_result.success else "failed",
                    "content_preview": clip_text(tool_result.content, 280),
                    "content": tool_result.content,
                }
                self._save_pass_record(
                    run_id=run.run_id,
                    phase=ContextPhase.ACTING,
                    tool_turn=tool_turn,
                    status="tool_executed",
                    objective=effective_objective,
                    strategy=latest_strategy,
                    notes=latest_notes,
                    selected_tools=[tool_call.tool_name],
                    tool_results=tool_results[-3:],
                    packet=acting_packet,
                )

                if self.orchestrator is None or not planner_enabled:
                    selected_tools = []
                    break

                observation_packet = self._compile_phase_packet(
                    context_engine=active_context_engine,
                    run_id=run.run_id,
                    request=request,
                    session=session,
                    phase=ContextPhase.VERIFICATION,
                    system_prompt=session_system_prompt,
                    effective_objective=effective_objective,
                    current_context=session.compressed_context,
                    allowed_tools=allowed_tools_list,
                    tool_results=tool_results,
                    tool_turn=tool_turn,
                    strategy=latest_strategy,
                    notes=latest_notes,
                    current_facts=current_facts,
                    conversation_tension=conversation_tension,
                    trace_phase_label="observation",
                )
                observation = await self.orchestrator.observe_with_context(
                    query=effective_objective,
                    tools_list=allowed_tools_list,
                    tool_results=tool_results,
                    model=request.planner_model or session_model,
                    compiled_context=observation_packet.content,
                )
                normalized_observation = self._normalize_observation_decision(observation)
                selected_tools = list(normalized_observation["selected_tools"])
                latest_strategy = str(normalized_observation["strategy"] or latest_strategy)
                latest_notes = str(normalized_observation["notes"] or latest_notes)
                latest_observation_status = str(normalized_observation["status"] or "")
                latest_observation_tools = list(selected_tools)
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="manager_observation",
                    data={
                        "tool_turn": tool_turn,
                        "status": normalized_observation["status"],
                        "raw_status": normalized_observation["raw_status"],
                        "strategy": normalized_observation["strategy"],
                        "notes": normalized_observation["notes"],
                        "confidence": normalized_observation["confidence"],
                        "should_continue": normalized_observation["should_continue"],
                        "tools": selected_tools,
                    },
                )
                self.run_store.save_checkpoint(
                    run_id=run.run_id,
                    phase="observation",
                    tool_turn=tool_turn,
                    status=normalized_observation["status"],
                    strategy=normalized_observation["strategy"],
                    notes=normalized_observation["notes"],
                    tools=selected_tools,
                    tool_results=summarize_tool_results(tool_results, limit=3, preview_chars=220),
                )
                self._save_pass_record(
                    run_id=run.run_id,
                    phase="observation",
                    tool_turn=tool_turn,
                    status=normalized_observation["status"],
                    objective=effective_objective,
                    strategy=latest_strategy,
                    notes=str(normalized_observation["notes"] or latest_notes),
                    selected_tools=selected_tools,
                    tool_results=tool_results[-3:],
                    packet=observation_packet,
                )
                if normalized_observation["should_continue"]:
                    strategy = str(normalized_observation["strategy"] or "Manager selected follow-up tools.")
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="plan",
                        data={"strategy": strategy, "tools": selected_tools},
                    )
                    yield {
                        "type": "plan",
                        "strategy": strategy,
                        "tools": selected_tools,
                        "confidence": normalized_observation["confidence"],
                    }
                if not normalized_observation["should_continue"]:
                    selected_tools = []

            final_response = ""
            response_packet = self._compile_phase_packet(
                context_engine=active_context_engine,
                run_id=run.run_id,
                request=request,
                session=session,
                phase=ContextPhase.RESPONSE,
                system_prompt=session_system_prompt,
                effective_objective=effective_objective,
                current_context=session.compressed_context,
                allowed_tools=allowed_tools_list,
                tool_results=tool_results,
                tool_turn=tool_turn,
                strategy=latest_strategy,
                notes=latest_notes,
                observation_status=latest_observation_status,
                observation_tools=latest_observation_tools,
                current_facts=current_facts,
                conversation_tension=conversation_tension,
            )
            final_messages = self._build_final_messages(
                session=session,
                system_prompt=session_system_prompt,
                user_message=request.user_message,
                effective_objective=effective_objective,
                tool_results=tool_results,
                context_block=response_packet.content,
                allowed_tools=[tool.get("name") for tool in allowed_tools_list if isinstance(tool, dict) and isinstance(tool.get("name"), str)],
                controller_summary=self._build_controller_summary(
                    status=latest_observation_status,
                    strategy=latest_strategy,
                    notes=latest_notes,
                    selected_tools=latest_observation_tools,
                ),
            )
            self.run_store.save_checkpoint(
                run_id=run.run_id,
                phase="response",
                status="streaming",
                tool_turn=tool_turn,
                tool_results=summarize_tool_results(tool_results, limit=5, preview_chars=220),
            )
            self._save_pass_record(
                run_id=run.run_id,
                phase=ContextPhase.RESPONSE,
                tool_turn=tool_turn,
                status="streaming",
                objective=effective_objective,
                strategy=latest_strategy,
                notes=latest_notes,
                selected_tools=[],
                tool_results=tool_results[-5:],
                packet=response_packet,
            )
            async for token in current_adapter.stream(
                model=clean_model,
                messages=final_messages,
                temperature=0.2,
                images=request.images,
            ):
                final_response += token
                yield {"type": "token", "content": token}

            response_summary = self._build_controller_summary(
                status=latest_observation_status,
                strategy=latest_strategy,
                notes=latest_notes,
                selected_tools=latest_observation_tools,
            )
            self.run_store.save_checkpoint(
                run_id=run.run_id,
                phase="response",
                status="complete",
                tool_turn=tool_turn,
                strategy=latest_strategy,
                notes=response_summary,
                tools=list(latest_observation_tools),
                tool_results=summarize_tool_results(tool_results, limit=5, preview_chars=220),
            )
            self._save_pass_record(
                run_id=run.run_id,
                phase=ContextPhase.RESPONSE,
                tool_turn=tool_turn,
                status="complete",
                objective=effective_objective,
                strategy=latest_strategy,
                notes=response_summary,
                selected_tools=list(latest_observation_tools),
                tool_results=tool_results[-5:],
                packet=response_packet,
                response_preview=final_response,
            )

            verification = {"supported": True, "issues": [], "confidence": 0.0}
            if self.orchestrator is not None:
                verification_packet = self._compile_phase_packet(
                    context_engine=active_context_engine,
                    run_id=run.run_id,
                    request=request,
                    session=session,
                    phase=ContextPhase.VERIFICATION,
                    system_prompt=session_system_prompt,
                    effective_objective=effective_objective,
                    current_context=session.compressed_context,
                    allowed_tools=allowed_tools_list,
                    tool_results=tool_results,
                    tool_turn=tool_turn,
                    strategy=latest_strategy,
                    notes=latest_notes,
                    observation_status=latest_observation_status,
                    observation_tools=latest_observation_tools,
                    final_response=final_response,
                    current_facts=current_facts,
                    conversation_tension=conversation_tension,
                )
                verification = await self.orchestrator.verify_with_context(
                    query=effective_objective,
                    response=final_response,
                    tool_results=tool_results,
                    model=request.planner_model or session_model,
                    compiled_context=verification_packet.content,
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="verification_result",
                    data={
                        "supported": bool(verification.get("supported", True)),
                        "issues": [str(item) for item in (verification.get("issues") or [])[:5]],
                        "issue_count": len(verification.get("issues") or []),
                        "confidence": verification.get("confidence"),
                    },
                )
                self.run_store.save_checkpoint(
                    run_id=run.run_id,
                    phase="verification",
                    tool_turn=tool_turn,
                    status="supported" if bool(verification.get("supported", True)) else "unsupported",
                    notes="; ".join(str(item) for item in (verification.get("issues") or [])[:3]),
                    confidence=verification.get("confidence"),
                    tool_results=summarize_tool_results(tool_results, limit=5, preview_chars=220),
                )
                self._save_pass_record(
                    run_id=run.run_id,
                    phase=ContextPhase.VERIFICATION,
                    tool_turn=tool_turn,
                    status="verified" if bool(verification.get("supported", True)) else "warning",
                    objective=effective_objective,
                    strategy=latest_strategy,
                    notes="; ".join(verification.get("issues") or []),
                    selected_tools=[],
                    tool_results=tool_results[-5:],
                    packet=verification_packet,
                    response_preview=final_response,
                )
            self.run_store.save_eval(
                run_id=run.run_id,
                session_id=session.id,
                eval_type="response_support",
                phase="verification",
                passed=bool(verification.get("supported", True)),
                owner_id=request.owner_id,
                score=float(verification.get("confidence") or 0.0),
                detail="; ".join(verification.get("issues") or []),
                metadata={"issues": verification.get("issues") or []},
            )
            verification_supported = bool(verification.get("supported", True))
            if not verification_supported:
                warning = list(verification.get("issues") or [])
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="verification_warning",
                    data={
                        "issues": warning,
                        "confidence": verification.get("confidence"),
                        "issue_count": len(warning),
                    },
                )
                yield {
                    "type": "verification_warning",
                    "issues": warning,
                    "confidence": verification.get("confidence"),
                }

            self.sessions.append_message(session.id, "assistant", final_response)
            if verification_supported:
                await self._commit_context(
                    context_engine=active_context_engine,
                    session_id=session.id,
                    request=request,
                    assistant_response=final_response,
                    current_context=session.compressed_context,
                    is_first_exchange=is_first_exchange,
                    tool_results=tool_results,
                    model=request.context_model or clean_model,
                    run_id=run.run_id,
                    deterministic_keyed_facts=deterministic_keyed_facts,
                    deterministic_voids=deterministic_voids,
                    current_facts=current_facts,
                    stance_signals=stance_signals,
                )
            else:
                self.run_store.save_checkpoint(
                    run_id=run.run_id,
                    phase="memory_commit",
                    status="blocked_verification",
                    tool_turn=tool_turn,
                    notes="Managed runtime skipped memory commit because verification returned unsupported.",
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="memory_commit_skipped",
                    data={
                        "reason": "verification_unsupported",
                        "issue_count": len(verification.get("issues") or []),
                        "confidence": verification.get("confidence"),
                    },
                )
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="assistant_response",
                data={"content": final_response, "content_length": len(final_response)},
            )
            self.run_store.finish_run(run.run_id, status="completed")
            yield {"type": "done", "run_id": run.run_id, "session_id": session.id}
        except Exception as exc:
            self.run_store.finish_run(run.run_id, status="failed", error=str(exc))
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="error",
                data={"message": str(exc)},
            )
            raise

    def _resolve_adapter(self, model: str) -> BaseLLMAdapter:
        if model and ":" in model:
            return create_adapter(provider=model)
        return self.adapter

    def _list_allowed_tools(self, allowed_tools: tuple[str, ...]) -> list[dict[str, Any]]:
        allowed = set(allowed_tools or ())
        if not allowed:
            return self.tool_registry.list_tools()
        return [tool for tool in self.tool_registry.list_tools() if tool.get("name") in allowed]

    def _get_schema_subset(self, allowed_tools: list[str]) -> list[dict[str, Any]]:
        subset = []
        for name in allowed_tools:
            tool = self.tool_registry.get(name)
            if tool is None:
                continue
            subset.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return subset

    async def _select_tool_call(
        self,
        *,
        adapter: BaseLLMAdapter,
        model: str,
        request: RunEngineRequest,
        session,
        allowed_tools: list[str],
        tool_results: list[dict[str, Any]],
        context_block: str,
    ) -> Optional[ToolCall]:
        available_names = [name for name in allowed_tools if self.tool_registry.get(name) is not None]
        if not available_names:
            return None
        effective_objective = self._resolve_effective_objective(
            request.user_message,
            list(getattr(session, "sliding_window", []) or []),
        )

        messages = [
            {
                "role": "system",
                "content": TOOL_ACTOR_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_actor_request_content(
                    request=request,
                    session=session,
                    effective_objective=effective_objective,
                    allowed_tools=available_names,
                    tool_results=tool_results,
                    context_block=context_block,
                ),
            },
        ]
        raw = ""
        try:
            raw = await adapter.complete(
                model=model,
                messages=messages,
                temperature=0.0,
                images=request.images,
                tools=self._get_schema_subset(available_names),
            )
        except Exception as exc:
            log(
                "run_engine",
                request.session_id,
                f"Tool actor completion failed: {exc}",
                level="warn",
                owner_id=request.owner_id,
            )

        call = extract_tool_call(raw, self.tool_registry) if raw else None
        if call is not None:
            return call
        fallback_ranked = self._rank_fallback_tool_candidates(effective_objective, available_names)
        for name in fallback_ranked:
            fallback = fallback_tool_call(name, effective_objective)
            if fallback is not None:
                return fallback
        return None

    def _build_actor_request_content(
        self,
        *,
        request: RunEngineRequest,
        session,
        effective_objective: str,
        allowed_tools: list[str],
        tool_results: list[dict[str, Any]],
        context_block: str,
    ) -> str:
        return build_actor_request_content(
            user_message=request.user_message,
            effective_objective=effective_objective,
            session_first_message=str(session.first_message or ""),
            allowed_tools=allowed_tools,
            tool_results=tool_results,
            context_block=context_block,
            clip_text=self._clip,
        )

    @staticmethod
    def _normalize_optional_limit(
        value: Optional[int],
        *,
        minimum: int,
        maximum: int,
    ) -> Optional[int]:
        if value is None:
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        return max(minimum, min(normalized, maximum))

    @staticmethod
    def _bootstrap_tools_for_turn(
        *,
        user_message: str,
        allowed_tools: list[dict[str, Any]],
        planner_enabled: bool,
        session_history: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        names = [
            str(item.get("name") or "").strip()
            for item in allowed_tools
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        if not names:
            return []

        lowered = str(user_message or "").strip().lower()
        if not lowered:
            return []

        if RunEngine._is_trivial_acknowledgement(lowered):
            return []

        effective_objective = RunEngine._resolve_effective_objective(
            user_message,
            session_history,
        )
        ranked = RunEngine._rank_fallback_tool_candidates(effective_objective, names)
        if ranked:
            return ranked[:1]

        if (
            not planner_enabled
            and re.search(
                r"\b(search|find|fetch|lookup|look up|investigate|research|compare|check|show|get)\b",
                effective_objective.lower(),
            )
            and "web_search" in names
        ):
            return ["web_search"]
        return []

    @staticmethod
    def _rank_fallback_tool_candidates(user_message: str, available_names: list[str]) -> list[str]:
        ordered = [str(name or "").strip() for name in available_names if str(name or "").strip()]
        if not ordered:
            return []
        available = set(ordered)
        lowered = str(user_message or "").lower()
        ranked: list[str] = []

        def add(name: str):
            if name in available and name not in ranked:
                ranked.append(name)

        if re.search(r"https?://\S+", lowered):
            add("web_fetch")
        if re.search(r"\b(weather|temperature|forecast|rain|snow|wind)\b", lowered):
            add("weather_fetch")
        if re.search(r"\b(image|photo|picture|screenshot|logo)\b", lowered):
            add("image_search")
        if re.search(r"\b(remember|memory|previously|earlier|recall)\b", lowered):
            add("query_memory")
            add("rag_search")
        if re.search(r"\b(near me|nearby|place|restaurant|hotel|map|route|directions)\b", lowered):
            add("places_search")
        if re.search(
            r"\b(current|latest|today|news|price|prices|stock|stocks|market|search|find|lookup|look up|research|investigate|compare)\b",
            lowered,
        ):
            add("web_search")

        return ranked

    def _build_final_messages(
        self,
        *,
        session,
        system_prompt: str,
        user_message: str,
        effective_objective: str,
        tool_results: list[dict[str, Any]],
        context_block: str,
        allowed_tools: Optional[list[str]] = None,
        controller_summary: str = "",
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_parts = [part for part in [system_prompt.strip(), FINAL_RESPONSE_PROMPT.strip()] if part]
        if context_block:
            system_parts.append(f"Context:\n{context_block}")
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        reminder = self._build_response_reminder(
            controller_summary=controller_summary,
            user_message=effective_objective,
            tool_results=tool_results,
            allowed_tools=list(allowed_tools or []),
        )
        if reminder:
            messages.append({"role": "system", "content": reminder})

        history = list(session.sliding_window or [])[-6:]
        for entry in history:
            role = str(entry.get("role") or "user")
            content = str(entry.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        if tool_results:
            evidence = "\n\n".join(
                f"Tool: {item.get('tool_name')}\nSuccess: {item.get('success')}\nContent:\n{self._clip(str(item.get('content') or ''), 4000)}"
                for item in tool_results[-5:]
            )
            messages.append({"role": "system", "content": f"Evidence gathered this run:\n{evidence}"})

        final_user_prompt = user_message
        if effective_objective.strip() and effective_objective.strip() != user_message.strip():
            final_user_prompt = (
                f"Current user turn:\n{user_message}\n\n"
                f"Resolved working objective:\n{effective_objective}\n\n"
                "Respond to the current user turn by advancing the resolved working objective."
            )
        messages.append({"role": "user", "content": final_user_prompt})
        return messages

    async def _commit_context(
        self,
        *,
        context_engine,
        session_id: str,
        request: RunEngineRequest,
        assistant_response: str,
        current_context: str,
        is_first_exchange: bool,
        tool_results: list[dict[str, Any]],
        model: str,
        run_id: str,
        deterministic_keyed_facts: list[dict[str, Any]],
        deterministic_voids: list[dict[str, Any]],
        current_facts: list[tuple[str, str, str]],
        stance_signals: list[dict[str, Any]],
    ) -> None:
        session = self.sessions.get(session_id, owner_id=request.owner_id)
        existing_candidate_signals = list(getattr(session, "candidate_signals", []) or [])
        existing_candidate_context = getattr(session, "candidate_context", "") if session is not None else ""

        if context_engine is None or not hasattr(context_engine, "compress_exchange"):
            commit_plan = build_deterministic_memory_commit(
                deterministic_keyed_facts=deterministic_keyed_facts,
                deterministic_voids=deterministic_voids,
                existing_candidate_signals=existing_candidate_signals,
                existing_candidate_context=existing_candidate_context,
                user_message=request.user_message,
                new_candidate_signals=stance_signals,
            )
            outcome = await apply_memory_commit(
                sessions=self.sessions,
                session_id=session_id,
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                turn=self._message_count(session_id, request.owner_id),
                run_id=run_id,
                user_message=request.user_message,
                assistant_response=assistant_response,
                graph=self.graph,
                plan=commit_plan,
                current_context=current_context,
            )
            if outcome.indexed_fact_keys:
                self._trace(
                    request=request,
                    run_id=run_id,
                    event_type="facts_indexed",
                    data={"facts": list(outcome.indexed_fact_keys), "context_lines": outcome.context_lines},
                )
            return
        maybe_result = context_engine.compress_exchange(
            user_message=request.user_message,
            assistant_response=assistant_response,
            current_context=current_context,
            is_first_exchange=is_first_exchange,
            model=model,
            grounding_text=build_grounding_text(tool_results, successful_only=False, separator="\n\n"),
        )
        if inspect.isawaitable(maybe_result):
            maybe_result = await maybe_result
        if not isinstance(maybe_result, tuple) or not maybe_result:
            commit_plan = build_deterministic_memory_commit(
                deterministic_keyed_facts=deterministic_keyed_facts,
                deterministic_voids=deterministic_voids,
                existing_candidate_signals=existing_candidate_signals,
                existing_candidate_context=existing_candidate_context,
                user_message=request.user_message,
                new_candidate_signals=stance_signals,
            )
            outcome = await apply_memory_commit(
                sessions=self.sessions,
                session_id=session_id,
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                turn=self._message_count(session_id, request.owner_id),
                run_id=run_id,
                user_message=request.user_message,
                assistant_response=assistant_response,
                graph=self.graph,
                plan=commit_plan,
                current_context=current_context,
            )
            if outcome.indexed_fact_keys:
                self._trace(
                    request=request,
                    run_id=run_id,
                    event_type="facts_indexed",
                    data={"facts": list(outcome.indexed_fact_keys), "context_lines": outcome.context_lines},
                )
            return
        commit_plan = plan_memory_commit(
            context_result=maybe_result,
            user_message=request.user_message,
            tool_results=tool_results,
            deterministic_keyed_facts=deterministic_keyed_facts,
            deterministic_voids=deterministic_voids,
            current_facts=current_facts,
            existing_candidate_signals=existing_candidate_signals,
            existing_candidate_context=existing_candidate_context,
            new_candidate_signals=stance_signals,
        )
        outcome = await apply_memory_commit(
            sessions=self.sessions,
            session_id=session_id,
            owner_id=request.owner_id,
            agent_id=request.agent_id,
            turn=self._message_count(session_id, request.owner_id),
            run_id=run_id,
            user_message=request.user_message,
            assistant_response=assistant_response,
            graph=self.graph,
            plan=commit_plan,
            current_context=current_context,
        )
        blocked_keyed_facts = list(outcome.blocked_keyed_facts or [])
        if blocked_keyed_facts:
            self._trace(
                request=request,
                run_id=run_id,
                event_type="memory_fact_filter",
                data={
                    "blocked": [
                        {
                            "subject": item.get("subject"),
                            "predicate": item.get("predicate"),
                            "object": item.get("object"),
                            "reason": item.get("grounding_reason"),
                        }
                        for item in blocked_keyed_facts[:10]
                    ],
                    "blocked_count": len(blocked_keyed_facts),
                    "candidate_context_lines": len([line for line in outcome.candidate_context.splitlines() if line.strip()]),
                    "candidate_signal_count": len(outcome.candidate_signals or []),
                },
            )
        if outcome.indexed_fact_keys:
            self._trace(
                request=request,
                run_id=run_id,
                event_type="facts_indexed",
                data={"facts": list(outcome.indexed_fact_keys), "context_lines": outcome.context_lines},
            )
        self.run_store.save_checkpoint(
            run_id=run_id,
            phase="memory_commit",
            status="committed",
            notes=(
                f"facts={len(outcome.merged_facts or [])} "
                f"voids={len(outcome.merged_voids or [])} "
                f"blocked={len(blocked_keyed_facts)}"
            ),
            candidate_facts=[
                str(item.get("fact") or "").strip()
                for item in blocked_keyed_facts[:6]
                if str(item.get("fact") or "").strip()
            ],
        )

    def _build_context_block(self, context: str) -> str:
        if self.context_engine is None or not hasattr(self.context_engine, "build_context_block"):
            return ""
        try:
            return str(self.context_engine.build_context_block(context) or "")
        except Exception:
            return ""

    def _compile_phase_packet(
        self,
        *,
        context_engine,
        run_id: str,
        request: RunEngineRequest,
        session,
        phase: ContextPhase,
        system_prompt: str,
        effective_objective: str,
        current_context: str,
        allowed_tools: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        tool_turn: int = 0,
        strategy: str = "",
        notes: str = "",
        observation_status: str = "",
        observation_tools: Optional[list[str]] = None,
        final_response: str = "",
        current_facts: Optional[list[tuple[str, str, str]]] = None,
        conversation_tension=None,
        trace_phase_label: Optional[str] = None,
    ) -> CompiledPassPacket:
        packet = build_phase_packet(
            context_engine=context_engine,
            inputs=PacketBuildInputs(
                request=request,
                session=session,
                phase=phase,
                system_prompt=system_prompt,
                effective_objective=effective_objective,
                current_context=current_context,
                allowed_tools=allowed_tools,
                tool_results=tool_results,
                tool_turn=tool_turn,
                strategy=strategy,
                notes=notes,
                observation_status=observation_status,
                observation_tools=list(observation_tools or []),
                final_response=final_response,
                current_facts=current_facts,
                conversation_tension=conversation_tension,
            ),
        )
        phase_trace = dict(packet.trace)
        if trace_phase_label:
            phase_trace["phase"] = trace_phase_label
        self._trace(
            request=request,
            run_id=run_id,
            event_type="phase_context",
            data=phase_trace,
        )
        return packet

    def _trace(self, *, request: RunEngineRequest, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        self.trace_store.append_event(
            request.agent_id,
            request.session_id,
            event_type,
            data,
            run_id=run_id,
            owner_id=request.owner_id,
        )

    def _save_pass_record(
        self,
        *,
        run_id: str,
        phase: ContextPhase | str,
        tool_turn: int,
        status: str,
        objective: str,
        strategy: str,
        notes: str,
        selected_tools: list[str],
        tool_results: list[dict[str, Any]],
        packet: CompiledPassPacket,
        response_preview: str = "",
    ) -> None:
        self.run_store.save_pass(
            run_id=run_id,
            phase=phase.value if isinstance(phase, ContextPhase) else str(phase),
            tool_turn=tool_turn,
            status=status,
            objective=objective,
            strategy=strategy,
            notes=notes,
            selected_tools=selected_tools,
            tool_results=summarize_tool_results(tool_results, limit=4, preview_chars=220),
            compiled_context=packet.trace,
            response_preview=clip_text(response_preview, 400),
        )

    @staticmethod
    def _normalize_observation_decision(observation: dict[str, Any]) -> dict[str, Any]:
        raw_status = str(observation.get("status") or "").strip().lower()
        strategy = str(observation.get("strategy") or "")
        notes = str(observation.get("notes") or "")
        selected_tools = [
            entry["name"]
            for entry in observation.get("tools", [])
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]

        if raw_status == "finalize":
            status = "finalize"
            selected_tools = []
        elif selected_tools:
            status = "continue"
        else:
            status = "finalize"

        return {
            "status": status,
            "raw_status": raw_status,
            "strategy": strategy,
            "notes": notes,
            "confidence": observation.get("confidence"),
            "selected_tools": selected_tools,
            "should_continue": status == "continue" and bool(selected_tools),
        }

    @staticmethod
    def _build_controller_summary(
        *,
        status: str,
        strategy: str,
        notes: str,
        selected_tools: list[str],
    ) -> str:
        lines: list[str] = []
        clean_status = str(status or "").strip()
        clean_strategy = str(strategy or "").strip()
        clean_notes = str(notes or "").strip()
        clean_tools = [str(item).strip() for item in selected_tools if str(item).strip()]

        if clean_status:
            lines.append(f"Manager status: {clean_status}")
        if clean_strategy:
            lines.append(f"Strategy: {clean_strategy}")
        if clean_tools:
            lines.append("Preferred next tools: " + ", ".join(clean_tools))
        if clean_notes:
            lines.append(f"Notes: {clean_notes}")
        return "\n".join(lines)

    @staticmethod
    def _build_response_reminder(
        controller_summary: str,
        user_message: str,
        tool_results: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> str:
        summary = str(controller_summary or "").strip()
        if not summary:
            return ""
        priority_guidance = RunEngine._build_evidence_priority_reminder(user_message, tool_results)
        guidance = "Use the controller handoff below to close the turn. Synthesize directly from the gathered evidence."
        if priority_guidance:
            guidance += f" {priority_guidance}"
        availability_guidance = RunEngine._build_tool_availability_reminder(user_message, allowed_tools)
        evidence_focus = RunEngine._build_ranked_evidence_focus(user_message, tool_results)
        lines = [
            "<system-reminder>",
            f"{guidance} Do not mention internal planning, checkpoints, reminders, or observation state.",
        ]
        if availability_guidance:
            lines.append(availability_guidance)
        if evidence_focus:
            lines.append("Evidence focus:")
            lines.extend(evidence_focus)
        lines.append(summary)
        lines.append("</system-reminder>")
        return "\n".join(lines)

    @staticmethod
    def _build_evidence_priority_reminder(user_message: str, tool_results: list[dict[str, Any]]) -> str:
        return build_evidence_priority_reminder(user_message, tool_results)

    @staticmethod
    def _build_ranked_evidence_focus(user_message: str, tool_results: list[dict[str, Any]]) -> list[str]:
        return build_evidence_focus_lines(
            user_message,
            tool_results,
            max_results=2,
            preview_chars=140,
        )

    @staticmethod
    def _build_tool_availability_reminder(user_message: str, allowed_tools: list[str]) -> str:
        lowered = str(user_message or "").lower()
        allowed = {str(item).strip() for item in allowed_tools if str(item).strip()}
        if "web_search" in allowed and re.search(r"\b(latest|current|today|news|price|prices|stock|stocks|market)\b", lowered):
            return "For this request, web_search is available in this runtime. Use gathered search evidence or acknowledge a missing result, but do not claim browsing is unavailable."
        return ""

    @staticmethod
    def _select_followup_tool_results(
        tool_results: list[dict[str, Any]],
        *,
        user_message: str,
        max_results: int = 4,
    ) -> list[dict[str, Any]]:
        return select_working_evidence(
            tool_results,
            user_message=user_message,
            max_results=max_results,
        )

    @staticmethod
    def _is_substantive_tool_result(item: dict[str, Any]) -> bool:
        return is_substantive_tool_result(item)

    @staticmethod
    def _tool_kind_priority(tool_name: str) -> int:
        return tool_kind_priority(tool_name)

    @staticmethod
    def _extract_exact_query_targets(user_message: str) -> list[str]:
        return extract_exact_query_targets(user_message)

    @staticmethod
    def _tool_result_matches_exact_target(item: dict[str, Any], exact_targets: list[str]) -> bool:
        return tool_result_matches_exact_target(item, exact_targets)

    def _current_facts(self, session_id: str, owner_id: Optional[str]) -> list[tuple[str, str, str]]:
        if self.graph is None:
            return []
        try:
            return list(self.graph.get_current_facts(session_id, owner_id=owner_id) or [])
        except Exception:
            return []

    def _message_count(self, session_id: str, owner_id: Optional[str]) -> int:
        session = self.sessions.get(session_id, owner_id=owner_id)
        return int(getattr(session, "message_count", 0) or 0)

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        return clip_text(text, max_chars)
