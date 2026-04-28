from __future__ import annotations

import inspect
import json
import re
from typing import Any, AsyncIterator, Optional

from config.logger import log
from engine.candidate_signals import extract_stance_signals
from engine.conversation_tension import analyze_conversation_tension
from engine.tokenization import get_token_encoding as _get_token_encoding
from engine.context_schema import ContextPhase
from engine.context_governor import ContextGovernor
from engine.deterministic_facts import extract_user_stated_fact_updates
from engine.direct_fact_policy import should_answer_direct_fact_from_memory
from engine.side_effect_guard import check_plan_for_side_effects, check_side_effect_claims
from engine.tool_loop_guard import ToolLoopGuard
from engine.tool_contract import (
    canonical_tool_call,
    clip_text,
    diagnose_tool_failure,
    enrich_tool_result_content,
    format_tool_result_line,
    is_retry_sensitive_tool,
    summarize_tool_results,
    tool_call_signature,
)
from llm.adapter_factory import create_adapter, strip_provider_prefix
from plugins.hook_registry import hooks
from llm.base_adapter import BaseLLMAdapter
from memory.semantic_graph import SemanticGraph
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolCall, ToolRegistry
from run_engine.context_packets import PacketBuildInputs, build_phase_packet
from run_engine.code_intent import CodeIntent, classify_code_intent, check_research_ambiguity
from run_engine.evidence_lane import (
    build_evidence_focus_lines,
    build_evidence_priority_reminder,
    extract_exact_query_targets,
    is_substantive_tool_result,
    select_working_evidence,
    tool_kind_priority,
    tool_result_matches_exact_target,
)
from run_engine.memory_pipeline import build_grounding_text
from run_engine.skill_loader import list_available_skills, load_skill_context
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
- If a direct URL already exists in the context or signals, use it exactly — do not invent a new one.
- Use only the allowed tools.
- Do not create files or code unless the user explicitly asked for a file, script, app, or other artifact.
- Return a tool call, not prose.

Reading tool result signals:
- [READ_MORE: <url>]    — call web_fetch with THAT EXACT URL. Do not substitute a different URL.
- [NEXT_PROBE: <q>]     — use that exact value as your next web_search query or web_fetch URL.
- [KEY_FACT: <text>]    — note this fact; no further fetching needed for this fact alone.
- [TRUNCATED: N chars]  — content was cut; call web_fetch on the same URL to get the rest.
- [NO_RESULTS]          — previous tool found nothing; try a different query or source.
- [AUTH_REQUIRED]       — page requires login; try web_search for cached or alternative source.

Failure signals (gather intel before retrying — do not repeat the same failed call):
- [ARG_ERROR: ...]      — call list_tools to confirm the tool's real signature, then retry with correct args.
- [UNKNOWN_TOOL: ...]   — the tool name is not registered. Call list_tools. Do not invent tool names.
- [EMBED_DOWN: ...]     — embedding service is unreachable; deterministic memory still works, skip vector recall.
- [PROVIDER_DOWN: ...]  — upstream provider is unavailable; switch source or wait.
- [NETWORK: ...]        — connection failed; retry once, then try a different source.
- [NOT_FOUND: ...]      — target does not exist; verify path/url before retrying.
- [TIMEOUT: ...]        — operation too slow; retry with narrower scope or different tool.
- [BAD_FORMAT: ...]     — re-encode arguments as valid JSON.
- [FAILURE: ...]        — generic failure; gather more context (list_tools, query_memory, read related state) before retrying.
When signals are present, act on the highest-priority one before inventing your own next step.

Task list rules (when an Active Task List is present in context):
- Before starting a task: call todo_update(task_id, "in_progress").
- Immediately after completing a task: call todo_update(task_id, "completed").
- Do not skip these calls — the task list drifts out of sync if you do.
- If unsure of the current task state, call todo_read first.

Memory correction rules:
- When the user corrects a fact ("actually it's X", "I moved to Y", "forget Z"):
  - Use shovs_memory_update to void the old value and write the new one.
  - Use shovs_memory_void when there is no replacement value (user says "forget it").
  - These tools handle temporal voiding — do not use store_memory alone for corrections.
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


_REDRAFT_ISSUE_THRESHOLD = 2
# Confidence below this triggers a redraft even when issue count is sub-threshold.
# Picked at 0.4 because the verifier defaults to 0.5 on parse-failure / no-info,
# so we only redraft when it's actively expressing low trust in the answer.
_REDRAFT_CONFIDENCE_THRESHOLD = 0.4


def _build_redraft_constraints(*, issues: list[str], side_effect_tripped: bool) -> str:
    bullets = "\n".join(f"- {str(issue)}" for issue in issues if str(issue).strip())
    if not bullets:
        bullets = "- response was not grounded in the gathered evidence"
    extra = ""
    if side_effect_tripped:
        extra = (
            "\n- You claimed a side effect (created / ran / installed / wrote / etc.) "
            "that has no successful tool result backing it. Do not claim any action you "
            "did not actually take in this run."
        )
    return (
        "Your previous response failed verification. Rewrite it now with these constraints:\n"
        f"{bullets}{extra}\n\n"
        "Rewrite rules:\n"
        "- Use only facts present in the evidence above.\n"
        "- If a piece of information is missing, say it is missing instead of inventing it.\n"
        "- Do not repeat the previous unsupported claims.\n"
        "- Be direct and concise."
    )


def _estimate_text_tokens(text: str) -> int:
    normalized = str(text or "")
    if not normalized:
        return 0
    encoding = _get_token_encoding()
    if encoding is None:
        return max(1, len(normalized) // 4)
    try:
        return len(encoding.encode(normalized))
    except Exception:
        return max(1, len(normalized) // 4)


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for item in messages or []:
        total += _estimate_text_tokens(str(item.get("content") or ""))
        total += 4
    return total


def _estimate_model_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return 0.0

    rates: dict[str, tuple[float, float]] = {
        "openai:gpt-5": (1.25 / 1_000_000, 10.0 / 1_000_000),
        "openai:gpt-4.1": (2.0 / 1_000_000, 8.0 / 1_000_000),
        "openai:gpt-4o": (2.5 / 1_000_000, 10.0 / 1_000_000),
        "groq:llama3-groq-tool-use": (0.0, 0.0),
        "groq:": (0.0, 0.0),
        "ollama:": (0.0, 0.0),
        "lmstudio:": (0.0, 0.0),
        "llamacpp:": (0.0, 0.0),
        "local_openai:": (0.0, 0.0),
    }
    matched = next((price for prefix, price in rates.items() if normalized.startswith(prefix)), (0.0, 0.0))
    input_rate, output_rate = matched
    return round((max(0, input_tokens) * input_rate) + (max(0, output_tokens) * output_rate), 8)


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
        self._context_governor = ContextGovernor(
            adapter=adapter,
            v1_engine=context_engine,
            semantic_graph=graph,
        )
        self._last_model_swap: Optional[dict[str, Any]] = None

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
        session_model: Optional[str] = None,
    ):
        """
        Resolve the context engine and coerce the compression model so an
        embed-only model never lands in the chat-completion compression slot.
        Capability swaps are surfaced via _last_model_swap for the caller to
        emit a trace event once a run_id exists.
        """
        from llm.model_capabilities import coerce_chat_model
        self._context_governor.set_adapter(self.adapter)
        fallback = session_model or compression_model or ""
        safe_model, swapped = coerce_chat_model(compression_model, fallback)
        if swapped:
            self._last_model_swap = {
                "slot": "context_model",
                "rejected": str(compression_model or ""),
                "fallback": str(safe_model or ""),
                "reason": "embedding-only model passed for chat compression",
            }
        else:
            self._last_model_swap = None
        return self._context_governor.resolve(mode, compression_model=safe_model or compression_model)

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
        # Guard: refuse to run a chat session on an embed-only model.
        from llm.model_capabilities import coerce_chat_model, is_chat_capable
        if not is_chat_capable(session_model):
            safe_session_model, _ = coerce_chat_model(session_model, "llama3.2")
            session_model = safe_session_model
        active_context_engine = self._resolve_context_engine(
            requested_context_mode,
            compression_model=request.context_model or session_model,
            session_model=session_model,
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

        # Surface any model-capability swap that happened during context-engine resolution.
        if self._last_model_swap:
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="model_capability_violation",
                data=dict(self._last_model_swap),
            )
            self._last_model_swap = None

        # Validate planner_model and embed_model up-front so a misconfigured
        # frontend selection surfaces in the trace instead of failing silently.
        from llm.model_capabilities import (
            is_chat_capable as _is_chat_ok,
            is_embed_capable as _is_embed_ok,
        )
        if request.planner_model and not _is_chat_ok(request.planner_model):
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="model_capability_violation",
                data={
                    "slot": "planner_model",
                    "rejected": request.planner_model,
                    "fallback": session_model,
                    "reason": "embedding-only model passed for planner",
                },
            )
        if request.embed_model and not _is_embed_ok(request.embed_model):
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="model_capability_violation",
                data={
                    "slot": "embed_model",
                    "rejected": request.embed_model,
                    "fallback": "nomic-embed-text",
                    "reason": "chat model passed where embedding model required",
                },
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
            failed_tool_names: set[str] = set()
            planner_enabled = bool(request.use_planner)
            allowed_tools_list = self._list_allowed_tools(request.allowed_tools)
            latest_strategy = ""
            latest_notes = ""
            latest_observation_status = ""
            latest_observation_tools: list[str] = []
            latest_missing_slots: list[str] = []
            tool_loop_guard = ToolLoopGuard()
            normalized_max_tool_calls = self._normalize_optional_limit(
                request.max_tool_calls,
                minimum=1,
                maximum=24,
            )
            max_turns = max(1, min(int(request.max_turns or 3), 6))
            if normalized_max_tool_calls is not None:
                max_turns = min(max_turns, normalized_max_tool_calls)
            current_facts = self._context_governor.get_current_facts(
                session.id,
                owner_id=request.owner_id,
            )
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
            direct_fact_memory_only = should_answer_direct_fact_from_memory(
                request.user_message,
                current_facts,
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

            # ── Skill discovery ──
            available_skills: list[dict[str, str]] = []
            active_skill_context = ""
            active_skill_name = ""
            sticky_skill_hint = str(getattr(session, "last_active_skill", "") or "").strip()
            workspace_path = str(getattr(request, "workspace_path", "") or "").strip() or None
            if workspace_path:
                skill_manifests = list_available_skills(workspace_path)
                _tool_names = [t.get("name", "") for t in (request.tools or [])]
                available_skills = [
                    {
                        "name": m.name,
                        "description": m.description,
                        "eligibility": m.eligibility,
                        "triggers": m.triggers,
                        "auto_activate": m.is_eligible_for_auto(
                            request.user_message, _tool_names
                        ),
                    }
                    for m in skill_manifests
                ]
                if available_skills:
                    auto_activated = [s["name"] for s in available_skills if s["auto_activate"]]
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="skills_discovered",
                        data={
                            "count": len(available_skills),
                            "skills": [s["name"] for s in available_skills],
                            "auto_activated": auto_activated,
                        },
                    )

            # ── Loci discovery — surfaces Memory Palace rooms to the planner ──
            available_loci: list[dict[str, str]] = []
            try:
                raw_loci = self.graph.list_loci(owner_id=request.owner_id)
                available_loci = [
                    {
                        "id": str(l.get("id", "")),
                        "name": str(l.get("name", "")),
                        "description": str(l.get("description", "") or ""),
                    }
                    for l in raw_loci
                    if l.get("id")
                ]
                if available_loci:
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="loci_discovered",
                        data={
                            "count": len(available_loci),
                            "loci": [l["id"] for l in available_loci],
                        },
                    )
            except Exception:
                available_loci = []

            # ── Code intent classification ──
            code_intent = classify_code_intent(
                user_message=request.user_message,
                effective_objective=effective_objective,
                tool_results=tool_results,
            )
            code_intent_note = code_intent.to_phase_note()
            execution_risk_tier = code_intent.to_risk_note()
            if code_intent.code_warranted:
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="code_intent_classified",
                    data={
                        "code_warranted": code_intent.code_warranted,
                        "reason": code_intent.reason,
                        "risk_tier": code_intent.execution_risk_tier,
                        "has_missing_context": code_intent.missing_context is not None,
                    },
                )

            # ── Clarification gate ──
            # Gate 1: code scope ambiguity (existing)
            _clarification_question = code_intent.missing_context
            _clarification_reason = code_intent.reason
            _clarification_risk = code_intent.execution_risk_tier

            # Gate 2: vague research/comparison queries — ask before wasting tool calls
            if not _clarification_question and not tool_results:
                _research_question = check_research_ambiguity(request.user_message)
                if _research_question:
                    _clarification_question = _research_question
                    _clarification_reason = "research scope is underspecified"
                    _clarification_risk = "none"

            if _clarification_question and not tool_results:
                yield {
                    "type": "clarification_needed",
                    "question": _clarification_question,
                    "reason": _clarification_reason,
                    "risk_tier": _clarification_risk,
                }

            # Default locus_id — overwritten by planner if a locus is targeted.
            planned_locus_id: str = ""
            # Argument clues from the planner (tool_name → hint string for actor).
            planned_argument_clues: dict[str, str] = {}
            # Full structured plan — populated by planner, used by verify skip logic.
            structured_plan: dict[str, Any] = {}
            # Route resolved from plan (used by hooks and verify skip).
            _plan_route: str = ""

            if not selected_tools and not direct_fact_memory_only and planner_enabled and self.orchestrator is not None:
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
                    active_skill_context=active_skill_context,
                    active_skill_name=active_skill_name,
                    code_intent_note=code_intent_note,
                    execution_risk_tier=execution_risk_tier,
                    available_loci=available_loci if available_loci else None,
                )
                _planner_model = (
                    request.planner_model
                    if request.planner_model and _is_chat_ok(request.planner_model)
                    else session_model
                )
                # Pull cross-run failure rates so the planner can avoid tools
                # that have been failing for this (owner, agent). Cheap read
                # with bounded lookback; missing/empty is fine — planner just
                # doesn't get the hint that turn.
                try:
                    _tool_failure_rates = self.run_store.get_tool_failure_rates(
                        owner_id=request.owner_id,
                        agent_id=request.agent_id,
                        lookback_days=7,
                        min_attempts=3,
                    )
                except Exception:
                    _tool_failure_rates = {}
                structured_plan = await self.orchestrator.plan_with_context(
                    query=effective_objective,
                    tools_list=allowed_tools_list,
                    model=_planner_model,
                    session_has_history=bool(session.full_history),
                    current_fact_count=len(current_facts),
                    failed_tools=sorted(failed_tool_names),
                    compiled_context=planning_packet.content,
                    skills_list=available_skills if available_skills else None,
                    available_loci=available_loci if available_loci else None,
                    code_intent_note=code_intent_note or "none",
                    risk_tier=execution_risk_tier or "standard",
                    tension_summary=conversation_tension.summary or "none",
                    tool_failure_rates=_tool_failure_rates,
                )

                # ── Capture planned locus_id for spatial memory routing ──
                planned_locus_id = str(structured_plan.get("locus_id", "")).strip()
                if planned_locus_id:
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="locus_targeted",
                        data={"locus_id": planned_locus_id},
                    )

                # ── Load selected skill ──
                # Planner is authoritative. If it picks no skill but the prior
                # turn had one (and it's still available), reuse it — prevents
                # multi-turn flows from losing skill context when the planner
                # forgets to re-pick mid-flow.
                planned_skill = str(structured_plan.get("skill", "")).strip()
                skill_source = "planner"
                if (
                    not planned_skill
                    and sticky_skill_hint
                    and any(s.get("name") == sticky_skill_hint for s in available_skills)
                ):
                    planned_skill = sticky_skill_hint
                    skill_source = "sticky_carryover"
                if planned_skill and workspace_path:
                    loaded_context = load_skill_context(planned_skill, workspace_path)
                    if loaded_context:
                        active_skill_context = loaded_context
                        active_skill_name = planned_skill
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="skill_loaded",
                            data={
                                "skill_name": planned_skill,
                                "chars": len(loaded_context),
                                "source": skill_source,
                            },
                        )
                # Persist back to session — empty when planner explicitly drops
                # the skill, so a stale skill cannot leak forward.
                session.last_active_skill = active_skill_name or None

                selected_tools = [
                    entry["name"]
                    for entry in structured_plan.get("tools", [])
                    if isinstance(entry, dict) and isinstance(entry.get("name"), str)
                ]
                # Capture argument clues keyed by tool name for the actor.
                planned_argument_clues: dict[str, str] = {
                    entry["name"]: str(entry.get("target_argument_clue", "")).strip()
                    for entry in structured_plan.get("tools", [])
                    if isinstance(entry, dict)
                    and isinstance(entry.get("name"), str)
                    and str(entry.get("target_argument_clue", "")).strip()
                }
                # Pre-execution side-effect advisory. Non-blocking — surfaces
                # when the plan picked write/destructive tools but the user
                # message has no action verb (or code_intent disagrees with
                # the plan's risk level). Trace-only; the post-hoc response
                # guard is still authoritative for actual blocking.
                _plan_advisory = check_plan_for_side_effects(
                    user_message=request.user_message,
                    selected_tools=selected_tools,
                    declared_risk_tier=execution_risk_tier or "",
                )
                if _plan_advisory["warnings"]:
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="plan_side_effect_advisory",
                        data={
                            "warnings": _plan_advisory["warnings"],
                            "max_tier": _plan_advisory["max_tier"],
                            "selected_tools": _plan_advisory["selected_tools"],
                            "declared_risk_tier": execution_risk_tier or "",
                        },
                    )
                _plan_route = str(structured_plan.get("route", "")).strip()
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
                    # ── Hook: plan_generated ──
                    hooks.emit_sync("plan_generated", {
                        "route": _plan_route,
                        "skill": planned_skill,
                        "tools": selected_tools,
                        "confidence": structured_plan.get("confidence"),
                        "strategy": strategy[:120],
                    }, run_id=run.run_id, session_id=request.session_id)
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

            if not selected_tools and not direct_fact_memory_only:
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
                    active_skill_context=active_skill_context,
                    active_skill_name=active_skill_name,
                    code_intent_note=code_intent_note,
                    execution_risk_tier=execution_risk_tier,
                    available_loci=available_loci if available_loci else None,
                    planned_locus_id=planned_locus_id,
                )
                tool_call = await self._select_tool_call(
                    adapter=current_adapter,
                    model=clean_model,
                    request=request,
                    session=session,
                    allowed_tools=selected_tools,
                    tool_results=tool_results,
                    context_block=acting_packet.content,
                    argument_clues=planned_argument_clues if planned_argument_clues else None,
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

                # ── Hook: tool_selected ──
                hooks.emit_sync("tool_selected", {
                    "tool_name": tool_call.tool_name,
                    "arguments_preview": str(tool_call.arguments or {})[:120],
                    "turn": tool_turn,
                }, run_id=run.run_id, session_id=request.session_id)

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
                    failed_tool_names.add(tool_call.tool_name)
                    result_payload = {
                        "tool_name": tool_call.tool_name,
                        "success": False,
                        "content": f"Tool validation failed: {validation_error}",
                        "arguments": dict(tool_call.arguments or {}),
                    }
                    tool_results.append(result_payload)
                    self._record_tool_outcome(
                        request=request,
                        tool_name=tool_call.tool_name,
                        success=False,
                        error_kind="validation_error",
                    )
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
                        failed_tool_names.add(tool_call.tool_name)
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
                        self._record_tool_outcome(
                            request=request,
                            tool_name=tool_call.tool_name,
                            success=False,
                            error_kind="duplicate_suppressed",
                        )
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
                        latest_missing_slots = list(normalized_observation.get("missing_slots") or [])
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
                        "_embed_model": (
                            request.embed_model
                            if request.embed_model and _is_embed_ok(request.embed_model)
                            else "nomic-embed-text"
                        ),
                        "_runtime_path": "run_engine",
                    },
                )
                # Enrich content with AI-readable signals ([READ_MORE], [KEY_FACT], etc.)
                # so the actor and observation phases know what to do next without
                # having to infer next steps from raw content alone.
                _raw_content = tool_result.content
                try:
                    _content_json = json.loads(_raw_content)
                    _is_truncated = bool(_content_json.get("truncated"))
                    _total_len = int(_content_json.get("total_length") or 0)
                    _shown_len = len(str(_content_json.get("content") or ""))
                    _truncated_chars = max(0, _total_len - _shown_len) if _is_truncated else 0
                except Exception:
                    _is_truncated = "[truncated" in _raw_content.lower()
                    _truncated_chars = 0
                if tool_result.success:
                    _enriched_content = enrich_tool_result_content(
                        tool_result.tool_name,
                        _raw_content,
                        arguments=dict(tool_call.arguments or {}),
                        is_truncated=_is_truncated,
                        truncated_chars=_truncated_chars,
                    )
                else:
                    # Annotate failures with structured recovery hints so the
                    # actor's signal-reading rules drive an intel-gathering
                    # retry instead of repeating the same call or hallucinating.
                    _diag_hints = diagnose_tool_failure(
                        tool_result.tool_name,
                        _raw_content,
                        arguments=dict(tool_call.arguments or {}),
                    )
                    if _diag_hints:
                        _enriched_content = (
                            (_raw_content or "").rstrip() + "\n\n" + "\n".join(_diag_hints)
                        )
                    else:
                        _enriched_content = _raw_content

                result_payload = {
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                    "content": _enriched_content,
                    "arguments": dict(tool_call.arguments or {}),
                }
                tool_results.append(result_payload)
                if not tool_result.success:
                    failed_tool_names.add(tool_result.tool_name)
                self._record_tool_outcome(
                    request=request,
                    tool_name=tool_result.tool_name,
                    success=bool(tool_result.success),
                    error_kind=("hard_failure" if (not tool_result.success and "HARD_FAILURE" in (tool_result.content or "")) else ""),
                )

                # ── Hook: tool_completed ──
                hooks.emit_sync("tool_completed", {
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                    "turn": tool_turn,
                }, run_id=run.run_id, session_id=request.session_id)

                # ── Hook: hard_failure ──
                # Fires when a tool result returns status=HARD_FAILURE in its
                # JSON payload. Subscribers can use this for circuit breakers,
                # alerting, or auto-rollback hooks.
                if not tool_result.success and "HARD_FAILURE" in (tool_result.content or ""):
                    hooks.emit_sync("hard_failure", {
                        "tool_name": tool_result.tool_name,
                        "turn": tool_turn,
                        "preview": clip_text(tool_result.content, 200),
                    }, run_id=run.run_id, session_id=request.session_id)

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
                stall_alert = tool_loop_guard.observe_result(
                    tool_name=tool_result.tool_name,
                    arguments=dict(tool_call.arguments or {}),
                    success=bool(tool_result.success),
                    content=str(tool_result.content or ""),
                )
                if stall_alert:
                    alert_tool_name = str(stall_alert.get("tool_name") or tool_result.tool_name or "unknown")
                    failed_tool_names.add(alert_tool_name)
                    latest_notes = str(stall_alert["message"] or "")
                    alert_payload = {
                        "tool_name": alert_tool_name,
                        "success": False,
                        "content": latest_notes,
                        "arguments": dict(tool_call.arguments or {}),
                    }
                    tool_results.append(alert_payload)
                    self._record_tool_outcome(
                        request=request,
                        tool_name=alert_tool_name,
                        success=False,
                        error_kind="logical_stall",
                    )
                    self.run_store.save_checkpoint(
                        run_id=run.run_id,
                        phase="acting",
                        tool_turn=tool_turn,
                        status="logical_stall_alert",
                        notes=latest_notes,
                        tools=[alert_tool_name],
                        tool_results=summarize_tool_results([alert_payload], limit=1, preview_chars=220),
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="logical_stall_alert",
                        data={
                            "tool_name": alert_tool_name,
                            "command": stall_alert.get("command", ""),
                            "message": latest_notes,
                            "suggested_pivot": stall_alert.get("suggested_pivot", ""),
                            "turn": tool_turn,
                        },
                    )
                    yield stall_alert
                    self._save_pass_record(
                        run_id=run.run_id,
                        phase=ContextPhase.ACTING,
                        tool_turn=tool_turn,
                        status="logical_stall_alert",
                        objective=effective_objective,
                        strategy=latest_strategy,
                        notes=latest_notes,
                        selected_tools=[alert_tool_name],
                        tool_results=tool_results[-3:],
                        packet=acting_packet,
                    )
                    selected_tools = []
                    break
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
                    missing_slots=latest_missing_slots,
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
                output_tokens=0,
                input_tokens=0,
                estimated_cost_usd=0.0,
                model=session_model,
            )
            # Only forward reasoning_enabled when the caller set it explicitly,
            # so adapters that don't accept the kwarg aren't broken.
            stream_kwargs: dict[str, Any] = {}
            if request.reasoning_enabled is not None:
                stream_kwargs["reasoning_enabled"] = request.reasoning_enabled
            async for token in current_adapter.stream(
                model=clean_model,
                messages=final_messages,
                temperature=0.2,
                images=request.images,
                **stream_kwargs,
            ):
                final_response += token
                yield {"type": "token", "content": token}

            response_summary = self._build_controller_summary(
                status=latest_observation_status,
                strategy=latest_strategy,
                notes=latest_notes,
                selected_tools=latest_observation_tools,
                missing_slots=latest_missing_slots,
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
                prompt_messages=final_messages,
                model=session_model,
            )

            verification = {"supported": True, "issues": [], "confidence": 0.0}

            # Deterministic side-effect claim guard — runs unconditionally when
            # tool_results exist. Catches the Replit/Gemini failure pattern
            # (response claims "created/ran/installed" with no supporting
            # successful tool result, or against a HARD_FAILURE).
            side_effect_check = check_side_effect_claims(final_response, tool_results)
            if not side_effect_check.get("supported", True):
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="side_effect_claim_unsupported",
                    data={
                        "issues": side_effect_check.get("issues", []),
                        "claims": side_effect_check.get("claims", []),
                        "hard_failures": side_effect_check.get("hard_failures", []),
                    },
                )

            # Skip verification for runs that need no external grounding:
            # direct-fact-memory-only answers come from deterministic facts,
            # and no-tool runs have nothing to ground-check.
            _plan_route = str(structured_plan.get("route_type", ""))
            _skip_verify = direct_fact_memory_only or (not tool_results and _plan_route in {"trivial_chat", "memory_recall"})
            if self.orchestrator is not None and not _skip_verify:
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
                    route_type=_plan_route,
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

            # Merge deterministic side-effect guard into the verification verdict.
            # Runs even when LLM verification was skipped, so trivial-chat / memory-recall
            # routes still cannot pass through with hallucinated side-effect claims.
            if not side_effect_check.get("supported", True):
                merged_issues = list(verification.get("issues") or [])
                merged_issues.extend(side_effect_check.get("issues", []))
                verification = {
                    "supported": False,
                    "issues": merged_issues,
                    "confidence": min(float(verification.get("confidence") or 0.0), 0.0),
                    "side_effect_guard": "tripped",
                }

            # Re-draft once when verification fails with high severity.
            # Triggered by side-effect guard tripping OR multiple verification issues.
            # Bound to a single attempt — if the redraft also fails, we ship with a warning.
            if not bool(verification.get("supported", True)):
                _initial_issues = list(verification.get("issues") or [])
                _side_effect_tripped = bool(verification.get("side_effect_guard") == "tripped")
                try:
                    _verifier_confidence = float(verification.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    _verifier_confidence = 0.0
                _low_confidence = _verifier_confidence < _REDRAFT_CONFIDENCE_THRESHOLD
                if (
                    _side_effect_tripped
                    or len(_initial_issues) >= _REDRAFT_ISSUE_THRESHOLD
                    or _low_confidence
                ):
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="redraft_triggered",
                        data={
                            "issues": [str(item) for item in _initial_issues[:5]],
                            "side_effect_guard": _side_effect_tripped,
                            "issue_count": len(_initial_issues),
                            "confidence": _verifier_confidence,
                            "low_confidence_trigger": _low_confidence,
                        },
                    )
                    yield {
                        "type": "redraft",
                        "reason": "verification_unsupported",
                        "issues": _initial_issues,
                        "side_effect_guard": _side_effect_tripped,
                        "confidence": _verifier_confidence,
                    }
                    constraints_msg = _build_redraft_constraints(
                        issues=_initial_issues,
                        side_effect_tripped=_side_effect_tripped,
                    )
                    redraft_messages = list(final_messages) + [
                        {"role": "system", "content": constraints_msg}
                    ]
                    final_response = ""
                    async for token in current_adapter.stream(
                        model=clean_model,
                        messages=redraft_messages,
                        temperature=0.1,
                        images=request.images,
                        **stream_kwargs,
                    ):
                        final_response += token
                        yield {"type": "token", "content": token}

                    # Re-evaluate ONCE — no further redrafts.
                    side_effect_check = check_side_effect_claims(final_response, tool_results)
                    if self.orchestrator is not None and not _skip_verify:
                        verification = await self.orchestrator.verify_with_context(
                            query=effective_objective,
                            response=final_response,
                            tool_results=tool_results,
                            model=request.planner_model or session_model,
                            compiled_context=verification_packet.content,
                            route_type=_plan_route,
                        )
                    else:
                        verification = {"supported": True, "issues": [], "confidence": 0.0}
                    if not side_effect_check.get("supported", True):
                        merged_issues = list(verification.get("issues") or [])
                        merged_issues.extend(side_effect_check.get("issues", []))
                        verification = {
                            "supported": False,
                            "issues": merged_issues,
                            "confidence": min(float(verification.get("confidence") or 0.0), 0.0),
                            "side_effect_guard": "tripped",
                        }
                    # Verifier self-consistency check. The initial verification
                    # said unsupported (that's why we redrafted). If the *post-
                    # redraft* verification flips back to supported with a
                    # different response body, that's normal — the redraft
                    # worked. But if the verifier flips supported↔unsupported
                    # while the response barely changed, the verifier itself
                    # is unreliable on this turn. Surface that signal so
                    # downstream consumers can decide whether to trust the
                    # final answer.
                    _post_redraft_supported = bool(verification.get("supported", True))
                    try:
                        _post_redraft_confidence = float(verification.get("confidence") or 0.0)
                    except (TypeError, ValueError):
                        _post_redraft_confidence = 0.0
                    _verifier_unstable = (
                        _post_redraft_supported != bool(False)  # initial was False to enter this block
                        and abs(_post_redraft_confidence - _verifier_confidence) >= 0.4
                    )
                    if _verifier_unstable:
                        # Confidence swing ≥ 0.4 between drafts on the same
                        # underlying evidence is the verifier disagreeing with
                        # itself, not the response actually improving.
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="verifier_unstable",
                            data={
                                "initial_supported": False,
                                "post_redraft_supported": _post_redraft_supported,
                                "initial_confidence": _verifier_confidence,
                                "post_redraft_confidence": _post_redraft_confidence,
                                "delta": abs(_post_redraft_confidence - _verifier_confidence),
                            },
                        )
                        yield {
                            "type": "verifier_unstable",
                            "initial_confidence": _verifier_confidence,
                            "post_redraft_confidence": _post_redraft_confidence,
                        }
                        # Pin to the more conservative read: if the second
                        # verification flipped to supported with a big jump,
                        # we don't fully trust it — keep some warning weight
                        # by capping confidence at the midpoint.
                        if _post_redraft_supported and _post_redraft_confidence > _verifier_confidence:
                            _conservative = (_post_redraft_confidence + _verifier_confidence) / 2.0
                            verification = {
                                **verification,
                                "confidence": _conservative,
                                "verifier_unstable": True,
                            }
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="redraft_result",
                        data={
                            "supported": bool(verification.get("supported", True)),
                            "issues": [str(item) for item in (verification.get("issues") or [])[:5]],
                            "issue_count": len(verification.get("issues") or []),
                            "verifier_unstable": bool(verification.get("verifier_unstable")),
                        },
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
                    planned_locus_id=planned_locus_id,
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
            # ── Hook: run_complete ──
            hooks.emit_sync("run_complete", {
                "run_id": run.run_id,
                "route": _plan_route,
                "tool_count": tool_call_count,
                "success": True,
            }, run_id=run.run_id, session_id=request.session_id)
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
        argument_clues: Optional[dict[str, str]] = None,
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
                    argument_clues=argument_clues,
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
            # Strict allowed-tool enforcement — actor cannot escape the
            # planner's tool selection by hallucinating a tool name that's
            # registered but not in the planner's narrowed list.
            if call.tool_name not in available_names:
                self._trace(
                    request=request,
                    run_id=getattr(request, "run_id", None) or "",
                    event_type="tool_selection_violation",
                    data={
                        "attempted_tool": call.tool_name,
                        "allowed_tools": list(available_names),
                    },
                )
                call = None
            else:
                return call
        fallback_ranked = self._rank_fallback_tool_candidates(effective_objective, available_names)
        for name in fallback_ranked:
            if name not in available_names:
                continue
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
        argument_clues: Optional[dict[str, str]] = None,
    ) -> str:
        return build_actor_request_content(
            user_message=request.user_message,
            effective_objective=effective_objective,
            session_first_message=str(session.first_message or ""),
            allowed_tools=allowed_tools,
            tool_results=tool_results,
            context_block=context_block,
            clip_text=self._clip,
            argument_clues=argument_clues,
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
        planned_locus_id: str = "",
    ) -> None:
        session = self.sessions.get(session_id, owner_id=request.owner_id)
        existing_candidate_signals = list(getattr(session, "candidate_signals", []) or [])
        existing_candidate_context = getattr(session, "candidate_context", "") if session is not None else ""

        if context_engine is None or not hasattr(context_engine, "compress_exchange"):
            commit_plan = self._context_governor.build_memory_commit_plan(
                user_message=request.user_message,
                tool_results=tool_results,
                deterministic_keyed_facts=deterministic_keyed_facts,
                deterministic_voids=deterministic_voids,
                current_facts=current_facts,
                existing_candidate_signals=existing_candidate_signals,
                existing_candidate_context=existing_candidate_context,
                new_candidate_signals=stance_signals,
                current_turn=self._message_count(session_id, request.owner_id),
            )
            outcome = await self._context_governor.apply_memory_commit(
                sessions=self.sessions,
                session_id=session_id,
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                turn=self._message_count(session_id, request.owner_id),
                run_id=run_id,
                user_message=request.user_message,
                assistant_response=assistant_response,
                plan=commit_plan,
                current_context=current_context,
                planned_locus_id=planned_locus_id,
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
            commit_plan = self._context_governor.build_memory_commit_plan(
                user_message=request.user_message,
                tool_results=tool_results,
                deterministic_keyed_facts=deterministic_keyed_facts,
                deterministic_voids=deterministic_voids,
                current_facts=current_facts,
                existing_candidate_signals=existing_candidate_signals,
                existing_candidate_context=existing_candidate_context,
                new_candidate_signals=stance_signals,
                current_turn=self._message_count(session_id, request.owner_id),
            )
            outcome = await self._context_governor.apply_memory_commit(
                sessions=self.sessions,
                session_id=session_id,
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                turn=self._message_count(session_id, request.owner_id),
                run_id=run_id,
                user_message=request.user_message,
                assistant_response=assistant_response,
                plan=commit_plan,
                current_context=current_context,
                planned_locus_id=planned_locus_id,
            )
            if outcome.indexed_fact_keys:
                self._trace(
                    request=request,
                    run_id=run_id,
                    event_type="facts_indexed",
                    data={"facts": list(outcome.indexed_fact_keys), "context_lines": outcome.context_lines},
                )
            return
        commit_plan = self._context_governor.build_memory_commit_plan(
            context_result=maybe_result,
            user_message=request.user_message,
            tool_results=tool_results,
            deterministic_keyed_facts=deterministic_keyed_facts,
            deterministic_voids=deterministic_voids,
            current_facts=current_facts,
            existing_candidate_signals=existing_candidate_signals,
            existing_candidate_context=existing_candidate_context,
            new_candidate_signals=stance_signals,
            current_turn=self._message_count(session_id, request.owner_id),
        )
        outcome = await self._context_governor.apply_memory_commit(
            sessions=self.sessions,
            session_id=session_id,
            owner_id=request.owner_id,
            agent_id=request.agent_id,
            turn=self._message_count(session_id, request.owner_id),
            run_id=run_id,
            user_message=request.user_message,
            assistant_response=assistant_response,
            plan=commit_plan,
            current_context=current_context,
            planned_locus_id=planned_locus_id,
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
        active_skill_context: str = "",
        active_skill_name: str = "",
        code_intent_note: str = "",
        execution_risk_tier: str = "",
        correction_turn: bool = False,
        direct_fact_memory_only: bool = False,
        available_loci: Optional[list[dict[str, str]]] = None,
        planned_locus_id: str = "",
    ) -> CompiledPassPacket:
        inferred_correction_turn = correction_turn or bool(
            re.search(r"\b(actually|instead|correction|updated|changed|not .* anymore|moved to|call me)\b", request.user_message or "", re.IGNORECASE)
        )
        inferred_direct_fact_memory_only = direct_fact_memory_only or should_answer_direct_fact_from_memory(
            request.user_message,
            current_facts,
        )
        packet = build_phase_packet(
            context_engine=context_engine,
            context_governor=self._context_governor,
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
                active_skill_context=active_skill_context,
                active_skill_name=active_skill_name,
                code_intent_note=code_intent_note,
                execution_risk_tier=execution_risk_tier,
                correction_turn=inferred_correction_turn,
                direct_fact_memory_only=inferred_direct_fact_memory_only,
                available_loci=available_loci,
                planned_locus_id=planned_locus_id,
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

    def _record_tool_outcome(
        self,
        *,
        request: RunEngineRequest,
        tool_name: str,
        success: bool,
        error_kind: str = "",
    ) -> None:
        """Best-effort write to ``tool_outcomes`` so the planner can read
        per-(owner, agent, tool) failure rates back next turn. Swallowed on
        error because this is observational — never block the loop on it."""
        try:
            self.run_store.record_tool_outcome(
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                tool_name=tool_name,
                success=bool(success),
                error_kind=error_kind,
            )
        except Exception:
            pass

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
        prompt_messages: Optional[list[dict[str, Any]]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        model: str = "",
    ) -> None:
        resolved_input_tokens = max(
            0,
            int(input_tokens if input_tokens is not None else _estimate_message_tokens(prompt_messages or [])),
        )
        resolved_output_tokens = max(
            0,
            int(output_tokens if output_tokens is not None else _estimate_text_tokens(response_preview)),
        )
        resolved_total_tokens = resolved_input_tokens + resolved_output_tokens
        resolved_cost = (
            float(estimated_cost_usd)
            if estimated_cost_usd is not None
            else _estimate_model_cost_usd(model, resolved_input_tokens, resolved_output_tokens)
        )
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
            input_tokens=resolved_input_tokens,
            output_tokens=resolved_output_tokens,
            total_tokens=resolved_total_tokens,
            estimated_cost_usd=resolved_cost,
        )

    @staticmethod
    def _normalize_observation_decision(observation: dict[str, Any]) -> dict[str, Any]:
        raw_status = str(observation.get("status") or "").strip().lower()
        strategy = str(observation.get("strategy") or "")
        notes = str(observation.get("notes") or "")
        missing_slots = observation.get("missing") or observation.get("missing_slots") or []
        if not isinstance(missing_slots, list):
            missing_slots = [str(missing_slots)]
        missing_slots = [str(item).strip() for item in missing_slots if str(item).strip()][:6]
        selected_tools = [
            entry["name"]
            for entry in observation.get("tools", [])
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]

        # Three-state status: finalize | continue | partial.
        # `partial` means the controller has *some* useful evidence but knows
        # specific slots are still missing — distinct from `continue` (no
        # answer yet, keep going) and `finalize` (good enough). It still
        # routes through the actor loop, but surfaces the missing slots so
        # the next plan can target them instead of re-running broad tools.
        if raw_status == "finalize":
            status = "finalize"
            selected_tools = []
        elif raw_status == "partial":
            # Partial requires either explicit follow-up tools OR named
            # missing slots. Without either it collapses to finalize so we
            # don't loop forever waiting for the controller to make up its
            # mind.
            if selected_tools or missing_slots:
                status = "partial"
            else:
                status = "finalize"
        elif selected_tools:
            status = "continue"
        else:
            status = "finalize"

        should_continue = (
            (status == "continue" and bool(selected_tools))
            or (status == "partial" and bool(selected_tools or missing_slots))
        )

        return {
            "status": status,
            "raw_status": raw_status,
            "strategy": strategy,
            "notes": notes,
            "confidence": observation.get("confidence"),
            "selected_tools": selected_tools,
            "missing_slots": missing_slots,
            "should_continue": should_continue,
        }

    @staticmethod
    def _build_controller_summary(
        *,
        status: str,
        strategy: str,
        notes: str,
        selected_tools: list[str],
        missing_slots: Optional[list[str]] = None,
    ) -> str:
        lines: list[str] = []
        clean_status = str(status or "").strip()
        clean_strategy = str(strategy or "").strip()
        clean_notes = str(notes or "").strip()
        clean_tools = [str(item).strip() for item in selected_tools if str(item).strip()]
        clean_missing = [str(item).strip() for item in (missing_slots or []) if str(item).strip()]

        if clean_status:
            lines.append(f"Manager status: {clean_status}")
        if clean_strategy:
            lines.append(f"Strategy: {clean_strategy}")
        if clean_missing:
            # When partial, surfacing the named gaps lets the actor target
            # exactly the missing slot instead of re-running broad searches.
            lines.append("Still missing: " + ", ".join(clean_missing))
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
        return self._context_governor.get_current_facts(session_id, owner_id=owner_id)

    def _message_count(self, session_id: str, owner_id: Optional[str]) -> int:
        session = self.sessions.get(session_id, owner_id=owner_id)
        return int(getattr(session, "message_count", 0) or 0)

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        return clip_text(text, max_chars)
