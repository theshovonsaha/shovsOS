from __future__ import annotations

import inspect
import asyncio
import json
import os
import re
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlsplit, urlunsplit

from config.logger import log
from engine.candidate_signals import extract_stance_signals
from engine.conversation_tension import analyze_conversation_tension, conversation_tension_audit_payload
from engine.tokenization import get_token_encoding as _get_token_encoding
from engine.context_schema import ContextPhase
from engine.context_governor import ContextGovernor
from engine.context_hygiene import should_run_llm_memory_compression, should_skip_memory_compression
from engine.deterministic_facts import extract_user_stated_fact_updates
from engine.direct_fact_policy import should_answer_direct_fact_from_memory
from engine.side_effect_guard import check_plan_for_side_effects, check_side_effect_claims
from engine.tool_loop_guard import ToolLoopGuard
from engine.response_guard import guard_final_response, is_small_or_local_model
from engine.tool_contract import (
    canonical_tool_call,
    clip_text,
    diagnose_tool_failure,
    enrich_tool_result_content,
    format_tool_result_line,
    is_retry_sensitive_tool,
    shape_tool_result_for_actor,
    summarize_tool_results,
    tool_call_signature,
)
from llm.adapter_factory import create_adapter, strip_provider_prefix
from plugins.hook_registry import hooks
from llm.base_adapter import BaseLLMAdapter, RateLimitError, ProviderError
from memory.semantic_graph import SemanticGraph
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from orchestration.capability_cards import render_capability_cards
from orchestration.workflow_patterns import get_workflow_pattern
from orchestration.workflow_templates import get_workflow_template
from plugins.tool_registry import ToolCall, ToolRegistry
from plugins.tools_web import _validate_fetch_url
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
from run_engine.pass_framework import build_pass_graph
from run_engine.search_query import compile_web_search_query
from run_engine.skill_loader import list_available_skills, load_skill_context
from run_engine.tool_selection import (
    build_actor_request_content,
    extract_tool_call,
    fallback_tool_call,
    summarize_arguments,
)
from run_engine.types import CompiledPassPacket, RunEngineRequest
from run_engine.ledger import RunLedger, ToolCallDraft, recovery_policy_for
from run_engine.workflow_contracts import infer_workflow_contract, update_contract_from_tool_results
from run_engine.control_policies import resolve_control_policy
from run_engine.turn_relation import classify_turn_relation
from run_engine.turn_policy import resolve_turn_policy
from run_engine.workflow_plugins import (
    default_tool_turn_budget as _plugin_default_tool_turn_budget,
    extract_mover_tickers_from_fetched_pages as _plugin_extract_mover_tickers_from_fetched_pages,
    extract_stock_tickers_from_tool_results as _plugin_extract_stock_tickers_from_tool_results,
    extract_urls_from_tool_results as _plugin_extract_urls_from_tool_results,
    explicit_stock_source_workflow_requested as _plugin_explicit_stock_source_workflow_requested,
    select_movers_source_url as _plugin_select_movers_source_url,
    select_workflow_override,
    select_workflow_plugin_contract,
    source_collection_contract_from_objective as _plugin_source_collection_contract_from_objective,
    stock_source_workflow_override as _plugin_stock_source_workflow_override,
    urls_by_entity_from_search_results as _plugin_urls_by_entity_from_search_results,
)


async def _stream_with_rate_limit_retry(
    adapter,
    *,
    model: str,
    messages: list[dict],
    **kwargs,
) -> AsyncIterator[str | dict[str, Any]]:
    try:
        async for chunk in adapter.stream(model=model, messages=messages, **kwargs):
            yield chunk
    except RateLimitError as exc:
        # First choice: fail OVER to another provider/model (a daily-quota 429 is
        # not fixed by waiting). Only fires when SHOVS_PROVIDER_FALLBACK_CHAIN is
        # configured; otherwise we fall back to the wait-and-retry-same behavior.
        from llm.adapter_factory import get_failover_adapter, get_default_model
        provider = model.split(":", 1)[0].lower() if ":" in model else ""
        failover = get_failover_adapter(provider) if provider else None
        if failover is not None:
            f_model = get_default_model(failover)
            yield {
                "type": "activity_short",
                "text": f"Rate limited on {provider or model} — failing over to {f_model}...",
            }
            async for chunk in failover.stream(model=f_model, messages=messages, **kwargs):
                yield chunk
            return
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
        wait = float(match.group(1)) if match else 10.0
        wait = min(wait, 60.0)
        yield {
            "type": "activity_short",
            "text": f"Rate limit hit, waiting {wait:.0f}s before retry...",
        }
        await asyncio.sleep(wait)
        async for chunk in adapter.stream(model=model, messages=messages, **kwargs):
            yield chunk


TOOL_ACTOR_PROMPT = """\
You are the Shovs Run Engine actor.

Your job is to choose exactly one next tool call from the allowed tools.
Use the current user objective, prior tool results, and the execution clue.

Rules:
- The allowed tools are real and available in this runtime right now.
- Prefer the smallest useful next action.
- For current or time-sensitive requests such as latest, current, today, news, prices, or market data, use an allowed web_search tool instead of saying you cannot browse.
- If the objective is ambiguous, underspecified, or contradicts existing memory, DO NOT guess. Stop calling tools so you can ask the user a clarifying question.
- If the objective is clear and can be advanced by a tool call, make the call. Do not ask unnecessary questions.
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
- Answer the user directly using the evidence gathered in this run and your active memory context.
- Maintain a coherent, natural conversational flow. Acknowledge the user's intent, provide context for your answers, and demonstrate understanding of the ongoing dialogue.
- If the objective is ambiguous, underspecified, or missing key details, ask clear, pointed questions to resolve the ambiguity before proceeding.
- If tools were available earlier in this run, do not claim you lack browsing or tool access.
- Do not mention hidden planning, phase names, checkpoints, or internal protocol.
- Do not fabricate completed actions, tool results, URLs, or files.
- If evidence is missing, say what is missing plainly.
- If the phase context shows drift or contradiction between the user's current turn and earlier user-stated facts, name that tension plainly.
- Do not optimize only for comfort or agreement. If the user's current claim conflicts with earlier facts or evidence, challenge it directly and ask for reconciliation when needed.
"""


_REDRAFT_ISSUE_THRESHOLD = 2
_URL_RE = re.compile(r"https?://[^\s\"'<>),\]]+", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "TODAY", "STOCK",
    "STOCKS", "NYSE", "NASDAQ", "AMEX", "ETF", "ETFS", "CEO", "CFO", "SEC",
    "US", "USA", "USD", "AI", "API", "URL", "JSON", "HTTP", "HTTPS",
}


def _safe_json_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def _tool_success_count(tool_results: list[dict[str, Any]], tool_name: str) -> int:
    return sum(
        1
        for item in tool_results
        if str(item.get("tool_name") or "") == tool_name and bool(item.get("success"))
    )


def _tool_argument_value(item: dict[str, Any], key: str) -> str:
    args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
    return str(args.get(key) or "").strip()


def _valid_fetch_url(url: str) -> str:
    ok, _error, normalized, _host = _validate_fetch_url(str(url or "").strip().rstrip(".,"))
    return normalized if ok else ""


def _canonical_fetch_url_for_loop(url: str) -> str:
    normalized = _valid_fetch_url(url)
    if not normalized:
        return ""
    try:
        parts = urlsplit(normalized)
        path = parts.path or ""
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return urlunsplit((parts.scheme, parts.netloc.lower(), path, parts.query, ""))
    except Exception:
        return normalized.rstrip("/")


def _canonical_tool_arguments_for_loop(tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = dict(arguments or {})
    if tool_name == "web_fetch":
        url = _canonical_fetch_url_for_loop(str(args.get("url") or ""))
        if url:
            args["url"] = url
    elif tool_name == "web_search" and args.get("query") is not None:
        args["query"] = compile_web_search_query(str(args.get("query") or ""))
    return args


def _extract_urls_from_tool_results(tool_results: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    return _plugin_extract_urls_from_tool_results(tool_results, limit=limit)


def _find_cached_retry_sensitive_result(
    tool_results: list[dict[str, Any]],
    *,
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a prior successful web result for the same exact tool call."""
    signature = tool_call_signature(tool_name, arguments or {})
    for item in reversed(tool_results or []):
        if not isinstance(item, dict) or not item.get("success"):
            continue
        prior_signature = tool_call_signature(
            str(item.get("tool_name") or ""),
            item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
        )
        if prior_signature == signature:
            return item
    return None


def _extract_mover_tickers_from_fetched_pages(tool_results: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    return _plugin_extract_mover_tickers_from_fetched_pages(tool_results, limit=limit)


def _select_movers_source_url(urls: list[str]) -> str:
    return _plugin_select_movers_source_url(urls)


def _extract_stock_tickers_from_tool_results(tool_results: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    return _plugin_extract_stock_tickers_from_tool_results(tool_results, limit=limit)


def _explicit_stock_source_workflow_requested(objective: str) -> bool:
    return _plugin_explicit_stock_source_workflow_requested(objective)


def _source_collection_contract_from_objective(objective: str) -> dict[str, int]:
    return _plugin_source_collection_contract_from_objective(objective)


def _default_tool_turn_budget(objective: str, requested_max_turns: Optional[int]) -> int:
    return _plugin_default_tool_turn_budget(objective, requested_max_turns)


_DISCLOSURE_OPENER_RE = re.compile(
    r"^\s*(?:i\s*am|i'?m|i\s+have|i\s+like|i\s+love|i\s+prefer|i\s+enjoy|"
    r"my\s+\w+(?:\s+\w+){0,3}\s+(?:is|are|was)|call\s+me)\b",
    re.IGNORECASE,
)
_RESEARCH_INTENT_RE = re.compile(
    r"\b(search|find|look(?:ing)?\s*(?:up|for)|lookup|google|research|investigate|compare|"
    r"recommend|suggest|need|want|how\s+(?:do|to|can|should)|what(?:'s|\s+is|\s+are|\s+should)|"
    r"where|when|who|which|why|best|top|news|price|prices|latest|current|today|trending|"
    r"help\s+me|show\s+me|give\s+me|fetch|get\s+me|tell\s+me\s+about|ideas|options|resources|"
    r"tutorial|guide|tips|advice)\b",
    re.IGNORECASE,
)


def _is_personal_disclosure_only(message: str) -> bool:
    """True for a first-person statement of fact/preference that should be
    remembered, not web-searched (e.g. "I am a photographer", "I like blue").

    Conservative on purpose: any question mark or research/request verb disables
    it, so genuine research intents ("I am looking for the best DAW") keep their
    web tools. The deterministic extractor still captures the fact regardless;
    this only suppresses a spurious web_search the planner may have chosen.
    """
    text = str(message or "").strip()
    if not text or "?" in text:
        return False
    if not _DISCLOSURE_OPENER_RE.search(text):
        return False
    if _RESEARCH_INTENT_RE.search(text):
        return False
    return len(text.split()) <= 12


def _urls_by_entity_from_search_results(
    *,
    tool_results: list[dict[str, Any]],
    entities: list[str],
    per_entity: int,
) -> dict[str, list[str]]:
    return _plugin_urls_by_entity_from_search_results(
        tool_results=tool_results,
        entities=entities,
        per_entity=per_entity,
    )


def _stock_source_workflow_override(
    *,
    objective: str,
    allowed_tools: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return _plugin_stock_source_workflow_override(
        objective=objective,
        allowed_tools=allowed_tools,
        tool_results=tool_results,
    )


def _should_finalize(
    observation_status: str,
    plan_steps: list[dict[str, Any]],
    tool_turn: int,
    max_tool_calls: Optional[int],
    session_id: str,
    task_tracker=None,
    contract_complete: Optional[bool] = None,
) -> tuple[bool, str]:
    """Decide whether the run should finalize this turn.

    Returns ``(should_finalize, reason)`` — reason is always logged into
    the trace and surfaced in the partial-completion event when finalize
    is forced by a hard cap. Reasons follow a fixed vocabulary so callers
    can route on ``startswith``:

    * ``no_plan_steps`` — backward compatible path; observation alone decides
    * ``contract_complete`` — deterministic workflow contract says enough
      evidence is gathered; finalize even if the observation wants more
    * ``clean_finalize`` — observation says finalize, no pending steps, no open todos
    * ``observation_driven`` — fallback when plan_steps complete but observation drove the call
    * ``pending_steps:[ids]`` — gate refuses finalize because steps remain
    * ``open_todos`` — gate refuses finalize because TaskTracker has active tasks
    * ``max_tool_calls_reached:N_steps_pending`` — hard cap forces finalize anyway

    ``contract_complete`` is the source_collection over-collection cure: when the
    deterministic completion gate (workflow_contracts) reports sufficient
    evidence — or its own hard escape trips — the loop stops collecting even if
    the observation model still says "continue".
    """
    wants_finalize = observation_status == "finalize" or contract_complete is True

    if not plan_steps:
        return wants_finalize, "no_plan_steps"

    pending = [s for s in plan_steps if s.get("status") == "pending"]

    # Hard cap reached — always finalize, but flag pending steps so the
    # response phase can be honest about what was not completed.
    if max_tool_calls is not None and tool_turn >= max_tool_calls:
        return True, f"max_tool_calls_reached:{len(pending)}_steps_pending"

    # Cross-turn TaskTracker open todos. Best-effort — treat tracker errors
    # as "no open todos" so a misbehaving tracker can't deadlock the loop.
    open_todos = False
    if task_tracker is not None and session_id:
        try:
            open_todos = bool(task_tracker.has_active_tasks(session_id))
        except Exception:
            open_todos = False

    # Deterministic contract authority: enough evidence exists, so finalize even
    # if plan steps remain pending. The response phase stays honest via the gate
    # reason and any open missing_slots.
    if contract_complete is True and not open_todos:
        return True, "contract_complete"

    if observation_status == "finalize" and not pending and not open_todos:
        return True, "clean_finalize"

    if pending:
        ids = [str(s.get("id") or "") for s in pending][:6]
        return False, f"pending_steps:{ids}"

    if open_todos:
        return False, "open_todos"

    return observation_status == "finalize", "observation_driven"


def _update_plan_step(
    plan_steps: list[dict[str, Any]],
    tool_name: str,
    success: bool,
    *,
    hard_failure: bool = False,
) -> Optional[str]:
    """Match the next pending plan_step for ``tool_name`` and update status.

    Returns the step id that was updated (for tracing) or None if no match.
    The matcher takes the FIRST pending step whose tool field equals
    ``tool_name`` — order of plan_steps is the planner's stated execution
    order, so this preserves intent.
    """
    if not plan_steps or not tool_name:
        return None
    target_status = "failed" if (hard_failure or not success) else "done"
    for step in plan_steps:
        if (
            isinstance(step, dict)
            and step.get("tool") == tool_name
            and step.get("status") == "pending"
        ):
            step["status"] = target_status
            return str(step.get("id") or "")
    return None
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


def _build_continuation_state_payload(
    *,
    reason: str,
    effective_objective: str,
    run_id: str,
    plan_steps: list[dict[str, Any]],
    missing_slots: list[str],
    strategy: str,
    notes: str,
    tool_results: list[dict[str, Any]],
    issues: Optional[list[str]] = None,
    target_state: str = "",
) -> dict[str, Any]:
    incomplete_steps = [
        {
            "id": str(step.get("id") or ""),
            "description": str(step.get("description") or ""),
            "tool": step.get("tool"),
            "status": str(step.get("status") or "pending"),
            "risk": str(step.get("risk") or "read_only"),
        }
        for step in (plan_steps or [])
        if isinstance(step, dict)
        and str(step.get("status") or "pending") not in {"done", "completed", "skipped"}
    ][:6]
    clean_missing = [str(item).strip() for item in (missing_slots or []) if str(item).strip()][:6]
    clean_issues = [str(item).strip() for item in (issues or []) if str(item).strip()][:6]
    if not incomplete_steps and not clean_missing and not clean_issues and not target_state:
        return {}

    next_action = ""
    if clean_missing:
        next_action = f"Gather missing data: {clean_missing[0]}"
    elif incomplete_steps:
        first = incomplete_steps[0]
        desc = str(first.get("description") or "").strip()
        tool = str(first.get("tool") or "").strip()
        next_action = f"Run {tool}: {desc}" if tool else desc
    elif clean_issues:
        next_action = f"Resolve verification issue: {clean_issues[0]}"
    elif target_state:
        next_action = f"Replan toward target state: {target_state}"

    evidence_summary = [
        format_tool_result_line(item, preview_chars=140, include_status_label=True)
        for item in select_working_evidence(
            tool_results,
            user_message=effective_objective,
            max_results=3,
        )
    ]

    return {
        "reason": str(reason or "unfinished_run"),
        "objective": str(effective_objective or ""),
        "run_id": str(run_id or ""),
        "target_state": str(target_state or ""),
        "next_action": next_action,
        "pending_steps": incomplete_steps,
        "missing_slots": clean_missing,
        "strategy": str(strategy or ""),
        "notes": str(notes or ""),
        "issues": clean_issues,
        "evidence_summary": evidence_summary,
    }


_CONTINUATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "get", "go", "have", "how", "i", "in", "is", "it", "me", "my", "near", "of",
    "on", "or", "please", "that", "the", "this", "to", "use", "we", "what", "with",
    "you",
}


def _continuation_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(text or "").lower())
        if token not in _CONTINUATION_STOPWORDS
    }


def _assess_continuation_state(
    *,
    user_message: str,
    continuation_state: dict[str, Any],
    ambiguous_followup: bool,
) -> dict[str, Any]:
    """Decide whether a durable unfinished plan should steer this turn.

    The policy is intentionally deterministic: explicit resume language wins,
    additive language can extend a related plan, and unrelated concrete tasks
    supersede the stale continuation state before planning begins.
    """
    if not isinstance(continuation_state, dict) or not continuation_state.get("objective"):
        return {"action": "none", "reason": "no_continuation_state", "effective_objective": str(user_message or "")}

    current = str(user_message or "").strip()
    previous_objective = str(continuation_state.get("objective") or "").strip()
    pending_steps = [item for item in (continuation_state.get("pending_steps") or []) if isinstance(item, dict)]
    missing_slots = [str(item).strip() for item in (continuation_state.get("missing_slots") or []) if str(item).strip()]
    lowered = current.lower()

    explicit_resume = bool(re.search(
        r"\b(continue|resume|keep going|finish|carry on|pick up|complete (it|that|the plan)|run the next|next step)\b",
        lowered,
    ))
    additive = bool(re.search(r"\b(also|and also|include|add|check|compare|what about|instead|same thing)\b", lowered))
    current_tokens = _continuation_tokens(current)
    objective_tokens = _continuation_tokens(previous_objective)
    denominator = max(1, min(len(current_tokens), len(objective_tokens)))
    overlap = len(current_tokens & objective_tokens) / denominator

    if ambiguous_followup or explicit_resume:
        return {
            "action": "resume",
            "reason": "explicit_or_ambiguous_resume",
            "effective_objective": previous_objective or current,
            "overlap": overlap,
            "pending_step_count": len(pending_steps),
            "missing_slot_count": len(missing_slots),
        }

    if additive and overlap >= 0.15:
        return {
            "action": "resume_with_update",
            "reason": "additive_related_turn",
            "effective_objective": f"{previous_objective}\nCurrent turn update: {current}",
            "overlap": overlap,
            "pending_step_count": len(pending_steps),
            "missing_slot_count": len(missing_slots),
        }

    if overlap >= 0.45:
        return {
            "action": "resume_with_update",
            "reason": "related_objective_overlap",
            "effective_objective": f"{previous_objective}\nCurrent turn update: {current}",
            "overlap": overlap,
            "pending_step_count": len(pending_steps),
            "missing_slot_count": len(missing_slots),
        }

    return {
        "action": "supersede",
        "reason": "new_turn_not_related_to_pending_plan",
        "effective_objective": current,
        "overlap": overlap,
        "pending_step_count": len(pending_steps),
        "missing_slot_count": len(missing_slots),
    }


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
        self._active_ledgers: dict[str, RunLedger] = {}
        self._background_memory_tasks: set[asyncio.Task] = set()

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
    def _trivial_response(message: str) -> str:
        lowered = str(message or "").strip().lower().strip(".! ")
        if lowered in {"thanks", "thank you"}:
            return "You're welcome."
        if lowered in {"ok", "okay", "cool", "nice", "yes", "yeah", "yep", "sure", "sounds good"}:
            return "Got it."
        return "Hi."

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
        # V3 is the unified convergent engine and the declared profile default
        # (agent_profiles.default_context_mode="v3", with active migration off
        # v1/v2). Fall back to it here too so unconfigured sessions get the
        # unified engine; v1/v2 remain selectable explicitly. V3 transparently
        # migrates older v1/v2 session blobs on first compress.
        requested_context_mode = str(
            request.context_mode or getattr(session, "context_mode", "v3") or "v3"
        ).strip().lower()
        if requested_context_mode not in {"v1", "v2", "v3"}:
            requested_context_mode = str(getattr(session, "context_mode", "v3") or "v3").strip().lower()
        if getattr(session, "context_mode", "v3") != requested_context_mode:
            self.sessions.set_context_mode(session.id, requested_context_mode)
            refreshed_session = self.sessions.get(session.id, owner_id=request.owner_id)
            if refreshed_session is not None:
                session = refreshed_session
        # Guard: refuse to run a chat session on an embed-only model and make
        # image handling explicit instead of silently dropping vision input.
        from config.config import cfg
        from llm.model_capabilities import coerce_chat_model, is_chat_capable, is_vision_capable
        if not is_chat_capable(session_model):
            safe_session_model, _ = coerce_chat_model(session_model, "llama3.2")
            session_model = safe_session_model
        vision_model_swap: dict[str, Any] | None = None
        if request.images and not is_vision_capable(session_model):
            configured_vision_model = str(getattr(cfg, "VISION_MODEL", "") or "").strip()
            if configured_vision_model and is_vision_capable(configured_vision_model):
                vision_model_swap = {
                    "slot": "model",
                    "rejected": session_model,
                    "fallback": configured_vision_model,
                    "reason": "image input requires a vision-capable model",
                }
                session_model = configured_vision_model
            else:
                yield {
                    "type": "error",
                    "message": (
                        "Image input requires a vision-capable model. Select a vision model "
                        "or set VISION_MODEL to a provider-prefixed vision model such as "
                        "gemini:gemini-1.5-flash, openai:gpt-4o-mini, or ollama:llava."
                    ),
                }
                return
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
        if vision_model_swap:
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="model_capability_violation",
                data=vision_model_swap,
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
        continuation_state = getattr(session, "continuation_state", {}) or {}
        continuation_gate = _assess_continuation_state(
            user_message=request.user_message,
            continuation_state=continuation_state if isinstance(continuation_state, dict) else {},
            ambiguous_followup=self._is_ambiguous_followup(request.user_message),
        )
        turn_relation = classify_turn_relation(
            user_message=request.user_message,
            continuation_state=continuation_state if isinstance(continuation_state, dict) else {},
            recent_turns=list(getattr(session, "sliding_window", []) or getattr(session, "full_history", []) or []),
            distant_memory_signals=[
                str(getattr(session, "compressed_context", "") or ""),
                str(getattr(session, "candidate_context", "") or ""),
            ],
            ambiguous_followup=self._is_ambiguous_followup(request.user_message),
        )
        if continuation_gate.get("action") in {"resume", "resume_with_update"}:
            effective_objective = str(continuation_gate.get("effective_objective") or "").strip() or effective_objective
        elif continuation_gate.get("action") == "supersede" and hasattr(self.sessions, "clear_continuation_state"):
            self.sessions.clear_continuation_state(session.id, owner_id=request.owner_id)
            session.continuation_state = {}
            effective_objective = str(getattr(turn_relation, "resolved_objective", "") or "").strip() or effective_objective
        elif str(getattr(turn_relation, "resolved_objective", "") or "").strip():
            effective_objective = str(turn_relation.resolved_objective).strip()

        try:
            tool_results: list[dict[str, Any]] = []
            selected_tools = list(request.forced_tools)
            failed_tool_names: set[str] = set()
            planner_enabled = bool(request.use_planner)
            allowed_tools_list = self._list_allowed_tools(request.allowed_tools)
            ledger_mode = str(os.getenv("LEDGER_MODE", request.ledger_mode or "shadow") or "shadow").strip().lower()
            if ledger_mode not in {"shadow", "ledger_enforced"}:
                ledger_mode = "shadow"
            ledger = RunLedger(
                run_id=run.run_id,
                session_id=session.id,
                turn_id=f"{session.id}:{int(getattr(session, 'message_count', 0) or 0)}",
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                objective=effective_objective,
                allowed_tools=[
                    str(item.get("name") or "")
                    for item in allowed_tools_list
                    if isinstance(item, dict) and item.get("name")
                ],
                ledger_mode=ledger_mode,
            )
            workflow_contract = infer_workflow_contract(
                effective_objective,
                allowed_tools=ledger.allowed_tools,
            )
            ledger.set_workflow_contract(workflow_contract)
            ledger.set_pass_graph(
                build_pass_graph(workflow_contract, workflow_template=request.workflow_template)
            )
            control_policy = resolve_control_policy(
                effective_objective,
                requested=request.control_policy,
                workflow_contract=workflow_contract,
                risk_policy=request.risk_policy,
                allowed_tools=ledger.allowed_tools,
            )
            ledger.set_control_policy(control_policy)
            ledger.set_turn_relation(turn_relation.to_dict())
            plugin_contract = select_workflow_plugin_contract(effective_objective)
            if plugin_contract:
                ledger.set_source_contract(
                    {
                        "plugin_id": plugin_contract.get("plugin_id"),
                        "plugin_contract": plugin_contract,
                        "forbid_unlocked_entity_drift": bool(
                            (plugin_contract.get("entity_lock_rules") or {}).get("forbid_unlocked_entity_drift", True)
                        ),
                    },
                    source="workflow_plugin_contract",
                )
            if control_policy.id == "react" and str(request.control_policy or "auto").lower().replace("-", "_") == "react":
                planner_enabled = False
            if workflow_contract.workflow_shape == "simple_chat":
                planner_enabled = False
                selected_tools = []
            self._active_ledgers[run.run_id] = ledger
            ledger.append_event(
                "intake",
                source="run_engine",
                data={
                    "user_message": request.user_message,
                    "effective_objective": effective_objective,
                    "context_mode": requested_context_mode,
                    "model": session_model,
                    "workflow_template": request.workflow_template,
                    "prompt_version": request.prompt_version,
                    "risk_policy": request.risk_policy,
                    "control_policy": control_policy.to_dict(),
                },
            )
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="turn_relation",
                data=turn_relation.to_dict(),
            )
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="control_policy",
                data=control_policy.to_dict(),
            )
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="policy_selected",
                data=control_policy.to_dict(),
            )
            if continuation_gate.get("action") != "none":
                ledger.append_event(
                    "continuation_gate",
                    source="run_engine",
                    status=str(continuation_gate.get("action") or "info"),
                    data={
                        "action": continuation_gate.get("action"),
                        "reason": continuation_gate.get("reason"),
                        "overlap": continuation_gate.get("overlap"),
                        "pending_step_count": continuation_gate.get("pending_step_count"),
                        "missing_slot_count": continuation_gate.get("missing_slot_count"),
                        "source_run_id": continuation_state.get("run_id") if isinstance(continuation_state, dict) else "",
                    },
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="continuation_gate",
                    data={
                        "action": continuation_gate.get("action"),
                        "reason": continuation_gate.get("reason"),
                        "effective_objective": effective_objective,
                        "overlap": continuation_gate.get("overlap"),
                        "pending_step_count": continuation_gate.get("pending_step_count"),
                        "missing_slot_count": continuation_gate.get("missing_slot_count"),
                        "source_run_id": continuation_state.get("run_id") if isinstance(continuation_state, dict) else "",
                    },
                )
            self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="intake")

            if (
                self._is_trivial_acknowledgement(request.user_message)
                and continuation_gate.get("action") == "none"
                and not request.forced_tools
            ):
                final_response = self._trivial_response(request.user_message)
                ledger.append_event(
                    "response",
                    source="run_engine",
                    status="trivial_chat",
                    data={"content_length": len(final_response)},
                )
                self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="trivial_chat")
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="route_decision",
                    data={
                        "route_type": "trivial_chat",
                        "reason": "short_acknowledgement_fast_path",
                        "tools_allowed": False,
                    },
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="assistant_response",
                    data={"content": final_response, "content_length": len(final_response)},
                )
                self.sessions.append_message(session.id, "assistant", final_response)
                self.run_store.finish_run(run.run_id, status="completed")
                yield {"type": "token", "content": final_response}
                yield {"type": "done", "run_id": run.run_id, "session_id": session.id}
                return

            workflow_template = get_workflow_template(request.workflow_template)
            workflow_pattern = get_workflow_pattern(workflow_template.workflow_pattern)
            if workflow_pattern:
                ledger.append_event(
                    "workflow_pattern",
                    source="workflow_template",
                    status="active",
                    data=workflow_pattern.to_dict(),
                )
                self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="workflow_pattern")
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
            max_turns = _default_tool_turn_budget(effective_objective, request.max_turns)
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
            turn_policy_intent = ""

            # ── Converged turn router ───────────────────────────────────────
            # One deterministic gate maps intent -> {tool palette, planner on/off,
            # answer-from-memory}. This is the single place that fixes routing
            # (memory/identity/meta never web-searched), context poisoning (no web
            # junk enters a "remember my name" turn), and looping (direct-fact and
            # chat take zero tools; disclosures store + acknowledge). Honoured only
            # when the caller did not force a specific tool set.
            if not request.forced_tools:
                turn_policy = resolve_turn_policy(
                    effective_objective,
                    user_message=request.user_message,
                    allowed_tools=[
                        str(t.get("name") or "")
                        for t in allowed_tools_list
                        if isinstance(t, dict)
                    ],
                    direct_fact_answerable=direct_fact_memory_only,
                    workflow_shape=workflow_contract.workflow_shape,
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="turn_policy",
                    data=turn_policy.to_dict(),
                )
                turn_policy_intent = str(turn_policy.intent or "")
                if turn_policy.tool_whitelist is not None:
                    _palette = set(turn_policy.tool_whitelist)
                    allowed_tools_list = [
                        t for t in allowed_tools_list
                        if isinstance(t, dict) and str(t.get("name") or "") in _palette
                    ]
                if not turn_policy.use_planner:
                    planner_enabled = False
                if turn_policy.answer_from_memory:
                    direct_fact_memory_only = True
                if turn_policy.tool_whitelist is not None and not allowed_tools_list:
                    selected_tools = []

            conversation_tension = analyze_conversation_tension(
                user_message=request.user_message,
                current_facts=current_facts,
                deterministic_keyed_facts=deterministic_keyed_facts,
                session_history=list(getattr(session, "full_history", []) or []),
                candidate_signals=list(getattr(session, "candidate_signals", []) or []),
                current_stance_signals=stance_signals,
            )
            # Slice 3: tag each extracted fact with source="user_direct" in
            # the trace event so the evidence ledger (Slice 7) and operator
            # views can distinguish direct user assertions from compression-
            # side proposals. The source label lives in the trace; the
            # semantic_graph schema is not changed (per convergence rule).
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
                            "source": "user_direct",
                        }
                        for item in deterministic_keyed_facts
                    ],
                    "voids": [
                        {**v, "source": "user_direct"} if isinstance(v, dict) else v
                        for v in deterministic_voids
                    ],
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
                _tool_names = list(request.allowed_tools or ())
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
            # Per-turn plan_steps spine — initialized empty so all code paths
            # (including the no-plan / fast-path return) can reference it
            # safely. Populated by planner output below; mutated in place by
            # _update_plan_step after each tool turn.
            plan_steps_state: list[dict[str, Any]] = []
            if (
                continuation_gate.get("action") in {"resume", "resume_with_update"}
                and isinstance(continuation_state, dict)
            ):
                plan_steps_state = [
                    dict(step)
                    for step in (continuation_state.get("pending_steps") or [])
                    if isinstance(step, dict)
                ]
                if plan_steps_state:
                    ledger.set_plan(plan_steps_state, source="continuation_state")
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="continuation_plan_seed")
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="continuation_plan_seeded",
                        data={
                            "pending_step_count": len(plan_steps_state),
                            "source_run_id": continuation_state.get("run_id") or "",
                        },
                    )
            planned_risk_tier: str = "read_only"
            # Route resolved from plan (used by hooks and verify skip).
            _plan_route: str = ""

            if not selected_tools and not direct_fact_memory_only and planner_enabled and self.orchestrator is not None:
                # Slice 5: spatial-aware planning. Pre-fetch the drawer of any
                # locus the user message targets (token-overlap detection),
                # plus 1-hop neighbors with score decay. The planner then sees
                # the actual workspace context — what facts are filed where —
                # before deciding what tools to use. Without this the planner
                # only sees a flat list of locus IDs and can't reason about
                # which one matches the question's substance.
                _spatial_drawers: list[dict[str, Any]] = []
                try:
                    if available_loci and self.graph is not None:
                        from memory.retrieval import detect_locus_by_overlap
                        _detected_locus = detect_locus_by_overlap(
                            request.user_message or "", list(available_loci)
                        )
                        if _detected_locus:
                            _primary = self.graph.get_compiled_drawer(_detected_locus)
                            if _primary:
                                _spatial_drawers.append({
                                    "locus_id": _detected_locus,
                                    "hop": 0,
                                    "score": 1.0,
                                    "content": _primary,
                                })
                            # 1-hop neighbors with score decay; cap fanout to
                            # keep the planning packet tight.
                            try:
                                _neighbors = self.graph.get_locus_neighbors(
                                    _detected_locus, max_depth=1, max_fanout=3,
                                )
                            except Exception:
                                _neighbors = []
                            for nbr_id, _depth, nbr_score in _neighbors:
                                nbr_drawer = self.graph.get_compiled_drawer(nbr_id)
                                if not nbr_drawer:
                                    continue
                                _spatial_drawers.append({
                                    "locus_id": nbr_id,
                                    "hop": 1,
                                    "score": float(nbr_score),
                                    "content": nbr_drawer,
                                })
                            if _spatial_drawers:
                                self._trace(
                                    request=request,
                                    run_id=run.run_id,
                                    event_type="spatial_drawers_loaded",
                                    data={
                                        "primary_locus": _detected_locus,
                                        "drawer_count": len(_spatial_drawers),
                                        "neighbor_ids": [d["locus_id"] for d in _spatial_drawers if d["hop"] > 0],
                                    },
                                )
                except Exception:
                    _spatial_drawers = []

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
                    spatial_drawers=_spatial_drawers if _spatial_drawers else None,
                    plan_steps=plan_steps_state,
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

                # ── Capture plan_steps spine (Slice 1) ──
                # Mutable per-turn list; runtime updates statuses as tools complete.
                # Copied so updates don't leak back into structured_plan if the
                # planner result is reused elsewhere (defensive).
                planner_plan_steps = [
                    dict(step) for step in (structured_plan.get("plan_steps") or [])
                    if isinstance(step, dict)
                ]
                if planner_plan_steps:
                    plan_steps_state = planner_plan_steps
                if not plan_steps_state and workflow_pattern:
                    selected_tool_names_for_pattern = [
                        entry.get("name")
                        for entry in structured_plan.get("tools", [])
                        if isinstance(entry, dict)
                    ]
                    if "shopping_advice" in selected_tool_names_for_pattern or "shopping_advice" in request.allowed_tools:
                        plan_steps_state = [dict(step) for step in workflow_pattern.default_plan_steps]
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="workflow_pattern_applied",
                            data={
                                "pattern_id": workflow_pattern.id,
                                "source": "runtime_fallback_plan_steps",
                                "step_count": len(plan_steps_state),
                            },
                        )
                planned_risk_tier = str(structured_plan.get("risk_tier", "read_only")).strip().lower() or "read_only"
                if plan_steps_state:
                    ledger.set_plan(
                        plan_steps_state,
                        source="planner" if planner_plan_steps else "continuation_or_workflow",
                    )
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="plan_steps")
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="plan_steps",
                        data={
                            "step_count": len(plan_steps_state),
                            "risk_tier": planned_risk_tier,
                            "steps": [
                                {"id": s.get("id"), "tool": s.get("tool"), "risk": s.get("risk")}
                                for s in plan_steps_state
                            ],
                        },
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="plan_committed",
                        data={
                            "step_count": len(plan_steps_state),
                            "risk_tier": planned_risk_tier,
                            "mutable_plan": bool(getattr(control_policy, "mutable_plan", True)),
                        },
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

                # Slice 6: dispatch gate. Promote the advisory to a real gate
                # for the highest-risk tier ONLY: when the plan picked
                # destructive tools AND the user message has no authorization
                # verb, refuse dispatch and ask for explicit confirmation
                # instead. Write/read_only plans pass through with the trace
                # advisory only — over-blocking would break legitimate
                # implicit-write turns ("fix the auth bug" + file_str_replace).
                _destructive_no_auth = (
                    _plan_advisory["max_tier"] == "destructive"
                    and not _plan_advisory["clear"]
                )
                if _destructive_no_auth:
                    _risky_tools = [
                        name for name in selected_tools
                        if name in {"bash"}  # only true-destructive surfaces
                    ]
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="dispatch_gate_blocked",
                        data={
                            "reason": "destructive_plan_requires_explicit_auth",
                            "risky_tools": _risky_tools,
                            "user_message_preview": (request.user_message or "")[:160],
                        },
                    )
                    yield {
                        "type": "clarification_needed",
                        "reason": "destructive_plan_requires_explicit_auth",
                        "risky_tools": _risky_tools,
                        "question": (
                            "I planned a destructive action ("
                            + ", ".join(_risky_tools)
                            + ") but your message has no explicit authorization verb. "
                            "Should I proceed? (Reply with an action verb like 'run', "
                            "'install', or 'delete' to confirm.)"
                        ),
                    }
                    selected_tools = []
                    plan_steps_state = []  # don't carry the destructive plan forward
                    ledger.update_plan_steps(plan_steps_state, source="dispatch_gate")
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="dispatch_gate_blocked")

                _plan_route = str(structured_plan.get("route", "")).strip()
                if structured_plan.get("strategy"):
                    strategy = str(structured_plan.get("strategy"))
                    latest_strategy = strategy
                    latest_notes = str(structured_plan.get("notes") or "")
                    ledger.append_event(
                        "plan",
                        source="planner",
                        status="planned",
                        data={
                            "strategy": strategy,
                            "tools": list(selected_tools),
                            "confidence": structured_plan.get("confidence"),
                            "route": _plan_route,
                        },
                    )
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="plan")
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

            if not selected_tools and not direct_fact_memory_only and turn_policy_intent != "conversation_recall":
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
                    ledger.append_event(
                        "plan",
                        source="run_engine_fallback",
                        status="planned_fallback",
                        data={"strategy": bootstrap_strategy, "tools": list(selected_tools)},
                    )
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="plan_fallback")
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

            # Personal disclosures ("I am a photographer", "I like blue") are facts
            # to remember, not research queries. The deterministic extractor already
            # captures them; drop any spurious web tool the planner chose so the
            # agent acknowledges + stores instead of web-searching the user.
            if selected_tools and _is_personal_disclosure_only(request.user_message):
                _filtered_tools = [
                    t for t in selected_tools
                    if t not in {"web_search", "web_fetch", "web_fetch_batch", "rag_search"}
                ]
                if _filtered_tools != selected_tools:
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="disclosure_route",
                        data={
                            "dropped": [t for t in selected_tools if t not in _filtered_tools],
                            "message": str(request.user_message or "")[:80],
                        },
                    )
                    selected_tools = _filtered_tools

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
                    plan_steps=plan_steps_state,
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
                    run_id=run.run_id,
                    run_ledger=ledger,
                )
                if tool_call is None:
                    break

                if not isinstance(tool_call.arguments, dict):
                    tool_call.arguments = {}
                original_tool_arguments = dict(tool_call.arguments or {})
                tool_call.arguments = _canonical_tool_arguments_for_loop(
                    tool_call.tool_name,
                    tool_call.arguments,
                )
                if (
                    tool_call.tool_name == "web_search"
                    and str(original_tool_arguments.get("query") or "") != str(tool_call.arguments.get("query") or "")
                ):
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="tool_argument_compiled",
                        data={
                            "tool_name": "web_search",
                            "field": "query",
                            "original": str(original_tool_arguments.get("query") or ""),
                            "compiled": str(tool_call.arguments.get("query") or ""),
                            "reason": "workflow text was reduced to a retrieval probe",
                        },
                    )
                if tool_call.tool_name == "web_search":
                    if request.search_backend and request.search_backend != "auto":
                        tool_call.arguments["backend"] = request.search_backend
                    if request.search_engine:
                        tool_call.arguments["search_engine"] = request.search_engine
                if tool_call.tool_name == "shopping_advice":
                    tool_call.arguments = self._normalize_shopping_arguments(
                        tool_call.arguments,
                        objective=effective_objective,
                    )

                gated_tool_call, gate_event = self._gate_tool_call_with_ledger(
                    request=request,
                    run_id=run.run_id,
                    ledger=ledger,
                    tool_call=tool_call,
                    tool_turn=tool_turn,
                )
                if gate_event:
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="policy_gate")
                    if gate_event.get("type") == "recovery_started":
                        yield gate_event
                if gated_tool_call is None:
                    yield gate_event or {
                        "type": "policy_violation",
                        "issue": "tool_call_blocked",
                        "recovery_class": "missing_input",
                        "turn": tool_turn,
                    }
                    selected_tools = []
                    break
                tool_call = gated_tool_call

                pre_signature = tool_call_signature(tool_call.tool_name, tool_call.arguments or {})
                if is_retry_sensitive_tool(tool_call.tool_name) and repeated_tool_signatures.get(pre_signature, 0) >= 1:
                    workflow_override = select_workflow_override(
                        objective=effective_objective,
                        allowed_tools=allowed_tools_list,
                        tool_results=tool_results,
                    )
                    clue_map = dict(workflow_override.get("argument_clues") or {})
                    pivot_tool = (workflow_override.get("selected_tools") or [tool_call.tool_name])[0] if workflow_override else tool_call.tool_name
                    pivot_clue = str(clue_map.get(pivot_tool) or "").strip()
                    if pivot_tool == "web_search" and pivot_clue:
                        pivot_args = {"query": pivot_clue}
                        if request.search_backend and request.search_backend != "auto":
                            pivot_args["backend"] = request.search_backend
                        if request.search_engine:
                            pivot_args["search_engine"] = request.search_engine
                        tool_call = ToolCall(
                            tool_name="web_search",
                            arguments=pivot_args,
                            raw_json=json.dumps({"tool": "web_search", "arguments": pivot_args}),
                        )
                        planned_argument_clues.update(clue_map)
                    elif pivot_tool == "web_fetch" and pivot_clue:
                        url_match = _URL_RE.search(pivot_clue)
                        if url_match:
                            url = _valid_fetch_url(url_match.group(0))
                            if url:
                                pivot_args = {"url": url}
                                tool_call = ToolCall(
                                    tool_name="web_fetch",
                                    arguments=pivot_args,
                                    raw_json=json.dumps({"tool": "web_fetch", "arguments": pivot_args}),
                                )
                                planned_argument_clues.update(clue_map)

                tool_call_count += 1

                # ── Hook: tool_selected ──
                hooks.emit_sync("tool_selected", {
                    "tool_name": tool_call.tool_name,
                    "arguments_preview": str(tool_call.arguments or {})[:120],
                    "turn": tool_turn,
                }, run_id=run.run_id, session_id=request.session_id)

                validation_error = self.tool_registry.validate_tool_call(tool_call)
                call_payload = canonical_tool_call(tool_call.tool_name, tool_call.arguments or {})
                tool_call_record = ledger.add_tool_call(
                    tool_call.tool_name,
                    tool_call.arguments or {},
                    source="actor",
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="tool_call",
                    data={**call_payload, "turn": tool_turn, "tool_call_id": tool_call_record.id},
                )
                self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="tool_call")
                yield {"type": "tool_call", **call_payload, "tool_call_id": tool_call_record.id}
                if validation_error:
                    failed_tool_names.add(tool_call.tool_name)
                    result_payload = {
                        "tool_name": tool_call.tool_name,
                        "success": False,
                        "content": f"Tool validation failed: {validation_error}",
                        "arguments": dict(tool_call.arguments or {}),
                    }
                    tool_results.append(result_payload)
                    result_record = ledger.link_tool_result(
                        tool_call_id=tool_call_record.id,
                        tool_name=tool_call.tool_name,
                        success=False,
                        status="failed",
                        summary=clip_text(result_payload["content"], 280),
                    )
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="tool_validation_failed")
                    self._record_tool_outcome(
                        request=request,
                        tool_name=tool_call.tool_name,
                        success=False,
                        error_kind="validation_error",
                    )
                    _update_plan_step(plan_steps_state, tool_call.tool_name, success=False)
                    ledger.update_plan_steps(plan_steps_state, source="run_engine")
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
                            "tool_call_id": tool_call_record.id,
                            "tool_result_id": result_record.id,
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
                        "tool_call_id": tool_call_record.id,
                        "tool_result_id": result_record.id,
                    }
                    break

                signature = tool_call_signature(tool_call.tool_name, tool_call.arguments or {})
                if is_retry_sensitive_tool(tool_call.tool_name):
                    repeat_count = repeated_tool_signatures.get(signature, 0)
                    if repeat_count >= 1:
                        cached_result = _find_cached_retry_sensitive_result(
                            tool_results,
                            tool_name=tool_call.tool_name,
                            arguments=tool_call.arguments or {},
                        )
                        duplicate_message = (
                            f"Duplicate {tool_call.tool_name} call reused from this run's cached tool result."
                            if cached_result
                            else (
                                f"Duplicate {tool_call.tool_name} call suppressed for this run. "
                                "Use the existing result or pivot to a different query/source."
                            )
                        )
                        duplicate_success = bool(cached_result)
                        if not duplicate_success:
                            failed_tool_names.add(tool_call.tool_name)
                        duplicate_content = (
                            str(cached_result.get("content") or "")
                            if cached_result
                            else duplicate_message
                        )
                        result_payload = {
                            "tool_name": tool_call.tool_name,
                            "success": duplicate_success,
                            "content": duplicate_content,
                            "arguments": dict(tool_call.arguments or {}),
                            "cached_duplicate": duplicate_success,
                        }
                        if cached_result and cached_result.get("actor_summary"):
                            result_payload["actor_summary"] = str(cached_result.get("actor_summary") or "")
                        if cached_result and isinstance(cached_result.get("extracted_urls"), list):
                            result_payload["extracted_urls"] = list(cached_result.get("extracted_urls") or [])
                        tool_results.append(result_payload)
                        result_record = ledger.link_tool_result(
                            tool_call_id=tool_call_record.id,
                            tool_name=tool_call.tool_name,
                            success=duplicate_success,
                            status="cached" if duplicate_success else "failed",
                            summary=clip_text(duplicate_message, 280),
                        )
                        if duplicate_success and is_substantive_tool_result(result_payload):
                            ledger.add_evidence_from_result(result_record)
                        self._trace_ledger(
                            request=request,
                            run_id=run.run_id,
                            ledger=ledger,
                            reason="duplicate_cached" if duplicate_success else "duplicate_suppressed",
                        )
                        self._record_tool_outcome(
                            request=request,
                            tool_name=tool_call.tool_name,
                            success=duplicate_success,
                            error_kind="" if duplicate_success else "duplicate_suppressed",
                        )
                        _update_plan_step(plan_steps_state, tool_call.tool_name, success=duplicate_success)
                        ledger.update_plan_steps(plan_steps_state, source="run_engine")
                        self.run_store.save_checkpoint(
                            run_id=run.run_id,
                            phase="acting",
                            tool_turn=tool_turn,
                            status="duplicate_cached" if duplicate_success else "duplicate_suppressed",
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
                                "success": duplicate_success,
                                "status": "cached" if duplicate_success else "failed",
                                "content_length": len(duplicate_content),
                                "content_preview": clip_text(duplicate_content, 280),
                                "cached_duplicate": duplicate_success,
                                "turn": tool_turn,
                                "tool_call_id": tool_call_record.id,
                                "tool_result_id": result_record.id,
                            },
                        )
                        yield {
                            "type": "tool_result",
                            "tool": tool_call.tool_name,
                            "tool_name": tool_call.tool_name,
                            "success": duplicate_success,
                            "status": "cached" if duplicate_success else "failed",
                            "content_preview": clip_text(duplicate_content, 280),
                            "content": duplicate_content,
                            "cached_duplicate": duplicate_success,
                            "tool_call_id": tool_call_record.id,
                            "tool_result_id": result_record.id,
                        }
                        self._save_pass_record(
                            run_id=run.run_id,
                            phase=ContextPhase.ACTING,
                            tool_turn=tool_turn,
                            status="duplicate_cached" if duplicate_success else "duplicate_suppressed",
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
                        workflow_override = select_workflow_override(
                            objective=effective_objective,
                            allowed_tools=allowed_tools_list,
                            tool_results=tool_results,
                        )
                        had_workflow_override = bool(workflow_override)
                        if workflow_override:
                            self._sync_workflow_override_to_ledger(
                                ledger=ledger,
                                workflow_override=workflow_override,
                                source="duplicate_recovery",
                            )
                            self._trace_workflow_override(
                                request=request,
                                run_id=run.run_id,
                                tool_turn=tool_turn,
                                workflow_override=workflow_override,
                                source="duplicate_recovery",
                            )
                            normalized_observation.update({
                                "status": workflow_override.get("status", normalized_observation["status"]),
                                "strategy": workflow_override.get("strategy", normalized_observation["strategy"]),
                                "notes": workflow_override.get("notes", normalized_observation["notes"]),
                                "selected_tools": list(workflow_override.get("selected_tools") or []),
                                "missing_slots": list(workflow_override.get("missing_slots") or []),
                                "should_continue": bool(workflow_override.get("selected_tools")),
                            })
                            planned_argument_clues.update(dict(workflow_override.get("argument_clues") or {}))
                        if (
                            normalized_observation["should_continue"]
                            and repeat_count >= 1
                            and not had_workflow_override
                        ):
                            stagnation_message = (
                                "Loop_Stagnation: duplicate tool recovery did not produce a new deterministic next action. "
                                "Stopping the tool loop and preserving continuation state instead of repeating the same call."
                            )
                            normalized_observation.update({
                                "status": "blocked",
                                "notes": stagnation_message,
                                "selected_tools": [],
                                "missing_slots": list(dict.fromkeys([
                                    *(normalized_observation.get("missing_slots") or []),
                                    "duplicate_loop",
                                ])),
                                "should_continue": False,
                            })
                            ledger.append_event(
                                "loop_stagnation",
                                source="duplicate_recovery",
                                status="blocked",
                                data={
                                    "tool": tool_call.tool_name,
                                    "signature": signature,
                                    "repeat_count": repeat_count + 1,
                                    "recovery_class": "duplicate_loop",
                                    "message": stagnation_message,
                                },
                            )
                            self._trace(
                                request=request,
                                run_id=run.run_id,
                                event_type="loop_stagnation",
                                data={
                                    "tool": tool_call.tool_name,
                                    "signature": signature,
                                    "repeat_count": repeat_count + 1,
                                    "recovery_class": "duplicate_loop",
                                    "message": stagnation_message,
                                },
                            )
                            yield {
                                "type": "loop_stagnation",
                                "tool": tool_call.tool_name,
                                "signature": signature,
                                "repeat_count": repeat_count + 1,
                                "recovery_class": "duplicate_loop",
                                "message": stagnation_message,
                            }
                        selected_tools = list(normalized_observation["selected_tools"])
                        latest_strategy = str(normalized_observation["strategy"] or latest_strategy)
                        latest_notes = str(normalized_observation["notes"] or latest_notes)
                        latest_observation_status = str(normalized_observation["status"] or "")
                        latest_observation_tools = list(selected_tools)
                        latest_missing_slots = list(normalized_observation.get("missing_slots") or [])
                        ledger.append_event(
                            "observation",
                            source="loop_manager",
                            status=latest_observation_status or "info",
                            data={
                                "strategy": normalized_observation["strategy"],
                                "notes": normalized_observation["notes"],
                                "confidence": normalized_observation["confidence"],
                                "should_continue": normalized_observation["should_continue"],
                                "tools": selected_tools,
                                "missing_slots": latest_missing_slots,
                            },
                        )
                        self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="observation")
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
                        "_ledger_authority": True,
                        "_ledger_tool_results": list(tool_results),
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

                # Slice 4: shape the tool output for the actor — KEY_LINES on
                # long bash, SUMMARY on web_search, HARD_FAILURE preservation,
                # plus a one-line actor_summary for the working_evidence
                # cross-tool accumulator. Shaping never raises; on failure
                # the original content is kept verbatim.
                _shaped_content, _actor_summary = shape_tool_result_for_actor(
                    tool_result.tool_name,
                    _enriched_content,
                    success=bool(tool_result.success),
                )
                result_payload = {
                    "tool_name": tool_result.tool_name,
                    "success": tool_result.success,
                    "content": _shaped_content,
                    "actor_summary": _actor_summary,
                    "arguments": dict(tool_call.arguments or {}),
                    "tool_call_id": tool_call_record.id,
                }
                if tool_result.tool_name in {"web_search", "web_fetch"}:
                    result_payload["extracted_urls"] = _extract_urls_from_tool_results(
                        [
                            {
                                "success": bool(tool_result.success),
                                "content": _enriched_content,
                            }
                        ],
                        limit=12,
                    )
                tool_results.append(result_payload)
                result_record = ledger.link_tool_result(
                    tool_call_id=tool_call_record.id,
                    tool_name=tool_result.tool_name,
                    success=bool(tool_result.success),
                    status="ok" if tool_result.success else "failed",
                    summary=clip_text(_actor_summary or _shaped_content, 280),
                )
                result_payload["tool_result_id"] = result_record.id
                if is_substantive_tool_result(result_payload):
                    ledger.add_evidence_from_result(result_record)
                self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="tool_result")
                if not tool_result.success:
                    failed_tool_names.add(tool_result.tool_name)
                self._record_tool_outcome(
                    request=request,
                    tool_name=tool_result.tool_name,
                    success=bool(tool_result.success),
                    error_kind=("hard_failure" if (not tool_result.success and "HARD_FAILURE" in (tool_result.content or "")) else ""),
                )
                _update_plan_step(
                    plan_steps_state,
                    tool_result.tool_name,
                    success=bool(tool_result.success),
                    hard_failure=("HARD_FAILURE" in (tool_result.content or "")),
                )
                ledger.update_plan_steps(plan_steps_state, source="run_engine")

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
                        "tool_call_id": tool_call_record.id,
                        "tool_result_id": result_record.id,
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
                    "tool_call_id": tool_call_record.id,
                    "tool_result_id": result_record.id,
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
                        "tool_call_id": tool_call_record.id,
                    }
                    tool_results.append(alert_payload)
                    alert_record = ledger.link_tool_result(
                        tool_call_id=tool_call_record.id,
                        tool_name=alert_tool_name,
                        success=False,
                        status="failed",
                        summary=clip_text(latest_notes, 280),
                    )
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="logical_stall")
                    self._record_tool_outcome(
                        request=request,
                        tool_name=alert_tool_name,
                        success=False,
                        error_kind="logical_stall",
                    )
                    _update_plan_step(plan_steps_state, alert_tool_name, success=False)
                    ledger.update_plan_steps(plan_steps_state, source="run_engine")
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
                            "tool_call_id": tool_call_record.id,
                            "tool_result_id": alert_record.id,
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
                # ── Efficiency: skip the LLM observe call when a deterministic
                # workflow override already decides the next step. For source/
                # stock workflows the override replaced the observe result
                # anyway, so that model request was pure waste — skipping it
                # removes one LLM call per tool turn (a real free-tier quota
                # saver) with identical routing behavior. Other turns still run
                # the LLM observer.
                workflow_override = select_workflow_override(
                    objective=effective_objective,
                    allowed_tools=allowed_tools_list,
                    tool_results=tool_results,
                )
                if workflow_override:
                    self._sync_workflow_override_to_ledger(
                        ledger=ledger,
                        workflow_override=workflow_override,
                        source="observation_override",
                    )
                    self._trace_workflow_override(
                        request=request,
                        run_id=run.run_id,
                        tool_turn=tool_turn,
                        workflow_override=workflow_override,
                        source="observation_override",
                    )
                    normalized_observation = self._normalize_observation_decision({
                        "status": workflow_override.get("status", "continue"),
                        "strategy": workflow_override.get("strategy", latest_strategy),
                        "notes": workflow_override.get("notes", latest_notes),
                        "tools": [{"name": t} for t in (workflow_override.get("selected_tools") or [])],
                        "missing": list(workflow_override.get("missing_slots") or []),
                    })
                    planned_argument_clues.update(dict(workflow_override.get("argument_clues") or {}))
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="observation_deterministic",
                        data={
                            "tool_turn": tool_turn,
                            "source": "workflow_override",
                            "status": normalized_observation["status"],
                            "saved_llm_call": True,
                        },
                    )
                else:
                    try:
                        observation = await self.orchestrator.observe_with_context(
                            query=effective_objective,
                            tools_list=allowed_tools_list,
                            tool_results=tool_results,
                            model=request.planner_model or session_model,
                            compiled_context=observation_packet.content,
                        )
                        normalized_observation = self._normalize_observation_decision(observation)
                    except (RateLimitError, ProviderError) as _obs_exc:
                        # Resilience: a provider 429 / 5xx on the observer must not
                        # kill the run (it's exactly what a real Gemini free-tier
                        # run hit). Fall back to a deterministic observation:
                        # finalize from the evidence we have, else keep going.
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="observation_fallback",
                            data={"tool_turn": tool_turn, "error": type(_obs_exc).__name__},
                        )
                        normalized_observation = self._normalize_observation_decision({
                            "status": "finalize" if tool_results else "continue",
                            "strategy": latest_strategy,
                            "notes": "deterministic observation (provider unavailable)",
                            "tools": [],
                            "missing": [],
                        })
                selected_tools = list(normalized_observation["selected_tools"])
                latest_strategy = str(normalized_observation["strategy"] or latest_strategy)
                latest_notes = str(normalized_observation["notes"] or latest_notes)
                latest_observation_status = str(normalized_observation["status"] or "")
                latest_observation_tools = list(selected_tools)
                latest_missing_slots = list(normalized_observation.get("missing_slots") or [])
                ledger.append_event(
                    "observation",
                    source="loop_manager",
                    status=latest_observation_status or "info",
                    data={
                        "strategy": normalized_observation["strategy"],
                        "notes": normalized_observation["notes"],
                        "confidence": normalized_observation["confidence"],
                        "should_continue": normalized_observation["should_continue"],
                        "tools": selected_tools,
                        "missing_slots": latest_missing_slots,
                    },
                )
                self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="observation")
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
                # ── Slice 2: finalize gate ──────────────────────────────
                # Cross-check the observation's status against plan_steps and
                # the cross-session TaskTracker. The observation alone is no
                # longer authoritative — finalize requires (status == finalize)
                # AND (all plan_steps done) AND (no open todos). If the gate
                # refuses, override `selected_tools` with the next pending
                # step's tool so the actor loop continues against that step.
                from memory.task_tracker import get_session_task_tracker as _get_tt
                _tracker_for_gate = None
                try:
                    _tracker_for_gate = _get_tt()
                except Exception:
                    _tracker_for_gate = None
                # ── Move C: recompute the deterministic workflow contract from the
                # accumulated tool results so its completion gate can finalize a
                # source/web collection instead of over-collecting to the cap. The
                # gate has its own hard escape (fetch target met OR turn budget),
                # so it can never lock the loop forever.
                _contract_complete = None
                try:
                    if workflow_contract is not None and workflow_contract.workflow_shape == "source_collection":
                        workflow_contract = update_contract_from_tool_results(
                            workflow_contract,
                            tool_results,
                            tool_turn=tool_turn,
                            max_tool_turns=normalized_max_tool_calls or max_turns,
                        )
                        ledger.set_workflow_contract(workflow_contract)
                        _contract_complete = workflow_contract.completion_gate.final_answer_allowed
                except Exception:
                    _contract_complete = None
                _gate_finalize, _gate_reason = _should_finalize(
                    observation_status=normalized_observation["status"],
                    plan_steps=plan_steps_state,
                    tool_turn=tool_turn,
                    max_tool_calls=normalized_max_tool_calls,
                    session_id=session.id,
                    task_tracker=_tracker_for_gate,
                    contract_complete=_contract_complete,
                )
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="finalize_gate",
                    data={
                        "should_finalize": _gate_finalize,
                        "reason": _gate_reason,
                        "tool_turn": tool_turn,
                        "pending_step_count": sum(
                            1 for s in plan_steps_state if s.get("status") == "pending"
                        ),
                    },
                )

                if (
                    not _gate_finalize
                    and _gate_reason.startswith("pending_steps")
                    and not (workflow_override and selected_tools)
                ):
                    # Force one more actor turn against the first pending step.
                    next_step = next(
                        (s for s in plan_steps_state if s.get("status") == "pending"), None
                    )
                    if next_step and next_step.get("tool"):
                        selected_tools = [next_step["tool"]]
                        latest_notes = (
                            f"{latest_notes}\nNext required: {next_step.get('description', '')}"
                            if latest_notes else
                            f"Next required: {next_step.get('description', '')}"
                        )
                    elif not selected_tools:
                        # No tool to drive the step — let observation decide,
                        # but don't deadlock.
                        _gate_finalize = True

                # Partial-completion event when the hard cap forces finalize
                # while plan_steps remain. This must be emitted before the
                # response phase so the response generator can read it via
                # the injected `latest_notes` and never claim full success.
                if _gate_finalize and _gate_reason.startswith("max_tool_calls_reached"):
                    completed = [
                        s.get("id") for s in plan_steps_state
                        if s.get("status") in {"done", "skipped"}
                    ]
                    incomplete = [
                        s.get("id") for s in plan_steps_state
                        if s.get("status") == "pending"
                    ]
                    incomplete_descs = [
                        s.get("description", "") for s in plan_steps_state
                        if s.get("status") == "pending"
                    ]
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="partial_completion",
                        data={
                            "completed_steps": completed,
                            "incomplete_steps": incomplete,
                            "reason": "max_tool_calls_reached",
                        },
                    )
                    yield {
                        "type": "partial_completion",
                        "completed_steps": completed,
                        "incomplete_steps": incomplete,
                        "reason": "max_tool_calls_reached",
                    }
                    if incomplete_descs:
                        latest_notes = (
                            f"{latest_notes}\nNote: {len(incomplete_descs)} planned step(s) "
                            f"could not be completed: {'; '.join(incomplete_descs[:3])}. "
                            "Report exactly what was done and what wasn't."
                        )

                _effective_should_continue = (
                    normalized_observation["should_continue"]
                    and not _gate_finalize
                ) or (not _gate_finalize and bool(selected_tools))

                if _effective_should_continue:
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
                else:
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
                plan_steps=plan_steps_state,
            )
            response_context_block = response_packet.content
            answer_patch_for_response = self._extract_answer_patch(tool_results)
            if answer_patch_for_response:
                response_context_block = ""
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="response_context_elided",
                    data={
                        "reason": "deterministic_answer_patch",
                        "patch_format": answer_patch_for_response.get("format"),
                        "original_chars": len(response_packet.content or ""),
                    },
                )
            elif self._should_use_lean_response_context(
                user_message=request.user_message,
                effective_objective=effective_objective,
                workflow_shape=workflow_contract.workflow_shape,
                tool_results=tool_results,
                selected_tools=selected_tools,
                current_facts=current_facts,
                continuation_state=continuation_state if isinstance(continuation_state, dict) else {},
                plan_steps=plan_steps_state,
                session_history=list(getattr(session, "sliding_window", []) or []),
            ):
                response_context_block = ""
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="response_context_elided",
                    data={
                        "reason": "lean_no_tool_turn",
                        "workflow_shape": workflow_contract.workflow_shape,
                        "original_chars": len(response_packet.content or ""),
                    },
                )
            final_messages = self._build_final_messages(
                session=session,
                system_prompt=session_system_prompt,
                user_message=request.user_message,
                effective_objective=effective_objective,
                tool_results=tool_results,
                context_block=response_context_block,
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
            buffer_final_response = is_small_or_local_model(session_model)
            buffered_tokens: list[str] = []
            async for token in _stream_with_rate_limit_retry(
                current_adapter,
                model=clean_model,
                messages=final_messages,
                temperature=0.2,
                images=request.images,
                **stream_kwargs,
            ):
                if isinstance(token, dict):
                    yield token
                    continue
                final_response += token
                if buffer_final_response:
                    buffered_tokens.append(token)
                else:
                    yield {"type": "token", "content": token}

            guarded_response = guard_final_response(
                final_response,
                user_message=request.user_message,
                model=session_model,
            )
            if guarded_response.changed:
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="final_response_guard",
                    data={
                        "issues": guarded_response.issues,
                        "replacement_used": guarded_response.replacement_used,
                        "buffered": buffer_final_response,
                    },
                )
                if not buffer_final_response:
                    yield {"type": "retract_last_tokens"}
                final_response = guarded_response.text
                yield {"type": "token", "content": final_response}
            elif buffer_final_response:
                for token in buffered_tokens:
                    yield {"type": "token", "content": token}

            ledger_response_check = ledger.response_support_check(final_response)
            if not ledger_response_check.get("supported", True):
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="completion_gate",
                    data={
                        "final_answer_allowed": False,
                        "issues": ledger_response_check.get("issues", []),
                        "successful_tool_result_ids": ledger_response_check.get("successful_tool_result_ids", []),
                        "ledger_mode": ledger.ledger_mode,
                    },
                )
                if ledger.ledger_mode == "ledger_enforced":
                    if not buffer_final_response:
                        yield {"type": "retract_last_tokens"}
                    final_response = self._ledger_incomplete_response(ledger)
                    yield {"type": "token", "content": final_response}

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
            # direct-fact-memory-only answers come from deterministic facts, and
            # no-tool runs have nothing to ground-check. When no tool ran AND the
            # deterministic guards already passed (ledger response support +
            # side-effect), the LLM verifier has nothing to add — skip it to save
            # one model call per run (the deterministic floor still holds). Runs
            # WITH tool results keep the LLM verifier so claims stay grounded.
            _plan_route = str(structured_plan.get("route_type", ""))
            _deterministic_response_clean = bool(
                ledger_response_check.get("supported", True)
                and side_effect_check.get("supported", True)
            )
            _skip_verify = (
                direct_fact_memory_only
                or (not tool_results and _plan_route in {"trivial_chat", "memory_recall"})
                or (not tool_results and _deterministic_response_clean)
            )
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

            if not ledger_response_check.get("supported", True):
                merged_issues = list(verification.get("issues") or [])
                merged_issues.extend(ledger_response_check.get("issues", []))
                verification = {
                    **verification,
                    "supported": False,
                    "verdict": "needs_replan",
                    "issues": list(dict.fromkeys(str(item) for item in merged_issues if str(item).strip())),
                    "missing_evidence": list(ledger_response_check.get("issues") or []),
                    "target_state": "gather",
                    "ledger_completion_gate": "blocked",
                }

            # Slice 4: 3-way verifier verdict routing.
            # The verifier now returns one of: supported | needs_redraft | needs_replan.
            # - supported   → COMMIT (handled by the post-redraft fall-through path)
            # - needs_redraft → REDRAFT (legacy path, evidence is sufficient)
            # - needs_replan → emit replan_recommended event + inject honest
            #   "missing evidence" note into the response. We do NOT re-enter the
            #   tool loop here (that's the caller's next-turn responsibility);
            #   we surface the signal so a managing orchestrator or the user can
            #   trigger the next plan with the missing evidence as the objective.
            _verifier_verdict = str(verification.get("verdict", "")).strip().lower()
            if _verifier_verdict == "needs_replan":
                _missing_evidence = list(verification.get("missing_evidence") or [])
                _target_state = str(verification.get("target_state") or "gather").strip().lower()
                self._trace(
                    request=request,
                    run_id=run.run_id,
                    event_type="replan_recommended",
                    data={
                        "target_state": _target_state,
                        "missing_evidence": _missing_evidence,
                        "issues": list(verification.get("issues") or [])[:5],
                        "confidence": verification.get("confidence"),
                    },
                )
                yield {
                    "type": "replan_recommended",
                    "target_state": _target_state,
                    "missing_evidence": _missing_evidence,
                    "issues": list(verification.get("issues") or [])[:5],
                    "confidence": verification.get("confidence"),
                }
                # Skip the redraft loop — redrafting can't produce missing
                # evidence. Continue to the standard "ship with warning" path
                # below, which will surface the warning to the user.

            # Re-draft once when verification fails with high severity.
            # Triggered by side-effect guard tripping OR multiple verification issues.
            # Bound to a single attempt — if the redraft also fails, we ship with a warning.
            # Only fires for verdict=needs_redraft (legacy unsupported case);
            # needs_replan is handled above (cross-turn re-plan via continuation_state).
            # The recovery budget comes from the active control policy
            # (max_recovery_rounds): a policy can set it to 0 to forbid in-turn
            # redraft and rely purely on the cross-turn re-plan path.
            _recovery_budget = int(getattr(control_policy, "max_recovery_rounds", 1) or 0)
            _redraft_eligible = (
                not bool(verification.get("supported", True))
                and _verifier_verdict != "needs_replan"
                and _recovery_budget >= 1
            )
            if _redraft_eligible:
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
                    redraft_buffered_tokens: list[str] = []
                    async for token in _stream_with_rate_limit_retry(
                        current_adapter,
                        model=clean_model,
                        messages=redraft_messages,
                        temperature=0.1,
                        images=request.images,
                        **stream_kwargs,
                    ):
                        if isinstance(token, dict):
                            yield token
                            continue
                        final_response += token
                        if buffer_final_response:
                            redraft_buffered_tokens.append(token)
                        else:
                            yield {"type": "token", "content": token}

                    guarded_redraft = guard_final_response(
                        final_response,
                        user_message=request.user_message,
                        model=session_model,
                    )
                    if guarded_redraft.changed:
                        self._trace(
                            request=request,
                            run_id=run.run_id,
                            event_type="final_response_guard",
                            data={
                                "issues": guarded_redraft.issues,
                                "replacement_used": guarded_redraft.replacement_used,
                                "buffered": buffer_final_response,
                                "redraft": True,
                            },
                        )
                        if not buffer_final_response:
                            yield {"type": "retract_last_tokens"}
                        final_response = guarded_redraft.text
                        yield {"type": "token", "content": final_response}
                    elif buffer_final_response:
                        for token in redraft_buffered_tokens:
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
            ledger.set_verification(
                supported=verification_supported,
                verdict=str(verification.get("verdict") or ("supported" if verification_supported else "needs_redraft")),
                issues=[str(item) for item in (verification.get("issues") or [])],
                confidence=verification.get("confidence"),
            )
            self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="verification")
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

            pending_or_failed_steps = [
                step for step in plan_steps_state
                if isinstance(step, dict)
                and str(step.get("status") or "pending") not in {"done", "completed", "skipped"}
            ]
            verification_missing_evidence = [
                str(item).strip()
                for item in (verification.get("missing_evidence") or [])
                if str(item).strip()
            ]
            continuation_missing_slots = list(dict.fromkeys(
                [str(item).strip() for item in latest_missing_slots if str(item).strip()]
                + verification_missing_evidence
            ))
            continuation_reason = ""
            if str(verification.get("verdict", "")).strip().lower() == "needs_replan":
                continuation_reason = "verification_needs_replan"
            elif pending_or_failed_steps:
                continuation_reason = "pending_plan_steps"
            elif continuation_missing_slots:
                continuation_reason = "missing_evidence_slots"
            elif not verification_supported:
                continuation_reason = "verification_unsupported"

            if continuation_reason:
                continuation_state = _build_continuation_state_payload(
                    reason=continuation_reason,
                    effective_objective=effective_objective,
                    run_id=run.run_id,
                    plan_steps=plan_steps_state,
                    missing_slots=continuation_missing_slots,
                    strategy=latest_strategy,
                    notes=latest_notes,
                    tool_results=tool_results,
                    issues=list(verification.get("issues") or []),
                    target_state=str(verification.get("target_state") or ""),
                )
                if continuation_state and hasattr(self.sessions, "update_continuation_state"):
                    ledger.set_continuation(continuation_state)
                    self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="continuation_state")
                    self.sessions.update_continuation_state(
                        session.id,
                        continuation_state,
                        owner_id=request.owner_id,
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="continuation_state_updated",
                        data={
                            "reason": continuation_reason,
                            "pending_step_count": len(continuation_state.get("pending_steps") or []),
                            "missing_slot_count": len(continuation_state.get("missing_slots") or []),
                            "target_state": continuation_state.get("target_state") or "",
                        },
                    )
            elif hasattr(self.sessions, "clear_continuation_state"):
                prior_continuation = bool(getattr(session, "continuation_state", {}) or {})
                self.sessions.clear_continuation_state(session.id, owner_id=request.owner_id)
                ledger.continuation_state = None
                if prior_continuation:
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="continuation_state_cleared",
                        data={"reason": "run_closed"},
                    )

            self.sessions.append_message(session.id, "assistant", final_response)
            if verification_supported:
                memory_commit_mode = str(
                    request.memory_commit_mode
                    or os.getenv("SHOVSOS_MEMORY_COMMIT_MODE", "sync")
                    or "sync"
                ).strip().lower()
                if memory_commit_mode not in {"sync", "async", "skip"}:
                    memory_commit_mode = "sync"
                commit_kwargs = {
                    "context_engine": active_context_engine,
                    "session_id": session.id,
                    "request": request,
                    "assistant_response": final_response,
                    "current_context": session.compressed_context,
                    "is_first_exchange": is_first_exchange,
                    "tool_results": tool_results,
                    "model": request.context_model or session_model,
                    "run_id": run.run_id,
                    "deterministic_keyed_facts": deterministic_keyed_facts,
                    "deterministic_voids": deterministic_voids,
                    "current_facts": current_facts,
                    "stance_signals": stance_signals,
                    "conversation_tension": conversation_tension,
                    "planned_locus_id": planned_locus_id,
                    "run_ledger": ledger,
                }
                if memory_commit_mode == "skip":
                    self.run_store.save_checkpoint(
                        run_id=run.run_id,
                        phase="memory_commit",
                        status="skipped",
                        tool_turn=tool_turn,
                        notes="Memory commit skipped by request.memory_commit_mode.",
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="memory_commit_skipped",
                        data={"reason": "request_memory_commit_mode_skip"},
                    )
                elif memory_commit_mode == "async":
                    self.run_store.save_checkpoint(
                        run_id=run.run_id,
                        phase="memory_commit",
                        status="deferred",
                        tool_turn=tool_turn,
                        notes="Memory commit deferred so response streaming can finish without waiting on compression or embeddings.",
                    )
                    self._trace(
                        request=request,
                        run_id=run.run_id,
                        event_type="memory_commit_deferred",
                        data={"reason": "async_memory_commit", "mode": memory_commit_mode},
                    )
                    yield {"type": "memory_commit_deferred", "run_id": run.run_id}
                    self._schedule_deferred_memory_commit(**commit_kwargs)
                else:
                    await self._commit_context(**commit_kwargs)
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
            ledger.append_event(
                "response",
                source="run_engine",
                status="complete",
                data={"content_length": len(final_response), "verification_supported": verification_supported},
            )
            self._trace_ledger(request=request, run_id=run.run_id, ledger=ledger, reason="run_complete")
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
            active_ledger = self._active_ledgers.get(run.run_id)
            if active_ledger is not None:
                active_ledger.append_event(
                    "run_failed",
                    source="run_engine",
                    status="error",
                    data={"message": str(exc)},
                )
                self._trace_ledger(request=request, run_id=run.run_id, ledger=active_ledger, reason="run_failed")
            self.run_store.finish_run(run.run_id, status="failed", error=str(exc))
            self._trace(
                request=request,
                run_id=run.run_id,
                event_type="error",
                data={"message": str(exc)},
            )
            raise
        finally:
            self._active_ledgers.pop(run.run_id, None)

    def _resolve_adapter(self, model: str) -> BaseLLMAdapter:
        if model and ":" in model:
            return create_adapter(provider=model.split(":", 1)[0].lower())
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
        run_id: str = "",
        run_ledger: Optional[RunLedger] = None,
    ) -> Optional[ToolCall]:
        available_names = [name for name in allowed_tools if self.tool_registry.get(name) is not None]
        if not available_names:
            return None
        effective_objective = ""
        if run_ledger is not None:
            effective_objective = str(getattr(run_ledger, "objective", "") or "").strip()
        if not effective_objective:
            effective_objective = self._resolve_effective_objective(
                request.user_message,
                list(getattr(session, "sliding_window", []) or []),
            )
        if run_ledger is not None:
            kernel_call = self._tool_call_from_language_kernel(
                request=request,
                run_id=run_id,
                ledger=run_ledger,
                available_names=available_names,
            )
            if kernel_call is not None:
                return kernel_call
        deterministic_call = self._deterministic_single_tool_call(
            request=request,
            run_id=run_id,
            available_names=available_names,
            effective_objective=effective_objective,
            context_block=context_block,
            argument_clues=argument_clues,
        )
        if deterministic_call is not None:
            return deterministic_call
        language_kernel_context = ""
        if run_ledger is not None:
            from run_engine.language_kernel import build_kernel_snapshot

            language_kernel_context = build_kernel_snapshot(
                run_ledger,
                ContextPhase.ACTING,
                query=effective_objective,
            ).render_for_actor()

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
                    language_kernel_context=language_kernel_context,
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
        draft = ToolCallDraft(
            tool_name=call.tool_name if call is not None else "",
            arguments=dict(call.arguments or {}) if call is not None and isinstance(call.arguments, dict) else {},
            raw=raw,
            source="actor",
            validation_error="" if call is not None else ("empty_actor_output" if not raw else "unparseable_tool_call"),
        )
        if call is not None:
            # Strict allowed-tool enforcement — actor cannot escape the
            # planner's tool selection by hallucinating a tool name that's
            # registered but not in the planner's narrowed list.
            if call.tool_name not in available_names:
                draft.validation_error = "tool_not_allowed"
                self._trace(
                    request=request,
                    run_id=run_id,
                    event_type="tool_selection_violation",
                    data={
                        "attempted_tool": call.tool_name,
                        "allowed_tools": list(available_names),
                    },
                )
                call = None
            else:
                self._trace(
                    request=request,
                    run_id=run_id,
                    event_type="tool_call_draft",
                    data={
                        "tool_name": draft.tool_name,
                        "arguments": draft.arguments,
                        "valid": draft.valid,
                        "validation_error": draft.validation_error,
                        "raw_preview": clip_text(raw, 500),
                    },
                )
                return call
        self._trace(
            request=request,
            run_id=run_id,
            event_type="tool_call_draft",
            data={
                "tool_name": draft.tool_name,
                "arguments": draft.arguments,
                "valid": False,
                "validation_error": draft.validation_error,
                "raw_preview": clip_text(raw, 500),
            },
        )
        if argument_clues:
            for name in available_names:
                clue = str(argument_clues.get(name) or "").strip()
                if not clue:
                    continue
                if name in {"web_search", "rag_search"}:
                    return ToolCall(
                        tool_name=name,
                        arguments={"query": clue},
                        raw_json=json.dumps({"tool": name, "arguments": {"query": clue}}),
                    )
                if name == "web_fetch":
                    url_match = _URL_RE.search(clue)
                    if url_match:
                        url = _canonical_fetch_url_for_loop(url_match.group(0))
                        if url:
                            return ToolCall(
                                tool_name=name,
                                arguments={"url": url},
                                raw_json=json.dumps({"tool": name, "arguments": {"url": url}}),
                            )
        fallback_ranked = self._rank_fallback_tool_candidates(effective_objective, available_names)
        if "web_fetch" in available_names and "web_fetch" not in fallback_ranked:
            if re.search(r"https?://[^\s\"'<>),]+", context_block or ""):
                fallback_ranked.insert(0, "web_fetch")
        for name in fallback_ranked:
            if name not in available_names:
                continue
            fallback_objective = effective_objective
            if name == "web_fetch" and not re.search(r"https?://\S+", fallback_objective or ""):
                context_url = re.search(r"https?://[^\s\"'<>),]+", context_block or "")
                if context_url:
                    fallback_objective = _canonical_fetch_url_for_loop(context_url.group(0)) or fallback_objective
            fallback = fallback_tool_call(name, fallback_objective)
            if fallback is not None:
                return fallback
        return None

    def _deterministic_single_tool_call(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        available_names: list[str],
        effective_objective: str,
        context_block: str,
        argument_clues: Optional[dict[str, str]] = None,
    ) -> Optional[ToolCall]:
        """Skip the actor LLM when a single allowed tool has deterministic args."""
        if len(available_names) != 1:
            return None
        name = available_names[0]
        call: Optional[ToolCall] = None
        clue = str((argument_clues or {}).get(name) or "").strip()
        if clue and name in {"web_search", "rag_search"}:
            call = fallback_tool_call(name, clue)
        elif clue and name == "web_fetch":
            url_match = _URL_RE.search(clue)
            if url_match:
                url = _canonical_fetch_url_for_loop(url_match.group(0))
                if url:
                    call = ToolCall(
                        tool_name=name,
                        arguments={"url": url},
                        raw_json=json.dumps({"tool": name, "arguments": {"url": url}, "source": "deterministic_single_tool"}),
                    )
        if call is None and name == "web_fetch" and not re.search(r"https?://\S+", effective_objective or ""):
            context_url = re.search(r"https?://[^\s\"'<>),]+", context_block or "")
            if context_url:
                url = _canonical_fetch_url_for_loop(context_url.group(0))
                if url:
                    call = ToolCall(
                        tool_name=name,
                        arguments={"url": url},
                        raw_json=json.dumps({"tool": name, "arguments": {"url": url}, "source": "deterministic_single_tool"}),
                    )
        if call is None:
            call = fallback_tool_call(name, effective_objective)
        if call is None:
            return None
        validation_error = self.tool_registry.validate_tool_call(call)
        if validation_error:
            self._trace(
                request=request,
                run_id=run_id,
                event_type="tool_call_deterministic_rejected",
                data={
                    "tool_name": name,
                    "validation_error": validation_error,
                    "arguments": dict(call.arguments or {}),
                },
            )
            return None
        self._trace(
            request=request,
            run_id=run_id,
            event_type="tool_call_deterministic",
            data={
                "tool_name": call.tool_name,
                "arguments": dict(call.arguments or {}),
                "source": "single_allowed_tool",
            },
        )
        return call

    def _tool_call_from_language_kernel(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        ledger: RunLedger,
        available_names: list[str],
    ) -> Optional[ToolCall]:
        next_action = ledger.next_required_action()
        tool_name = str(next_action.get("tool") or "").strip()
        arguments = next_action.get("arguments") if isinstance(next_action.get("arguments"), dict) else {}
        if not tool_name or tool_name not in available_names or not arguments:
            return None
        validation = ledger.validate_tool_call_against_policy(tool_name, dict(arguments))
        if not validation.valid:
            return None
        payload = {
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "reason": str(next_action.get("reason") or "language_kernel_next_action"),
            "missing_slots": list(next_action.get("missing_slots") or []),
            "policy": str(getattr(ledger.control_policy, "id", "") or ""),
        }
        self._trace(
            request=request,
            run_id=run_id,
            event_type="language_kernel_next_action",
            data=payload,
        )
        ledger.append_event(
            "language_kernel_next_action",
            source="language_kernel",
            status="selected",
            data=payload,
        )
        return ToolCall(
            tool_name=tool_name,
            arguments=dict(arguments),
            raw_json=json.dumps({"tool": tool_name, "arguments": dict(arguments), "source": "language_kernel"}),
        )

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
        language_kernel_context: str = "",
    ) -> str:
        capability_context = render_capability_cards(
            allowed_tools=allowed_tools,
            workflow_template=request.workflow_template,
        )
        return build_actor_request_content(
            user_message=request.user_message,
            effective_objective=effective_objective,
            session_first_message=str(session.first_message or ""),
            allowed_tools=allowed_tools,
            tool_results=tool_results,
            context_block=context_block,
            clip_text=self._clip,
            argument_clues=argument_clues,
            capability_context=capability_context,
            language_kernel_context=language_kernel_context,
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
        if re.search(
            r"\b(generate|create|make|design|render|draw|produce)\b.*\b(image|photo|picture|logo|mockup|illustration|asset)\b",
            lowered,
        ) or re.search(
            r"\b(image|photo|picture|logo|mockup|illustration|asset)\b.*\b(generate|create|make|design|render|draw|produce)\b",
            lowered,
        ):
            add("image_generate")
        if re.search(r"\b(image|photo|picture|screenshot|logo)\b", lowered):
            add("image_search")
        if re.search(
            r"\b(top\s+)?(?:gainers?|losers?|movers?|most\s+active|market\s+movers?)\b",
            lowered,
        ) and re.search(r"\b(stock|stocks|ticker|tickers|market|markets|today|latest|current)\b", lowered):
            add("alpha_vantage_movers")
        if re.search(
            r"\b(finance|financial|snapshot|stock|stocks|ticker|tickers|quote|price|fundamental|fundamentals|"
            r"earnings|revenue|market\s+cap|valuation|sentiment|analyst|analysis|report)\b",
            lowered,
        ):
            add("finance_snapshot")
            add("alpha_vantage_quote")
            add("alpha_vantage_overview")
            add("alpha_vantage_news")
        if re.search(r"\b(remember|memory|previously|earlier|recall)\b", lowered):
            add("query_memory")
            add("rag_search")
        # Conversation-history / meta requests ("read recent chat", "what did we
        # talk about", "summarize our conversation") must hit memory, never get
        # web-searched literally. Rank query_memory before the web_search rule so
        # it wins the single fallback slot.
        if re.search(
            r"\b(recent|previous|prior|last|earlier|past|our|the)\b[\w\s]{0,20}\b(chat|conversation|conversations|message|messages|discussion|thread|history)\b",
            lowered,
        ) or re.search(
            r"\bwhat did (?:we|i|you)\b|\bwhat (?:were|was) (?:we|i|you)\b[\w\s]{0,16}\b(?:talking|chat(?:t)?ing|discussing)\b|"
            r"\b(?:you|u)\s+(?:forgot|lost|missed)\b[\w\s]{0,24}\b(?:chat|conversation|context|thread|talking|chat(?:t)?ing|discussing)\b|"
            r"\b(?:we|you|i)\s+were\b[\w\s]{0,12}\b(?:talking|chat(?:t)?ing|discussing)\b|"
            r"\b(chat|conversation) history\b|\bread (?:the )?(?:recent |last )?(?:chat|conversation)\b",
            lowered,
        ):
            add("query_memory")
        if re.search(r"\b(near me|nearby|place|restaurant|hotel|map|route|directions)\b", lowered):
            add("places_search")
        if _source_collection_contract_from_objective(user_message):
            add("source_collect")
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

        answer_patch = self._extract_answer_patch(tool_results)
        if answer_patch:
            messages.append({
                "role": "system",
                "content": (
                    "Answer patch:\n"
                    f"{json.dumps(answer_patch, ensure_ascii=False, indent=2)}\n\n"
                    "Use the answer_patch as the source of structured facts. Write a concise buyer answer from it. "
                    "Do not cite links outside verified_urls. Do not invent prices, availability, purchases, or product facts."
                ),
            })

        final_user_prompt = user_message
        if effective_objective.strip() and effective_objective.strip() != user_message.strip():
            final_user_prompt = (
                f"Current user turn:\n{user_message}\n\n"
                f"Resolved working objective:\n{effective_objective}\n\n"
                "Respond to the current user turn by advancing the resolved working objective."
            )
        messages.append({"role": "user", "content": final_user_prompt})
        return messages

    @staticmethod
    def _should_use_lean_response_context(
        *,
        user_message: str,
        effective_objective: str,
        workflow_shape: str,
        tool_results: list[dict[str, Any]],
        selected_tools: list[str],
        current_facts: Optional[list[tuple[str, str, str]]],
        continuation_state: dict[str, Any],
        plan_steps: Optional[list[dict[str, Any]]],
        session_history: list[dict[str, Any]],
    ) -> bool:
        if tool_results or selected_tools or current_facts or continuation_state or plan_steps:
            return False
        if len(session_history or []) > 2:
            return False
        text = f"{user_message}\n{effective_objective}".lower()
        if len(text) > 360:
            return False
        if workflow_shape in {"source_collection", "research_report", "coding_change", "memory_correction"}:
            return False
        if re.search(
            r"\b(search|fetch|research|analy[sz]e|compare|latest|today|current|news|price|stock|stocks|"
            r"quote|finance|financial|fundamental|weather|near me|remember|memory|file|code|implement|fix)\b",
            text,
        ):
            return False
        return True

    @staticmethod
    def _extract_answer_patch(tool_results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        for item in reversed(tool_results or []):
            if item.get("tool_name") not in {"shopping_advice", "finance_snapshot"}:
                continue
            try:
                content = json.loads(str(item.get("content") or "{}"))
            except Exception:
                continue
            if isinstance(content, dict) and isinstance(content.get("answer_patch"), dict):
                return {
                    "verified_urls": content.get("verified_urls") or [],
                    "source_urls": content.get("answer_patch", {}).get("source_urls") or content.get("source_urls") or [],
                    "warnings": content.get("warnings") or [],
                    "provider": content.get("provider"),
                    "symbol": content.get("symbol"),
                    **content["answer_patch"],
                }
        return None

    @staticmethod
    def _normalize_shopping_arguments(arguments: dict[str, Any], *, objective: str) -> dict[str, Any]:
        args = dict(arguments or {})
        text = str(objective or "")
        lowered = text.lower()
        if not str(args.get("query") or "").strip():
            cleaned = re.sub(
                r"\b(find|buy|get|recommend|best|cheap|near me|near|under|for me|please|check|compare)\b",
                " ",
                lowered,
                flags=re.IGNORECASE,
            )
            args["query"] = re.sub(r"\s+", " ", cleaned).strip() or text[:120]
        if not str(args.get("budget") or "").strip():
            budget_match = re.search(r"(?:under|below|less than|up to|max(?:imum)?)\s+([$€£]?\s?\d[\d,]*(?:\.\d{2})?\s?(?:cad|usd|eur|gbp)?)", text, re.IGNORECASE)
            if budget_match:
                args["budget"] = budget_match.group(1).strip()
        if not str(args.get("location") or "").strip():
            location_match = re.search(r"\b(?:near|in|around)\s+([A-Z][A-Za-z .'-]{2,}(?:,\s?[A-Z]{2})?)", text)
            if location_match:
                location = re.split(
                    r"\s+\b(?:at|from|for|under|with|and|or)\b\s+",
                    location_match.group(1).strip(" ."),
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip(" .")
                args["location"] = location
            elif re.search(r"\btoronto\b", lowered):
                args["location"] = "Toronto"
        if not args.get("stores"):
            store_hits: list[str] = []
            store_aliases = {
                "costco": "Costco",
                "canadian tire": "Canadian Tire",
                "shoppers": "Shoppers",
                "shoppers drug mart": "Shoppers",
                "metro": "Metro",
                "dollarama": "Dollarama",
                "walmart": "Walmart",
                "best buy": "Best Buy",
                "bestbuy": "Best Buy",
            }
            for phrase, label in store_aliases.items():
                if phrase in lowered and label not in store_hits:
                    store_hits.append(label)
            if store_hits:
                args["stores"] = store_hits
        if not str(args.get("region") or "").strip() and re.search(r"\b(canada|toronto|ontario|cad)\b", lowered):
            args["region"] = "Canada"
        return args

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
        conversation_tension=None,
        planned_locus_id: str = "",
        run_ledger: Optional[RunLedger] = None,
    ) -> None:
        if should_skip_memory_compression(request.user_message, assistant_response):
            self.run_store.save_checkpoint(
                run_id=run_id,
                phase="memory_commit",
                status="skipped_low_value",
                notes="Low-value social turn skipped before context packet compilation.",
            )
            self._trace(
                request=request,
                run_id=run_id,
                event_type="memory_commit_skipped",
                data={"reason": "low_value_social_turn"},
            )
            return
        session = self.sessions.get(session_id, owner_id=request.owner_id)
        existing_candidate_signals = list(getattr(session, "candidate_signals", []) or [])
        existing_candidate_context = getattr(session, "candidate_context", "") if session is not None else ""
        memory_packet = None
        if session is not None:
            memory_packet = self._compile_phase_packet(
                context_engine=context_engine,
                run_id=run_id,
                request=request,
                session=session,
                phase=ContextPhase.MEMORY_COMMIT,
                system_prompt=request.system_prompt,
                effective_objective=request.user_message,
                current_context=current_context,
                allowed_tools=[],
                tool_results=tool_results,
                tool_turn=0,
                strategy="Commit only verified durable facts, voids, and candidate signals.",
                notes="Storage must preserve contradiction state and avoid committing unsupported tool or meta claims.",
                final_response=assistant_response,
                current_facts=current_facts,
                conversation_tension=conversation_tension,
                planned_locus_id=planned_locus_id,
            )

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
            self._trace_memory_commit_plan(
                request=request,
                run_id=run_id,
                plan=commit_plan,
                conversation_tension=conversation_tension,
                memory_packet=memory_packet,
                path="deterministic_governor",
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
            if run_ledger is not None:
                self._record_memory_outcome(
                    request=request,
                    run_id=run_id,
                    ledger=run_ledger,
                    outcome=outcome,
                    path="deterministic_governor",
                )
            return
        current_turn = self._message_count(session_id, request.owner_id)
        compression_mode = str(os.getenv("SHOVSOS_LLM_COMPRESSION_MODE", "adaptive") or "adaptive")
        try:
            compression_interval = int(os.getenv("SHOVSOS_LLM_COMPRESSION_INTERVAL", "6") or "6")
        except ValueError:
            compression_interval = 6
        should_compress_with_llm = should_run_llm_memory_compression(
            request.user_message,
            assistant_response,
            is_first_exchange=is_first_exchange,
            deterministic_fact_count=len(deterministic_keyed_facts or []),
            void_count=len(deterministic_voids or []),
            candidate_signal_count=len(stance_signals or []),
            turn=current_turn,
            interval=max(0, compression_interval),
            mode=compression_mode,
        )
        if not should_compress_with_llm:
            commit_plan = self._context_governor.build_memory_commit_plan(
                user_message=request.user_message,
                tool_results=tool_results,
                deterministic_keyed_facts=deterministic_keyed_facts,
                deterministic_voids=deterministic_voids,
                current_facts=current_facts,
                existing_candidate_signals=existing_candidate_signals,
                existing_candidate_context=existing_candidate_context,
                new_candidate_signals=stance_signals,
                current_turn=current_turn,
            )
            self._trace_memory_commit_plan(
                request=request,
                run_id=run_id,
                plan=commit_plan,
                conversation_tension=conversation_tension,
                memory_packet=memory_packet,
                path="adaptive_deterministic_governor",
            )
            self._trace(
                request=request,
                run_id=run_id,
                event_type="llm_compression_skipped",
                data={
                    "reason": "adaptive_gate",
                    "mode": compression_mode,
                    "turn": current_turn,
                    "interval": compression_interval,
                    "deterministic_fact_count": len(deterministic_keyed_facts or []),
                    "void_count": len(deterministic_voids or []),
                    "candidate_signal_count": len(stance_signals or []),
                },
            )
            outcome = await self._context_governor.apply_memory_commit(
                sessions=self.sessions,
                session_id=session_id,
                owner_id=request.owner_id,
                agent_id=request.agent_id,
                turn=current_turn,
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
            if run_ledger is not None:
                self._record_memory_outcome(
                    request=request,
                    run_id=run_id,
                    ledger=run_ledger,
                    outcome=outcome,
                    path="adaptive_deterministic_governor",
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
            self._trace_memory_commit_plan(
                request=request,
                run_id=run_id,
                plan=commit_plan,
                conversation_tension=conversation_tension,
                memory_packet=memory_packet,
                path="compression_empty_governor",
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
            if run_ledger is not None:
                self._record_memory_outcome(
                    request=request,
                    run_id=run_id,
                    ledger=run_ledger,
                    outcome=outcome,
                    path="compression_empty_governor",
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
        self._trace_memory_commit_plan(
            request=request,
            run_id=run_id,
            plan=commit_plan,
            conversation_tension=conversation_tension,
            memory_packet=memory_packet,
            path="compression_plus_governor",
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
        if run_ledger is not None:
            self._record_memory_outcome(
                request=request,
                run_id=run_id,
                ledger=run_ledger,
                outcome=outcome,
                path="compression_plus_governor",
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

    def _schedule_deferred_memory_commit(self, **kwargs) -> None:
        async def _runner() -> None:
            request = kwargs.get("request")
            run_id = str(kwargs.get("run_id") or "")
            try:
                await self._commit_context(**kwargs)
                if request is not None and run_id:
                    self._trace(
                        request=request,
                        run_id=run_id,
                        event_type="memory_commit_completed",
                        data={"mode": "async"},
                    )
            except Exception as exc:
                if run_id:
                    try:
                        self.run_store.save_checkpoint(
                            run_id=run_id,
                            phase="memory_commit",
                            status="failed",
                            notes=f"Deferred memory commit failed: {exc}",
                        )
                    except Exception:
                        pass
                if request is not None and run_id:
                    self._trace(
                        request=request,
                        run_id=run_id,
                        event_type="memory_commit_failed",
                        data={"mode": "async", "message": str(exc)},
                    )

        task = asyncio.create_task(_runner())
        self._background_memory_tasks.add(task)
        task.add_done_callback(self._background_memory_tasks.discard)

    def _trace_memory_commit_plan(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        plan,
        conversation_tension=None,
        memory_packet: Optional[CompiledPassPacket] = None,
        path: str = "",
    ) -> None:
        merged_facts = list(getattr(plan, "merged_facts", None) or [])
        merged_voids = list(getattr(plan, "merged_voids", None) or [])
        blocked_keyed_facts = list(getattr(plan, "blocked_keyed_facts", None) or [])
        candidate_signals = list(getattr(plan, "candidate_signals", None) or [])
        memory_decisions = list(getattr(plan, "memory_decisions", None) or [])
        data = {
            "path": path,
            "fact_count": len(merged_facts),
            "void_count": len(merged_voids),
            "blocked_count": len(blocked_keyed_facts),
            "candidate_signal_count": len(candidate_signals),
            "memory_decision_count": len(memory_decisions),
            "candidate_context_lines": len(
                [line for line in str(getattr(plan, "candidate_context", "") or "").splitlines() if line.strip()]
            ),
            "memory_decisions": memory_decisions[:12],
            "facts": [
                {
                    "subject": item.get("subject"),
                    "predicate": item.get("predicate"),
                    "object": item.get("object"),
                    "key": item.get("key") or item.get("fact"),
                }
                for item in merged_facts[:10]
            ],
            "voids": [
                {
                    "subject": item.get("subject"),
                    "predicate": item.get("predicate"),
                    "reason": item.get("reason") or item.get("source") or "",
                }
                for item in merged_voids[:10]
            ],
            "blocked": [
                {
                    "subject": item.get("subject"),
                    "predicate": item.get("predicate"),
                    "object": item.get("object"),
                    "reason": item.get("grounding_reason"),
                }
                for item in blocked_keyed_facts[:10]
            ],
            "conversation_tension": conversation_tension_audit_payload(conversation_tension),
        }
        if memory_packet is not None:
            data["phase_packet"] = {
                "phase": memory_packet.phase.value,
                "included_ids": [
                    item.get("item_id")
                    for item in memory_packet.trace.get("included", [])
                    if item.get("item_id")
                ],
                "excluded_ids": [
                    item.get("item_id")
                    for item in memory_packet.trace.get("excluded", [])
                    if item.get("item_id")
                ],
            }
        self._trace(request=request, run_id=run_id, event_type="memory_commit_plan", data=data)

    def _trace_workflow_override(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        tool_turn: int,
        workflow_override: dict[str, Any],
        source: str,
    ) -> None:
        """Emit a frontend-friendly diagnostic for deterministic workflow control.

        This is deliberately non-authoritative: it mirrors the runtime override
        so Trace Monitor can later show why the loop pivoted without requiring
        UI code to parse notes or raw model observations.
        """
        if not workflow_override:
            return
        status = str(workflow_override.get("status") or "")
        strategy = str(workflow_override.get("strategy") or "")
        selected_tools = list(workflow_override.get("selected_tools") or [])
        missing_slots = list(workflow_override.get("missing_slots") or [])
        entities = list(workflow_override.get("tickers") or workflow_override.get("entities") or [])
        log(
            "run_engine",
            request.session_id,
            f"Source contract {status or 'observed'}: {strategy or 'workflow override'}",
            level=("warn" if status in {"missing", "blocked"} else "info"),
            owner_id=request.owner_id,
            run_id=run_id,
            turn=tool_turn,
            selected_tools=selected_tools,
            missing_slots=missing_slots,
            entities=entities,
        )
        self._trace(
            request=request,
            run_id=run_id,
            event_type="source_contract",
            data={
                "source": source,
                "turn": tool_turn,
                "status": status,
                "strategy": strategy,
                "notes": str(workflow_override.get("notes") or ""),
                "selected_tools": selected_tools,
                "missing_slots": missing_slots,
                "argument_clues": dict(workflow_override.get("argument_clues") or {}),
                "entities": entities,
                "source_contract": dict(workflow_override.get("source_contract") or {}),
            },
        )

    def _sync_workflow_override_to_ledger(
        self,
        *,
        ledger: RunLedger,
        workflow_override: dict[str, Any],
        source: str,
    ) -> None:
        if not workflow_override:
            return
        entities = [
            str(item).strip()
            for item in (workflow_override.get("tickers") or workflow_override.get("entities") or [])
            if str(item).strip()
        ]
        if entities:
            ledger.lock_entities(entities, source=source)
        source_contract = dict(workflow_override.get("source_contract") or {})
        selected_tools = list(workflow_override.get("selected_tools") or [])
        argument_clues = dict(workflow_override.get("argument_clues") or {})
        next_tool = str(selected_tools[0] if selected_tools else "").strip()
        next_arguments: dict[str, Any] = {}
        next_clue = str(argument_clues.get(next_tool) or "").strip() if next_tool else ""
        if next_tool == "web_search" and next_clue:
            next_arguments = {"query": next_clue}
        elif next_tool == "web_fetch" and next_clue:
            url_match = _URL_RE.search(next_clue)
            if url_match:
                url = _canonical_fetch_url_for_loop(url_match.group(0))
                if url:
                    next_arguments = {"url": url}
        allowed_fetch_urls = []
        if next_tool == "web_fetch" and next_arguments.get("url"):
            allowed_fetch_urls.append(next_arguments["url"])
        allowed_by_entity = workflow_override.get("allowed_fetch_urls_by_entity")
        if isinstance(allowed_by_entity, dict):
            source_contract["allowed_fetch_urls_by_entity"] = {
                str(entity): [
                    canonical
                    for canonical in (
                        _canonical_fetch_url_for_loop(str(url))
                        for url in (urls if isinstance(urls, list) else [])
                    )
                    if canonical
                ]
                for entity, urls in allowed_by_entity.items()
            }
            for urls in allowed_by_entity.values():
                if isinstance(urls, list):
                    allowed_fetch_urls.extend(
                        canonical
                        for canonical in (_canonical_fetch_url_for_loop(str(url)) for url in urls)
                        if canonical
                    )
        if source_contract or entities or next_tool:
            ledger.set_source_contract(
                {
                    **source_contract,
                    "plugin_id": workflow_override.get("plugin_id") or ledger.source_contract.get("plugin_id"),
                    "missing_slots": list(workflow_override.get("missing_slots") or []),
                    "next_tool": next_tool,
                    "next_arguments": next_arguments,
                    "next_reason": str(workflow_override.get("strategy") or ""),
                    "allowed_fetch_urls": list(dict.fromkeys(url for url in allowed_fetch_urls if url)),
                    "forbid_unlocked_entity_drift": True,
                },
                source=source,
            )

    def _gate_tool_call_with_ledger(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        ledger: RunLedger,
        tool_call: ToolCall,
        tool_turn: int,
    ) -> tuple[Optional[ToolCall], Optional[dict[str, Any]]]:
        validation = ledger.validate_tool_call_against_policy(
            tool_call.tool_name,
            tool_call.arguments if isinstance(tool_call.arguments, dict) else {},
        )
        if validation.valid:
            self._trace(
                request=request,
                run_id=run_id,
                event_type="tool_call_validated",
                data={
                    "tool_name": tool_call.tool_name,
                    "turn": tool_turn,
                    "policy": str(getattr(ledger.control_policy, "id", "") or ""),
                },
            )
            return tool_call, None

        ledger.record_policy_violation(
            issue=validation.issue,
            recovery_class=validation.recovery_class,
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments if isinstance(tool_call.arguments, dict) else {},
            expected_tool=validation.expected_tool,
            expected_arguments=validation.expected_arguments,
            message=validation.message,
        )
        violation_payload = {
            "type": "policy_violation",
            "issue": validation.issue,
            "recovery_class": validation.recovery_class,
            "recovery_policy": recovery_policy_for(validation.recovery_class).to_dict(),
            "tool_name": tool_call.tool_name,
            "arguments": dict(tool_call.arguments or {}),
            "expected_tool": validation.expected_tool,
            "expected_arguments": dict(validation.expected_arguments or {}),
            "message": validation.message,
            "turn": tool_turn,
            "ledger_mode": ledger.ledger_mode,
        }
        self._trace(request=request, run_id=run_id, event_type="policy_violation", data=violation_payload)
        if ledger.ledger_mode != "ledger_enforced":
            return tool_call, violation_payload

        if validation.expected_tool and validation.expected_arguments:
            recovered = ToolCall(
                tool_name=validation.expected_tool,
                arguments=dict(validation.expected_arguments),
                raw_json=json.dumps({"tool": validation.expected_tool, "arguments": validation.expected_arguments}),
            )
            recovery_payload = {
                **violation_payload,
                "type": "recovery_started",
                "recovered_tool": recovered.tool_name,
                "recovered_arguments": dict(recovered.arguments or {}),
            }
            ledger.append_event("recovery_started", source="policy_gate", status="active", data=recovery_payload)
            self._trace(request=request, run_id=run_id, event_type="recovery_started", data=recovery_payload)
            return recovered, recovery_payload

        return None, violation_payload

    def _ledger_incomplete_response(self, ledger: RunLedger) -> str:
        gate = ledger.completion_gate()
        next_action = ledger.next_required_action()
        missing = ", ".join(str(item) for item in gate.get("missing_slots") or next_action.get("missing_slots") or [])
        action = ""
        if next_action.get("tool"):
            action = f" Next required action: {next_action.get('tool')} {next_action.get('arguments') or {}}."
        if not missing:
            missing = "required evidence"
        return f"I cannot finalize this as complete yet. Missing: {missing}.{action}"

    def _record_memory_outcome(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        ledger: RunLedger,
        outcome,
        path: str,
    ) -> None:
        fact_count = len(getattr(outcome, "merged_facts", None) or [])
        void_count = len(getattr(outcome, "merged_voids", None) or [])
        blocked_count = len(getattr(outcome, "blocked_keyed_facts", None) or [])
        graph_error = str(getattr(outcome, "graph_error", "") or "")
        index_error = str(getattr(outcome, "index_error", "") or "")
        status = "committed"
        if graph_error or index_error:
            status = "error"
        elif blocked_count:
            status = "partial"
        ledger.add_memory_write(
            status=status,
            summary=f"{path}: facts={fact_count} voids={void_count} blocked={blocked_count}",
            data={
                "path": path,
                "fact_count": fact_count,
                "void_count": void_count,
                "blocked_count": blocked_count,
                "candidate_signal_count": len(getattr(outcome, "candidate_signals", None) or []),
                "indexed_fact_count": len(getattr(outcome, "indexed_fact_keys", None) or []),
                "graph_error": graph_error,
                "index_error": index_error,
            },
        )
        self._trace_ledger(request=request, run_id=run_id, ledger=ledger, reason="memory_commit")

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
        plan_steps: Optional[list[dict[str, Any]]] = None,
        spatial_drawers: Optional[list[dict[str, Any]]] = None,
        run_ledger: Optional[RunLedger] = None,
    ) -> CompiledPassPacket:
        active_ledger = run_ledger or self._active_ledgers.get(run_id)
        if active_ledger is not None:
            active_ledger.set_phase(phase, source="phase_packet")
            control_policy_id = str(getattr(getattr(active_ledger, "control_policy", None), "id", "") or "")
            if control_policy_id == "graph_harness":
                graph_payload = active_ledger.record_graph_phase(
                    phase,
                    marker=f"{phase.value}:{tool_turn}",
                    source="phase_packet",
                )
                if graph_payload:
                    self._trace(
                        request=request,
                        run_id=run_id,
                        event_type="pass_graph_execution",
                        data=graph_payload,
                    )
        capability_context = render_capability_cards(
            allowed_tools=[
                str(tool.get("name") or "")
                for tool in allowed_tools
                if isinstance(tool, dict) and tool.get("name")
            ],
            workflow_template=request.workflow_template,
        )
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
                capability_context=capability_context,
                code_intent_note=code_intent_note,
                execution_risk_tier=execution_risk_tier,
                correction_turn=inferred_correction_turn,
                direct_fact_memory_only=inferred_direct_fact_memory_only,
                available_loci=available_loci,
                planned_locus_id=planned_locus_id,
                plan_steps=plan_steps,
                spatial_drawers=spatial_drawers,
                run_ledger=active_ledger,
            ),
        )
        phase_trace = dict(packet.trace)
        if trace_phase_label:
            phase_trace["phase"] = trace_phase_label
        if active_ledger is not None:
            phase_trace["run_ledger"] = active_ledger.to_phase_packet(phase)
        self._trace(
            request=request,
            run_id=run_id,
            event_type="phase_context",
            data=phase_trace,
        )
        self._trace(
            request=request,
            run_id=run_id,
            event_type="phase_packet",
            data={
                "phase": phase_trace.get("phase") or phase.value,
                "content_chars": len(packet.content or ""),
                "included": phase_trace.get("included", []),
                "excluded": phase_trace.get("excluded", []),
                "summary": phase_trace.get("summary", {}),
                "run_ledger": active_ledger.to_phase_packet(phase) if active_ledger is not None else None,
            },
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

    def _trace_ledger(
        self,
        *,
        request: RunEngineRequest,
        run_id: str,
        ledger: RunLedger,
        reason: str,
    ) -> None:
        self._trace(
            request=request,
            run_id=run_id,
            event_type="run_ledger",
            data={**ledger.to_trace_payload(), "reason": reason},
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
