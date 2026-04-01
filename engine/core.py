"""
AgentCore v3 — intelligence fixes + structured tracing
"""

import asyncio
import hashlib
import inspect
import json
import os
import re
import tiktoken
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from llm.base_adapter import BaseLLMAdapter
from llm.llm_adapter     import OllamaAdapter, LLMError, RateLimitError, ProviderError
from engine.context_engine  import ContextEngine
from engine.context_engine_v2 import ContextEngineV2
from engine.context_engine_v3 import ContextEngineV3
from engine.context_compiler import compile_context_items
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from engine.compression_fact_policy import finalize_compression_fact_records
from engine.direct_fact_policy import should_answer_direct_fact_from_memory
from engine.deterministic_facts import (
    extract_user_stated_fact_updates,
    is_redundant_user_alias_text,
    merge_fact_records,
    merge_void_records,
)
from orchestration.session_manager import SessionManager
from orchestration.run_store import LoopCheckpoint, get_run_store
from plugins.tool_registry   import ToolRegistry
from guardrails.middleware import GuardrailMiddleware
from memory.vector_engine   import VectorEngine
from memory.semantic_graph  import SemanticGraph
from llm.adapter_factory    import create_adapter, strip_provider_prefix
from engine.circuit_breaker import CircuitBreaker
from orchestration.orchestrator import AgenticOrchestrator
from config.config          import cfg
from config.logger          import log
from config.trace_store     import get_trace_store
from memory.task_tracker    import get_session_task_tracker

# Strict token safety limits for providers (especially Groq/Cloud)
# These represent the TOTAL tokens allowed per request (TPM/Budget)
TOKEN_SAFETY_LIMITS: dict[str, int] = {
    "gpt-4o":                  128_000,
    "gpt-4o-mini":             128_000,
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant":    128_000,
    "llama3.2":                32_000,
    "deepseek-r1":             64_000,
    "moonshotai/kimi-k2-instruct-0905": 9_000,  # Strict Groq limit (10k TPM)
    "qwen2.5-coder:7b":        10_000,
    "qwen2.5-coder:3b":        6_000,
    "gemma2:2b":               6_000,
    "_default":                16_000, 
}
MAX_OLDEST_MESSAGE_TRUNCATION_CHARS = 1500
RAG_INDEX_TIMEOUT_SECONDS = 2.0
SEMANTIC_GRAPH_THRESHOLD = 0.4
DEDUP_PREFIX_LENGTH = 60
DEFAULT_PLANNER_FALLBACK_MODEL = "groq:llama-3.3-70b-versatile"
SYSTEM_PROMPT_CHAR_BUDGET = max(4000, int(getattr(cfg, "SYSTEM_PROMPT_CHAR_BUDGET", 9000)))
CANDIDATE_CONTEXT_MAX_LINES = 12
COMPRESSION_INTERVAL = max(1, int(getattr(cfg, "COMPRESSION_INTERVAL", 4)))
COMPRESSION_WINDOW_THRESHOLD = min(
    1.0,
    max(0.1, float(getattr(cfg, "COMPRESSION_WINDOW_THRESHOLD", 0.8))),
)
ENABLE_MANIFEST_PROTOCOL = bool(getattr(cfg, "ENABLE_MANIFEST_PROTOCOL", True))
TRIVIAL_QUERY_RE = re.compile(
    r"^(?:(?:hi|hello|hey|yo)(?:\s+\w{1,20}){0,2}|ok|okay|k|kk|cool|nice|thanks|thank you|thx|got it|sure)[!. ]*$",
    re.IGNORECASE,
)
MEMORY_QUERY_RE = re.compile(
    r"\b(remember|recall|earlier|before|last time|you said|i told|we discussed|previously|what did i tell)\b",
    re.IGNORECASE,
)
URL_QUERY_RE = re.compile(r"https?://", re.IGNORECASE)
MULTISTEP_QUERY_RE = re.compile(
    r"\b(then|after that|afterwards|step by step|plan|research|summarize|save|write|create|build|compare|analyze|fix|implement)\b",
    re.IGNORECASE,
)
DIRECT_FACT_QUERY_RE = re.compile(
    r"\b(current|latest|news|today|recent|price|who is|what is|when is|where is|which)\b",
    re.IGNORECASE,
)
ARTIFACT_REQUEST_RE = re.compile(
    r"\b(html|svg|ui|app|dashboard|website|webpage|component|code|file|script|markdown|json|xml)\b",
    re.IGNORECASE,
)
CONVERSATIONAL_QUERY_RE = re.compile(
    r"\b(hey|hi|hello|yo|bro|sup|what'?s up|how are you|what happened|where were you|you there|come back)\b",
    re.IGNORECASE,
)
CORRECTION_QUERY_RE = re.compile(
    r"\b(actually|now|instead|correction|updated|i moved|not .* anymore|i live in|i am in)\b",
    re.IGNORECASE,
)
INTERNAL_EXECUTION_CHATTER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"i(?:'| a)?m sorry,? but i can'?t assist with that\.?"
    r"|i already have an execution plan.*"
    r"|the results? (?:from .*?)?are available in the evidence packet.*"
    r"|please let me know what specific information.*"
    r"|what specific information are you looking for.*"
    r"|sure,? i can research .* for you\..*"
    r")\s*(?=\n|$)",
    re.IGNORECASE,
)
SYSTEM_ECHO_BLOCK_RE = re.compile(
    r"\s*(?:<SYSTEM_EVIDENCE_PACKET>.*?</SYSTEM_EVIDENCE_PACKET>|<SYSTEM_OBSERVATION>.*?</SYSTEM_OBSERVATION>|<system_observation>.*?</system_observation>|<tool_call>.*?</tool_call>|</?arg_key>|</?arg_value>)\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

def _get_token_encoding():
    """
    Resolve a safe tokenizer without raising.
    Falls back to a minimal char-based encoder if tiktoken mappings are unavailable.
    """
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None

def _truncate_for_model(content: str, model: str) -> str:
    """Truncate tool result to fit within the model's token context limit."""
    encoding = _get_token_encoding()
        
    limit = TOKEN_SAFETY_LIMITS.get(model, TOKEN_SAFETY_LIMITS["_default"])
    # Tool results should only take up at most 70% of the total budget 
    # to leave room for history/system prompts.
    safe_tool_limit = int(limit * 0.70)
    if encoding is None:
        # Approximate 1 token ~= 4 chars if tokenizer isn't available.
        safe_chars = safe_tool_limit * 4
        if len(content) <= safe_chars:
            return content
        keep = int(safe_chars * 0.45)
        prefix = content[:keep]
        suffix = content[-keep:]
    else:
        tokens = encoding.encode(content)
        if len(tokens) <= safe_tool_limit:
            return content
        keep = int(safe_tool_limit * 0.45)
        prefix = encoding.decode(tokens[:keep])
        suffix = encoding.decode(tokens[-keep:])
        truncated_amount = len(tokens) - safe_tool_limit
        return f"{prefix}\n\n[...Token Budget Exceeded: {truncated_amount} tokens truncated...]\n\n{suffix}"
    
    truncated_amount = len(content) - safe_chars
    return f"{prefix}\n\n[...Token Budget Exceeded: ~{truncated_amount} chars truncated...]\n\n{suffix}"

def _enforce_total_budget(messages: list[dict], model: str) -> list[dict]:
    """
    Ensure the total token count of the message list fits within the model's safety limit.
    If not, iteratively drops or truncates the sliding window/system context.
    """
    encoding = _get_token_encoding()

    limit = TOKEN_SAFETY_LIMITS.get(model, TOKEN_SAFETY_LIMITS["_default"])
    
    def total_tokens(msgs):
        if encoding is None:
            # Approximation fallback: ~1 token per 4 chars.
            return sum(len(m.get("content", "")) // 4 for m in msgs)
        return sum(len(encoding.encode(m["content"])) for m in msgs)

    # If already safe, return
    if total_tokens(messages) <= limit:
        return messages

    log("llm", "budget", f"Total tokens ({total_tokens(messages)}) exceeds model limit ({limit}). Truncating...", level="warn")

    # 1. Truncate the combined system message (context/RAG/tools) first if it's huge
    if messages and messages[0]["role"] == "system":
        if encoding is None:
            sys_content = messages[0]["content"]
            sys_limit_chars = int(limit * 0.6) * 4
            if len(sys_content) > sys_limit_chars:
                keep_chars = int(limit * 0.3) * 4
                prefix = sys_content[:keep_chars]
                suffix = sys_content[-keep_chars:]
                messages[0]["content"] = f"{prefix}\n\n[...System Context Truncated...]\n\n{suffix}"
        else:
            sys_tokens = encoding.encode(messages[0]["content"])
            if len(sys_tokens) > limit * 0.6:
                # Keep first 30% and last 30% of system prompt
                keep = int(limit * 0.3)
                prefix = encoding.decode(sys_tokens[:keep])
                suffix = encoding.decode(sys_tokens[-keep:])
                messages[0]["content"] = f"{prefix}\n\n[...System Context Truncated...]\n\n{suffix}"

    # 2. If still over, drop oldest non-user messages from sliding window (middle of list)
    # We keep the first system message (index 0) and the last user message (index -1)
    while total_tokens(messages) > limit and len(messages) > 2:
        dropped = False
        for i in range(1, len(messages) - 1):
            role = messages[i].get("role", "")
            content = messages[i].get("content", "")
            is_tool_result = "SYSTEM_TOOL_RESULT" in content
            if role == "assistant" and not is_tool_result:
                removed = messages.pop(i)
                dropped = True
                log("llm", "budget", f"Dropped message to fit budget: {removed.get('role')}", level="info")
                break
        if not dropped:
            # Nothing safe to drop — truncate oldest remaining message.
            if len(messages) > 2:
                messages[1]["content"] = (
                    messages[1].get("content", "")[:MAX_OLDEST_MESSAGE_TRUNCATION_CHARS]
                    + "\n[truncated for budget]"
                )
            break

    return messages

# ── Trace Logger ─────────────────────────────────────────────────────────────
_TRACE_DIR = os.getenv("TRACE_DIR", "./logs")
os.makedirs(_TRACE_DIR, exist_ok=True)
_TRACE_PATH = os.path.join(_TRACE_DIR, "agent_trace.jsonl")
_SYSTEM_REMINDER_BLOCK_RE = re.compile(r"\s*<system-reminder>.*?</system-reminder>\s*", flags=re.IGNORECASE | re.DOTALL)
_TRACE_STORE = get_trace_store()

def _trace(
    agent_id: str,
    session_id: str,
    event_type: str,
    data: dict,
    *,
    run_id: Optional[str] = None,
    owner_id: Optional[str] = None,
):
    """Append structured trace event to the scalable trace store."""
    pass_index = None
    if isinstance(data, dict):
        if isinstance(data.get("pass_index"), int):
            pass_index = data["pass_index"]
        elif isinstance(data.get("turn"), int):
            pass_index = data["turn"]

    try:
        _TRACE_STORE.append_event(
            agent_id=agent_id,
            session_id=session_id,
            event_type=event_type,
            data=data,
            pass_index=pass_index,
            run_id=run_id,
            owner_id=owner_id,
        )
    except Exception:
        pass  # Never crash on trace failure

def _normalize_retrieval_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized

def _retrieval_fingerprint(item: dict) -> str:
    doc_id = str(item.get("id") or "").strip().lower()
    if doc_id:
        return f"id:{doc_id}"
    text = _normalize_retrieval_text(
        item.get("text")
        or f"{item.get('key', '')} {item.get('anchor', '')}"
    )
    if not text:
        return ""
    return text[:240]

def _truncate_prompt_section(content: str, budget: int, *, preserve_ends: bool = False) -> str:
    if budget <= 0 or len(content) <= budget:
        return content
    if preserve_ends and budget >= 120:
        keep = max(40, (budget - 32) // 2)
        return f"{content[:keep]}\n[...truncated...]\n{content[-keep:]}"
    keep = max(32, budget - 18)
    return f"{content[:keep]}\n[...truncated...]"


def _load_bootstrap_documents(
    workspace_path: Optional[str],
    bootstrap_files: Optional[list[str]],
    bootstrap_max_chars: int,
) -> list[dict[str, str]]:
    root = str(workspace_path or "").strip()
    if not root:
        return []

    try:
        workspace = Path(root).expanduser().resolve()
    except Exception:
        return []
    if not workspace.exists() or not workspace.is_dir():
        return []

    requested = [str(name or "").strip() for name in (bootstrap_files or []) if str(name or "").strip()]
    if not requested:
        return []

    docs: list[dict[str, str]] = []
    max_total = max(1000, int(bootstrap_max_chars or 8000))
    remaining = max_total

    for name in requested:
        if remaining <= 0:
            break
        candidate = (workspace / name).resolve()
        try:
            candidate.relative_to(workspace)
        except Exception:
            continue
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not content:
            continue
        take = min(remaining, max(400, remaining))
        snippet = _truncate_prompt_section(content, take, preserve_ends=True)
        docs.append({
            "name": candidate.name,
            "path": str(candidate),
            "content": snippet,
        })
        remaining -= len(snippet) + 64
    return docs

def sanitize_user_message(content: str) -> str:
    """
    Phase 4: Prompt Injection Sanitization.
    Strips raw XML tool tags and fake metadata blocks from user input 
    to prevent the user from spoofing tool results or system directives.
    """
    if not content:
        return content
    # Strip <SYSTEM_TOOL_RESULT> tags
    content = re.sub(r'</?SYSTEM_TOOL_RESULT[^>]*>', '[censored tag]', content, flags=re.IGNORECASE)
    # Strip fake thinking blocks
    content = re.sub(r'</?think>', '[censored tag]', content, flags=re.IGNORECASE)
    # Strip fake boundary markers
    content = content.replace("---", "- - -")
    return content

DEFAULT_SYSTEM_PROMPT = """\
You are Shovs, a Language OS assistant running inside a controlled multi-phase runtime.

Core Directives:
- CURRENT TURN FIRST: The latest user objective is the primary source of truth for this response. Use memory/history to support the current turn, not to override it.
- EXECUTION DISCIPLINE: Either perform the next step or answer the user. Do not narrate hidden plans, packet names, schemas, or internal execution state.
- TOOL CALLS: When a tool is materially needed, output ONLY one valid JSON tool call: {"tool": "<name>", "arguments": {<args>}}. No markdown. No preamble. No extra text.
- QUERY FIDELITY: Preserve exact entities, domains, URLs, file names, tickers, and user-supplied keywords unless verified evidence justifies a change. Do not silently rename or broaden the target.
- RESPONSES: Default to normal plain text. Do not output HTML, SVG, app fragments, dashboards, faux logs, or code blocks unless the user explicitly asks for a visual artifact, file, app, code, or markup.
- ACCURACY: Never fabricate tool results, files, searches, completed work, or prior execution.
- EVIDENCE USE: If the runtime says evidence is already gathered or tools are disabled, synthesize directly from that evidence. Do not mention internal packets, reminders, planner strategy, or observation state.
- WEB FETCH RULE: Only call web_fetch with URLs that already appeared in a prior web_search result in this conversation. Never invent or guess URLs.
- DELEGATION: Use `delegate_to_agent` only when a specialized agent is clearly a better fit than continuing inside the current run.
"""

MODEL_PROFILE_BUDGETS: dict[str, dict[str, int]] = {
    "small_local": {
        "system_chars": 5200,
        "history_chars_per_msg": 1600,
        "followup_results": 3,
        "followup_chars": 1100,
    },
    "tool_native_local": {
        "system_chars": 6200,
        "history_chars_per_msg": 2200,
        "followup_results": 4,
        "followup_chars": 1400,
    },
    "local_standard": {
        "system_chars": 7200,
        "history_chars_per_msg": 2400,
        "followup_results": 4,
        "followup_chars": 1500,
    },
    "frontier_native": {
        "system_chars": SYSTEM_PROMPT_CHAR_BUDGET,
        "history_chars_per_msg": 4000,
        "followup_results": 5,
        "followup_chars": 2200,
    },
    "frontier_standard": {
        "system_chars": SYSTEM_PROMPT_CHAR_BUDGET,
        "history_chars_per_msg": 4000,
        "followup_results": 4,
        "followup_chars": 1800,
    },
}

def _count_context_items(raw_context: str) -> int:
    """Count visible memory units across V1 bullets and V2 JSON context."""
    if not raw_context:
        return 0
    try:
        payload = json.loads(raw_context)
        if isinstance(payload, dict) and payload.get("__v2__"):
            modules = payload.get("modules", {})
            return len(modules) if isinstance(modules, dict) else 0
    except Exception:
        pass
    return len([l for l in raw_context.split("\n") if l.strip()])


MAX_TOOL_TURNS = max(8, int(os.getenv("MAX_TOOL_TURNS", "8")))

def _ev(type_: str, **kw) -> dict:
    return {"type": type_, **kw}

def _to_bool(val) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)

def _is_trivial_query(message: str) -> bool:
    return bool(TRIVIAL_QUERY_RE.fullmatch((message or "").strip()))

def _prefers_plaintext_chat(message: str, route_type: str) -> bool:
    normalized = (message or "").strip()
    if not normalized:
        return True
    if ARTIFACT_REQUEST_RE.search(normalized):
        return False
    if route_type == "trivial_chat":
        return True
    if len(normalized.split()) <= 14 and CONVERSATIONAL_QUERY_RE.search(normalized):
        return True
    return False

def _tool_turn_budget(model: str, adapter: Optional[BaseLLMAdapter]) -> int:
    adapter_name = (adapter.__class__.__name__ if adapter else "").lower()
    model_name = (model or "").lower()
    if "groq" in adapter_name or model_name.startswith("groq:"):
        return 3
    if "ollama" in adapter_name or model_name.startswith("ollama:"):
        return 4
    if "anthropic" in adapter_name or model_name.startswith("anthropic:"):
        return 8
    return MAX_TOOL_TURNS


def _classify_model_profile(adapter: Optional[BaseLLMAdapter], model: str) -> str:
    adapter_name = (adapter.__class__.__name__ if adapter else "").lower()
    model_name = (model or "").lower()
    raw_base_url = getattr(adapter, "base_url", "")
    base_url = raw_base_url.strip().lower() if isinstance(raw_base_url, str) else ""
    local_base_url = base_url.startswith((
        "http://127.0.0.1",
        "http://localhost",
        "https://127.0.0.1",
        "https://localhost",
    ))
    is_local = (
        "ollama" in adapter_name
        or local_base_url
        or model_name.startswith(("ollama:", "lmstudio:", "llamacpp:", "local_openai:"))
    )
    native_tools = _should_use_native_tools(adapter, model)

    size_match = re.search(r"(^|[^0-9])(\d+(?:\.\d+)?)b([^a-z0-9]|$)", model_name)
    size_b = float(size_match.group(2)) if size_match else None
    if is_local and size_b is not None and size_b <= 4.0:
        return "small_local"
    if is_local and native_tools:
        return "tool_native_local"
    if is_local:
        return "local_standard"
    if native_tools:
        return "frontier_native"
    return "frontier_standard"


def _model_profile_budget(model_profile: str, key: str, fallback: int) -> int:
    return int(MODEL_PROFILE_BUDGETS.get(model_profile, {}).get(key, fallback))

def _adaptive_tool_turn_budget(
    model: str,
    adapter: Optional[BaseLLMAdapter],
    *,
    route_type: str = "open_ended",
    forced_tools: Optional[list[str]] = None,
    user_message: str = "",
) -> int:
    budget = _tool_turn_budget(model, adapter)
    lowered_tools = {name.lower() for name in (forced_tools or [])}
    lowered_message = (user_message or "").lower()

    file_heavy = bool(
        {"file_view", "file_str_replace", "file_create"} & lowered_tools
        or (
            re.search(r"\b(check|read|open|inspect|review|scan|list)\b", lowered_message)
            and re.search(r"\b(file|files|folder|directory|md|markdown)\b", lowered_message)
        )
    )
    research_heavy = bool(
        {"web_search", "web_fetch", "rag_search"} & lowered_tools
        or (
            route_type in {"multi_step", "open_ended"}
            and re.search(r"\b(research|gather|intel|investigate|compare|analyze|deep[- ]?dive)\b", lowered_message)
        )
    )

    if file_heavy:
        budget += 2
    if research_heavy:
        budget += 1

    return min(max(budget, 3), 8)


def _normalize_optional_limit(value: Any, *, minimum: int, maximum: int) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        normalized = int(value)
    except Exception:
        return None
    if normalized <= 0:
        return None
    return max(minimum, min(maximum, normalized))

def _should_compress_exchange(session, is_first_exchange: bool) -> bool:
    if is_first_exchange:
        return True
    turn_number = max(1, session.message_count // 2)
    if turn_number % COMPRESSION_INTERVAL == 0:
        return True
    sliding_window_size = max(1, int(getattr(cfg, "SLIDING_WINDOW_SIZE", len(session.sliding_window) or 1)))
    window_limit = max(1, int(sliding_window_size * COMPRESSION_WINDOW_THRESHOLD))
    return len(session.sliding_window) >= window_limit

def _classify_route(
    query: str,
    *,
    session_has_history: bool,
    current_fact_count: int,
    active_task_count: int,
) -> str:
    normalized = (query or "").strip()
    if not normalized:
        return "trivial_chat"
    if _is_trivial_query(normalized):
        return "trivial_chat"
    if URL_QUERY_RE.search(normalized):
        return "url_fetch"
    if CORRECTION_QUERY_RE.search(normalized):
        return "memory_recall"
    if MEMORY_QUERY_RE.search(normalized) and (session_has_history or current_fact_count > 0):
        return "memory_recall"
    if active_task_count > 0 and MULTISTEP_QUERY_RE.search(normalized):
        return "multi_step"
    if MULTISTEP_QUERY_RE.search(normalized):
        return "multi_step"
    if DIRECT_FACT_QUERY_RE.search(normalized):
        return "direct_fact"
    return "open_ended"

def _build_retrieval_policy(route_type: str, force_memory: bool = False) -> dict[str, object]:
    policy = {
        "should_retrieve": False,
        "use_vector": False,
        "use_graph": False,
        "use_session_rag": False,
        "top_n": max(1, int(getattr(cfg, "RETRIEVAL_TOP_K_DEFAULT", 5))),
    }
    if route_type == "memory_recall" or force_memory:
        policy.update({
            "should_retrieve": True,
            "use_vector": True,
            "use_graph": True,
            "use_session_rag": True,
            "top_n": max(1, int(getattr(cfg, "RETRIEVAL_TOP_K_MEMORY", 6))),
        })
    elif route_type == "direct_fact":
        policy.update({
            "should_retrieve": False,
            "use_vector": False,
            "use_graph": False,
            "use_session_rag": False,
            "top_n": max(1, int(getattr(cfg, "RETRIEVAL_TOP_K_FACT", 4))),
        })
    elif route_type == "multi_step":
        policy.update({
            "should_retrieve": True,
            "use_vector": True,
            "use_graph": True,
            "use_session_rag": True,
        })
    elif route_type == "open_ended":
        policy.update({
            "should_retrieve": True,
            "use_vector": True,
            "use_graph": True,
            "use_session_rag": False,
        })
    elif route_type == "url_fetch":
        policy.update({
            "should_retrieve": False,
            "use_vector": False,
            "use_graph": False,
            "use_session_rag": False,
        })
    return policy


def _should_index_tool_result(tool_name: str, content: str) -> bool:
    if len((content or "").strip()) < 120:
        return False
    if tool_name in {"todo_write", "todo_update", "query_memory", "store_memory"}:
        return False
    return True

def _tool_result_content_type(tool_name: str) -> str:
    if tool_name in {"web_fetch", "web_search", "image_search"}:
        return "web_evidence"
    if tool_name in {"file_view", "file_create", "file_str_replace", "pdf_processor"}:
        return "file_evidence"
    if tool_name == "bash":
        return "execution_output"
    return "tool_result"


def _sanitize_known_tool_result_payload(tool_name: str, content: str) -> Optional[str]:
    raw = (content or "").strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        return None

    payload_type = str(payload.get("type") or "").strip().lower()

    if tool_name in {"web_search", "image_search"}:
        if payload_type.endswith("_error"):
            return json.dumps({
                "type": payload_type or f"{tool_name}_error",
                "query": payload.get("query", ""),
                "backend": payload.get("backend", ""),
                "error": payload.get("error", ""),
            })
        results = []
        for item in (payload.get("results") or [])[:3]:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": str(item.get("snippet", ""))[:220],
            })
        context_summary = payload.get("context_summary", "")
        if isinstance(context_summary, dict):
            context_summary = {
                "requested_results": context_summary.get("requested_results"),
                "curated_results": context_summary.get("curated_results"),
                "unique_domains": (context_summary.get("unique_domains") or [])[:6],
            }
        else:
            context_summary = str(context_summary)[:280]
        return json.dumps({
            "type": payload_type or f"{tool_name}_results",
            "query": payload.get("query", ""),
            "backend": payload.get("backend", ""),
            "engine": payload.get("engine", ""),
            "context_summary": context_summary,
            "results": results,
        })

    if tool_name == "web_fetch":
        if payload_type.endswith("_error"):
            return json.dumps({
                "type": payload_type or "web_fetch_error",
                "url": payload.get("url", ""),
                "error": payload.get("error", ""),
            })
        content_preview = str(payload.get("content", "")).strip()
        if len(content_preview) > 1200:
            content_preview = content_preview[:1200].rstrip() + "\n[...sanitized-preview-truncated...]"
        return json.dumps({
            "type": payload_type or "web_fetch_result",
            "url": payload.get("url", ""),
            "final_url": payload.get("final_url", payload.get("url", "")),
            "title": payload.get("title", ""),
            "backend": payload.get("backend", ""),
            "status_code": payload.get("status_code", ""),
            "truncated": bool(payload.get("truncated")),
            "total_length": payload.get("total_length", 0),
            "content_preview": content_preview,
        })

    return None


def _sanitize_task_state_result(tool_name: str, content: str) -> Optional[str]:
    raw = (content or "").strip()
    if not raw or tool_name not in {"todo_write", "todo_update"}:
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    topic = ""
    tasks: list[str] = []
    for line in lines:
        if line.lower().startswith("workflow topic:"):
            topic = line.split(":", 1)[1].strip()
        elif line.startswith("- ["):
            tasks.append(line)
    summary = {
        "type": "task_state_update",
        "tool": tool_name,
        "topic": topic,
        "task_count": len(tasks),
        "tasks_preview": tasks[:3],
    }
    return json.dumps(summary)


def _compact_tool_result_for_followup(tool_name: str, content: str) -> str:
    raw = (content or "").strip()
    if not raw:
        return ""

    sanitized_known = _sanitize_known_tool_result_payload(tool_name, raw)
    if sanitized_known:
        return sanitized_known

    sanitized_task_state = _sanitize_task_state_result(tool_name, raw)
    if sanitized_task_state:
        return sanitized_task_state

    if tool_name in {"web_search", "image_search"}:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                compact = {
                    "type": payload.get("type", f"{tool_name}_results"),
                    "query": payload.get("query", ""),
                    "engine": payload.get("engine", ""),
                    "context_summary": payload.get("context_summary", ""),
                    "results": [],
                }
                for item in (payload.get("results") or [])[:3]:
                    if not isinstance(item, dict):
                        continue
                    compact["results"].append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": str(item.get("snippet", ""))[:220],
                    })
                return json.dumps(compact)
        except Exception:
            pass

    limit = 1200
    if tool_name in {"file_create", "file_str_replace"}:
        limit = 500
    elif tool_name in {"file_view", "bash"}:
        limit = 1000
    elif tool_name in {"query_memory", "todo_write", "todo_update"}:
        limit = 900

    if len(raw) <= limit:
        return raw
    return f"{raw[:limit].rstrip()}\n[...observation-truncated...]"


def _merge_candidate_context(existing: str, blocked_records: list[dict]) -> str:
    lines = [line.strip() for line in (existing or "").splitlines() if line.strip()]
    seen = {line.lower() for line in lines}
    for record in blocked_records or []:
        fact = str(record.get("fact") or " ".join(
            part for part in (
                record.get("subject"),
                record.get("predicate"),
                record.get("object"),
            ) if part
        )).strip()
        if not fact:
            continue
        reason = str(record.get("grounding_reason") or "candidate")
        line = f"- Candidate: {fact} (reason={reason})"
        if line.lower() in seen:
            continue
        lines.append(line)
        seen.add(line.lower())
    return "\n".join(lines[-CANDIDATE_CONTEXT_MAX_LINES:])


def _tool_result_previews(tool_results: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for item in (tool_results or [])[:limit]:
        previews.append({
            "tool_name": item.get("tool_name"),
            "success": bool(item.get("success")),
            "preview": str(item.get("content") or item.get("preview") or "")[:220],
        })
    return previews


def _checkpoint_candidate_facts(checkpoint: Optional[LoopCheckpoint]) -> list[str]:
    if checkpoint is None:
        return []
    return [
        str(item).strip()
        for item in (checkpoint.candidate_facts or [])
        if str(item).strip()
    ]


def _build_followup_evidence_packet(
    tool_results: list[dict[str, Any]],
    *,
    max_results: int = 4,
    max_total_chars: int = 1800,
) -> str:
    sections: list[str] = []
    used = 0
    for item in (tool_results or [])[:max_results]:
        tool_name = str(item.get("tool_name") or "unknown")
        success = "ok" if item.get("success") else "failed"
        preview = _compact_tool_result_for_followup(tool_name, str(item.get("content") or ""))
        preview = preview.strip()
        if not preview:
            preview = "[empty]"
        block = (
            f"Tool: {tool_name}\n"
            f"Status: {success}\n"
            f"Evidence:\n{preview}"
        )
        additional = len(block) + (2 if sections else 0)
        if used + additional > max_total_chars:
            break
        sections.append(block)
        used += additional
    if not sections:
        return ""
    return "<SYSTEM_EVIDENCE_PACKET>\n" + "\n\n".join(sections) + "\n</SYSTEM_EVIDENCE_PACKET>"


def _extract_exact_query_targets(user_message: str) -> list[str]:
    text = (user_message or "").lower()
    targets = re.findall(r"\b[a-z0-9][a-z0-9.-]*\.[a-z]{2,}\b", text)
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        normalized = target.strip().lower()
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def _tool_result_matches_exact_target(item: dict[str, Any], exact_targets: list[str]) -> bool:
    if not exact_targets:
        return False
    haystacks = [
        str(item.get("content") or "").lower(),
        json.dumps(item.get("arguments") or {}, ensure_ascii=False).lower(),
    ]
    return any(target in haystack for target in exact_targets for haystack in haystacks)


def _tool_kind_priority(tool_name: str) -> int:
    priority = {
        "web_fetch": 0,
        "file_view": 1,
        "web_search": 2,
        "image_search": 3,
        "query_memory": 5,
        "todo_update": 6,
        "todo_write": 7,
        "store_memory": 8,
    }
    return priority.get(tool_name, 4)


def _select_followup_tool_results(
    tool_results: list[dict[str, Any]],
    *,
    user_message: str,
    max_results: int = 4,
) -> list[dict[str, Any]]:
    if not tool_results:
        return []

    exact_targets = _extract_exact_query_targets(user_message)
    scored: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []
    for idx, item in enumerate(tool_results):
        tool_name = str(item.get("tool_name") or "unknown")
        success = bool(item.get("success"))
        substantive = tool_name not in {"todo_write", "todo_update", "query_memory", "store_memory"}
        exact_match = _tool_result_matches_exact_target(item, exact_targets)
        score = (
            0 if success else 1,
            0 if substantive else 1,
            0 if exact_match else 1,
            _tool_kind_priority(tool_name),
        )
        scored.append((score, idx, item))

    scored.sort(key=lambda row: (row[0], -row[1]))

    selected: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()
    for _, _, item in scored:
        tool_name = str(item.get("tool_name") or "unknown")
        preview = _compact_tool_result_for_followup(tool_name, str(item.get("content") or ""))
        signature = (tool_name, preview[:180])
        if signature in seen_signatures:
            continue
        selected.append(item)
        seen_signatures.add(signature)
        if len(selected) >= max_results:
            break

    if any(bool(item.get("success")) for item in selected) and len(selected) < max_results:
        latest_failure = next(
            (item for item in reversed(tool_results) if not bool(item.get("success"))),
            None,
        )
        if latest_failure:
            tool_name = str(latest_failure.get("tool_name") or "unknown")
            preview = _compact_tool_result_for_followup(tool_name, str(latest_failure.get("content") or ""))
            signature = (tool_name, preview[:180])
            if signature not in seen_signatures:
                selected.append(latest_failure)

    return selected


def _dedupe_tool_names(tool_names: Optional[list[str]]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for name in tool_names or []:
        normalized = str(name or "").strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _normalize_forced_tools_for_task_state(
    forced_tools: Optional[list[str]],
    *,
    has_tasks: bool,
) -> list[str]:
    normalized = _dedupe_tool_names(forced_tools)
    if not has_tasks:
        return normalized
    return [name for name in normalized if name != "todo_write"]


def _has_substantive_tool_evidence(tool_results: list[dict[str, Any]]) -> bool:
    administrative_tools = {"todo_write", "todo_update", "query_memory", "store_memory"}
    return any(
        bool(item.get("success")) and str(item.get("tool_name") or "") not in administrative_tools
        for item in (tool_results or [])
    )


def _requests_synthesis(user_message: str) -> bool:
    return bool(
        re.search(
            r"\b(tldr|tl;dr|summary|summarize|report|compare|analyze|analysis)\b",
            (user_message or "").lower(),
        )
    )


def _build_task_progress_instruction(
    *,
    route_type: str,
    user_message: str,
    tool_results: list[dict[str, Any]],
    has_active_tasks: bool,
) -> str:
    requested_synthesis = _requests_synthesis(user_message)
    task_bootstrapped = any(
        bool(item.get("success")) and str(item.get("tool_name") or "") == "todo_write"
        for item in (tool_results or [])
    )
    substantive_evidence = _has_substantive_tool_evidence(tool_results)

    if task_bootstrapped and not substantive_evidence:
        return (
            "Task state is now initialized. Do not call todo_write again in this run. "
            "Next, call the next substantive tool needed for the user's request, preserving the user's exact entities/domains when forming queries, "
            "or use todo_update to mark progress if a task status changed. "
            "If another tool is needed, respond with ONLY one valid JSON tool invocation and no commentary."
        )

    if substantive_evidence:
        if requested_synthesis:
            return (
                "You now have substantive evidence. Do not call todo_write again in this run. "
                "If the gathered evidence is enough, write the requested summary/report/TLDR directly. "
                "Only call one more substantive tool if a specific missing gap remains."
            )
        if has_active_tasks or route_type == "multi_step":
            return (
                "You now have substantive evidence. Do not call todo_write again in this run. "
                "Prefer either the next substantive tool or a concise user-facing synthesis. "
                "Use todo_update only if task progress actually changed."
            )

    return ""


def _should_include_task_reminder(
    *,
    route_type: str,
    user_message: str,
    tool_results: list[dict[str, Any]],
    has_active_tasks: bool,
) -> bool:
    if not has_active_tasks:
        return False
    if _has_substantive_tool_evidence(tool_results) and _requests_synthesis(user_message):
        return False
    if route_type == "direct_fact" and _has_substantive_tool_evidence(tool_results):
        return False
    return True


def _is_synthetic_followup_message(message: dict[str, Any]) -> bool:
    role = str(message.get("role") or "")
    content = str(message.get("content") or "")
    stripped = content.strip()
    if role == "assistant":
        return stripped.startswith('{"tool"') or stripped.startswith('{"tool_calls"')
    if role == "user":
        return (
            "<SYSTEM_TOOL_RESULT" in content
            or "<SYSTEM_EVIDENCE_PACKET>" in content
            or "<SYSTEM_OBSERVATION>" in content
            or "Tool budget reached for this turn." in content
        )
    return False


def _prune_managed_followup_messages(
    messages: list[dict[str, Any]],
    *,
    keep_recent_natural: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not messages:
        return messages, {"removed_messages": 0, "natural_messages_kept": 0}

    kept_system: list[dict[str, Any]] = []
    body = messages
    if messages and str(messages[0].get("role") or "") == "system":
        kept_system = [messages[0]]
        body = messages[1:]

    natural_messages = [
        message
        for message in body
        if not _is_synthetic_followup_message(message)
    ]
    trimmed_natural = natural_messages[-keep_recent_natural:]
    pruned = kept_system + trimmed_natural
    removed = max(0, len(messages) - len(pruned))
    return pruned, {
        "removed_messages": removed,
        "natural_messages_kept": len(trimmed_natural),
    }


def _sanitize_followup_messages(
    messages: list[dict[str, Any]],
    *,
    forced_tools: Optional[list[str]],
    user_message: str,
    route_type: str,
    latest_tool_results: list[dict[str, Any]],
    has_tasks: bool,
    keep_recent_natural: int = 4,
) -> tuple[list[dict[str, Any]], Optional[list[str]], dict[str, Any]]:
    sanitized_messages, prune_stats = _prune_managed_followup_messages(
        messages,
        keep_recent_natural=keep_recent_natural,
    )
    sanitized_forced_tools = _normalize_forced_tools_for_task_state(
        forced_tools,
        has_tasks=has_tasks,
    )
    substantive_evidence = _has_substantive_tool_evidence(latest_tool_results)
    requested_synthesis = _requests_synthesis(user_message)
    if substantive_evidence and requested_synthesis and sanitized_forced_tools:
        sanitized_forced_tools = [
            name
            for name in sanitized_forced_tools
            if name not in {"todo_update", "query_memory", "store_memory"}
        ]
    return sanitized_messages, (sanitized_forced_tools or None), {
        **prune_stats,
        "forced_tools_before": len(_dedupe_tool_names(forced_tools)),
        "forced_tools_after": len(sanitized_forced_tools),
        "substantive_evidence": substantive_evidence,
        "requested_synthesis": requested_synthesis,
        "has_tasks": has_tasks,
        "route_type": route_type,
    }


def _json_payload_or_none(raw: str) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_run_artifact_candidates(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    content: str,
) -> list[dict[str, Any]]:
    raw = str(content or "")
    preview = _compact_tool_result_for_followup(tool_name, raw)[:600]
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""
    size_bytes = len(raw.encode("utf-8")) if raw else 0
    artifacts: list[dict[str, Any]] = []

    def add_candidate(
        artifact_type: str,
        label: str,
        *,
        storage_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        artifacts.append({
            "artifact_type": artifact_type,
            "label": label,
            "storage_path": storage_path,
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "preview": preview,
            "metadata": dict(metadata or {}),
        })

    if tool_name in {"file_create", "file_view", "file_str_replace"}:
        path = str(arguments.get("path") or arguments.get("filename") or "").strip()
        if path:
            add_candidate(
                "file",
                path,
                storage_path=path,
                metadata={"path": path, "tool_name": tool_name},
            )
        return artifacts

    if tool_name == "pdf_processor":
        path = str(
            arguments.get("output_path")
            or arguments.get("path")
            or arguments.get("save_path")
            or ""
        ).strip()
        if path:
            add_candidate(
                "file",
                path,
                storage_path=path,
                metadata={"path": path, "tool_name": tool_name},
            )
        return artifacts

    if tool_name == "web_fetch":
        url = str(arguments.get("url") or "").strip()
        add_candidate(
            "web_page",
            url or "web_fetch",
            metadata={"url": url, "tool_name": tool_name},
        )
        return artifacts

    if tool_name in {"web_search", "image_search"}:
        payload = _json_payload_or_none(raw) or {}
        query = str(arguments.get("query") or payload.get("query") or "").strip()
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        add_candidate(
            "search_results",
            query or tool_name,
            metadata={
                "query": query,
                "engine": payload.get("engine", ""),
                "result_count": len(results),
                "tool_name": tool_name,
            },
        )
        return artifacts

    if tool_name == "generate_app":
        payload = _json_payload_or_none(raw) or {}
        label = str(payload.get("filename") or payload.get("title") or "generated_app").strip()
        storage_path = str(payload.get("path") or "").strip() or None
        add_candidate(
            "app",
            label,
            storage_path=storage_path,
            metadata={"tool_name": tool_name},
        )
        return artifacts

    return artifacts

def _should_use_native_tools(adapter: Optional[BaseLLMAdapter], model: str) -> bool:
    adapter_name = (adapter.__class__.__name__ if adapter else "").lower()
    model_name = (model or "").lower()
    if "ollama" in adapter_name or model_name.startswith("ollama:"):
        return False
    if "groq" in adapter_name or model_name.startswith("groq:"):
        return False
    return True

def _should_prefer_single_loop(
    adapter: Optional[BaseLLMAdapter],
    model: str,
) -> bool:
    adapter_name = (adapter.__class__.__name__ if adapter else "").lower()
    model_name = (model or "").lower()
    raw_base_url = getattr(adapter, "base_url", "")
    base_url = raw_base_url.strip().lower() if isinstance(raw_base_url, str) else ""
    local_base_url = base_url.startswith((
        "http://127.0.0.1",
        "http://localhost",
        "https://127.0.0.1",
        "https://localhost",
    ))
    return (
        "ollama" in adapter_name
        or local_base_url
        or model_name.startswith(("ollama:", "lmstudio:", "llamacpp:", "local_openai:"))
    )


def _is_context_overflow_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "context length" in text
        or "maximum context length" in text
        or "prompt is too long" in text
        or "number of tokens to keep" in text
        or "input too long" in text
    )


def _shrink_messages_for_context_retry(messages: list[dict]) -> tuple[list[dict], dict[str, int]]:
    if not messages:
        return messages, {"system_chars": 0, "history_count": 0, "user_chars": 0}

    system_content = str(messages[0].get("content", ""))
    compact_system = _truncate_prompt_section(system_content, 2600, preserve_ends=True)

    history = [
        {"role": msg.get("role", "user"), "content": str(msg.get("content", ""))}
        for msg in messages[1:-1]
        if msg.get("role") in {"user", "assistant"} and str(msg.get("content", "")).strip()
    ]
    compact_history = [
        {
            "role": msg["role"],
            "content": _truncate_prompt_section(msg["content"], 900, preserve_ends=False),
        }
        for msg in history[-2:]
    ]

    last_message = messages[-1]
    compact_last = {
        "role": last_message.get("role", "user"),
        "content": _truncate_prompt_section(str(last_message.get("content", "")), 1400, preserve_ends=True),
    }

    reduced = [{"role": "system", "content": compact_system}, *compact_history, compact_last]
    stats = {
        "system_chars": len(compact_system),
        "history_count": len(compact_history),
        "user_chars": len(compact_last["content"]),
    }
    return reduced, stats

def _should_inject_task_state(route_type: str, task_tracker, session_id: str) -> bool:
    if not task_tracker.has_tasks(session_id):
        return False
    if bool(getattr(cfg, "ENABLE_TASK_STALENESS_GUARD", True)) and task_tracker.is_stale(
        session_id,
        int(getattr(cfg, "TASK_STALE_SECONDS", 1800)),
    ):
        return route_type == "multi_step"
    return route_type in {"multi_step", "open_ended"}

def _is_correction_turn(message: str) -> bool:
    return bool(CORRECTION_QUERY_RE.search((message or "").strip()))


def _normalize_loop_mode(
    requested_mode: Any,
    *,
    use_planner: bool,
    has_orchestrator: bool,
    is_trivial_turn: bool,
    adapter: Optional[BaseLLMAdapter] = None,
    model: str = "",
) -> tuple[str, str]:
    raw = str(requested_mode or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "managed": "managed",
        "manager": "managed",
        "phased": "managed",
        "single": "single",
        "simple": "single",
        "direct": "single",
    }
    requested = aliases.get(raw, "auto")
    if is_trivial_turn or not has_orchestrator:
        effective = "single"
    elif requested == "managed":
        effective = "managed"
    elif requested == "single":
        effective = "single"
    elif _should_prefer_single_loop(adapter, model):
        effective = "single"
    else:
        effective = "managed" if use_planner else "single"
    return requested, effective

def _tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(
            {
                "tool": str(tool_name or ""),
                "arguments": arguments or {},
            },
            sort_keys=True,
            default=str,
        )
    except Exception:
        return f"{tool_name}:{repr(arguments)}"

def _is_retry_sensitive_tool(tool_name: str) -> bool:
    return str(tool_name or "").lower() in {"web_search", "web_fetch"}

def _looks_like_tool_instability(tool_name: str, content: str) -> bool:
    text = str(content or "").lower()
    if not text:
        return False
    if "tool_output_validation_error" in text:
        return True
    if "rate limit" in text or "too many requests" in text or "429" in text:
        return True
    if "network error" in text or "connection error" in text or "timed out" in text:
        return True
    if _is_retry_sensitive_tool(tool_name) and "returned no results" in text:
        return True
    return False

class AgentCore:

    def __init__(
        self,
        adapter:         OllamaAdapter,
        context_engine:  ContextEngine,
        session_manager: SessionManager,
        tool_registry:   ToolRegistry,
        middleware:      Optional[GuardrailMiddleware] = None,
        orchestrator:    Optional[AgenticOrchestrator] = None,
        default_model:   str = "llama3.2",
        embed_model:     str = "nomic-embed-text",
        default_system_prompt: str = None,
        workspace_path: Optional[str] = None,
        bootstrap_files: Optional[list[str]] = None,
        bootstrap_max_chars: int = 8000,
    ):
        self.adapter   = adapter
        self.ctx_eng   = context_engine
        self.sessions  = session_manager
        self.tools     = tool_registry
        self.middleware = middleware
        self.orch      = orchestrator
        self.def_model = default_model
        self.embed_model = embed_model
        self.def_system_prompt = default_system_prompt or DEFAULT_SYSTEM_PROMPT
        self.workspace_path = workspace_path
        self.bootstrap_files = list(bootstrap_files or [])
        self.bootstrap_max_chars = int(bootstrap_max_chars or 8000)

        self.circuit_breaker = CircuitBreaker(threshold=3)
        self.graph = SemanticGraph()
        self._v1_engine = context_engine
        self._v2_engine = None  # lazy init
        self._v3_engine = None  # lazy init
        self._last_context_compilation: dict = {}

    def set_context_engine(self, mode: str):
        """Switch between V1 (linear), V2 (convergent), and V3 (hybrid) context engines."""
        if mode == "v3":
            if self._v3_engine is None:
                self._v3_engine = ContextEngineV3(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=self.def_model,
                )
            self.ctx_eng = self._v3_engine
            print("[AgentCore] Context engine: V3 (Hybrid Durable + Convergent)")
            return

        if mode == "v2":
            if self._v2_engine is None:
                self._v2_engine = ContextEngineV2(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=self.def_model,
                )
            self.ctx_eng = self._v2_engine
            print("[AgentCore] Context engine: V2 (Convergent Graph)")
            return

        self.ctx_eng = self._v1_engine
        print("[AgentCore] Context engine: V1 (Linear Compression)")

    def _resolve_context_engine(self, mode: str) -> ContextEngine:
        """Return the correct context engine for this request without mutating shared request state."""
        if mode == "v3":
            if self._v3_engine is None:
                self._v3_engine = ContextEngineV3(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=self.def_model,
                )
            return self._v3_engine

        if mode == "v2":
            if self._v2_engine is None:
                self._v2_engine = ContextEngineV2(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=self.def_model,
                )
            return self._v2_engine

        return self._v1_engine

    def _is_session_interrupted(self, session_id: str) -> bool:
        checker = getattr(self.sessions, "is_interrupted", None)
        if not callable(checker):
            return False
        try:
            result = checker(session_id)
            return result is True or result == 1
        except Exception:
            return False

    def _clear_session_interrupt(self, session_id: str) -> None:
        clearer = getattr(self.sessions, "clear_interrupt", None)
        if not callable(clearer):
            return
        try:
            clearer(session_id)
        except Exception:
            pass

    def _direct_tool_hints(
        self,
        query: str,
        *,
        tools_list: list[dict],
        session_has_history: bool,
        current_fact_count: int,
        failed_tools: Optional[list[str]] = None,
    ) -> list[str]:
        """Deterministic tool hints for direct mode when planner is disabled."""
        available = {
            t.get("name")
            for t in tools_list
            if isinstance(t, dict) and isinstance(t.get("name"), str)
        }
        failed = set(failed_tools or [])
        q = (query or "").lower().strip()

        if not q:
            return []
        if re.fullmatch(r"(hi|hello|hey|yo|thanks|thank you|ok|okay)[!. ]*", q):
            return []

        suggestions: list[str] = []

        # Memory-first bias for ongoing conversations.
        if (
            bool(MEMORY_QUERY_RE.search(query))
            and "query_memory" in available
            and "query_memory" not in failed
        ):
            suggestions.append("query_memory")

        if re.search(r"https?://", q):
            if "web_fetch" in available and "web_fetch" not in failed:
                suggestions.append("web_fetch")
        elif (
            re.search(r"\b[a-z0-9][a-z0-9.-]*\.[a-z]{2,}\b", q)
            and re.search(r"\b(research|investigate|evaluate|assess|pricing|price|plan|cost|review|good|trust|trustworthy|recommend|intel|report)\b", q)
        ):
            for candidate in ("web_search", "web_fetch"):
                if candidate in available and candidate not in failed:
                    suggestions.append(candidate)
        elif re.search(r"\b(research|gather intel|intel|investigate|analyze|analysis|assess|evaluate|compare|deep[- ]?dive|report)\b", q):
            if "web_search" in available and "web_search" not in failed:
                suggestions.append("web_search")
        elif re.search(r"\b(news|latest|current|today|who is|what is|when is|where is)\b", q):
            if "web_search" in available and "web_search" not in failed:
                suggestions.append("web_search")

        if re.search(r"\b(weather|temperature|forecast)\b", q):
            if "weather_fetch" in available and "weather_fetch" not in failed:
                suggestions.append("weather_fetch")

        if re.search(r"\b(image|photo|picture|logo)\b", q):
            if "image_search" in available and "image_search" not in failed:
                suggestions.append("image_search")

        if re.search(r"\b(create|build|generate).*\b(html|dashboard|app|webpage)\b", q):
            for candidate in ("generate_app", "file_create"):
                if candidate in available and candidate not in failed:
                    suggestions.append(candidate)
                    break

        ordered = []
        seen = set()
        for name in suggestions:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    async def chat_stream(
        self,
        user_message:   str,
        session_id:     Optional[str] = None,
        agent_id:       str = "default",
        model:          Optional[str] = None,
        system_prompt:  Optional[str] = None,
        search_backend: Optional[str] = None,
        search_engine:  Optional[str] = None, # Added param
        images:         Optional[list[str]] = None,
        force_memory:   bool = False,
        forced_tools:   Optional[list[str]] = None,
        **kw
    ) -> AsyncIterator[dict]:

        model         = model or self.def_model
        system_prompt = system_prompt or self.def_system_prompt
        owner_id      = kw.get("owner_id")
        run_id        = kw.get("run_id") or str(uuid.uuid4())
        parent_run_id = kw.get("parent_run_id")
        agent_revision = kw.get("agent_revision")

        session = self.sessions.get_or_create(
            session_id=session_id,
            model=model,
            system_prompt=system_prompt,
            agent_id=agent_id,
            owner_id=owner_id,
        )
        sid = session.id
        run_store = get_run_store()
        run_store.start_run(
            run_id=run_id,
            session_id=sid,
            agent_id=agent_id,
            owner_id=owner_id,
            agent_revision=agent_revision,
            parent_run_id=parent_run_id,
            model=model,
        )

        run_status = "completed"
        run_error: Optional[str] = None

        def _emit(event: dict) -> dict:
            payload = dict(event)
            payload.setdefault("run_id", run_id)
            return payload

        def trace_event(event_type: str, data: dict):
            _trace(
                agent_id,
                sid,
                event_type,
                data,
                run_id=run_id,
                owner_id=owner_id,
            )

        def trace_phase(phase: ContextPhase | str, status: str, **data):
            phase_name = phase.value if isinstance(phase, ContextPhase) else str(phase)
            payload = {
                "phase": phase_name,
                "status": status,
            }
            payload.update(data)
            trace_event("run_phase", payload)

        def finish_run() -> None:
            try:
                run_store.finish_run(run_id, status=run_status, error=run_error)
            except Exception:
                pass

        # Resolve context engine per request to avoid cross-session races.
        ctx_mode = getattr(session, "context_mode", "v1")
        ctx_eng = self._resolve_context_engine(ctx_mode)
        
        # New: Layer settings (passed via kwargs or extracted from request)
        use_planner    = _to_bool(kw.get("use_planner", True))
        planner_model  = kw.get("planner_model")
        context_model  = kw.get("context_model") or getattr(cfg, "CONTEXT_MODEL", None) or model
        requested_loop_mode = kw.get("loop_mode", "auto")
        max_tool_calls = _normalize_optional_limit(kw.get("max_tool_calls"), minimum=1, maximum=24)
        max_turns = _normalize_optional_limit(kw.get("max_turns"), minimum=2, maximum=12)

        # BUG FIX: Update session model when user switches models mid-conversation
        if session.model != model:
            log("agent", sid, f"Model switched: {session.model} → {model}")
            session.model = model
            self.sessions.update_model(sid, model)

        yield _emit(_ev("session", session_id=sid, agent_id=session.agent_id))

        log("agent", sid, f"Turn started · agent={agent_id} · model={model}")

        full_response = ""
        error_msg     = None
        unified_hits: list[dict] = []
        incomplete_turn = False
        failed_turn_recorded = False
        successful_tool_evidence: list[str] = []
        all_tool_results: list[dict[str, Any]] = []
        loop_checkpoint: Optional[LoopCheckpoint] = run_store.latest_checkpoint(run_id)

        async with session.lock:
            # ── Dynamic Adapter Selection ─────────────────────────────────────
            # Smart switching logic with resolution priority:
            # 1. Explicit model request
            # 2. Session model
            # 3. Agent default model
            # 4. System default (llama3.2)
            
            resolved_model = model or session.model or self.def_model or "llama3.2"
            known_providers = {
                "ollama",
                "openai",
                "local_openai",
                "lmstudio",
                "llamacpp",
                "groq",
                "gemini",
                "anthropic",
            }
            resolved_lower = resolved_model.lower()
            has_provider_prefix = (
                any(resolved_lower.startswith(f"{p}:") for p in known_providers)
                or any(resolved_lower.startswith(f"{p}/") for p in known_providers)
                or resolved_lower in known_providers
            )
            # If the model has no provider prefix, keep the agent's current adapter.
            # This preserves provider context for bare model ids like "llama-3.3-70b-versatile".
            if has_provider_prefix:
                current_use_adapter = create_adapter(provider=resolved_model)
            else:
                current_use_adapter = self.adapter
            
            # Use clean model name for API calls
            clean_model = strip_provider_prefix(resolved_model)
            self.adapter = current_use_adapter
            
            # ── CRITICAL: Propagate dynamic adapter to subsystems ─────────
            # Without this, ContextEngine and Orchestrator remain stuck on 
            # the original OllamaAdapter even when user selects Groq/Claude/Gemini.
            ctx_eng.set_adapter(current_use_adapter)
            if self.orch:
                self.orch.set_adapter(current_use_adapter)
            
            # ── Sanitization ─────────
            user_message = sanitize_user_message(user_message)
            is_trivial_turn = _is_trivial_query(user_message)
            is_correction_turn = _is_correction_turn(user_message)

            current_facts = self.graph.get_current_facts(sid, owner_id=owner_id)
            anticipated_deterministic_facts, anticipated_deterministic_voids = extract_user_stated_fact_updates(
                user_message,
                current_facts=current_facts,
            )
            failed_tools = self.circuit_breaker.get_failed_tools(sid)
            task_tracker = get_session_task_tracker()
            task_bootstrap_done = task_tracker.has_tasks(sid)
            task_summary = task_tracker.summary(sid)
            route_type = _classify_route(
                user_message,
                session_has_history=bool(getattr(session, "full_history", [])),
                current_fact_count=len(current_facts),
                active_task_count=task_summary.get("active_count", 0),
            ) if bool(getattr(cfg, "ENABLE_DETERMINISTIC_ROUTING", True)) else "open_ended"
            direct_fact_memory_only = (
                route_type == "direct_fact"
                and should_answer_direct_fact_from_memory(user_message, current_facts)
            )
            blocked_tools_for_turn: set[str] = set()
            if anticipated_deterministic_facts or anticipated_deterministic_voids:
                blocked_tools_for_turn.add("store_memory")
            retrieval_policy = _build_retrieval_policy(route_type, force_memory=force_memory)
            task_state = ""
            requested_loop_mode, effective_loop_mode = _normalize_loop_mode(
                requested_loop_mode,
                use_planner=use_planner,
                has_orchestrator=bool(self.orch),
                is_trivial_turn=is_trivial_turn,
                adapter=current_use_adapter,
                model=resolved_model,
            )
            model_profile = _classify_model_profile(current_use_adapter, resolved_model)
            manager_loop_enabled = effective_loop_mode == "managed"
            forced_tools = _normalize_forced_tools_for_task_state(
                forced_tools,
                has_tasks=task_tracker.has_tasks(sid),
            ) or None

            trace_event("route_decision", {
                "route_type": route_type,
                "session_has_history": bool(getattr(session, "full_history", [])),
                "current_fact_count": len(current_facts),
                "active_task_count": task_summary.get("active_count", 0),
                "query": user_message[:200],
            })
            trace_event("retrieval_policy", retrieval_policy)
            trace_event("direct_fact_memory_guard", {
                "enabled": direct_fact_memory_only,
                "fact_count": len(current_facts),
                "query": user_message[:200],
            })
            trace_event("runtime_loop_mode", {
                "requested": requested_loop_mode,
                "effective": effective_loop_mode,
                "use_planner": bool(use_planner),
                "has_orchestrator": bool(self.orch),
                "is_trivial_turn": is_trivial_turn,
                "model_profile": model_profile,
            })
            trace_event("model_execution_profile", {
                "model": resolved_model,
                "profile": model_profile,
                "native_tools": _should_use_native_tools(current_use_adapter, resolved_model),
            })

            def record_failed_user_turn(note: str = "") -> None:
                nonlocal failed_turn_recorded
                if failed_turn_recorded:
                    return
                try:
                    last_message = session.full_history[-1] if session.full_history else None
                    if not (
                        isinstance(last_message, dict)
                        and last_message.get("role") == "user"
                        and str(last_message.get("content") or "") == user_message
                    ):
                        self.sessions.append_message(sid, "user", user_message)
                    if note:
                        trace_event("failed_turn_recorded", {"note": note[:240]})
                    failed_turn_recorded = True
                except Exception as persist_err:
                    log("agent", sid, f"Failed to record interrupted turn: {persist_err}", level="warn")

            def save_loop_checkpoint(
                phase: str,
                *,
                tool_turn: int = 0,
                status: str = "",
                strategy: str = "",
                notes: str = "",
                confidence: Optional[float] = None,
                tools: Optional[list[str]] = None,
                tool_results: Optional[list[dict[str, Any]]] = None,
                candidate_facts: Optional[list[str]] = None,
            ) -> LoopCheckpoint:
                nonlocal loop_checkpoint
                loop_checkpoint = run_store.save_checkpoint(
                    run_id=run_id,
                    phase=phase,
                    tool_turn=tool_turn,
                    status=status,
                    strategy=strategy,
                    notes=notes,
                    confidence=confidence,
                    tools=tools,
                    tool_results=tool_results,
                    candidate_facts=candidate_facts,
                )
                trace_event("loop_checkpoint", {
                    "phase": loop_checkpoint.phase,
                    "checkpoint_id": loop_checkpoint.checkpoint_id,
                    "tool_turn": loop_checkpoint.tool_turn,
                    "status": loop_checkpoint.status,
                    "tool_count": len(loop_checkpoint.tools or []),
                    "candidate_fact_count": len(loop_checkpoint.candidate_facts or []),
                })
                return loop_checkpoint

            # ── Agentic Planning (Manager Agent) ──────────────────────────────
            should_call_planner = (
                manager_loop_enabled
                and not forced_tools
                and not is_trivial_turn
                and route_type in {"multi_step", "open_ended"}
            )
            if should_call_planner:
                trace_phase(ContextPhase.PLANNING, "start", route_type=route_type)
                log("agent", sid, "Planner active · determining strategy...", level="info")
                # Use override planner_model if provided
                use_p_model = planner_model or DEFAULT_PLANNER_FALLBACK_MODEL
                planning_context = self._compile_phase_context(
                    phase=ContextPhase.PLANNING,
                    system_prompt=session.system_prompt,
                    context=session.compressed_context,
                    user_message=user_message,
                    first_message=session.first_message,
                    message_count=session.message_count,
                    unified_hits=unified_hits,
                    force_memory=force_memory,
                    forced_tools=forced_tools,
                    current_facts=current_facts,
                    task_state=task_state,
                    candidate_context=session.candidate_context,
                    direct_fact_memory_only=direct_fact_memory_only,
                    loop_checkpoint=loop_checkpoint,
                    correction_turn=is_correction_turn,
                    route_type=route_type,
                    ctx_engine=ctx_eng,
                    model_profile=model_profile,
                )
                trace_event("phase_context", planning_context)
                structured_plan = None
                plan_with_context_fn = getattr(self.orch, "plan_with_context", None)
                if callable(plan_with_context_fn):
                    maybe_structured = plan_with_context_fn(
                        query=user_message,
                        tools_list=self.tools.list_tools(),
                        model=use_p_model,
                        session_has_history=bool(getattr(session, "full_history", [])),
                        current_fact_count=len(current_facts),
                        failed_tools=failed_tools,
                        compiled_context=planning_context.get("content", ""),
                    )
                    if inspect.isawaitable(maybe_structured):
                        maybe_structured = await maybe_structured

                    if isinstance(maybe_structured, dict):
                        structured_plan = maybe_structured

                clues: list[str] = []
                if structured_plan is not None:
                    planned_tools = []
                    for t in structured_plan.get("tools", []):
                        if isinstance(t, dict) and isinstance(t.get("name"), str):
                            t_name = t.get("name")
                            planned_tools.append(t_name)
                            t_clue = str(t.get("target_argument_clue", "")).strip()
                            if t_clue:
                                clues.append(f"[{t_name} target: {t_clue}]")
                else:
                    planned_tools = []
                    plan_fn = getattr(self.orch, "plan", None)
                    if callable(plan_fn):
                        maybe_plan = plan_fn(user_message, self.tools.list_tools(), model=use_p_model)
                        if inspect.isawaitable(maybe_plan):
                            maybe_plan = await maybe_plan
                        if isinstance(maybe_plan, list):
                            planned_tools = [name for name in maybe_plan if isinstance(name, str)]

                    structured_plan = {
                        "strategy": f"Using {', '.join(planned_tools)} to resolve query.",
                        "tools": [{"name": name, "priority": "medium", "reason": "legacy planner"} for name in planned_tools],
                        "force_memory": False,
                        "confidence": None,
                    }
                if planned_tools:
                    planned_tools = _normalize_forced_tools_for_task_state(
                        planned_tools,
                        has_tasks=task_tracker.has_tasks(sid),
                    )
                    strategy = structured_plan.get("strategy") or f"Using {', '.join(planned_tools)} to resolve query."
                    if clues:
                        strategy += " | EXECUTION CLUES: " + " ".join(clues)
                    yield _emit(_ev("plan", strategy=strategy, tools=planned_tools, confidence=structured_plan.get("confidence")))
                    forced_tools = planned_tools
                if structured_plan.get("force_memory"):
                    force_memory = True
                    retrieval_policy = _build_retrieval_policy(route_type, force_memory=True)
                save_loop_checkpoint(
                    "planning",
                    status="planned",
                    strategy=str(structured_plan.get("strategy") or ""),
                    confidence=structured_plan.get("confidence"),
                    tools=list(planned_tools),
                    candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
                )
                trace_phase(
                    ContextPhase.PLANNING,
                    "complete",
                    route_type=route_type,
                    tool_count=len(planned_tools),
                    force_memory=bool(structured_plan.get("force_memory")),
                )
            elif not manager_loop_enabled and not is_trivial_turn:
                log("agent", sid, "Planner bypassed (Direct Mode)")
                if not forced_tools:
                    direct_tools = self._direct_tool_hints(
                        user_message,
                        tools_list=self.tools.list_tools(),
                        session_has_history=bool(getattr(session, "full_history", [])),
                        current_fact_count=len(current_facts),
                        failed_tools=failed_tools,
                    )
                    if direct_tools:
                        forced_tools = direct_tools
                        yield _emit(_ev(
                            "plan",
                            strategy="Planner disabled; using direct-mode tool hints.",
                            tools=direct_tools,
                            confidence=0.35,
                        ))
            elif manager_loop_enabled:
                save_loop_checkpoint(
                    "planning",
                    status="deterministic_route",
                    strategy=f"Planner skipped for route {route_type}; runtime will follow deterministic policy.",
                    confidence=0.8 if route_type in {"direct_fact", "url_fetch", "memory_recall"} else 0.4,
                    tools=list(forced_tools or []),
                    candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
                )

            if route_type == "multi_step" and not task_tracker.has_tasks(sid):
                if "todo_write" in {t.get("name") for t in self.tools.list_tools()}:
                    existing = list(forced_tools or [])
                    if "todo_write" not in existing:
                        forced_tools = ["todo_write"] + existing
                yield _emit(_ev(
                    "plan",
                    strategy="Multi-step workflow detected; initialize task state before continuing.",
                    tools=list(forced_tools or []),
                    confidence=0.9,
                ))

            async def manager_observe_step(
                *,
                tool_turn: int,
                tool_calls: list[dict[str, Any]],
                tool_results: list[dict[str, Any]],
            ) -> dict[str, Any]:
                nonlocal loop_checkpoint
                if not manager_loop_enabled:
                    return {}
                observe_with_context_fn = getattr(self.orch, "observe_with_context", None)
                if not callable(observe_with_context_fn):
                    return {}

                trace_phase("observation", "start", tool_turn=tool_turn, result_count=len(tool_results))
                observation_context = self._compile_phase_context(
                    phase=ContextPhase.VERIFICATION,
                    system_prompt=session.system_prompt,
                    context=session.compressed_context,
                    user_message=user_message,
                    first_message=session.first_message,
                    message_count=session.message_count,
                    unified_hits=unified_hits,
                    force_memory=force_memory,
                    forced_tools=forced_tools,
                    current_facts=current_facts,
                    task_state=task_state,
                    candidate_context=session.candidate_context,
                    direct_fact_memory_only=direct_fact_memory_only,
                    loop_checkpoint=loop_checkpoint,
                    correction_turn=is_correction_turn,
                    route_type=route_type,
                    ctx_engine=ctx_eng,
                    model_profile=model_profile,
                )
                observation_context["phase"] = "observation"
                trace_event("phase_context", observation_context)

                observation = observe_with_context_fn(
                    query=user_message,
                    tools_list=self.tools.list_tools(),
                    tool_results=tool_results,
                    model=planner_model or DEFAULT_PLANNER_FALLBACK_MODEL,
                    compiled_context=observation_context.get("content", ""),
                )
                if inspect.isawaitable(observation):
                    observation = await observation
                if not isinstance(observation, dict):
                    observation = {}
                trace_event("manager_observation", {
                    "tool_turn": tool_turn,
                    **observation,
                })
                save_loop_checkpoint(
                    "observation",
                    tool_turn=tool_turn,
                    status=str(observation.get("status") or ""),
                    strategy=str(observation.get("strategy") or ""),
                    notes=str(observation.get("notes") or ""),
                    confidence=observation.get("confidence"),
                    tools=[
                        item.get("name")
                        for item in (observation.get("tools") or [])
                        if isinstance(item, dict) and isinstance(item.get("name"), str)
                    ],
                    tool_results=_tool_result_previews(tool_results),
                    candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
                )
                trace_phase(
                    "observation",
                    "complete",
                    tool_turn=tool_turn,
                    decision_status=observation.get("status"),
                    tool_count=len(observation.get("tools") or []),
                    confidence=observation.get("confidence"),
                )
                return observation

            if bool(getattr(cfg, "ENABLE_TASK_STALENESS_GUARD", True)) and task_tracker.is_stale(
                sid,
                int(getattr(cfg, "TASK_STALE_SECONDS", 1800)),
            ) and route_type not in {"multi_step", "open_ended"}:
                task_tracker.clear(sid)
                task_summary = task_tracker.summary(sid)

            task_state = task_tracker.render(sid) if _should_inject_task_state(route_type, task_tracker, sid) else ""
            trace_event("task_policy", {
                "route_type": route_type,
                "has_tasks": task_tracker.has_tasks(sid),
                "task_summary": task_summary,
                "task_state_injected": bool(task_state),
            })

            if not is_trivial_turn and bool(retrieval_policy.get("should_retrieve")):
                unified_hits = await self._unified_retrieve(
                    query=user_message,
                    session_id=sid,
                    agent_id=session.agent_id,
                    top_n=int(retrieval_policy.get("top_n", 5)),
                    use_vector=bool(retrieval_policy.get("use_vector")),
                    use_graph=bool(retrieval_policy.get("use_graph")),
                    use_session_rag=bool(retrieval_policy.get("use_session_rag")),
                    owner_id=owner_id,
                )
            if unified_hits:
                keys = [h.get("key", "?") for h in unified_hits[:3]]
                log("rag", sid, f"Retrieved {len(unified_hits)} unified hits: {', '.join(keys)}", level="ok")
                trace_event("retrieval_sources_used", {
                    "route_type": route_type,
                    "sources": sorted({h.get("source", "unknown") for h in unified_hits}),
                    "count": len(unified_hits),
                })
            else:
                if is_trivial_turn:
                    log("rag", sid, "Skipping retrieval for trivial turn")
                elif not bool(retrieval_policy.get("should_retrieve")):
                    log("rag", sid, f"Skipping retrieval for route {route_type}")
                else:
                    log("rag", sid, "No unified memory hits found")

            trace_phase(ContextPhase.ACTING, "start", route_type=route_type)
            save_loop_checkpoint(
                "acting",
                tool_turn=loop_checkpoint.tool_turn if loop_checkpoint else 0,
                status="running",
                strategy=f"Actor pass starting for route {route_type}.",
                tools=list(forced_tools or []),
                candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
            )
            messages = self._build_messages(
                system_prompt=session.system_prompt,
                context=session.compressed_context,
                sliding_window=session.sliding_window,
                user_message=user_message,
                first_message=session.first_message,
                message_count=session.message_count,
                unified_hits=unified_hits,
                force_memory=force_memory,
                forced_tools=forced_tools,
                current_facts=current_facts,
                task_state=task_state,
                candidate_context=session.candidate_context,
                direct_fact_memory_only=direct_fact_memory_only,
                loop_checkpoint=loop_checkpoint,
                correction_turn=is_correction_turn,
                route_type=route_type,
                ctx_engine=ctx_eng,
                phase=ContextPhase.ACTING,
                model_profile=model_profile,
            )
            if self._last_context_compilation:
                trace_event("compiled_context", self._last_context_compilation)

            system_chars = len(messages[0]["content"]) if messages else 0
            history_chars = sum(len(m.get("content", "")) for m in messages[1:-1]) if len(messages) > 2 else 0
            prompt_tokens_est = sum(len(m["content"]) for m in messages) // 4
            log(
                "llm",
                sid,
                f"Prompt built · {len(messages)} messages · ~{prompt_tokens_est} tokens est.",
                message_count=len(messages),
                estimated_tokens=prompt_tokens_est,
                system_chars=system_chars,
                history_chars=history_chars,
                rag_anchor_count=len(unified_hits or []),
            )

            trace_event("prompt_components", {
                "model": model,
                "model_profile": model_profile,
                "message_count": len(messages),
                "estimated_tokens": prompt_tokens_est,
                "system_chars": system_chars,
                "history_chars": history_chars,
                "rag_anchor_count": len(unified_hits or []),
                "fact_count": len(current_facts or []),
            })

            # Trace: log the actual prompt chain sent to LLM with flags
            trace_event("prompt_chain", {
                "model": model,
                "model_profile": model_profile,
                "message_count": len(messages),
                "estimated_tokens": prompt_tokens_est,
                "has_rag": len(unified_hits) > 0,
                "has_facts": len(current_facts) > 0,
                "has_memory": len(session.compressed_context) > 0,
                "system_prompt_preview": messages[0]["content"][:300] if messages else "",
                "user_message": user_message[:200],
            })
            trace_phase(ContextPhase.ACTING, "complete", message_count=len(messages), estimated_tokens=prompt_tokens_est)

            try:
                async for event in self._agent_loop(
                    model=clean_model,
                    messages=messages,
                    adapter=current_use_adapter,
                    search_backend=search_backend,
                    search_engine=search_engine, # Pass down
                    images=images,
                    agent_id=session.agent_id,
                    session_id=sid,
                    route_type=route_type,
                    forced_tools=forced_tools,
                    user_message=user_message,
                    run_id=run_id,
                    owner_id=owner_id,
                    observe_step=manager_observe_step if manager_loop_enabled else None,
                    max_tool_calls=max_tool_calls,
                    max_llm_turns=max_turns,
                    prior_tool_results=all_tool_results,
                    model_profile=model_profile,
                    suppress_tools=direct_fact_memory_only,
                    blocked_tools=blocked_tools_for_turn,
                ):
                    yield _emit(event)
                    if event["type"] == "token":
                        full_response += event["content"]
                    elif event["type"] == "tool_result" and event.get("success"):
                        content = str(event.get("content") or "").strip()
                        if content:
                            successful_tool_evidence.append(content[:4000])
                        all_tool_results.append({
                            "tool_name": event.get("tool_name") or event.get("tool"),
                            "success": bool(event.get("success")),
                            "content": event.get("content") or "",
                        })
                    elif event["type"] == "tool_result":
                        all_tool_results.append({
                            "tool_name": event.get("tool_name") or event.get("tool"),
                            "success": bool(event.get("success")),
                            "content": event.get("content") or "",
                        })
                    elif event["type"] == "_retract_response":
                        # BUG FIX: clean tool-call JSON from full_response
                        full_response = event.get("clean_response", full_response)
                    elif event["type"] == "_tool_budget_exhausted":
                        incomplete_turn = True
                    elif event["type"] == "error":
                        error_msg = event["message"]
                save_loop_checkpoint(
                    "acting",
                    tool_turn=loop_checkpoint.tool_turn if loop_checkpoint else 0,
                    status="complete",
                    notes=f"tool_results={len(all_tool_results)}",
                    tools=list(forced_tools or []),
                    tool_results=_tool_result_previews(all_tool_results),
                    candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
                )
            except (RateLimitError, ProviderError) as e:
                # ── Tiered Fallback: Cloud → Local ─────────────────────────────
                # If cloud provider is down or limited, fallback to Ollama.
                if not isinstance(current_use_adapter, OllamaAdapter):
                    log("llm", sid, f"Provider failed ({type(e).__name__}) — triggering local fallback", level="warn")
                    yield _emit(_ev("error", message=f"Cloud provider is temporarily unavailable ({str(e)}). Falling back to local Ollama..."))
                    
                    fallback_adapter = OllamaAdapter()
                    fallback_model   = "llama3.2" # Default reliable local model
                    fallback_max_tool_calls = min(max_tool_calls or 2, 2)
                    fallback_max_turns = min(max_turns or 2, 2)
                    
                    try:
                        async for event in self._agent_loop(
                            model=fallback_model,
                            messages=messages,
                            adapter=fallback_adapter,
                            search_backend=search_backend,
                            search_engine=search_engine,
                            images=images,
                            agent_id=session.agent_id,
                            session_id=sid,
                            route_type=route_type,
                            forced_tools=forced_tools,
                            user_message=user_message,
                            run_id=run_id,
                            owner_id=owner_id,
                            observe_step=None,
                            max_tool_calls=fallback_max_tool_calls,
                            max_llm_turns=fallback_max_turns,
                            prior_tool_results=all_tool_results,
                            model_profile=_classify_model_profile(fallback_adapter, fallback_model),
                        ):
                            yield _emit(event)
                            if event["type"] == "token":
                                full_response += event["content"]
                            elif event["type"] == "tool_result" and event.get("success"):
                                content = str(event.get("content") or "").strip()
                                if content:
                                    successful_tool_evidence.append(content[:4000])
                                all_tool_results.append({
                                    "tool_name": event.get("tool_name") or event.get("tool"),
                                    "success": bool(event.get("success")),
                                    "content": event.get("content") or "",
                                })
                            elif event["type"] == "tool_result":
                                all_tool_results.append({
                                    "tool_name": event.get("tool_name") or event.get("tool"),
                                    "success": bool(event.get("success")),
                                    "content": event.get("content") or "",
                                })
                            elif event["type"] == "_retract_response":
                                full_response = event.get("clean_response", full_response)
                            elif event["type"] == "_tool_budget_exhausted":
                                incomplete_turn = True
                        
                        # If we reached here, fallback succeeded
                        log("llm", sid, "Local fallback successful", level="ok")
                    except Exception as fallback_err:
                        log("llm", sid, f"Fallback failed: {fallback_err}", level="error")
                        run_status = "failed"
                        run_error = f"Fallback also failed: {fallback_err}"
                        record_failed_user_turn(run_error)
                        yield _emit(_ev("error", message=f"Fallback also failed: {fallback_err}"))
                else:
                    log("llm", sid, f"Local provider error: {e}", level="error")
                    run_status = "failed"
                    run_error = str(e)
                    record_failed_user_turn(run_error)
                    yield _emit(_ev("error", message=str(e)))
                finish_run()
                return
            except LLMError as e:
                log("llm", sid, f"LLM error: {e}", level="error")
                run_status = "failed"
                run_error = str(e)
                record_failed_user_turn(run_error)
                yield _emit(_ev("error", message=str(e)))
                finish_run()
                return
            except asyncio.TimeoutError:
                log("llm", sid, "LLM generation timed out after 60s", level="error")
                run_status = "failed"
                run_error = "Generation timed out. The model might be overloaded or the connection was lost."
                record_failed_user_turn(run_error)
                yield _emit(_ev("error", message="Generation timed out. The model might be overloaded or the connection was lost."))
                finish_run()
                return
            except Exception as e:
                log("agent", sid, f"Unexpected loop error: {e}", level="error")
                run_status = "failed"
                run_error = f"Internal Error: {e}"
                record_failed_user_turn(run_error)
                yield _emit(_ev("error", message=f"Internal Error: {e}"))
                finish_run()
                return

        if error_msg:
            lowered = (error_msg or "").lower()
            run_status = "interrupted" if "stopped" in lowered or "interrupt" in lowered else "failed"
            run_error = error_msg
            record_failed_user_turn(error_msg)
            finish_run()
            return

        if ENABLE_MANIFEST_PROTOCOL:
            try:
                from engine.manifest_parser import get_manifest_parser

                parser = get_manifest_parser()
                manifest = parser.extract(full_response)
                if manifest:
                    await parser.store(
                        manifest,
                        session_id=sid,
                        turn=session.message_count + 1,
                        agent_id=agent_id,
                        owner_id=owner_id,
                        run_id=run_id,
                        user_message=user_message,
                        grounding_text="\n".join(successful_tool_evidence),
                    )
                    full_response = parser.strip(full_response)
            except Exception as e:
                log("agent", sid, f"Manifest parse/store failed: {e}", level="warn")

        # BUG FIX: strip any remaining tool-call JSON and internal reasoning from full_response before storing
        clean_response = self._strip_internal_execution_chatter(
            self._strip_tool_json(self._strip_reasoning(full_response)),
            has_tool_results=bool(all_tool_results),
        )
        
        # — ENHANCEMENT: Automated Citation Verification —
        # Detects if the model contradicted historical facts injected via RAG
        if unified_hits:
            citations_valid = self._verify_citations(clean_response, unified_hits)
            if not citations_valid:
                log("rag", sid, "Citation contradiction detected — appending soft correction", level="warn")
                clean_response += (
                    "\n\n[Note: Some details above may conflict with previously stored context. "
                    "Please prioritize established facts from the conversation memory.]"
                )
        
        # Guard: If response is only tool calls, we still want a placeholder for continuity
        if not clean_response and full_response:
            clean_response = "[Tool Execution Turn]"

        verify_with_context_fn = getattr(self.orch, "verify_with_context", None) if self.orch else None
        if manager_loop_enabled and all_tool_results and callable(verify_with_context_fn):
            trace_phase(ContextPhase.VERIFICATION, "start", result_count=len(all_tool_results))
            verification_context = self._compile_phase_context(
                phase=ContextPhase.VERIFICATION,
                system_prompt=session.system_prompt,
                context=session.compressed_context,
                user_message=user_message,
                first_message=session.first_message,
                message_count=session.message_count,
                unified_hits=unified_hits,
                force_memory=force_memory,
                forced_tools=forced_tools,
                current_facts=current_facts,
                task_state=task_state,
                candidate_context=session.candidate_context,
                direct_fact_memory_only=direct_fact_memory_only,
                loop_checkpoint=loop_checkpoint,
                correction_turn=is_correction_turn,
                route_type=route_type,
                ctx_engine=ctx_eng,
            )
            trace_event("phase_context", verification_context)
            verification = verify_with_context_fn(
                query=user_message,
                response=clean_response,
                tool_results=all_tool_results,
                model=planner_model or DEFAULT_PLANNER_FALLBACK_MODEL,
                compiled_context=verification_context.get("content", ""),
            )
            if inspect.isawaitable(verification):
                verification = await verification
            if not isinstance(verification, dict):
                verification = {"supported": True, "issues": [], "confidence": 0.0}
            trace_event("verification_result", verification)
            try:
                evaluation = run_store.save_eval(
                    run_id=run_id,
                    session_id=sid,
                    owner_id=owner_id,
                    eval_type="response_support",
                    phase="verification",
                    passed=bool(verification.get("supported", True)),
                    score=verification.get("confidence"),
                    detail="; ".join(str(item) for item in (verification.get("issues") or [])[:3]),
                    metadata={
                        "issue_count": len(verification.get("issues") or []),
                        "issues": [str(item) for item in (verification.get("issues") or [])[:5]],
                    },
                )
                trace_event("run_eval", {
                    "eval_id": evaluation.eval_id,
                    "phase": evaluation.phase,
                    "eval_type": evaluation.eval_type,
                    "passed": evaluation.passed,
                    "score": evaluation.score,
                })
            except Exception as e:
                log("run", sid, f"Failed to persist run eval: {e}", level="warn")
            if not verification.get("supported", True):
                incomplete_turn = True
                issues = verification.get("issues") or []
                if issues:
                    clean_response += (
                        "\n\n[Verification warning: "
                        + "; ".join(str(item) for item in issues[:2])
                        + "]"
                    )
                yield _emit(_ev("verification_warning", issues=issues, confidence=verification.get("confidence")))
            save_loop_checkpoint(
                "verification",
                tool_turn=loop_checkpoint.tool_turn if loop_checkpoint else 0,
                status="supported" if verification.get("supported", True) else "unsupported",
                strategy=loop_checkpoint.strategy if loop_checkpoint else "",
                notes="; ".join(str(item) for item in (verification.get("issues") or [])[:3]),
                confidence=verification.get("confidence"),
                tools=list(loop_checkpoint.tools or []) if loop_checkpoint else [],
                tool_results=list(loop_checkpoint.tool_results or []) if loop_checkpoint else _tool_result_previews(all_tool_results),
                candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
            )
            trace_phase(
                ContextPhase.VERIFICATION,
                "complete",
                supported=bool(verification.get("supported", True)),
                issue_count=len(verification.get("issues") or []),
                confidence=verification.get("confidence"),
            )

        is_first_exchange = self.sessions.append_message(sid, "user", user_message)
        self.sessions.append_message(sid, "assistant", clean_response)

        log("agent", sid, f"Response complete · {len(clean_response)} chars · first_exchange={is_first_exchange}")

        # Trace: log the stored response
        trace_event("assistant_response", {
            "content": clean_response[:500],
            "raw_length": len(full_response),
            "clean_length": len(clean_response),
            "was_cleaned": len(full_response) != len(clean_response),
        })
        if run_id and clean_response.strip():
            try:
                response_text = clean_response.strip()
                response_artifact = run_store.save_artifact(
                    run_id=run_id,
                    session_id=sid,
                    owner_id=owner_id,
                    tool_name="assistant_response",
                    artifact_type="assistant_response",
                    label="final_response",
                    content_hash=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                    size_bytes=len(response_text.encode("utf-8")),
                    preview=response_text[:600],
                    metadata={
                        "clean_length": len(response_text),
                        "route_type": route_type,
                        "managed_loop": manager_loop_enabled,
                    },
                )
                trace_event("run_artifact", {
                    "artifact_id": response_artifact.artifact_id,
                    "artifact_type": response_artifact.artifact_type,
                    "label": response_artifact.label,
                    "size_bytes": response_artifact.size_bytes,
                })
            except Exception as e:
                log("run", sid, f"Failed to persist assistant response artifact: {e}", level="warn")
        response_context = self._compile_phase_context(
            phase=ContextPhase.RESPONSE,
            system_prompt=session.system_prompt,
            context=session.compressed_context,
            user_message=user_message,
            first_message=session.first_message,
            message_count=session.message_count,
            unified_hits=unified_hits,
            force_memory=force_memory,
            forced_tools=forced_tools,
            current_facts=current_facts,
            task_state=task_state,
            candidate_context=session.candidate_context,
            direct_fact_memory_only=direct_fact_memory_only,
            loop_checkpoint=loop_checkpoint,
            correction_turn=is_correction_turn,
            route_type=route_type,
            ctx_engine=ctx_eng,
            model_profile=model_profile,
        )
        trace_event("phase_context", response_context)
        trace_phase(ContextPhase.RESPONSE, "complete", response_length=len(clean_response))
        trace_event("memory_write_policy", {
            "route_type": route_type,
            "should_compress": (not is_trivial_turn) and not incomplete_turn,
            "response_length": len(clean_response),
            "retrieval_used": len(unified_hits),
            "incomplete_turn": incomplete_turn,
        })

        should_compress = (not is_trivial_turn) and (not incomplete_turn) and _should_compress_exchange(session, is_first_exchange)
        deterministic_keyed_facts, deterministic_voids = anticipated_deterministic_facts, anticipated_deterministic_voids
        trace_event("deterministic_fact_extractor", {
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
            "voids": deterministic_voids,
        })
        memory_commit_context = self._compile_phase_context(
            phase=ContextPhase.MEMORY_COMMIT,
            system_prompt=session.system_prompt,
            context=session.compressed_context,
            user_message=user_message,
            first_message=session.first_message,
            message_count=session.message_count,
            unified_hits=unified_hits,
            force_memory=force_memory,
            forced_tools=forced_tools,
            current_facts=current_facts,
            task_state=task_state,
            candidate_context=session.candidate_context,
            direct_fact_memory_only=direct_fact_memory_only,
            loop_checkpoint=loop_checkpoint,
            correction_turn=is_correction_turn,
            route_type=route_type,
            ctx_engine=ctx_eng,
            model_profile=model_profile,
        )
        trace_event("phase_context", memory_commit_context)
        trace_phase(ContextPhase.MEMORY_COMMIT, "start", should_compress=should_compress)
        keyed_facts = list(deterministic_keyed_facts)
        voids = list(deterministic_voids)
        blocked_keyed_facts: list[dict] = []
        if should_compress:
            yield _emit(_ev("compressing"))
            try:
                # Decouple: Background compression uses the provided context model layer.
                # Fallback logic: ensure the target model matches the active adapter.
                from llm.adapter_factory import get_default_model
                default_fallback = get_default_model(current_use_adapter)
            
                # Smart fallback: if context_model is deepseek but adapter is not ollama, fallback.
                adapter_name = current_use_adapter.__class__.__name__.lower()
                is_cloud = any(p in adapter_name for p in ["groq", "openai", "gemini", "anthropic"])
            
                compression_target = context_model or model or default_fallback
            
                if is_cloud and "deepseek" in compression_target.lower():
                    # Force fallback for cloud regions to the main model.
                    compression_target = model or default_fallback
            
                log("ctx", sid, f"Compressing exchange with {compression_target} (adapter: {adapter_name})...")
                updated_ctx, compression_keyed_facts, compression_voids = await ctx_eng.compress_exchange(
                    user_message=user_message,
                    assistant_response=clean_response,
                    current_context=session.compressed_context,
                    is_first_exchange=is_first_exchange,
                    model=compression_target,
                    grounding_text="\n".join(successful_tool_evidence),
                )
                compression_keyed_facts, blocked_keyed_facts = finalize_compression_fact_records(
                    compression_keyed_facts,
                    user_message=user_message,
                    grounding_text="\n".join(successful_tool_evidence),
                    deterministic_facts=deterministic_keyed_facts,
                    current_facts=current_facts,
                )
                keyed_facts = merge_fact_records(deterministic_keyed_facts, compression_keyed_facts)
                voids = merge_void_records(deterministic_voids, compression_voids)
                self.sessions.update_context(sid, updated_ctx)
                lines = _count_context_items(updated_ctx)
                log("ctx", sid, f"Context updated · {lines} lines · {len(keyed_facts)} new facts, {len(voids)} voids", level="ok")
                if blocked_keyed_facts:
                    candidate_context = _merge_candidate_context(session.candidate_context, blocked_keyed_facts)
                    self.sessions.update_candidate_context(sid, candidate_context)
                    session.candidate_context = candidate_context
                    candidate_facts = [
                        str(item.get("fact") or "").strip()
                        for item in blocked_keyed_facts[:6]
                        if str(item.get("fact") or "").strip()
                    ]
                    merged_candidate_facts: list[str] = []
                    seen_candidate_facts: set[str] = set()
                    for item in _checkpoint_candidate_facts(loop_checkpoint) + candidate_facts:
                        lowered = item.lower()
                        if lowered in seen_candidate_facts:
                            continue
                        seen_candidate_facts.add(lowered)
                        merged_candidate_facts.append(item)
                    save_loop_checkpoint(
                        "commit",
                        tool_turn=loop_checkpoint.tool_turn if loop_checkpoint else 0,
                        status=loop_checkpoint.status if loop_checkpoint else "candidate_update",
                        strategy=loop_checkpoint.strategy if loop_checkpoint else "",
                        notes=loop_checkpoint.notes if loop_checkpoint else "",
                        confidence=loop_checkpoint.confidence if loop_checkpoint else None,
                        tools=list(loop_checkpoint.tools or []) if loop_checkpoint else [],
                        tool_results=list(loop_checkpoint.tool_results or []) if loop_checkpoint else [],
                        candidate_facts=merged_candidate_facts[:6],
                    )
                    trace_event("memory_fact_filter", {
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
                        "candidate_context_lines": len([line for line in candidate_context.splitlines() if line.strip()]),
                    })

            except Exception as e:
                log("ctx", sid, f"Compression/indexing error: {e}", level="error")
                lines = 0
        else:
            lines = _count_context_items(session.compressed_context)
            log("ctx", sid, "Skipping compression for this turn")
            keyed_facts = merge_fact_records(deterministic_keyed_facts)
            voids = merge_void_records(deterministic_voids)

        if (keyed_facts or voids) and not should_compress:
            log("ctx", sid, f"Persisting deterministic facts without compression · {len(keyed_facts)} facts, {len(voids)} voids")

        if keyed_facts or voids:
            try:
                sg = self.graph
                for void in voids:
                    sg.void_temporal_fact(
                        sid,
                        void["subject"],
                        void["predicate"],
                        session.message_count,
                        owner_id=owner_id,
                    )
                    log("ctx", sid, f"Voided fact: {void['subject']} | {void['predicate']}", level="ok")
                for item in keyed_facts:
                    if item.get("subject") and item.get("predicate") and item.get("subject") != "General":
                        sg.add_temporal_fact(
                            sid,
                            item["subject"],
                            item["predicate"],
                            item.get("object", ""),
                            session.message_count,
                            owner_id=owner_id,
                            run_id=run_id,
                        )
            except Exception as e:
                log("ctx", sid, f"Database deterministic facts error: {e}", level="error")

        if keyed_facts:
            try:
                ve = VectorEngine(sid, agent_id=session.agent_id, owner_id=owner_id)
                anchor_text = f"User: {user_message}\nAssistant: {clean_response}"
                for item in keyed_facts:
                    await ve.index(
                        key=item["key"],
                        anchor=anchor_text,
                        metadata={"fact": item["fact"], "run_id": run_id, "owner_id": owner_id or ""},
                    )
                log("rag", sid, f"Indexed {len(keyed_facts)} facts into vector store", level="ok")

                trace_event("facts_indexed", {
                    "facts": [f["key"] for f in keyed_facts],
                    "context_lines": lines,
                })
            except Exception as e:
                log("rag", sid, f"Fact indexing error: {e}", level="error")
        save_loop_checkpoint(
            "commit",
            tool_turn=loop_checkpoint.tool_turn if loop_checkpoint else 0,
            status="compressed" if should_compress else "skipped",
            notes=f"context_lines={lines}",
            tools=list(loop_checkpoint.tools or []) if loop_checkpoint else [],
            tool_results=list(loop_checkpoint.tool_results or []) if loop_checkpoint else _tool_result_previews(all_tool_results),
            candidate_facts=_checkpoint_candidate_facts(loop_checkpoint),
        )
        trace_phase(ContextPhase.MEMORY_COMMIT, "complete", context_lines=lines, compressed=should_compress)

        yield _emit(_ev("context_updated", lines=lines))
        yield _emit(_ev("done"))
        log("agent", sid, "Turn complete")
        finish_run()

    @staticmethod
    def _strip_tool_json(text: str) -> str:
        """Remove tool-call JSON from response text before storing. Upgraded for V10 logic."""
        if not text: return ""
        
        # Use the same brace-counting scanner from ToolRegistry to find all JSON objects
        # This handles nested braces correctly (which regex can't)
        from plugins.tool_registry import _extract_json_objects
        candidates = _extract_json_objects(text)
        
        # Remove any JSON block that looks like a tool call
        cleaned = text
        for obj, raw_str in candidates:
            if "tool" in obj and "arguments" in obj:
                cleaned = cleaned.replace(raw_str, "")
        
        # Cleanup whitespace artifacts
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove internal reasoning blocks wrapped in <think> tags."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    @staticmethod
    def _strip_internal_execution_chatter(text: str, *, has_tool_results: bool = False) -> str:
        if not text:
            return ""
        cleaned = INTERNAL_EXECUTION_CHATTER_RE.sub("\n", text)
        if has_tool_results:
            cleaned = SYSTEM_ECHO_BLOCK_RE.sub("\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if has_tool_results and cleaned.lower() in {
            "i'd be happy to help you with that.",
            "sure, i can help with that.",
            "i can help with that.",
        }:
            return ""
        return cleaned

    def _verify_citations(self, response: str, anchors: list[dict]) -> bool:
        """
        Conservative contradiction check.
        Only flags explicit logical negations of stored facts.
        Returns False only on high-confidence direct contradictions.
        """
        if not anchors or not response:
            return True

        negation_pairs = [
            (r"\bmy name is (\w+)\b", r"\byour name is (\w+)\b"),
            (r"\bi (like|love|prefer|use)\b", r"\bi (hate|dislike|avoid|don't use|never)\b"),
        ]

        resp_lower = response.lower()
        for anchor in anchors:
            anchor_text = (anchor.get("anchor", "") + " " + anchor.get("key", "")).lower()
            for pos_pat, neg_pat in negation_pairs:
                if re.search(pos_pat, anchor_text) and re.search(neg_pat, resp_lower):
                    return False
        return True

    async def _unified_retrieve(
        self,
        query: str,
        session_id: str,
        agent_id: str,
        owner_id: Optional[str] = None,
        top_n: int = 5,
        use_vector: bool = True,
        use_graph: bool = True,
        use_session_rag: bool = True,
    ) -> list[dict]:
        """
        Query vector, semantic-graph, and session-rag memories.
        Deduplicate and optionally rerank into a unified hit list.
        """
        results: list[dict] = []
        vector_results: list[dict] = []
        bm25_results: list[dict] = []

        try:
            if not use_vector:
                raise RuntimeError("vector retrieval disabled")
            ve = VectorEngine(session_id, agent_id=agent_id, model=self.embed_model, owner_id=owner_id)
            ve_hits = await ve.query(query, limit=6)
            for h in ve_hits:
                item = {
                    "text": f"{h.get('key', '')}: {h.get('anchor', '')}".strip(),
                    "source": "vector",
                    "kind": "recall_anchor",
                    "raw": h,
                    "id": h.get("id") or h.get("key", ""),
                    "key": h.get("key", ""),
                    "anchor": h.get("anchor", ""),
                    "metadata": h.get("metadata", {}),
                    "score_hint": h.get("metadata", {}).get("score"),
                }
                results.append(item)
                vector_results.append(item)
        except Exception:
            pass

        try:
            if not use_vector:
                raise RuntimeError("bm25 retrieval disabled")
            from memory.bm25_engine import BM25Engine

            keyword_hits = BM25Engine(session_id=session_id, agent_id=agent_id, owner_id=owner_id).search(query, top_k=6)
            for h in keyword_hits:
                item = {
                    "text": f"{h.get('key', '')}: {h.get('anchor', '')}".strip(),
                    "source": "bm25",
                    "kind": "keyword_anchor",
                    "raw": h,
                    "id": h.get("id") or h.get("key", ""),
                    "key": h.get("key", ""),
                    "anchor": h.get("anchor", ""),
                    "metadata": h.get("metadata", {}),
                    "score_hint": h.get("score"),
                }
                results.append(item)
                bm25_results.append(item)
        except Exception:
            pass

        try:
            if not use_graph:
                raise RuntimeError("graph retrieval disabled")
            sg_hits = await self.graph.traverse(query, top_k=6, threshold=SEMANTIC_GRAPH_THRESHOLD, owner_id=owner_id)
            for h in sg_hits:
                results.append({
                    "text": f"{h.get('subject', '')} {h.get('predicate', '')} {h.get('object', '')}".strip(),
                    "source": "graph",
                    "kind": "durable_fact",
                    "raw": h,
                    "id": f"{h.get('subject', '')}|{h.get('predicate', '')}|{h.get('object', '')}",
                    "key": f"{h.get('subject', '')} {h.get('predicate', '')}".strip(),
                    "anchor": h.get("object", ""),
                    "metadata": {"score": h.get("score"), "kind": "semantic_graph"},
                    "score_hint": h.get("score"),
                })
        except Exception:
            pass

        try:
            if not use_session_rag:
                raise RuntimeError("session rag retrieval disabled")
            from memory.session_rag import get_session_rag
            rag = get_session_rag(session_id, owner_id=owner_id)
            rag_hits = await rag.query(query, top_k=6)
            for h in rag_hits:
                results.append({
                    "text": h.get("content", ""),
                    "source": "rag",
                    "kind": "session_evidence",
                    "raw": h,
                    "id": (
                        f"{h.get('filename') or h.get('source') or h.get('tool_name') or 'rag'}::"
                        f"{(h.get('content', '') or '')[:80]}"
                    ),
                    "key": h.get("tool_name") or h.get("source") or "session_rag",
                    "anchor": h.get("content", ""),
                    "metadata": {
                        "source": h.get("source", ""),
                        "tool_name": h.get("tool_name", ""),
                        "filename": h.get("filename", ""),
                        "content_type": h.get("content_type", ""),
                        "indexed_at": h.get("indexed_at", ""),
                        "score": h.get("score"),
                    },
                    "score_hint": h.get("score"),
                })
        except Exception:
            pass

        if not results:
            return []

        deduped: list[dict] = []
        seen_fingerprints: set[str] = set()
        for r in results:
            fingerprint = _retrieval_fingerprint(r)
            if not fingerprint:
                continue
            r["dedup_fingerprint"] = fingerprint
            if any(
                fingerprint[:DEDUP_PREFIX_LENGTH] in seen
                or seen[:DEDUP_PREFIX_LENGTH] in fingerprint
                for seen in seen_fingerprints
            ):
                continue
            deduped.append(r)
            seen_fingerprints.add(fingerprint)

        current_facts = self.graph.get_current_facts(session_id, owner_id=owner_id)
        deduped = [
            item
            for item in deduped
            if not is_redundant_user_alias_text(
                " ".join(
                    part for part in (
                        str(item.get("text") or "").strip(),
                        str(item.get("key") or "").strip(),
                        str(item.get("anchor") or "").strip(),
                    )
                    if part
                ),
                current_facts=current_facts,
            )
        ]

        ranked_retrieval = [items for items in (vector_results, bm25_results) if items]
        if ranked_retrieval:
            try:
                from memory.bm25_engine import reciprocal_rank_fusion

                fused = reciprocal_rank_fusion(*ranked_retrieval, top_n=max(top_n, 6))
                fused_fingerprints = {
                    _retrieval_fingerprint(item)
                    for item in fused
                }
                deduped = fused + [
                    item for item in deduped
                    if _retrieval_fingerprint(item) not in fused_fingerprints
                ]
            except Exception:
                pass

        try:
            from memory.reranker import rerank
            return rerank(query, deduped, text_key="text", top_n=top_n)
        except Exception:
            return deduped[:top_n]

    async def _agent_loop(
        self,
        model:          str,
        messages:       list[dict],
        adapter:        Optional[BaseLLMAdapter] = None,
        search_backend: Optional[str] = None,
        search_engine:  Optional[str] = None,
        images:         Optional[list[str]] = None,
        agent_id:       str = "default",
        session_id:     str = "unknown",
        run_id:         Optional[str] = None,
        owner_id:       Optional[str] = None,
        route_type:     str = "open_ended",
        forced_tools:   Optional[list[str]] = None,
        user_message:   str = "",
        observe_step:   Optional[Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]] = None,
        max_tool_calls: Optional[int] = None,
        max_llm_turns:  Optional[int] = None,
        prior_tool_results: Optional[list[dict[str, Any]]] = None,
        model_profile: str = "frontier_standard",
        suppress_tools: bool = False,
        blocked_tools: Optional[set[str]] = None,
    ) -> AsyncIterator[dict]:
        tool_turn = 0
        llm_turn = 0
        tool_call_count = 0
        repeated_tool_signatures: dict[str, int] = {}
        instability_score = 0
        failed_tools: dict[str, int] = {}
        prompt_retry_count = 0
        forced_tool_retry_used = False
        sid = session_id
        
        current_adapter = adapter or self.adapter
        if not adapter and ":" in model:
            from llm.adapter_factory import create_adapter
            current_adapter = create_adapter(model.split(":", 1)[0])
            
        clean_model = strip_provider_prefix(model)
        max_tool_turns = _adaptive_tool_turn_budget(
            model,
            current_adapter,
            route_type=route_type,
            forced_tools=forced_tools,
            user_message=user_message,
        )
        if max_llm_turns is None:
            max_llm_turns = max_tool_turns + 1
        tools_enabled = self.tools.has_tools() and not suppress_tools
        budget_exhausted = False
        task_bootstrap_done = get_session_task_tracker().has_tasks(sid)
        accumulated_tool_results: list[dict[str, Any]] = list(prior_tool_results or [])
        blocked_tools = {
            str(name or "").strip().lower()
            for name in (blocked_tools or set())
            if str(name or "").strip()
        }

        while True:
            # ── INTERRUPT CHECK ──────────────────────────────────────────────
            if self._is_session_interrupted(sid):
                log("agent", sid, "Execution cancelled by user", level="warn")
                yield _ev("error", message="Execution stopped.")
                self._clear_session_interrupt(sid)
                break

            remaining_turns = max(1, int(max_llm_turns) - llm_turn)
            turn_tools_enabled = tools_enabled and remaining_turns > 1

            turn_buffer = ""
            turn_images = images if tool_turn == 0 else None

            # ── CONTEXT SAFETY: Global Budget Enforcement ───────────────
            # Ensure the total tokens (system + tools + history) fits model limit.
            messages = _enforce_total_budget(messages, model)
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            prompt_tokens_est = prompt_chars // 4

            log(
                "llm",
                sid,
                f"Streaming turn {llm_turn} · model={model}",
                pass_index=llm_turn,
                message_count=len(messages),
                prompt_chars=prompt_chars,
                estimated_tokens=prompt_tokens_est,
            )

            _trace(agent_id or "default", sid, "llm_pass_start", {
                "turn": llm_turn,
                "model": model,
                "model_profile": model_profile,
                "message_count": len(messages),
                "prompt_chars": prompt_chars,
                "estimated_tokens": prompt_tokens_est,
            }, run_id=run_id, owner_id=owner_id)

            # Trace the full prompt that is actually sent after budget enforcement.
            _trace(agent_id or "default", sid, "llm_prompt", {
                "turn": llm_turn,
                "model": model,
                "model_profile": model_profile,
                "message_count": len(messages),
                "estimated_tokens": prompt_tokens_est,
                "messages": messages,
            }, run_id=run_id, owner_id=owner_id)

            # Watchdog: ensure the adapter stream doesn't hang forever
            # Python 3.10 compatible — asyncio.timeout() requires 3.11+
            try:
                # Pass tool schemas for native tool calling support
                tool_schemas = self.tools.get_schemas() if tools_enabled and self.tools.has_tools() else None
                if not turn_tools_enabled:
                    tool_schemas = None
                if not _should_use_native_tools(current_adapter, model):
                    tool_schemas = None
                stream_coro = current_adapter.stream(
                    model=model, 
                    messages=messages, 
                    images=turn_images,
                    tools=tool_schemas,
                    interrupt_check=lambda: self.sessions.is_interrupted(sid)
                )
                token_buffer = []
                
                in_thought = False
                
                async def _consume_stream():
                    nonlocal in_thought
                    async for token in stream_coro:
                        # Check for interruption mid-stream
                        if self._is_session_interrupted(sid):
                            yield _ev("error", message="Execution interrupted by user.")
                            return

                        # Handle structured events from modernized adapters
                        if token in {"<THOUGHT>", "<think>"}:
                            in_thought = True
                            yield _ev("thought_start")
                            continue
                        elif token in {"</THOUGHT>", "</think>"}:
                            in_thought = False
                            yield _ev("thought_end")
                            continue
                        
                        # Handle native tool calls (yielded as JSON string by adapter)
                        if token.startswith('{"tool_calls":'):
                            try:
                                yield {"type": "_native_tool_call", "data": json.loads(token)}
                                continue
                            except: pass

                        if in_thought:
                            yield _ev("thought", content=token)
                        else:
                            token_buffer.append(token)
                            yield _ev("token", content=token)
                
                async for event in _consume_stream():
                    if event["type"] == "_native_tool_call":
                        # Convert native tool call to our internal format immediately
                        native_calls = event["data"]["tool_calls"]
                        for nc in native_calls:
                            fn = nc.get("function", {})
                            call_json = json.dumps({
                                "tool": fn.get("name"),
                                "arguments": fn.get("arguments")
                            })
                            turn_buffer += f"\n{call_json}\n"
                        continue
                        
                    if event["type"] == "token":
                        turn_buffer += event["content"]
                    
                    yield event
                    
            except asyncio.TimeoutError:
                log("llm", sid, "Stream watchdog triggered: model stalled", level="error")
                yield _ev("error", message="LLM Stream stalled. Consider trying a different model or context window.")
                break
            except Exception as e:
                if _is_context_overflow_error(e) and prompt_retry_count < 2:
                    prompt_retry_count += 1
                    messages, shrink_stats = _shrink_messages_for_context_retry(messages)
                    log(
                        "llm",
                        sid,
                        f"Prompt overflow detected · retrying with reduced context ({prompt_retry_count}/2)",
                        level="warn",
                        **shrink_stats,
                    )
                    _trace(
                        agent_id or "default",
                        sid,
                        "prompt_overflow_retry",
                        {
                            "turn": llm_turn,
                            "model": model,
                            "retry_count": prompt_retry_count,
                            **shrink_stats,
                        },
                        run_id=run_id,
                        owner_id=owner_id,
                    )
                    continue
                raise

            log(
                "llm",
                sid,
                f"Turn {llm_turn} complete · {len(turn_buffer)} chars generated",
                level="ok",
                pass_index=llm_turn,
                completion_chars=len(turn_buffer),
            )
            _trace(agent_id or "default", sid, "llm_pass_complete", {
                "turn": llm_turn,
                "model": model,
                "completion_chars": len(turn_buffer),
            }, run_id=run_id, owner_id=owner_id)
            prompt_retry_count = 0
            llm_turn += 1

            calls = self.tools.detect_tool_calls(turn_buffer)
            if blocked_tools and calls:
                blocked_calls = [call for call in calls if call.tool_name.lower() in blocked_tools]
                if blocked_calls:
                    clean = self._strip_detected_tool_json(turn_buffer, calls)
                    yield _ev("retract_last_tokens")
                    yield {"type": "_retract_response", "clean_response": clean}
                    if clean:
                        messages.append({"role": "assistant", "content": clean})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Do not call these tools on this turn: {', '.join(sorted({call.tool_name for call in blocked_calls}))}. "
                            "The runtime will handle deterministic memory updates automatically for explicit user-stated facts and corrections. "
                            "Answer directly if you can, or choose a different substantive tool."
                        ),
                    })
                    _trace(
                        agent_id or "default",
                        sid,
                        "blocked_tool_call_suppressed",
                        {
                            "turn": llm_turn,
                            "blocked_tools": sorted({call.tool_name for call in blocked_calls}),
                        },
                        run_id=run_id,
                        owner_id=owner_id,
                    )
                    continue
            if tool_turn >= max_tool_turns:
                if calls and not budget_exhausted:
                    log("agent", sid, f"Max tool turns ({max_tool_turns}) reached", level="warn")
                    clean = self._strip_detected_tool_json(turn_buffer, calls)
                    yield _ev("retract_last_tokens")
                    yield {"type": "_retract_response", "clean_response": clean}
                    yield {
                        "type": "_tool_budget_exhausted",
                        "tool_turns": tool_turn,
                        "max_tool_turns": max_tool_turns,
                        "pending_tools": [call.tool_name for call in calls],
                    }

                    if clean:
                        messages.append({"role": "assistant", "content": clean})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Tool budget reached for this turn. Do not call any more tools. "
                            "Using only the gathered results already in this conversation, give the user a concise final answer. "
                            "If anything remains incomplete, say exactly what is missing."
                        ),
                    })
                    tools_enabled = False
                    budget_exhausted = True
                    continue

                log("agent", sid, f"Max tool turns ({max_tool_turns}) reached", level="warn")
                break

            if not turn_tools_enabled or not self.tools.has_tools():
                if calls and not budget_exhausted:
                    clean = self._strip_detected_tool_json(turn_buffer, calls)
                    yield _ev("retract_last_tokens")
                    yield {"type": "_retract_response", "clean_response": clean}
                    if clean:
                        messages.append({"role": "assistant", "content": clean})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Execution budget for additional tool steps has been reached. "
                            "Do not call any more tools. Using only the evidence already gathered in this run, "
                            "give the user the best concise final answer you can. "
                            "If anything remains incomplete, say exactly what is missing."
                        ),
                    })
                    tools_enabled = False
                    budget_exhausted = True
                    max_llm_turns = max(int(max_llm_turns), llm_turn + 1)
                    _trace(
                        agent_id or "default",
                        sid,
                        "terminal_tool_call_recovered",
                        {
                            "turn": llm_turn,
                            "tool_turn": tool_turn,
                            "pending_tools": [call.tool_name for call in calls],
                        },
                        run_id=run_id,
                        owner_id=owner_id,
                    )
                    continue
                break

            if not calls:
                if forced_tools and not forced_tool_retry_used and tool_turn == 0 and tools_enabled:
                    forced_tool_retry_used = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "You already have an execution plan. "
                            f"Call one of these tools now: {', '.join(forced_tools)}.\n"
                            "Respond with ONLY a single valid JSON tool invocation and nothing else."
                        ),
                    })
                    _trace(
                        agent_id or "default",
                        sid,
                        "forced_tool_retry",
                        {
                            "turn": llm_turn,
                            "tools": list(forced_tools),
                            "reason": "planned_tools_not_executed",
                        },
                        run_id=run_id,
                        owner_id=owner_id,
                    )
                    continue
                log("agent", sid, "No tool call detected — turn complete")
                break

            if max_tool_calls is not None:
                remaining_tool_calls = max_tool_calls - tool_call_count
                if remaining_tool_calls <= 0:
                    log("agent", sid, f"Max tool calls ({max_tool_calls}) reached", level="warn")
                    clean = self._strip_detected_tool_json(turn_buffer, calls)
                    yield _ev("retract_last_tokens")
                    yield {"type": "_retract_response", "clean_response": clean}
                    yield {
                        "type": "_tool_budget_exhausted",
                        "tool_turns": tool_turn,
                        "max_tool_turns": max_tool_turns,
                        "pending_tools": [call.tool_name for call in calls],
                    }
                    if clean:
                        messages.append({"role": "assistant", "content": clean})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Tool call limit reached for this turn. Do not call any more tools. "
                            "Using only the gathered results already in this conversation, give the user the best concise answer you can. "
                            "If anything remains incomplete, say exactly what is missing."
                        ),
                    })
                    tools_enabled = False
                    budget_exhausted = True
                    continue
                if len(calls) > remaining_tool_calls:
                    calls = calls[:remaining_tool_calls]

            tool_call_count += len(calls)

            for call in calls:
                yield _ev("tool_call", tool=call.tool_name, tool_name=call.tool_name, arguments=call.arguments)
            
            yield _ev("retract_last_tokens")

            # BUG FIX: emit internal event so chat_stream can clean full_response
            # Strip the tool-call JSONs that were already streamed to the UI
            clean = self._strip_detected_tool_json(turn_buffer, calls)
            yield {"type": "_retract_response", "clean_response": clean}

            tool_turn += 1

            # Execute all tools concurrently or sequentially
            combined_results = []
            tool_results_batch: list[dict[str, Any]] = []
            
            for call in calls:
                signature = _tool_call_signature(call.tool_name, call.arguments)
                if call.tool_name == "todo_write" and tool_turn > 0 and (
                    task_bootstrap_done or get_session_task_tracker().has_tasks(sid)
                ):
                    from plugins.tool_registry import ToolResult
                    result = ToolResult(
                        tool_name=call.tool_name,
                        success=False,
                        content=(
                            "todo_write already initialized task state earlier in this run. "
                            "Use todo_update to change task status, or proceed with the next substantive tool/result synthesis."
                        ),
                    )
                    combined_results.append(
                        f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                        f"{_compact_tool_result_for_followup(call.tool_name, result.content)}\n"
                        f"</SYSTEM_TOOL_RESULT>"
                    )
                    yield _ev("tool_result", tool_name=call.tool_name, success=False, content=result.content)
                    tool_results_batch.append({
                        "tool_name": call.tool_name,
                        "success": False,
                        "content": result.content,
                        "arguments": dict(call.arguments),
                    })
                    _trace(
                        agent_id or "default",
                        sid,
                        "task_bootstrap_suppressed",
                        {
                            "turn": tool_turn,
                            "reason": "todo_write_already_initialized",
                        },
                        run_id=run_id,
                        owner_id=owner_id,
                    )
                    continue
                if _is_retry_sensitive_tool(call.tool_name):
                    repeat_count = repeated_tool_signatures.get(signature, 0)
                    if repeat_count >= 1:
                        from plugins.tool_registry import ToolResult
                        result = ToolResult(
                            tool_name=call.tool_name,
                            success=False,
                            content=(
                                f"Duplicate {call.tool_name} call suppressed for this run. "
                                "Use the existing result or pivot to a different query/source."
                            ),
                        )
                        combined_results.append(
                            f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                            f"{_compact_tool_result_for_followup(call.tool_name, result.content)}\n"
                            f"</SYSTEM_TOOL_RESULT>"
                        )
                        yield _ev("tool_result", tool_name=call.tool_name, success=False, content=result.content)
                        tool_results_batch.append({
                            "tool_name": call.tool_name,
                            "success": False,
                            "content": result.content,
                            "arguments": dict(call.arguments),
                        })
                        instability_score += 1
                        continue
                    repeated_tool_signatures[signature] = repeat_count + 1

                if self.circuit_breaker.is_open(sid, call.tool_name):
                    from plugins.tool_registry import ToolResult
                    result = ToolResult(
                        tool_name=call.tool_name,
                        success=False,
                        content=(
                            f"Circuit breaker is OPEN for '{call.tool_name}'. "
                            "Choose an alternative tool."
                        ),
                    )
                    combined_results.append(
                        f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                        f"{_compact_tool_result_for_followup(call.tool_name, result.content)}\n"
                        f"</SYSTEM_TOOL_RESULT>"
                        f"{self.circuit_breaker.get_pivot_message(call.tool_name)}"
                    )
                    yield _ev("tool_result", tool_name=call.tool_name, success=False, content=result.content)
                    tool_results_batch.append({
                        "tool_name": call.tool_name,
                        "success": False,
                        "content": result.content,
                        "arguments": dict(call.arguments),
                    })
                    continue

                if call.tool_name == "web_search":
                    if search_backend and search_backend != "auto":
                        call.arguments["backend"] = search_backend
                    if search_engine:
                        call.arguments["search_engine"] = search_engine # Inject into kwargs

                args_summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in call.arguments.items())
                log("tool", sid, f"Calling {call.tool_name}({args_summary})")
                _trace(agent_id or "default", sid, "tool_call", {
                    "turn": tool_turn,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "arguments_summary": args_summary,
                }, run_id=run_id, owner_id=owner_id)

                yield _ev("tool_running", tool_name=call.tool_name)
                # Pass context (session_id, agent_id) to tool execution for hierarchy support
                context = {
                    "_session_id": sid,
                    "_agent_id": agent_id,
                    "_run_id": run_id,
                    "_owner_id": owner_id,
                    "_embed_model": getattr(self, "embed_model", "nomic-embed-text"),
                }
                validation_error = self.tools.validate_tool_call(call)
                if validation_error:
                    from plugins.tool_registry import ToolResult
                    result = ToolResult(
                        tool_name=call.tool_name,
                        success=False,
                        content=f"Tool validation failed: {validation_error}",
                    )
                else:
                    result = None
                
                # Use GuardrailMiddleware if available
                if result is None and self.middleware:
                    async for event in self.middleware.execute_stream(
                        call,
                        session_id = sid,
                        agent_id   = agent_id,
                        context    = context,
                    ):
                        # Forward middleware events (confirmations, blocks, etc.)
                        # We yield them all, but the tool_result specifically will also 
                        # be used to populate our 'result' object for unified yielding below.
                        if event["type"] != "tool_result":
                            yield event
                        
                        if event["type"] in ("tool_blocked", "confirmation_denied", "confirmation_timeout"):
                            from plugins.tool_registry import ToolResult
                            result = ToolResult(
                                tool_name = call.tool_name,
                                success   = False,
                                content   = event.get("reason", event["type"]),
                            )
                            break
                        
                        if event["type"] == "tool_result":
                            from plugins.tool_registry import ToolResult
                            result = ToolResult(
                                tool_name = call.tool_name,
                                success   = event["success"],
                                content   = event["content"],
                            )
                            break
                elif result is None:
                    result = await self.tools.execute(call, context=context)

                if result.success:
                    log("tool", sid, f"{call.tool_name} → {len(result.content)} chars returned", level="ok")
                    if call.tool_name == "todo_write":
                        task_bootstrap_done = True
                        forced_tools = _normalize_forced_tools_for_task_state(
                            forced_tools,
                            has_tasks=True,
                        ) or None
                else:
                    log("tool", sid, f"{call.tool_name} failed: {result.content[:120]}", level="error")

                _trace(agent_id or "default", sid, "tool_result", {
                    "turn": tool_turn,
                    "tool_name": call.tool_name,
                    "success": result.success,
                    "content_length": len(result.content),
                    "content_preview": result.content[:500],
                }, run_id=run_id, owner_id=owner_id)

                is_validation_failure = (
                    not result.success
                    and isinstance(result.content, str)
                    and result.content.startswith("Tool validation failed:")
                )

                if not result.success:
                    if is_validation_failure:
                        # Model formatted/argument error: don't poison circuit breaker health.
                        is_open = False
                    else:
                        is_open = self.circuit_breaker.record_failure(sid, call.tool_name)
                else:
                    self.circuit_breaker.record_success(sid, call.tool_name)
                    is_open = False
                if not result.success and _looks_like_tool_instability(call.tool_name, result.content):
                    instability_score += 1

                pivot_msg = ""
                if is_open:
                    pivot_msg = self.circuit_breaker.get_pivot_message(call.tool_name)

                # Truncate content for specific models to prevent context explosion
                truncated_content = _truncate_for_model(
                    _compact_tool_result_for_followup(call.tool_name, result.content),
                    clean_model,
                )

                # Unified yield for tool result - handles both middleware and direct execution
                yield _ev("tool_result", tool_name=call.tool_name, success=result.success, content=result.content)
                tool_results_batch.append({
                    "tool_name": call.tool_name,
                    "success": bool(result.success),
                    "content": result.content,
                    "arguments": dict(call.arguments),
                })
                if result.success and run_id:
                    try:
                        for artifact_candidate in _build_run_artifact_candidates(
                            tool_name=call.tool_name,
                            arguments=dict(call.arguments),
                            content=result.content,
                        ):
                            artifact = get_run_store().save_artifact(
                                run_id=run_id,
                                session_id=sid,
                                owner_id=owner_id,
                                tool_name=call.tool_name,
                                artifact_type=artifact_candidate["artifact_type"],
                                label=artifact_candidate["label"],
                                storage_path=artifact_candidate.get("storage_path"),
                                content_hash=artifact_candidate.get("content_hash", ""),
                                size_bytes=int(artifact_candidate.get("size_bytes") or 0),
                                preview=str(artifact_candidate.get("preview") or ""),
                                metadata=artifact_candidate.get("metadata") or {},
                            )
                            _trace(agent_id or "default", sid, "run_artifact", {
                                "artifact_id": artifact.artifact_id,
                                "tool_name": call.tool_name,
                                "artifact_type": artifact.artifact_type,
                                "label": artifact.label,
                                "storage_path": artifact.storage_path,
                                "size_bytes": artifact.size_bytes,
                            }, run_id=run_id, owner_id=owner_id)
                    except Exception as e:
                        log("run", sid, f"Failed to persist run artifact for {call.tool_name}: {e}", level="warn")

                # — INTEGRATION: Auto-index significant tool results into session RAG —
                if result.success and _should_index_tool_result(call.tool_name, result.content):
                    try:
                        from memory.session_rag import get_session_rag
                        rag = get_session_rag(sid, owner_id=owner_id)
                        # Await indexing with bounded timeout to avoid indexing races.
                        try:
                            await asyncio.wait_for(
                                rag.index(
                                    content=result.content,
                                    source=call.tool_name,
                                    tool_name=call.tool_name,
                                    content_type=_tool_result_content_type(call.tool_name),
                                    run_id=run_id,
                                ),
                                timeout=RAG_INDEX_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            pass
                    except Exception as e:
                        log("rag", sid, f"Failed to index tool result: {e}", level="warn")

                # ── Persist tool result to dedicated DB ───────────────────
                try:
                    from memory.tool_results_db import ToolResultsDB
                    result_type = "text"
                    if call.tool_name == "generate_app":
                        result_type = "app_view"
                    elif call.tool_name in ("web_search", "image_search"):
                        result_type = "search"
                    elif call.tool_name in ("file_create", "file_view", "file_str_replace"):
                        result_type = "file_op"
                    elif call.tool_name == "bash":
                        result_type = "bash"
                    
                    ToolResultsDB().store(
                        session_id=sid,
                        tool_name=call.tool_name,
                        arguments=call.arguments,
                        result=result.content[:10000],  # Cap at 10KB per result
                        success=result.success,
                        result_type=result_type,
                        agent_id=agent_id,
                    )
                except Exception as e:
                    log("tool", sid, f"Failed to persist tool result: {e}", level="error")

                
                # Assemble the XML result block for this specific tool call
                combined_results.append(
                    f"<SYSTEM_TOOL_RESULT name=\"{call.tool_name}\">\n"
                    f"{truncated_content}\n"
                    f"</SYSTEM_TOOL_RESULT>"
                    f"{pivot_msg}"
                )

            task_tracker = get_session_task_tracker()
            has_tasks_now = task_tracker.has_tasks(sid)
            has_active_tasks_now = task_tracker.has_active_tasks(sid)
            current_run_tool_results = accumulated_tool_results + list(tool_results_batch)
            followup_tool_results = _select_followup_tool_results(
                current_run_tool_results,
                user_message=user_message,
            )
            task_reminder = ""
            if _should_include_task_reminder(
                route_type=route_type,
                user_message=user_message,
                tool_results=followup_tool_results,
                has_active_tasks=has_active_tasks_now,
            ):
                task_reminder = (
                    "\n\n<system-reminder>\n"
                    f"{task_tracker.render(sid)}\n"
                    "</system-reminder>"
                )

            observation_reminder = ""
            post_tool_instruction = (
                "Use the result(s) above to continue the same task. "
                "Either call the next needed tool with one valid JSON tool invocation, or answer the user directly if the evidence is enough. "
                "Do not repeat tool JSON, schemas, hidden packet names, or internal notes."
            )
            if any(
                str(item.get("tool_name") or "") == "web_fetch"
                and bool(item.get("success"))
                and _tool_result_matches_exact_target(item, _extract_exact_query_targets(user_message))
                for item in followup_tool_results
            ):
                post_tool_instruction += (
                    " Prioritize verified exact-domain fetch evidence over noisier search results when deciding what is real, active, or trustworthy."
                )
            observation_digest = ""
            use_compact_evidence_packet = callable(observe_step) or model_profile in {
                "small_local",
                "tool_native_local",
                "local_standard",
            }
            if callable(observe_step):
                try:
                    observation = observe_step(
                        tool_turn=tool_turn,
                        tool_calls=[
                            {
                                "tool_name": call.tool_name,
                                "arguments": dict(call.arguments),
                            }
                            for call in calls
                        ],
                        tool_results=followup_tool_results,
                    )
                    if inspect.isawaitable(observation):
                        observation = await observation
                    if isinstance(observation, dict) and observation:
                        observation_tools = [
                            item.get("name")
                            for item in (observation.get("tools") or [])
                            if isinstance(item, dict) and isinstance(item.get("name"), str)
                        ]
                        observation_tools = _normalize_forced_tools_for_task_state(
                            observation_tools,
                            has_tasks=task_tracker.has_tasks(sid),
                        )
                        yield _ev(
                            "manager_observation",
                            status=observation.get("status"),
                            strategy=observation.get("strategy", ""),
                            tools=observation_tools,
                            confidence=observation.get("confidence"),
                        )
                        notes = str(observation.get("notes", "")).strip()
                        strategy = str(observation.get("strategy", "")).strip()
                        if observation.get("status") == "finalize":
                            tools_enabled = False
                            reminder_lines = ["Manager observation: enough evidence gathered for this run."]
                            if strategy:
                                reminder_lines.append(strategy)
                            if notes:
                                reminder_lines.append(notes)
                            reminder_lines.append(
                                "Do not call any more tools. Use only the gathered evidence above to answer the user."
                            )
                            observation_reminder = (
                                "\n\n<system-reminder>\n"
                                + "\n".join(reminder_lines)
                                + "\n</system-reminder>"
                            )
                            post_tool_instruction = (
                                "Use only the gathered evidence above and give the user a concise final answer. "
                                "Do not call any more tools and do not mention internal execution state."
                            )
                        elif observation_tools or notes or strategy:
                            reminder_lines = ["Manager observation: continue the same run using the latest evidence."]
                            if strategy:
                                reminder_lines.append(strategy)
                            if observation_tools:
                                reminder_lines.append(
                                    "If another tool is needed, prefer only: " + ", ".join(observation_tools)
                                )
                            if notes:
                                reminder_lines.append(notes)
                            observation_reminder = (
                                "\n\n<system-reminder>\n"
                                + "\n".join(reminder_lines)
                                + "\n</system-reminder>"
                            )
                except Exception as e:
                    log("agent", sid, f"Manager observation hook failed: {e}", level="warn")

            batch_failure_count = sum(1 for item in tool_results_batch if not item.get("success"))
            task_progress_instruction = _build_task_progress_instruction(
                route_type=route_type,
                user_message=user_message,
                tool_results=followup_tool_results,
                has_active_tasks=has_active_tasks_now,
            )
            if task_progress_instruction and "Do not call any more tools." not in post_tool_instruction:
                post_tool_instruction = task_progress_instruction
            if tools_enabled and instability_score >= 2:
                tools_enabled = False
                post_tool_instruction = (
                    "Recent tool/provider failures indicate unstable external services. "
                    "Do not call more tools. Using only the gathered evidence above, answer the user clearly "
                    "and state exactly what could not be verified because of provider or network failures."
                )
                observation_reminder = (
                    "\n\n<system-reminder>\n"
                    "Runtime degraded to answer-only mode after repeated tool/provider instability.\n"
                    "Stop retrying the same failing searches or fetches in this run.\n"
                    "</system-reminder>"
                )
                _trace(
                    agent_id or "default",
                    sid,
                    "runtime_degraded",
                    {
                        "reason": "tool_instability",
                        "tool_turn": tool_turn,
                        "instability_score": instability_score,
                        "batch_failure_count": batch_failure_count,
                    },
                    run_id=run_id,
                    owner_id=owner_id,
                )

            digest_lines = []
            for item in followup_tool_results[:4]:
                tool_name = str(item.get("tool_name") or "unknown")
                success = "ok" if item.get("success") else "failed"
                preview = _compact_tool_result_for_followup(tool_name, str(item.get("content") or ""))
                preview = preview.replace("\n", " ")
                if len(preview) > 240:
                    preview = preview[:237].rstrip() + "..."
                digest_lines.append(f"- {tool_name} [{success}]: {preview}")
            if digest_lines:
                observation_digest = (
                    "\n\n<SYSTEM_OBSERVATION>\n"
                    "Recent tool digest:\n"
                    + "\n".join(digest_lines)
                    + "\n</SYSTEM_OBSERVATION>"
                )

            evidence_payload = "\n\n".join(combined_results)
            compact_packet = _build_followup_evidence_packet(
                followup_tool_results,
                max_results=_model_profile_budget(model_profile, "followup_results", 4),
                max_total_chars=_model_profile_budget(model_profile, "followup_chars", 1800),
            )
            if compact_packet:
                evidence_payload = compact_packet
            if use_compact_evidence_packet:
                _trace(
                    agent_id or "default",
                    sid,
                    "managed_evidence_packet",
                    {
                        "tool_turn": tool_turn,
                        "result_count": len(followup_tool_results),
                        "payload_chars": len(evidence_payload),
                        "compact": True,
                    },
                    run_id=run_id,
                    owner_id=owner_id,
                )
            else:
                # Append the agent's raw JSONs back into the message history for the direct/single loop.
                messages.append({"role": "assistant", "content": "\n".join(c.raw_json for c in calls)})

            messages, forced_tools, sanitation_stats = _sanitize_followup_messages(
                messages,
                forced_tools=forced_tools,
                user_message=user_message,
                route_type=route_type,
                latest_tool_results=followup_tool_results,
                has_tasks=has_tasks_now,
            )
            _trace(
                agent_id or "default",
                sid,
                "followup_context_sanitized",
                {
                    "tool_turn": tool_turn,
                    **sanitation_stats,
                    "message_count_after": len(messages),
                    "task_reminder_included": bool(task_reminder),
                },
                run_id=run_id,
                owner_id=owner_id,
            )
            if use_compact_evidence_packet:
                _trace(
                    agent_id or "default",
                    sid,
                    "managed_prompt_minimization",
                    {
                        "tool_turn": tool_turn,
                        "removed_messages": sanitation_stats.get("removed_messages", 0),
                        "natural_messages_kept": sanitation_stats.get("natural_messages_kept", 0),
                        "message_count_after": len(messages),
                    },
                    run_id=run_id,
                    owner_id=owner_id,
                )
            accumulated_tool_results = current_run_tool_results
            
            # Inject the combined tool results block for the next turn
            messages.append({
                "role": "user",
                "content": (
                    evidence_payload +
                    observation_digest +
                    task_reminder +
                    observation_reminder +
                    f"\n\n{post_tool_instruction}"
                ),
            })

    def _build_historical_context_content(
        self,
        unified_hits: Optional[list[dict]],
        current_facts: Optional[list[tuple[str, str, str]]],
        *,
        correction_turn: bool,
    ) -> str:
        historical_anchors = list(unified_hits or [])
        if current_facts and historical_anchors:
            fact_strings = {f"{s} {p} {o}".lower() for (s, p, o) in current_facts}
            filtered_anchors = []
            for anchor in historical_anchors:
                anchor_text = (anchor.get("key", "") + " " + anchor.get("anchor", "")).lower()
                already_covered = any(
                    all(word in anchor_text for word in fact.split() if len(word) > 3)
                    for fact in fact_strings
                )
                if not already_covered:
                    filtered_anchors.append(anchor)
            historical_anchors = filtered_anchors

        if not historical_anchors:
            return ""

        anchor_parts = []
        total_rag_chars = 0
        max_rag_chars = 1800 if correction_turn else 3200

        for anchor in historical_anchors:
            key = anchor.get("key", "Context")
            fact = anchor.get("metadata", {}).get("fact", "")
            exchange = anchor.get("anchor", "")
            source = anchor.get("source", "memory")
            kind = anchor.get("kind", "context")

            if len(exchange) > 600:
                exchange = f"{exchange[:400]}\n[...truncated...]\n{exchange[-200:]}"

            part = (
                f"SOURCE: {source}\n"
                f"TYPE: {kind}\n"
                f"CONCEPT: {key}\n"
                f"FACT: {fact}\n"
                f"EXCHANGE:\n{exchange}"
            )
            if total_rag_chars + len(part) > max_rag_chars:
                omitted = len(historical_anchors) - len(anchor_parts)
                if omitted > 0:
                    anchor_parts.append(
                        f"[...{omitted} more historical anchors omitted to save context...]"
                    )
                break

            anchor_parts.append(part)
            total_rag_chars += len(part)

        return "\n\n---\n".join(anchor_parts)

    def _build_context_items(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        first_message: Optional[str] = None,
        message_count: int = 0,
        unified_hits: Optional[list[dict]] = None,
        force_memory: bool = False,
        forced_tools: Optional[list[str]] = None,
        current_facts: Optional[list[tuple[str, str, str]]] = None,
        task_state: Optional[str] = None,
        candidate_context: Optional[str] = None,
        loop_checkpoint: Optional[LoopCheckpoint] = None,
        correction_turn: bool = False,
        route_type: str = "open_ended",
        direct_fact_memory_only: bool = False,
        ctx_engine: Optional[ContextEngine] = None,
        model_profile: str = "frontier_standard",
    ) -> list[ContextItem]:
        if ctx_engine is None:
            raise RuntimeError(
                "ctx_eng must be resolved per-request via _resolve_context_engine() "
                "and passed explicitly — do not use shared instance state"
            )

        now = datetime.now().strftime("%A, %B %d, %Y")
        items: list[ContextItem] = [
            ContextItem(
                item_id="runtime_metadata",
                kind=ContextKind.RUNTIME,
                title="Runtime Metadata",
                content=f"Current Date: {now}",
                source="runtime",
                priority=10,
                max_chars=200,
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.MEMORY_COMMIT,
                    ContextPhase.VERIFICATION,
                }),
                trace_id="runtime:date",
                provenance={"route_type": route_type},
            ),
            ContextItem(
                item_id="core_instruction",
                kind=ContextKind.INSTRUCTION,
                title="Core Instruction",
                content=system_prompt,
                source="system_prompt",
                priority=20,
                max_chars=2400,
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                }),
                trace_id="instruction:system_prompt",
                provenance={"route_type": route_type},
            ),
        ]

        bootstrap_docs = _load_bootstrap_documents(
            self.workspace_path,
            self.bootstrap_files,
            self.bootstrap_max_chars,
        )
        if bootstrap_docs:
            summary_lines = [f"- {doc['name']}" for doc in bootstrap_docs]
            items.append(
                ContextItem(
                    item_id="agent_bootstrap_summary",
                    kind=ContextKind.ENVIRONMENT,
                    title="Agent Bootstrap",
                    content=(
                        "This agent profile has workspace bootstrap docs loaded for the current run:\n"
                        + "\n".join(summary_lines)
                        + "\nUse them as agent-specific operating context, not as universal truth."
                    ),
                    source="bootstrap_loader",
                    priority=19,
                    max_chars=700,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="environment:bootstrap_summary",
                    provenance={"workspace_path": self.workspace_path, "count": len(bootstrap_docs)},
                )
            )
            for index, doc in enumerate(bootstrap_docs):
                lowered_name = doc["name"].lower()
                priority = 23
                if lowered_name == "identity.md":
                    priority = 21
                elif lowered_name == "agents.md":
                    priority = 22
                items.append(
                    ContextItem(
                        item_id=f"bootstrap_doc_{index}",
                        kind=ContextKind.ENVIRONMENT,
                        title=f"Bootstrap: {doc['name']}",
                        content=doc["content"],
                        source=doc["path"],
                        priority=priority,
                        max_chars=max(600, self.bootstrap_max_chars // max(1, len(bootstrap_docs))),
                        phase_visibility=frozenset({
                            ContextPhase.PLANNING,
                            ContextPhase.ACTING,
                            ContextPhase.RESPONSE,
                            ContextPhase.VERIFICATION,
                        }),
                        trace_id=f"environment:bootstrap:{doc['name']}",
                        provenance={"path": doc["path"]},
                    )
                )

        if _prefers_plaintext_chat(user_message, route_type):
            items.append(
                ContextItem(
                    item_id="conversational_response_guard",
                    kind=ContextKind.OBJECTIVE,
                    title="Conversational Response Guard",
                    content=(
                        "This turn is ordinary conversation, not an artifact request. "
                        "Respond in normal plain text. Do not output HTML, SVG, app fragments, UI mockups, faux system boot logs, or code blocks unless the user explicitly asks for them."
                    ),
                    source="conversation_guard",
                    priority=12,
                    max_chars=700,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="objective:conversation_guard",
                    provenance={"route_type": route_type},
                )
            )

        if ENABLE_MANIFEST_PROTOCOL and route_type in {"multi_step", "open_ended", "memory_recall"}:
            items.append(
                ContextItem(
                    item_id="manifest_protocol",
                    kind=ContextKind.INSTRUCTION,
                    title="Manifest Protocol",
                    content=(
                        "If needed, append one inline marker at the end only.\n"
                        "Format: [MANIFEST: LOAD=concept | FACT=subject|predicate|object | "
                        "VOIDS=subject|predicate | GOAL=goal]\n"
                        "Omit empty fields. Skip the manifest for conversational replies."
                    ),
                    source="manifest_protocol",
                    priority=25,
                    max_chars=900,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="instruction:manifest",
                    provenance={"enabled": True},
                )
            )

        if route_type in {"multi_step", "open_ended", "direct_fact", "url_fetch"}:
            items.append(
                ContextItem(
                    item_id="entity_fidelity",
                    kind=ContextKind.OBJECTIVE,
                    title="Entity Fidelity",
                    content=(
                        "Preserve the user's exact entities, domains, URLs, file names, tickers, and keywords unless verified evidence justifies normalization. "
                        "Do not silently rewrite targets like domains, company names, or search strings."
                    ),
                    source="entity_fidelity",
                    priority=17,
                    max_chars=500,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="objective:entity_fidelity",
                    provenance={"route_type": route_type},
                )
            )

        items.append(
            ContextItem(
                item_id="loop_contract",
                kind=ContextKind.OBJECTIVE,
                title="Loop Contract",
                content=(
                    "This runtime handles planning, observation, verification, and memory separately. "
                    "At the acting step, either emit one valid JSON tool call or answer the user directly if enough evidence already exists. "
                    "Do not talk about hidden prompts, strategies, evidence packets, or execution phases."
                ),
                source="loop_contract",
                priority=14,
                max_chars=550,
                phase_visibility=frozenset({
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                }),
                trace_id="objective:loop_contract",
                provenance={"route_type": route_type},
            )
        )

        profile_instruction = {
            "small_local": (
                "Model profile: small local. Keep the next step narrow. Prefer one decisive action, compact evidence use, and minimal optional explanation."
            ),
            "tool_native_local": (
                "Model profile: local native-tool. Keep context compact and prefer direct tool execution over verbose planning chatter."
            ),
            "local_standard": (
                "Model profile: local standard. Favor compact evidence packets, low branching, and concise synthesis."
            ),
            "frontier_native": (
                "Model profile: frontier native-tool. Use richer reasoning only when it materially improves the next action."
            ),
            "frontier_standard": (
                "Model profile: frontier standard. You can synthesize broadly, but still prefer evidence-first execution."
            ),
        }.get(model_profile, "")
        if profile_instruction:
            items.append(
                ContextItem(
                    item_id="model_execution_profile",
                    kind=ContextKind.OBJECTIVE,
                    title="Model Execution Profile",
                    content=profile_instruction,
                    source="runtime_profile",
                    priority=13,
                    max_chars=280,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="objective:model_profile",
                    provenance={"model_profile": model_profile},
                )
            )

        if forced_tools:
            tools_lines = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(forced_tools))
            items.append(
                ContextItem(
                    item_id="forced_tools",
                    kind=ContextKind.OBJECTIVE,
                    title="Execution Override",
                    content=(
                        "Use the following tools in the order listed below when there are dependencies between them.\n"
                        f"{tools_lines}\n"
                        "If you are at the acting step and the runtime has not already disabled tools, your next output should be exactly one valid JSON tool invocation and nothing else.\n"
                        "Do not narrate intended actions, repeat tool schemas, or provide conversational filler before the tool call."
                    ),
                    source="forced_tools",
                    priority=15,
                    max_chars=1200,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="objective:forced_tools",
                    provenance={"forced_tools": list(forced_tools)},
                )
            )

        if force_memory:
            items.append(
                ContextItem(
                    item_id="force_memory",
                    kind=ContextKind.OBJECTIVE,
                    title="Priority Instruction",
                    content=(
                        "CRITICAL: The user has requested FORCED MEMORY. "
                        "You MUST use the `query_memory` tool immediately to search for any relevant past context "
                        "before providing a finalized answer. Do not skip this step."
                    ),
                    source="force_memory",
                    priority=16,
                    max_chars=700,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                    }),
                    trace_id="objective:force_memory",
                    provenance={"forced": True},
                )
            )

        if correction_turn:
            items.append(
                ContextItem(
                    item_id="correction_turn",
                    kind=ContextKind.OBJECTIVE,
                    title="Current Turn Override",
                    content=(
                        "The current user message appears to correct or update prior information. "
                        "For this response, treat the latest user-provided fact as more authoritative than retrieved historical context. "
                        "Do not restate superseded facts unless explicitly contrasting old vs new. "
                        "Prefer current-turn facts over older tool outputs and historical anchors whenever they conflict."
                    ),
                    source="correction_turn",
                    priority=18,
                    max_chars=1000,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.MEMORY_COMMIT,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="objective:correction",
                    provenance={"user_message_preview": user_message[:120]},
                )
            )

        if direct_fact_memory_only:
            items.append(
                ContextItem(
                    item_id="direct_fact_memory_only",
                    kind=ContextKind.OBJECTIVE,
                    title="Trusted Fact Answer",
                    content=(
                        "This direct fact question is already answerable from the trusted deterministic facts in context. "
                        "Answer from those facts only. Do not call web_search, web_fetch, query_memory, or any other tool for this turn."
                    ),
                    source="direct_fact_memory_only",
                    priority=24,
                    max_chars=420,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="objective:direct_fact_memory_only",
                    provenance={"route_type": route_type},
                )
            )

        if first_message is not None:
            total_turns = max(1, (message_count + 1) // 2)
            items.append(
                ContextItem(
                    item_id="session_anchor",
                    kind=ContextKind.WORKING,
                    title="Session Anchor",
                    content=f"First message: \"{first_message}\"\nTotal turns so far: {total_turns}",
                    source="session",
                    priority=30,
                    max_chars=800,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                    }),
                    trace_id="working:session_anchor",
                    provenance={"turn_count": total_turns},
                )
            )

        if current_facts:
            fact_lines = [f"FACT: {s} {p} {o}".strip() for (s, p, o) in current_facts]
            items.append(
                ContextItem(
                    item_id="deterministic_facts",
                    kind=ContextKind.MEMORY,
                    title="Deterministic Facts",
                    content=(
                        "The following facts are currently true and override any prior memory:\n"
                        + "\n".join(fact_lines)
                    ),
                    source="semantic_graph",
                    priority=22,
                    max_chars=1200,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.MEMORY_COMMIT,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="memory:deterministic_facts",
                    provenance={"fact_count": len(current_facts)},
                )
            )

        if task_state:
            items.append(
                ContextItem(
                    item_id="session_task_state",
                    kind=ContextKind.WORKING,
                    title="Session Task State",
                    content=(
                        "Keep this list current during multi-step workflows.\n"
                        "Use todo_write only to initialize or intentionally replace the full task list.\n"
                        "Once tasks already exist, prefer todo_update for status changes and substantive tools for actual work.\n"
                        "Task tracking supports execution; it is not the final answer.\n"
                        f"{task_state}"
                    ),
                    source="task_tracker",
                    priority=26,
                    max_chars=1200,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                    }),
                    trace_id="working:task_state",
                    provenance={"has_task_state": True},
                )
            )

        if candidate_context:
            items.append(
                ContextItem(
                    item_id="candidate_context",
                    kind=ContextKind.WORKING,
                    title="Candidate Signals",
                    content=(
                        "These are low-confidence candidate facts/signals. "
                        "Use them as hints for planning or verification, not as deterministic truth.\n"
                        f"{candidate_context}"
                    ),
                    source="candidate_context",
                    priority=34,
                    max_chars=900,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="working:candidate_context",
                    provenance={"line_count": len([line for line in candidate_context.splitlines() if line.strip()])},
                )
            )

        if loop_checkpoint:
            observation_lines = [f"Manager status: {loop_checkpoint.status or 'unknown'}"]
            if loop_checkpoint.strategy:
                observation_lines.append(f"Strategy: {loop_checkpoint.strategy}")
            if loop_checkpoint.tools:
                observation_lines.append(
                    "Preferred next tools: " + ", ".join(str(name) for name in loop_checkpoint.tools)
                )
            if loop_checkpoint.notes:
                observation_lines.append(f"Notes: {loop_checkpoint.notes}")
            tool_result_lines = []
            for item in (loop_checkpoint.tool_results or [])[:4]:
                tool_name = str(item.get("tool_name") or "unknown")
                success = "ok" if item.get("success") else "failed"
                preview = str(item.get("preview") or "").strip()
                tool_result_lines.append(f"- {tool_name} [{success}]: {preview}")
            if tool_result_lines:
                observation_lines.append("Recent tool observations:")
                observation_lines.extend(tool_result_lines)
            if loop_checkpoint.candidate_facts:
                observation_lines.append("Candidate facts under review:")
                observation_lines.extend(f"- {item}" for item in loop_checkpoint.candidate_facts[:4])
            items.append(
                ContextItem(
                    item_id="observation_state",
                    kind=ContextKind.WORKING,
                    title="Observation State",
                    content="\n".join(observation_lines),
                    source="manager_observation",
                    priority=24,
                    max_chars=1000,
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.MEMORY_COMMIT,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="working:observation_state",
                    provenance={
                        "status": loop_checkpoint.status,
                        "tool_turn": loop_checkpoint.tool_turn,
                        "tool_count": len(loop_checkpoint.tools or []),
                        "candidate_fact_count": len(loop_checkpoint.candidate_facts or []),
                    },
                )
            )

        historical_content = self._build_historical_context_content(
            unified_hits,
            current_facts,
            correction_turn=correction_turn,
        )
        if historical_content:
            items.append(
                ContextItem(
                    item_id="historical_context",
                    kind=ContextKind.MEMORY,
                    title="Historical Context",
                    content=historical_content,
                    source="retrieval",
                    priority=40,
                    max_chars=1800 if correction_turn else 3200,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="memory:historical_context",
                    provenance={"anchor_count": len(unified_hits or [])},
                )
            )

        ctx_items = []
        build_context_items = getattr(ctx_engine, "build_context_items", None)
        if callable(build_context_items):
            try:
                ctx_items = [
                    item for item in (build_context_items(context) or [])
                    if isinstance(item, ContextItem)
                ]
            except Exception:
                ctx_items = []
        if not ctx_items:
            ctx_block = ctx_engine.build_context_block(context)
            if ctx_block:
                ctx_items = [
                    ContextItem(
                        item_id="context_engine_memory",
                        kind=ContextKind.MEMORY,
                        title="",
                        content=ctx_block,
                        source="context_engine",
                        priority=60,
                        max_chars=1800,
                        phase_visibility=frozenset({
                            ContextPhase.PLANNING,
                            ContextPhase.ACTING,
                            ContextPhase.RESPONSE,
                            ContextPhase.VERIFICATION,
                        }),
                        trace_id="memory:context_engine",
                        provenance={"engine": ctx_engine.__class__.__name__},
                        formatted=True,
                    )
                ]
        items.extend(ctx_items)

        allowed_tools = set(forced_tools or []) if forced_tools else None
        tools_block = self.tools.build_tools_block(allowed_names=allowed_tools)
        if tools_block:
            items.append(
                ContextItem(
                    item_id="available_tools",
                    kind=ContextKind.ENVIRONMENT,
                    title="",
                    content=tools_block,
                    source="tool_registry",
                    priority=70,
                    max_chars=6000,
                    phase_visibility=frozenset({ContextPhase.ACTING}),
                    trace_id="environment:tools",
                    provenance={"forced_only": bool(allowed_tools)},
                    formatted=True,
                )
            )

        return items

    def _build_messages(
        self,
        system_prompt:      str,
        context:            str,
        sliding_window:     list[dict],
        user_message:       str,
        first_message:      Optional[str] = None,
        message_count:      int = 0,
        unified_hits:       Optional[list[dict]] = None,
        force_memory:       bool = False,
        forced_tools:       Optional[list[str]] = None,
        current_facts:      Optional[list[tuple[str, str, str]]] = None,
        task_state:         Optional[str] = None,
        candidate_context:  Optional[str] = None,
        loop_checkpoint:    Optional[LoopCheckpoint] = None,
        correction_turn:    bool = False,
        route_type:         str = "open_ended",
        direct_fact_memory_only: bool = False,
        ctx_engine:         Optional[ContextEngine] = None,
        phase:             ContextPhase = ContextPhase.ACTING,
        model_profile:     str = "frontier_standard",
    ) -> list[dict]:
        context_items = self._build_context_items(
            system_prompt=system_prompt,
            context=context,
            user_message=user_message,
            first_message=first_message,
            message_count=message_count,
            unified_hits=unified_hits,
            force_memory=force_memory,
            forced_tools=forced_tools,
            current_facts=current_facts,
            task_state=task_state,
            candidate_context=candidate_context,
            loop_checkpoint=loop_checkpoint,
            correction_turn=correction_turn,
            route_type=route_type,
            direct_fact_memory_only=direct_fact_memory_only,
            ctx_engine=ctx_engine,
            model_profile=model_profile,
        )
        compiled = compile_context_items(
            context_items,
            phase=phase,
            char_budget=_model_profile_budget(model_profile, "system_chars", SYSTEM_PROMPT_CHAR_BUDGET),
            truncate_section=lambda content, budget: _truncate_prompt_section(
                content,
                budget,
                preserve_ends=True,
            ),
        )
        self._last_context_compilation = compiled.to_trace_payload()
        self._last_context_compilation["model_profile"] = model_profile

        messages = [{"role": "system", "content": compiled.content}]

        max_chars_per_window_msg = _model_profile_budget(model_profile, "history_chars_per_msg", 4000)
        last_reminder_index = -1
        for i, message in enumerate(sliding_window):
            if message.get("role") == "user" and "<system-reminder>" in message.get("content", ""):
                last_reminder_index = i

        retained_history = 0
        truncated_history = 0
        for i, message in enumerate(sliding_window):
            if message["role"] in ("user", "assistant"):
                content = message["content"]
                if message["role"] == "user" and i != last_reminder_index and "<system-reminder>" in content:
                    content = _SYSTEM_REMINDER_BLOCK_RE.sub("\n", content).strip()
                    if not content:
                        continue
                if len(content) > max_chars_per_window_msg:
                    half = max_chars_per_window_msg // 2
                    content = f"{content[:half]}\n\n[...large block truncated...]\n\n{content[-half:]}"
                    truncated_history += 1
                retained_history += 1
                messages.append({"role": message["role"], "content": content})
        messages.append({"role": "user", "content": user_message})

        self._last_context_compilation["history"] = {
            "retained_count": retained_history,
            "truncated_count": truncated_history,
            "input_count": len(sliding_window),
            "max_chars_per_window_msg": max_chars_per_window_msg,
        }
        return messages

    def _compile_phase_context(
        self,
        *,
        phase: ContextPhase,
        system_prompt: str,
        context: str,
        user_message: str,
        first_message: Optional[str] = None,
        message_count: int = 0,
        unified_hits: Optional[list[dict]] = None,
        force_memory: bool = False,
        forced_tools: Optional[list[str]] = None,
        current_facts: Optional[list[tuple[str, str, str]]] = None,
        task_state: Optional[str] = None,
        candidate_context: Optional[str] = None,
        loop_checkpoint: Optional[LoopCheckpoint] = None,
        correction_turn: bool = False,
        route_type: str = "open_ended",
        direct_fact_memory_only: bool = False,
        ctx_engine: Optional[ContextEngine] = None,
        model_profile: str = "frontier_standard",
    ) -> dict:
        context_items = self._build_context_items(
            system_prompt=system_prompt,
            context=context,
            user_message=user_message,
            first_message=first_message,
            message_count=message_count,
            unified_hits=unified_hits,
            force_memory=force_memory,
            forced_tools=forced_tools,
            current_facts=current_facts,
            task_state=task_state,
            candidate_context=candidate_context,
            loop_checkpoint=loop_checkpoint,
            correction_turn=correction_turn,
            route_type=route_type,
            direct_fact_memory_only=direct_fact_memory_only,
            ctx_engine=ctx_engine,
            model_profile=model_profile,
        )
        compiled = compile_context_items(
            context_items,
            phase=phase,
            char_budget=_model_profile_budget(model_profile, "system_chars", SYSTEM_PROMPT_CHAR_BUDGET),
            truncate_section=lambda content, budget: _truncate_prompt_section(
                content,
                budget,
                preserve_ends=True,
            ),
        )
        payload = compiled.to_trace_payload()
        payload["model_profile"] = model_profile
        payload["content"] = compiled.content
        return payload

    @staticmethod
    def _strip_detected_tool_json(text: str, calls: list) -> str:
        clean = text or ""
        for call in calls:
            raw_json = getattr(call, "raw_json", "")
            if raw_json and raw_json in clean:
                clean = clean[:clean.index(raw_json)].rstrip()
        return AgentCore._strip_internal_execution_chatter(clean)
