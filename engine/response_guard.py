from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_INTERNAL_PHRASES = (
    "canonical run ledger",
    "run ledger",
    "phase packet",
    "memory_commit",
    "tool_call_id",
    "tool_result_id",
    "hidden architecture",
)


_ARCHITECTURE_QUERY_TERMS = (
    "architecture",
    "run ledger",
    "phase packet",
    "runtime",
    "trace",
    "traces",
    "tool_call_id",
    "tool result",
    "memory_commit",
)


@dataclass
class ResponseGuardResult:
    text: str
    changed: bool = False
    issues: list[str] = field(default_factory=list)
    replacement_used: bool = False


def is_small_or_local_model(model: str, profile: str = "") -> bool:
    profile_l = str(profile or "").lower()
    model_l = str(model or "").lower()
    if profile_l in {"small_local", "tool_native_local"}:
        return True
    if model_l.startswith(("ollama:", "lmstudio:", "llamacpp:", "local_openai:")):
        return bool(re.search(r"(?::|[-_])(1b|2b|3b|mini)\b", model_l)) or "llama3.2" in model_l
    return False


_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
# Tag-wrapped tool calls emitted as text by reasoning models that the runtime
# never executed, e.g. gpt-oss `<tool_code>{...}</tool_code>` or
# `<tool_call>{...}</tool_call>`. These must never be shown as the answer.
_TOOL_TAG_BLOCK_RE = re.compile(
    r"<(tool_code|tool_call|tool_use|function_call)>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_JSON_KEYS = ("tool_use", "tool_calls", "tool_call", "tool_name", "function")


def _loads_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_embedded_json_object(text: str) -> tuple[str, dict[str, Any]]:
    """Find a top-level JSON object embedded inside prose.

    Reasoning models (e.g. qwen3 thinking mode) sometimes emit a final answer
    that is prose/CoT followed by a literal tool-call blob like
    ``{"thought": "...", "tool_use": {"function": "store_memory", ...}}``. The
    whole text is not valid JSON, so ``_loads_json_object`` misses it. This
    isolates the ``{...}`` span and parses it so the guard can strip it.
    """
    s = str(text or "")
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "", {}
    candidate = s[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except Exception:
        return "", {}
    return candidate, (parsed if isinstance(parsed, dict) else {})


def _looks_like_embedded_tool_json(parsed: dict[str, Any]) -> bool:
    if not parsed:
        return False
    return any(key in parsed for key in _TOOL_JSON_KEYS)


def looks_like_tool_json(text: str) -> bool:
    payload = _loads_json_object(text)
    if not payload:
        return False
    if "tool_calls" in payload:
        return True
    if payload.get("tool") or payload.get("tool_name"):
        return isinstance(payload.get("arguments"), dict) or "arguments" in payload
    if payload.get("function") and isinstance(payload.get("function"), dict):
        return True
    return False


def _fallback_from_tool_json(text: str, *, user_message: str = "") -> str:
    payload = _loads_json_object(text)
    if not payload:
        return ""
    args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    tool = str(payload.get("tool") or payload.get("tool_name") or "").strip().lower()

    if tool in {"greet", "greeting", "hello"}:
        message = str(args.get("message") or args.get("text") or "").strip()
        return message or "Hi."

    preferred_name = str(args.get("preferred_name") or args.get("name") or "").strip()
    if preferred_name:
        if re.search(r"\b(call|name|what should you call)\b", user_message or "", re.IGNORECASE):
            return f"You should be called {preferred_name}."
        return preferred_name

    message = str(args.get("message") or args.get("text") or "").strip()
    if tool in {"greeting", "hello"} and message:
        return message

    query = str(args.get("query") or args.get("keywords") or "").strip()
    if not query:
        query = str(args.get("q") or "").strip()
    name_match = re.search(r"(?:preferred\s+name|name)\s*:\s*([A-Z][\w.-]+)", query)
    if name_match:
        return f"You should be called {name_match.group(1)}."
    if query:
        return f"I need to answer directly from the available context. The relevant query was: {query}."

    return "I do not have a valid final answer yet."


def _strip_internal_sentences(text: str) -> tuple[str, bool]:
    raw = str(text or "")
    if not raw.strip():
        return "", False
    parts = re.split(r"(?<=[.!?])\s+", raw.strip())
    kept: list[str] = []
    changed = False
    for part in parts:
        lowered = part.lower()
        if any(phrase in lowered for phrase in _INTERNAL_PHRASES):
            changed = True
            continue
        kept.append(part)
    clean = " ".join(kept).strip()
    return clean, changed


def guard_final_response(
    text: str,
    *,
    user_message: str = "",
    model: str = "",
    model_profile: str = "",
    strict: bool | None = None,
) -> ResponseGuardResult:
    """Validate and clean final user-visible text.

    This guard is output-only. It does not change planning, acting, tool
    selection, or evidence gathering. It catches final-answer leaks that are
    common with small/local models: tool-call-shaped JSON and hidden runtime
    implementation language.
    """

    original = str(text or "")
    clean = original.strip()
    issues: list[str] = []
    changed = False
    replacement_used = False
    strict_mode = is_small_or_local_model(model, model_profile) if strict is None else bool(strict)

    # 1) Strip leaked reasoning blocks (qwen/deepseek thinking mode etc.) and
    #    tag-wrapped tool calls the runtime never executed (gpt-oss
    #    <tool_code>...). These must never reach the user as the final answer.
    lowered_clean = clean.lower()
    if "<think" in lowered_clean:
        stripped_think = _THINK_BLOCK_RE.sub("", clean).strip()
        if stripped_think != clean:
            issues.append("leaked_reasoning_block")
            clean = stripped_think
            changed = True
    if any(tag in clean.lower() for tag in ("<tool_code", "<tool_call", "<tool_use", "<function_call")):
        stripped_tool_tag = _TOOL_TAG_BLOCK_RE.sub("", clean).strip()
        if stripped_tool_tag != clean:
            issues.append("leaked_tool_call_tag")
            clean = stripped_tool_tag
            changed = True

    if looks_like_tool_json(clean):
        issues.append("tool_json_final_response")
        fallback = _fallback_from_tool_json(clean, user_message=user_message)
        clean = fallback or ""
        changed = True
        replacement_used = True
    else:
        # 2) A tool-call blob leaked at the end of an otherwise-prose answer
        #    (e.g. reasoning followed by {"tool_use": {"function": ...}}). Remove
        #    the blob from the visible text; if nothing meaningful remains, fall
        #    back so the raw JSON is never shown.
        embedded_json, embedded = _extract_embedded_json_object(clean)
        if embedded_json and _looks_like_embedded_tool_json(embedded):
            prose = clean.replace(embedded_json, "").strip()
            issues.append("embedded_tool_json")
            changed = True
            if prose:
                clean = prose
            else:
                clean = _fallback_from_tool_json(embedded_json, user_message=user_message) or ""
                replacement_used = True

    query_allows_architecture = any(term in str(user_message or "").lower() for term in _ARCHITECTURE_QUERY_TERMS)
    if clean and (strict_mode or not query_allows_architecture) and not query_allows_architecture:
        stripped, stripped_changed = _strip_internal_sentences(clean)
        if stripped_changed:
            issues.append("hidden_runtime_language")
            clean = stripped or clean
            changed = True

    if not clean and original.strip():
        clean = "I do not have a valid final answer yet."
        issues.append("empty_after_response_guard")
        changed = True
        replacement_used = True

    return ResponseGuardResult(
        text=clean,
        changed=changed,
        issues=issues,
        replacement_used=replacement_used,
    )
