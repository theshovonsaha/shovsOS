"""
Model Capability Registry
-------------------------
Classifies model identifiers as `chat`, `embed`, or `both` so the runtime can
refuse to route an embedding-only model into a chat slot (and vice versa).

Patterns are checked against the bare model name AFTER stripping any
"provider:" prefix. Anything unknown defaults to `chat` — chat models are
the common case and we do not want to hard-fail on unfamiliar identifiers.
Vision is tracked as an additional capability because a model can be both
chat-capable and image-capable.
"""

from __future__ import annotations

import re
from typing import Literal

Capability = Literal["chat", "embed", "both"]

# Patterns that strongly indicate an embedding-only model.
_EMBED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bembed", re.IGNORECASE),
    re.compile(r"^nomic-embed", re.IGNORECASE),
    re.compile(r"^bge-", re.IGNORECASE),
    re.compile(r"^e5-", re.IGNORECASE),
    re.compile(r"^gte-", re.IGNORECASE),
    re.compile(r"text-embedding-", re.IGNORECASE),
    re.compile(r"\bmxbai-embed", re.IGNORECASE),
    re.compile(r"\bsentence-transformers\b", re.IGNORECASE),
)

# Explicit overrides — model IDs we know are chat even if they look embed-ish.
_CHAT_OVERRIDES: frozenset[str] = frozenset({})
_KNOWN_PROVIDERS: frozenset[str] = frozenset({
    "ollama",
    "openai",
    "local_openai",
    "lmstudio",
    "llamacpp",
    "groq",
    "gemini",
    "anthropic",
    "nvidia",
    "bridge",
})


_VISION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI multimodal families
    re.compile(r"gpt-4o", re.IGNORECASE),
    re.compile(r"gpt-4\.1", re.IGNORECASE),
    re.compile(r"gpt-5", re.IGNORECASE),
    re.compile(r"\bo[3-9]\b", re.IGNORECASE),
    # Gemini multimodal families
    re.compile(r"gemini-(?:1\.5|2\.0|2\.5|3)", re.IGNORECASE),
    re.compile(r"gemini.*vision", re.IGNORECASE),
    # Anthropic Claude 3+ accepts image parts.
    re.compile(r"claude-3", re.IGNORECASE),
    re.compile(r"claude-(?:opus|sonnet|haiku)-4", re.IGNORECASE),
    # Common local/Ollama vision model names.
    re.compile(r"\bllava\b", re.IGNORECASE),
    re.compile(r"\bmoondream\b", re.IGNORECASE),
    re.compile(r"\bminicpm-v\b", re.IGNORECASE),
    re.compile(r"\bqwen2(?:\.5)?-vl\b", re.IGNORECASE),
    re.compile(r"\bqwen-vl\b", re.IGNORECASE),
    re.compile(r"\bbakllava\b", re.IGNORECASE),
    re.compile(r"\bvision\b", re.IGNORECASE),
)


def _strip_provider(model: str) -> str:
    if ":" in model:
        provider, bare = model.split(":", 1)
        if provider.strip().lower() in _KNOWN_PROVIDERS:
            return bare.strip()
    return model.strip()


def classify_model(model: str | None) -> Capability:
    """Return the capability of a (possibly provider-prefixed) model id."""
    if not model:
        return "chat"
    bare = _strip_provider(model)
    if not bare:
        return "chat"
    if bare.lower() in _CHAT_OVERRIDES:
        return "chat"
    for pattern in _EMBED_PATTERNS:
        if pattern.search(bare):
            return "embed"
    return "chat"


def is_chat_capable(model: str | None) -> bool:
    return classify_model(model) in ("chat", "both")


def is_embed_capable(model: str | None) -> bool:
    return classify_model(model) in ("embed", "both")


def is_vision_capable(model: str | None) -> bool:
    if not model or not is_chat_capable(model):
        return False
    bare = _strip_provider(model)
    return any(pattern.search(bare) for pattern in _VISION_PATTERNS)


def capability_flags(model: str | None) -> dict[str, bool | str]:
    return {
        "model": model or "",
        "chat": is_chat_capable(model),
        "embed": is_embed_capable(model),
        "vision": is_vision_capable(model),
        "reasoning": supports_reasoning(model),
        "class": classify_model(model),
    }


def coerce_chat_model(model: str | None, fallback: str) -> tuple[str, bool]:
    """
    Return (model_to_use, was_replaced).
    If `model` is embed-only, fall back to `fallback` and signal the swap.
    """
    if model and is_chat_capable(model):
        return model, False
    return fallback, bool(model) and not is_chat_capable(model)


def coerce_embed_model(model: str | None, fallback: str) -> tuple[str, bool]:
    """
    Return (model_to_use, was_replaced).
    If `model` is chat-only (and looks like a chat model — no embed signal),
    fall back to `fallback`.
    """
    if model and is_embed_capable(model):
        return model, False
    if not model:
        return fallback, False
    return fallback, True


def coerce_vision_model(model: str | None, fallback: str) -> tuple[str, bool]:
    if model and is_vision_capable(model):
        return model, False
    if fallback and is_vision_capable(fallback):
        return fallback, bool(model)
    return model or fallback, False


# ── Reasoning capability ────────────────────────────────────────────────────
# Patterns that match models with provider-side reasoning / extended thinking.
# When ``supports_reasoning`` returns False the engine can skip threading the
# flag through entirely. When it returns True, the adapter is responsible for
# mapping ``reasoning_enabled`` to the right native knob.
#
# This list is conservative — false negatives just mean we leave the flag at
# provider default for that model (safe), while false positives could enable
# a knob the model doesn't actually support (sometimes harmless, sometimes
# an API error). When in doubt, omit.
_REASONING_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI reasoning models
    re.compile(r"^o[1-9](?:-|$)", re.IGNORECASE),       # o1, o3, o4, …
    re.compile(r"^gpt-5", re.IGNORECASE),               # gpt-5 series
    # Anthropic extended thinking — Claude 3.7+ / Claude 4
    re.compile(r"claude-3-7", re.IGNORECASE),
    re.compile(r"claude-(?:opus|sonnet|haiku)-4", re.IGNORECASE),
    # Gemini thinking — 2.5+
    re.compile(r"gemini-2\.5", re.IGNORECASE),
    re.compile(r"gemini-3", re.IGNORECASE),
    # DeepSeek-R1 family
    re.compile(r"deepseek.*r1", re.IGNORECASE),
    re.compile(r"\br1\b", re.IGNORECASE),
    # Qwen QwQ / Qwen3 thinking
    re.compile(r"qwq", re.IGNORECASE),
    re.compile(r"qwen3", re.IGNORECASE),
    # Ollama tagged thinking models
    re.compile(r"thinking", re.IGNORECASE),
)


def supports_reasoning(model: str | None) -> bool:
    """Return True if ``model`` is known to expose a reasoning / extended-
    thinking knob the adapter can flip with ``reasoning_enabled``.

    Used by callers that want to avoid paying for the kwarg on models where
    it would be a no-op. Adapters still tolerate the flag on any model — the
    contract is "ignore silently when unsupported", not "raise."
    """
    if not model:
        return False
    bare = _strip_provider(model)
    if not bare:
        return False
    return any(p.search(bare) for p in _REASONING_PATTERNS)


# Current-generation Anthropic models use ADAPTIVE thinking and reject the old
# ``{type:"enabled", budget_tokens:N}`` shape (400). Opus 4.6+, Sonnet 4.6+, and
# the Fable/Mythos 5 family. Older Claude (3.7 / 4.0 / 4.5) still take budget_tokens.
_ADAPTIVE_THINKING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"claude-opus-4-(?:[6-9]|\d\d)\b", re.IGNORECASE),
    re.compile(r"claude-sonnet-4-(?:[6-9]|\d\d)\b", re.IGNORECASE),
    re.compile(r"claude-(?:fable|mythos)-\d", re.IGNORECASE),
)

# Models that reject sampling params (temperature / top_p / top_k) with a 400:
# Opus 4.7 / 4.8 and the Fable/Mythos 5 family. (Opus 4.6 and Sonnet 4.6 still
# accept temperature.)
_NO_SAMPLING_PARAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"claude-opus-4-(?:[7-9]|\d\d)\b", re.IGNORECASE),
    re.compile(r"claude-(?:fable|mythos)-\d", re.IGNORECASE),
)


def uses_adaptive_thinking(model: str | None) -> bool:
    """True for current Claude models that require ``thinking:{type:"adaptive"}``
    and 400 on ``budget_tokens``."""
    if not model:
        return False
    bare = _strip_provider(model)
    return any(p.search(bare) for p in _ADAPTIVE_THINKING_PATTERNS)


def rejects_sampling_params(model: str | None) -> bool:
    """True for models that 400 when sent temperature / top_p / top_k
    (Opus 4.7+, Fable 5, Mythos 5)."""
    if not model:
        return False
    bare = _strip_provider(model)
    return any(p.search(bare) for p in _NO_SAMPLING_PARAM_PATTERNS)
