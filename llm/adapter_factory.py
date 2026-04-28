"""
Adapter Factory
----------------
Instantiates the correct LLM adapter based on environment configuration.

Priority order (first configured wins):
  1. LMSTUDIO_BASE_URL set → LM Studio (OpenAI-compatible)
  2. LLAMACPP_BASE_URL set → llama.cpp (OpenAI-compatible)
  3. OPENAI_BASE_URL set → Local OpenAI-compatible server
  4. OPENAI_API_KEY set → OpenAI cloud
  5. GROQ_API_KEY set → GroqLLMAdapter
  6. GEMINI_API_KEY set → GeminiAdapter

Override with: LLM_PROVIDER=ollama|openai|local_openai|lmstudio|llamacpp|groq|gemini|anthropic
"""

import logging
import os
import time
from llm.base_adapter import BaseLLMAdapter

log = logging.getLogger("shovs.adapter_factory")

# ── Global Adapter Cache ──────────────────────────────────────────────────
_ADAPTER_CACHE: dict[str, BaseLLMAdapter] = {}
_OPENAI_COMPAT_ALIASES = {"openai", "local_openai", "lmstudio", "llamacpp"}
_KNOWN_PROVIDERS = {
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
}

# ── Failover tracking ─────────────────────────────────────────────────────
# Maps provider → (failure_count, last_failure_ts)
_PROVIDER_FAILURES: dict[str, tuple[int, float]] = {}
_FAILOVER_COOLDOWN_SECS = int(os.getenv("SHOVS_PROVIDER_COOLDOWN", "120"))
_FAILOVER_THRESHOLD = int(os.getenv("SHOVS_PROVIDER_FAIL_THRESHOLD", "3"))

# Ordered fallback chain read from env: comma-separated provider names
# e.g. SHOVS_PROVIDER_FALLBACK_CHAIN=anthropic,groq,openai,ollama
_FALLBACK_CHAIN: list[str] = [
    p.strip().lower()
    for p in os.getenv("SHOVS_PROVIDER_FALLBACK_CHAIN", "").split(",")
    if p.strip()
]


def record_provider_failure(provider: str) -> None:
    """
    Record a failure for a provider. After _FAILOVER_THRESHOLD consecutive
    failures the provider is considered degraded and will be skipped in
    get_failover_adapter() for _FAILOVER_COOLDOWN_SECS.
    """
    count, _ = _PROVIDER_FAILURES.get(provider, (0, 0.0))
    _PROVIDER_FAILURES[provider] = (count + 1, time.monotonic())
    if count + 1 >= _FAILOVER_THRESHOLD:
        log.warning(
            "Provider '%s' has failed %d times — marked degraded for %ds",
            provider, count + 1, _FAILOVER_COOLDOWN_SECS,
        )


def record_provider_success(provider: str) -> None:
    """Reset failure count for a provider after a successful call."""
    if provider in _PROVIDER_FAILURES:
        del _PROVIDER_FAILURES[provider]


def _is_degraded(provider: str) -> bool:
    count, last_ts = _PROVIDER_FAILURES.get(provider, (0, 0.0))
    if count < _FAILOVER_THRESHOLD:
        return False
    return (time.monotonic() - last_ts) < _FAILOVER_COOLDOWN_SECS


def get_failover_adapter(current_provider: str) -> BaseLLMAdapter | None:
    """
    Return the next healthy adapter in the fallback chain, skipping
    `current_provider` and any degraded providers.

    Returns None if no healthy fallback is available.
    Configure the chain via SHOVS_PROVIDER_FALLBACK_CHAIN env var.
    Example: SHOVS_PROVIDER_FALLBACK_CHAIN=anthropic,groq,openai,ollama
    """
    if not _FALLBACK_CHAIN:
        return None

    for candidate in _FALLBACK_CHAIN:
        if candidate == current_provider:
            continue
        if _is_degraded(candidate):
            log.debug("Failover candidate '%s' is degraded — skipping", candidate)
            continue
        try:
            adapter = create_adapter(candidate)
            log.warning("Failing over from '%s' to '%s'", current_provider, candidate)
            return adapter
        except Exception as exc:
            log.warning("Failover candidate '%s' failed to instantiate: %s", candidate, exc)
            continue

    log.error("No healthy failover provider found after exhausting chain: %s", _FALLBACK_CHAIN)
    return None


def _build_openai_compat_adapter(provider: str):
    from llm.openai_adapter import OpenAIAdapter

    if provider == "lmstudio":
        return OpenAIAdapter(
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        )
    if provider == "llamacpp":
        return OpenAIAdapter(
            api_key=os.getenv("LLAMACPP_API_KEY", "llama.cpp"),
            base_url=os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1"),
        )
    if provider == "local_openai":
        return OpenAIAdapter(
            api_key=os.getenv("OPENAI_API_KEY", "local"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return OpenAIAdapter(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

def create_adapter(provider: str = None) -> BaseLLMAdapter:
    """
    Create (or return cached) LLM adapter based on provider string.
    """
    # 1. Resolve provider identifier
    target_provider = "auto"
    if provider:
        p = provider.strip()
        p_part = p.lower()
        if ":" in p:
            p_part = p.split(":", 1)[0].lower()
        elif "/" in p:
            p_part = p.split("/", 1)[0].lower()

        if p_part in _KNOWN_PROVIDERS:
            target_provider = p_part
        else:
            target_provider = p.lower()
    else:
        target_provider = os.getenv("LLM_PROVIDER", "auto")

    # 2. Determine actual provider class to use
    final_provider = target_provider
    if target_provider == "auto":
        if os.getenv("LMSTUDIO_BASE_URL"):
            final_provider = "lmstudio"
        elif os.getenv("LLAMACPP_BASE_URL"):
            final_provider = "llamacpp"
        elif os.getenv("OPENAI_BASE_URL"):
            final_provider = "local_openai"
        elif os.getenv("OPENAI_API_KEY"):
            final_provider = "openai"
        elif os.getenv("GROQ_API_KEY"): final_provider = "groq"
        elif os.getenv("GEMINI_API_KEY"): final_provider = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"): final_provider = "anthropic"
        elif os.getenv("NVIDIA_API_KEY"): final_provider = "nvidia"
        else: final_provider = "ollama"

    # 3. Cache lookup
    if final_provider in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[final_provider]

    # 4. Instantiate and cache
    if final_provider in _OPENAI_COMPAT_ALIASES:
        adapter = _build_openai_compat_adapter(final_provider)
    elif final_provider == "groq":
        from llm.groq_adapter import GroqLLMAdapter
        adapter = GroqLLMAdapter()
    elif final_provider == "gemini":
        from llm.gemini_adapter import GeminiAdapter
        adapter = GeminiAdapter()
    elif final_provider == "anthropic":
        from llm.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter()
    elif final_provider == "nvidia":
        from llm.nvidia_adapter import NvidiaAdapter
        adapter = NvidiaAdapter()
    elif final_provider == "bridge":
        from llm.bridge_adapter import BridgeAdapter
        adapter = BridgeAdapter()
    else:
        from llm.llm_adapter import OllamaAdapter
        adapter = OllamaAdapter()

    _ADAPTER_CACHE[final_provider] = adapter
    return adapter


def strip_provider_prefix(model_name: str) -> str:
    """
    Removes the "provider:" prefix if present.
    Example: "groq:llama-..." -> "llama-..."
    """
    if not model_name:
        return model_name

    if ":" in model_name:
        parts = model_name.split(":", 1)
        if parts[0].lower() in _KNOWN_PROVIDERS:
            return parts[1]
    if "/" in model_name:
        parts = model_name.split("/", 1)
        if parts[0].lower() in _KNOWN_PROVIDERS:
            return parts[1]
    return model_name


def get_default_model(adapter: BaseLLMAdapter) -> str:
    """Return sensible default model for each provider type."""
    from config.config import cfg
    from llm.llm_adapter import OllamaAdapter
    from llm.openai_adapter import OpenAIAdapter
    from llm.groq_adapter import GroqLLMAdapter

    if isinstance(adapter, OllamaAdapter):
        return cfg.OLLAMA_DEFAULT_MODEL
    if isinstance(adapter, OpenAIAdapter):
        return cfg.OPENAI_DEFAULT_MODEL
    if isinstance(adapter, GroqLLMAdapter):
        return cfg.GROQ_DEFAULT_MODEL
    
    from llm.gemini_adapter import GeminiAdapter
    if isinstance(adapter, GeminiAdapter):
        return cfg.GEMINI_DEFAULT_MODEL
    
    from llm.anthropic_adapter import AnthropicAdapter
    if isinstance(adapter, AnthropicAdapter):
        return cfg.ANTHROPIC_DEFAULT_MODEL

    return cfg.DEFAULT_MODEL
