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

import os
from llm.base_adapter import BaseLLMAdapter

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
    from llm.llm_adapter import OllamaAdapter
    from llm.openai_adapter import OpenAIAdapter
    from llm.groq_adapter import GroqLLMAdapter

    if isinstance(adapter, OllamaAdapter):
        return os.getenv("DEFAULT_MODEL", "llama3.2")
    if isinstance(adapter, OpenAIAdapter):
        return os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    if isinstance(adapter, GroqLLMAdapter):
        return os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
    
    from llm.gemini_adapter import GeminiAdapter
    if isinstance(adapter, GeminiAdapter):
        return os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
    
    from llm.anthropic_adapter import AnthropicAdapter
    if isinstance(adapter, AnthropicAdapter):
        return os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-latest")

    return "llama3.2"
