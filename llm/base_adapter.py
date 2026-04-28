"""
BaseLLMAdapter — Abstract interface for LLM providers
------------------------------------------------------
Swap LLM backends by implementing this interface.
The rest of the system only depends on this ABC.

Implementations:
  - OllamaAdapter (llm_adapter.py) — local Ollama
  - OpenAIAdapter (openai_adapter.py) — OpenAI / Azure
  - GroqLLMAdapter (groq_adapter.py) — Groq cloud
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class LLMError(Exception):
    """General error for LLM failures."""
    pass


class RateLimitError(LLMError):
    """Raised when the provider returns a 429 (Rate Limit)."""
    pass


class ProviderError(LLMError):
    """Raised when the provider is down or returns a 5xx error."""
    pass


class BaseLLMAdapter(ABC):
    """Universal LLM adapter interface.

    Internal protocol: list[{role: system|user|assistant, content: str}].

    Cross-cutting kwargs are part of the contract so the engine never has to
    branch on adapter class:

    * ``reasoning_enabled`` — opt-in/out of provider-side chain-of-thought.
      ``True`` enables, ``False`` disables, ``None`` = leave at provider default.
      Adapters map this to their native flag (Ollama ``think``, Anthropic
      extended ``thinking``, Gemini ``thinking_config``, OpenAI
      ``reasoning_effort`` for o-series/gpt-5, Groq ``reasoning_format``).
      On models that don't support reasoning, the adapter MUST silently
      ignore the flag — never raise.
    * ``**_extra_kwargs`` — forward-compat sink. Future engine knobs can be
      threaded through without breaking older adapters: an adapter that
      doesn't recognize a kwarg simply absorbs it. Adapters MAY shadow
      individual kwargs by promoting them to named parameters.

    The two patterns together mean: the engine can always call
    ``adapter.stream(reasoning_enabled=…, future_knob=…)`` and the call
    succeeds regardless of which provider answers it.
    """

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        reasoning_enabled: Optional[bool] = None,
        **_extra_kwargs,
    ) -> str:
        """Non-streaming completion. Returns full response string."""
        ...

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        interrupt_check: Optional[object] = None,  # Optional[Callable[[], bool]]
        reasoning_enabled: Optional[bool] = None,
        **_extra_kwargs,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields string tokens."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return available model names."""
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the provider is reachable."""
        ...
