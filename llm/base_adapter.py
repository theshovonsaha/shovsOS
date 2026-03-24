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
    """
    Universal LLM adapter interface.
    Internal protocol: list[{role: system|user|assistant, content: str}]
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
