"""
OpenAI-compatible LLM Adapter
------------------------------
Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API
(Together, Fireworks, local vLLM, etc.)

Requires: pip install openai
Env vars: OPENAI_API_KEY, OPENAI_BASE_URL (optional for custom endpoints)
"""

import asyncio
import os
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class OpenAIAdapter(BaseLLMAdapter):

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ) -> str:
        client = self._get_client()
        msgs = self._prepare_messages(messages, images)
        kwargs = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        last_err: Exception = RuntimeError("no attempts made")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                # When the model responds with native tool calls,
                # serialize them so extract_tool_call() can parse downstream.
                if msg.tool_calls:
                    import json as _json
                    normalized = []
                    for tc in msg.tool_calls:
                        fn = tc.function
                        normalized.append({
                            "type": "function",
                            "function": {
                                "name": fn.name or "",
                                "arguments": fn.arguments or "{}",
                            },
                        })
                    if normalized:
                        return _json.dumps({"tool_calls": normalized})
                return msg.content or ""
            except Exception as e:
                last_err = e
                if i < len(RETRY_DELAYS) - 1:
                    await asyncio.sleep(delay)

        raise self._wrap_error(last_err)

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        interrupt_check: Optional[object] = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        msgs = self._prepare_messages(messages, images)
        kwargs = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if interrupt_check and callable(interrupt_check) and interrupt_check():
                    break
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
                
                # Handle native tool calls from OpenAI
                if delta and delta.tool_calls:
                    import json
                    yield json.dumps({"tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in delta.tool_calls
                    ]})
        except Exception as e:
            raise self._wrap_error(e) from e

    def _wrap_error(self, e: Exception) -> LLMError:
        """Helper to map OpenAI errors to our internal exceptions."""
        err_str = str(e).lower()
        if "rate_limit" in err_str or "429" in err_str:
            return RateLimitError(f"OpenAI Rate Limit: {e}")
        if "500" in err_str or "503" in err_str or "service_unavailable" in err_str:
            return ProviderError(f"OpenAI Provider Error: {e}")
        return LLMError(f"OpenAI Error: {e}")

    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            models = await client.models.list()
            ids = [m.id for m in models.data]
            if self.base_url:
                return ids
            return [m for m in ids if "gpt" in m or "o1" in m or "o3" in m]
        except Exception:
            return ["gpt-4o", "gpt-4o-mini", "o3-mini"] if not self.base_url else []

    async def health(self) -> bool:
        if not self.api_key and not self.base_url:
            return False
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False

    def _prepare_messages(self, messages: list[dict], images: Optional[list[str]]) -> list[dict]:
        """Convert internal protocol to OpenAI format, including vision if needed."""
        if not images:
            return messages
        # Attach images to the last user message as content parts
        msgs = [m.copy() for m in messages]
        for msg in reversed(msgs):
            if msg["role"] == "user":
                content_parts = [{"type": "text", "text": msg["content"]}]
                for img_b64 in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                msg["content"] = content_parts
                break
        return msgs
