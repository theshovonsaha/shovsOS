"""
Groq LLM Adapter
-----------------
Fast inference via Groq cloud API.
Uses the Groq Python SDK for chat completions (NOT compound-beta — 
that's used for web search in tools_web.py, not as a general LLM).

Requires: pip install groq  (already in requirements.txt)
Env vars: GROQ_API_KEY
"""

import asyncio
import os
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class GroqLLMAdapter(BaseLLMAdapter):

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self.api_key, timeout=self.timeout)
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
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = "none"
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        last_err: Exception = RuntimeError("no attempts")
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
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = "none"
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
                    content = delta.content
                    if "<think>" in content:
                        yield "<THOUGHT>"
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        yield content.replace("</think>", "")
                        yield "</THOUGHT>"
                        content = ""
                    if content:
                        yield content
                
                # Handle native tool calls from Groq
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
        """Helper to map Groq/OpenAI errors to our internal exceptions."""
        err_str = str(e).lower()
        # Groq SDK uses its own exception types, but we can check the message/class
        if "rate_limit" in err_str or "429" in err_str:
            return RateLimitError(f"Groq Rate Limit: {e}")
        if "500" in err_str or "503" in err_str or "service_unavailable" in err_str:
            return ProviderError(f"Groq Provider Error: {e}")
        return LLMError(f"Groq Error: {e}")

    async def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            client = self._get_client()
            models = await client.models.list()
            # Filter and sort to prioritize production models
            ids = [m.id for m in models.data]
            return ids if ids else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        except Exception:
            return [
                "llama-3.3-70b-versatile", 
                "llama-3.1-8b-instant", 
                "mixtral-8x7b-32768", 
                "groq/compound-mini"
            ]

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False
