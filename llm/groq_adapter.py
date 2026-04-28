"""
Groq LLM Adapter
-----------------
Fast inference via Groq cloud API.
Uses the Groq Python SDK (OpenAI-compatible chat completions).

Requires: pip install groq
Env vars: GROQ_API_KEY

API notes:
  - Streaming tool_calls arrive as fragmented deltas indexed by `tc.index` —
    accumulate by index, emit once on `finish_reason == "tool_calls"`.
  - DeepSeek-R1 distill models stream `<think>...</think>` reasoning inline;
    we route those into a <THOUGHT> segment without losing surrounding content.
"""

import asyncio
import json
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        last_err: Exception = RuntimeError("no attempts")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                if msg.tool_calls:
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
                        return json.dumps({"tool_calls": normalized})
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        in_thought = False
        tool_call_acc: dict[int, dict] = {}
        finish_reason: Optional[str] = None

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if interrupt_check and callable(interrupt_check) and interrupt_check():
                    break
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta and delta.content:
                    content = delta.content
                    # Robust <think>...</think> handling that survives token splits.
                    while content:
                        if in_thought:
                            close_idx = content.find("</think>")
                            if close_idx == -1:
                                yield content
                                content = ""
                            else:
                                if close_idx > 0:
                                    yield content[:close_idx]
                                yield "</THOUGHT>"
                                in_thought = False
                                content = content[close_idx + len("</think>"):]
                        else:
                            open_idx = content.find("<think>")
                            if open_idx == -1:
                                yield content
                                content = ""
                            else:
                                if open_idx > 0:
                                    yield content[:open_idx]
                                yield "<THOUGHT>"
                                in_thought = True
                                content = content[open_idx + len("<think>"):]

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        slot = tool_call_acc.setdefault(
                            idx, {"id": None, "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

            if tool_call_acc and finish_reason == "tool_calls":
                normalized = []
                for idx in sorted(tool_call_acc):
                    slot = tool_call_acc[idx]
                    normalized.append({
                        "id": slot["id"],
                        "type": "function",
                        "function": {
                            "name": slot["name"],
                            "arguments": slot["arguments"] or "{}",
                        },
                    })
                yield json.dumps({"tool_calls": normalized})
        except Exception as e:
            raise self._wrap_error(e) from e

    def _wrap_error(self, e: Exception) -> LLMError:
        err_str = str(e).lower()
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
            ids = [m.id for m in models.data]
            return ids if ids else self._default_models()
        except Exception:
            return self._default_models()

    @staticmethod
    def _default_models() -> list[str]:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "qwen-2.5-32b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
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
