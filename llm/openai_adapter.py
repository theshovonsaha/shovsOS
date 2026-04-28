"""
OpenAI-compatible LLM Adapter
------------------------------
Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API
(Together, Fireworks, vLLM, LM Studio, etc.)

Requires: pip install openai
Env vars: OPENAI_API_KEY, OPENAI_BASE_URL (optional for custom endpoints)

API notes (verified against current OpenAI Chat Completions API):
  - Reasoning models (o1, o3, o4, gpt-5 family) require `max_completion_tokens`
    instead of `max_tokens`, and reject `temperature` other than the default.
  - Streaming tool_calls arrive as fragmented deltas indexed by `tc.index` —
    arguments must be concatenated character-by-character before the call is
    well-formed JSON. Yielding each delta as a separate JSON object breaks
    downstream parsers.
  - Vision images are attached as `image_url` content parts on the last user turn.
"""

import asyncio
import json
import os
import re
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError

RETRY_DELAYS = [0.5, 1.5, 3.0]

# Reasoning models — these reject `temperature` and require `max_completion_tokens`.
_REASONING_MODEL_RE = re.compile(r"^(?:o1|o3|o4|gpt-5)(?:-|$)", re.IGNORECASE)


def _is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.match(model or ""))


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

    def _build_kwargs(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict]],
        images: Optional[list[str]],
        stream: bool = False,
    ) -> dict:
        msgs = self._prepare_messages(messages, images)
        kwargs: dict = {
            "model": model,
            "messages": msgs,
        }
        if stream:
            kwargs["stream"] = True

        is_reasoning = _is_reasoning_model(model)
        if is_reasoning:
            # Reasoning models only accept the default temperature (1.0).
            # Token budget uses a different parameter name.
            if max_tokens:
                kwargs["max_completion_tokens"] = int(max_tokens)
        else:
            kwargs["temperature"] = temperature
            if max_tokens:
                kwargs["max_tokens"] = int(max_tokens)

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return kwargs

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
        kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            stream=False,
        )

        last_err: Exception = RuntimeError("no attempts made")
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
        kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            stream=True,
        )

        # Tool call deltas arrive fragmented; accumulate by index, emit once.
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
                    yield delta.content

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
                    args = slot["arguments"] or "{}"
                    normalized.append({
                        "id": slot["id"],
                        "type": "function",
                        "function": {
                            "name": slot["name"],
                            "arguments": args,
                        },
                    })
                yield json.dumps({"tool_calls": normalized})
        except Exception as e:
            raise self._wrap_error(e) from e

    def _wrap_error(self, e: Exception) -> LLMError:
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
            # Filter to chat-capable families (skip embeddings, tts, whisper).
            return [
                m for m in ids
                if any(p in m for p in ("gpt-4", "gpt-5", "o1", "o3", "o4", "chatgpt"))
            ]
        except Exception:
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4.1",
                "gpt-4.1-mini",
                "o3-mini",
                "o4-mini",
            ] if not self.base_url else []

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
        msgs = [m.copy() for m in messages]
        for msg in reversed(msgs):
            if msg["role"] == "user":
                content_parts = [{"type": "text", "text": msg["content"]}]
                for img_b64 in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                msg["content"] = content_parts
                break
        return msgs
