"""
Anthropic Claude LLM Adapter
------------------------------
Requires: pip install anthropic
Env vars: ANTHROPIC_API_KEY

API notes (verified against current Messages API):
  - System prompt is a separate `system` field, not a message role.
  - Anthropic requires strict role alternation; consecutive same-role messages
    are rejected. We merge them client-side.
  - Streaming tool_use blocks arrive as fragmented `input_json_delta` events
    on the raw event stream. The `text_stream` helper drops these entirely,
    so we use the lower-level event iterator and accumulate by content-block index.
  - First message must be `user`; we backfill if the trimmed history starts on `assistant`.
"""

import asyncio
import json
import os
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class AnthropicAdapter(BaseLLMAdapter):

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise LLMError("anthropic package not installed. Run: pip install anthropic")
            if not self.api_key:
                raise LLMError("ANTHROPIC_API_KEY is not set.")
            self._client = AsyncAnthropic(api_key=self.api_key, timeout=self.timeout)
        return self._client

    @staticmethod
    def _convert_tools(tools: Optional[list[dict]]) -> list[dict]:
        if not tools:
            return []
        out = []
        for t in tools:
            if "function" in t:
                fn = t["function"]
                out.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        return out

    @staticmethod
    def _maybe_attach_thinking(
        kwargs: dict,
        *,
        model: str,
        reasoning_enabled: Optional[bool],
        max_tokens: int,
    ) -> None:
        """Attach Anthropic extended-thinking config when applicable.

        Extended thinking is supported on Claude 3.7 and Claude 4 family
        models. We require ``reasoning_enabled is True`` to enable, and
        ``False`` is a no-op (Anthropic has no "force off" flag — extended
        thinking is opt-in, off by default). Budget is bounded by the
        request's ``max_tokens`` minus a 1024 floor for the visible reply.
        """
        if reasoning_enabled is not True:
            return
        from llm.model_capabilities import supports_reasoning
        if not supports_reasoning(model):
            return
        # Reserve at least 1024 tokens for the visible response, give the
        # rest to the thinking budget. Anthropic requires budget < max_tokens.
        budget = max(1024, int(max_tokens) - 1024)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # Extended thinking requires temperature=1 per Anthropic docs.
        kwargs["temperature"] = 1.0

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        reasoning_enabled: Optional[bool] = None,
        **_extra_kwargs,
    ) -> str:
        client = self._get_client()
        msgs, system_prompt = self._prepare_messages(messages, images)

        last_err: Exception = RuntimeError("no attempts made")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens or 4096,
                    "messages": msgs,
                    "temperature": temperature,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                anthropic_tools = self._convert_tools(tools)
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools
                self._maybe_attach_thinking(
                    kwargs, model=model, reasoning_enabled=reasoning_enabled,
                    max_tokens=max_tokens or 4096,
                )

                resp = await client.messages.create(**kwargs)
                if resp.stop_reason == "tool_use":
                    tool_calls = []
                    for content in resp.content:
                        if content.type == "tool_use":
                            tool_calls.append({
                                "id": content.id,
                                "type": "function",
                                "function": {
                                    "name": content.name,
                                    "arguments": json.dumps(content.input),
                                },
                            })
                    if tool_calls:
                        return json.dumps({"tool_calls": tool_calls})

                # Concatenate all text blocks (some responses have multiple).
                text_parts = [c.text for c in resp.content if getattr(c, "type", "") == "text"]
                return "".join(text_parts)
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
        max_tokens: Optional[int] = 4096,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        interrupt_check: Optional[object] = None,
        reasoning_enabled: Optional[bool] = None,
        **_extra_kwargs,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        msgs, system_prompt = self._prepare_messages(messages, images)

        kwargs = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "messages": msgs,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        self._maybe_attach_thinking(
            kwargs, model=model, reasoning_enabled=reasoning_enabled,
            max_tokens=max_tokens or 4096,
        )

        # Per-block accumulators keyed by content_block index.
        # text blocks → yielded incrementally; tool_use → buffered then emitted at block_stop.
        tool_blocks: dict[int, dict] = {}

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if interrupt_check and callable(interrupt_check) and interrupt_check():
                        break
                    etype = getattr(event, "type", "")

                    if etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            tool_blocks[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }
                    elif etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        dtype = getattr(delta, "type", "") if delta else ""
                        if dtype == "text_delta" and delta.text:
                            yield delta.text
                        elif dtype == "input_json_delta" and getattr(delta, "partial_json", None):
                            slot = tool_blocks.get(event.index)
                            if slot is not None:
                                slot["arguments"] += delta.partial_json
                    elif etype == "content_block_stop":
                        slot = tool_blocks.get(event.index)
                        if slot is not None:
                            yield json.dumps({
                                "tool_calls": [{
                                    "id": slot["id"],
                                    "type": "function",
                                    "function": {
                                        "name": slot["name"],
                                        "arguments": slot["arguments"] or "{}",
                                    },
                                }]
                            })
        except Exception as e:
            raise self._wrap_error(e) from e

    def _wrap_error(self, e: Exception) -> LLMError:
        err_str = str(e).lower()
        if "rate_limit" in err_str or "429" in err_str:
            return RateLimitError(f"Anthropic Rate Limit: {e}")
        if "500" in err_str or "503" in err_str or "service_unavailable" in err_str or "overloaded" in err_str:
            return ProviderError(f"Anthropic Provider Error: {e}")
        return LLMError(f"Anthropic Error: {e}")

    async def list_models(self) -> list[str]:
        # Anthropic exposes a models list endpoint on the SDK.
        try:
            client = self._get_client()
            page = await client.models.list()
            return [m.id for m in getattr(page, "data", [])]
        except Exception:
            return [
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-haiku-4-5",
                "claude-sonnet-4-0",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ]

    async def health(self) -> bool:
        return bool(self.api_key)

    def _prepare_messages(
        self,
        messages: list[dict],
        images: Optional[list[str]],
    ) -> tuple[list[dict], Optional[str]]:
        """Convert to Anthropic format:
        1. Separate system message into 'system' param.
        2. Merge consecutive same-role messages (Anthropic requires alternation).
        3. Backfill leading user turn if needed.
        4. Attach images to the latest user turn.
        """
        system_prompt = ""
        raw_msgs = []

        for m in messages:
            if m["role"] == "system":
                if system_prompt:
                    system_prompt += "\n\n" + m["content"]
                else:
                    system_prompt = m["content"]
            else:
                raw_msgs.append({"role": m["role"], "content": m["content"]})

        merged: list[dict] = []
        for msg in raw_msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append({"role": msg["role"], "content": msg["content"]})

        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "[Continue from previous context]"})

        if images:
            for msg in reversed(merged):
                if msg["role"] == "user":
                    content_parts = [{"type": "text", "text": msg["content"]}]
                    for img_b64 in images:
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        })
                    msg["content"] = content_parts
                    break

        return merged, (system_prompt or None)
