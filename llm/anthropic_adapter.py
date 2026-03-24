"""
Anthropic Claude LLM Adapter
------------------------------
Requires: pip install anthropic
Env vars: ANTHROPIC_API_KEY
"""

import asyncio
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

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
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
                    
                if tools:
                    anthropic_tools = []
                    for t in tools:
                        if "function" in t:
                            fn = t["function"]
                            anthropic_tools.append({
                                "name": fn["name"],
                                "description": fn.get("description", ""),
                                "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
                            })
                    if anthropic_tools:
                        kwargs["tools"] = anthropic_tools
                        
                resp = await client.messages.create(**kwargs)
                if resp.stop_reason == "tool_use":
                    import json
                    tool_calls = []
                    for content in resp.content:
                        if content.type == "tool_use":
                            tool_calls.append({
                                "id": content.id,
                                "type": "function",
                                "function": {
                                    "name": content.name,
                                    "arguments": json.dumps(content.input)
                                }
                            })
                    if tool_calls:
                        return json.dumps({"tool_calls": tool_calls})
                
                return resp.content[0].text if resp.content else ""
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
    ) -> AsyncIterator[str]:
        client = self._get_client()
        msgs, system_prompt = self._prepare_messages(messages, images)
        
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens or 4096,
                "messages": msgs,
                "temperature": temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            if tools:
                anthropic_tools = []
                for t in tools:
                    if "function" in t:
                        fn = t["function"]
                        anthropic_tools.append({
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
                        })
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools
                
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if interrupt_check and callable(interrupt_check) and interrupt_check():
                        break
                    yield text
                
                final_message = await stream.get_final_message()
                if final_message.stop_reason == "tool_use":
                    import json
                    tool_calls = []
                    for content in final_message.content:
                        if content.type == "tool_use":
                            tool_calls.append({
                                "id": content.id,
                                "type": "function",
                                "function": {
                                    "name": content.name,
                                    "arguments": json.dumps(content.input)
                                }
                            })
                    if tool_calls:
                        yield json.dumps({"tool_calls": tool_calls})
                        
        except Exception as e:
            raise self._wrap_error(e) from e

    def _wrap_error(self, e: Exception) -> LLMError:
        """Helper to map Anthropic errors to our internal exceptions."""
        err_str = str(e).lower()
        if "rate_limit" in err_str or "429" in err_str:
            return RateLimitError(f"Anthropic Rate Limit: {e}")
        if "500" in err_str or "503" in err_str or "service_unavailable" in err_str or "overloaded" in err_str:
            return ProviderError(f"Anthropic Provider Error: {e}")
        return LLMError(f"Anthropic Error: {e}")

    async def list_models(self) -> list[str]:
        # Anthropic doesn't have a dynamic list endpoint yet, return common ones
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

    async def health(self) -> bool:
        return bool(self.api_key)

    def _prepare_messages(self, messages: list[dict], images: Optional[list[str]]) -> tuple[list[dict], Optional[str]]:
        """
        Convert to Anthropic format:
        1. Separate system message (passed as 'system' param).
        2. Format user/assistant turns.
        3. Merge consecutive same-role messages (Anthropic requires strict alternation).
        4. Handle images if present.
        """
        system_prompt = ""
        raw_msgs = []
        
        for m in messages:
            if m["role"] == "system":
                # Anthropic takes system as a separate param, not a message
                if system_prompt:
                    system_prompt += "\n\n" + m["content"]
                else:
                    system_prompt = m["content"]
            else:
                raw_msgs.append({"role": m["role"], "content": m["content"]})
        
        # Merge consecutive same-role messages (Anthropic rejects them)
        merged = []
        for msg in raw_msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append({"role": msg["role"], "content": msg["content"]})
        
        # Ensure first message is always 'user' (Anthropic requirement)
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "[Continue from previous context]"})
        
        if images:
            # Handle vision (only on the latest user message for now)
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
                            }
                        })
                    msg["content"] = content_parts
                    break
                    
        return merged, system_prompt
