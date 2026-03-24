"""
Universal LLM Adapter — Ollama
-------------------------------
All is language and protocols.
Translates the internal message protocol → Ollama API.
To swap providers: rewrite only this file.

Internal protocol: list[{role: system|user|assistant, content: str}]
"""

import httpx
import json
import asyncio
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError  # re-export for back-compat


OLLAMA_BASE    = "http://localhost:11434"
RETRY_DELAYS   = [0.5, 1.5, 3.0]


class OllamaAdapter(BaseLLMAdapter):


    def __init__(self, base_url: str = OLLAMA_BASE, timeout: float = 120.0):
        self.base_url = base_url
        self.timeout  = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _raise_for_status(self, response):
        """
        Handle both normal httpx responses and AsyncMock-based test doubles.
        """
        maybe = response.raise_for_status()
        if asyncio.iscoroutine(maybe):
            await maybe

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Non-streaming completion with retry. Returns full response string."""
        payload = self._payload(model, messages, temperature, max_tokens, stream=False, images=images, tools=tools)
        client = self._get_client()
        last_err: Exception = RuntimeError("no attempts made")

        for i, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await client.post("/api/chat", json=payload)
                await self._raise_for_status(resp)
                return resp.json()["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise LLMError(f"LLM rejected: {e.response.status_code} - {e.response.text}") from e
                last_err = e
            except Exception as e:
                last_err = e

            if i < len(RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)

        raise LLMError(f"LLM failed after retries: {last_err}")

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
        """
        Streaming completion with reasoning extraction.
        Yields tokens for text, and special markers for reasoning:
        <THOUGHT>thinking...</THOUGHT>
        """
        payload = self._payload(model, messages, temperature, max_tokens, stream=True, images=images, tools=tools)
        client = self._get_client()
        
        in_thought = False
        
        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                await self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line: continue
                    
                    # 0. Check for interruption mid-stream
                    if interrupt_check and callable(interrupt_check) and interrupt_check():
                        break
                        
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError: continue
                    
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    
                    # 0. Handle Native Thought Field (some Ollama models)
                    if chunk.get("thought"):
                        yield f"<THOUGHT>{chunk['thought']}</THOUGHT>"
                    
                    # 1. Handle Native Tool Calls (Ollama 0.5+)
                    if "tool_calls" in msg:
                        # Yield as a formatted JSON block that our core detector can catch,
                        # but wrap it so we know it came from a native call.
                        yield json.dumps({"tool_calls": msg["tool_calls"]})
                        continue

                    # 2. Extract Reasoning (DeepSeek style)
                    if "<think>" in content:
                        in_thought = True
                        yield "<THOUGHT>"
                        content = content.replace("<think>", "")
                    
                    if "</think>" in content:
                        thought_content = content[:content.index("</think>")]
                        if thought_content: yield thought_content
                        yield "</THOUGHT>"
                        content = content[content.index("</think>")+8:]
                        in_thought = False

                    if content:
                        yield content
                        
                    if chunk.get("done"):
                        break
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise LLMError(f"LLM rejected stream: {e.response.status_code}") from e
            raise LLMError(f"LLM stream failed: {e}") from e
        except Exception as e:
             raise LLMError(f"LLM stream connection failed: {e}") from e

    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            resp = await client.get("/api/tags")
            await self._raise_for_status(resp)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return ["llama3.2", "deepseek-r1:8b"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def _payload(self, model, messages, temperature, max_tokens, stream, images=None, tools=None):
        p: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": 32000, # Expanded default context for local models
            },
            "keep_alive": "5m", # Keep model loaded for 5 minutes
        }
        if max_tokens:
            p["options"]["num_predict"] = max_tokens
        
        # Native Tool Calling (Ollama 0.5+)
        if tools:
            p["tools"] = tools

        # Ollama vision: attach images to the last user message
        if images:
            for msg in reversed(p["messages"]):
                if msg["role"] == "user":
                    msg["images"] = images
                    break
        return p
