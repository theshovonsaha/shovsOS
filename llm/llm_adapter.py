"""
Universal LLM Adapter — Ollama
-------------------------------
All is language and protocols.
Translates the internal message protocol → Ollama API.

Internal protocol: list[{role: system|user|assistant, content: str}]

Notes on Ollama API (verified against /api/chat semantics):
  - num_ctx is per-call; the server defaults to 2048. We probe model metadata via
    /api/show once per (base_url, model) and cache the result so we don't waste
    memory on small models or truncate on big ones.
  - keep_alive accepts duration strings ("5m") or seconds; we keep the model warm.
  - think: True/False (Ollama 0.7+) toggles reasoning output for thinking models.
  - format: "json" forces structured JSON output when callers want strict shape.
  - Streaming tool_calls: Ollama emits the full tool_calls object on the final
    chunk (not deltas), so accumulation is unnecessary on this provider.
"""

import asyncio
import json
import os
import re
from typing import AsyncIterator, Optional

import httpx

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError  # re-export for back-compat


OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RETRY_DELAYS = [0.5, 1.5, 3.0]

# How long the model stays resident in VRAM/RAM after a request.
# Longer = better prefix-cache reuse across an interactive session;
# shorter = frees memory faster on shared boxes.
# Override via env: OLLAMA_KEEP_ALIVE="30m" or "1h" or "0" to evict immediately.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

# Patterns of model names that use thinking-style reasoning channels.
# These models benefit from `think: true` (Ollama 0.7+) so reasoning is surfaced
# in a separate `thinking`/`thought` field instead of leaking into content.
_THINKING_MODEL_PATTERNS = re.compile(
    r"^(?:deepseek-r1|qwq|gpt-oss|magistral|qwen3-?\d+b-?thinking|phi4-reasoning)",
    re.IGNORECASE,
)

# Tiered context-window fallbacks when /api/show metadata is unavailable.
# Picks a sensible default from the model name family — never larger than the
# model can handle, and never wastes RAM on tiny models.
_CTX_TIERS = (
    (re.compile(r"\b(?:0\.5b|1b|1\.5b|2b|3b)\b", re.IGNORECASE), 8192),
    (re.compile(r"\b(?:7b|8b|9b|llama3\.2|llama3\.1)\b", re.IGNORECASE), 32768),
    (re.compile(r"\b(?:13b|14b|20b|22b|27b|32b|34b)\b", re.IGNORECASE), 65536),
    (re.compile(r"\b(?:70b|72b|110b|405b|gpt-oss)\b", re.IGNORECASE), 131072),
)


def _is_thinking_model(model: str) -> bool:
    return bool(_THINKING_MODEL_PATTERNS.search(model or ""))


# Cross-chunk-safe <think>...</think> extraction.
# Inline reasoning tags can split across SSE chunks ("<thi" + "nk>"). The previous
# implementation only matched when both tags landed in the same chunk, so partial
# tags leaked as visible characters and broke the consumer chat. We hold back a
# small tail (one byte less than the longest tag) so a straddled tag is never
# yielded prematurely.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TAG_HOLDBACK = max(len(_THINK_OPEN), len(_THINK_CLOSE)) - 1


def _drain_thought_buffer(
    buffer: str, in_thought: bool, *, force_flush: bool
) -> tuple[list[str], str, bool]:
    """Pull every safely-emittable chunk out of `buffer`.

    Returns (chunks_to_yield, remaining_buffer, new_in_thought).
    Wraps thought spans in <THOUGHT>...</THOUGHT> so downstream stays format-stable
    with the native-thinking branch.
    """
    chunks: list[str] = []
    while True:
        if in_thought:
            idx = buffer.find(_THINK_CLOSE)
            if idx >= 0:
                if idx > 0:
                    chunks.append(buffer[:idx])
                chunks.append("</THOUGHT>")
                buffer = buffer[idx + len(_THINK_CLOSE):]
                in_thought = False
                continue
            if len(buffer) > _TAG_HOLDBACK:
                chunks.append(buffer[:-_TAG_HOLDBACK])
                buffer = buffer[-_TAG_HOLDBACK:]
            elif force_flush and buffer:
                chunks.append(buffer)
                buffer = ""
            break
        else:
            idx = buffer.find(_THINK_OPEN)
            if idx >= 0:
                if idx > 0:
                    chunks.append(buffer[:idx])
                chunks.append("<THOUGHT>")
                buffer = buffer[idx + len(_THINK_OPEN):]
                in_thought = True
                continue
            if len(buffer) > _TAG_HOLDBACK:
                chunks.append(buffer[:-_TAG_HOLDBACK])
                buffer = buffer[-_TAG_HOLDBACK:]
            elif force_flush and buffer:
                chunks.append(buffer)
                buffer = ""
            break
    return chunks, buffer, in_thought


def _default_ctx_for_model(model: str) -> int:
    name = (model or "").lower()
    for pattern, ctx in _CTX_TIERS:
        if pattern.search(name):
            return ctx
    return 8192  # Conservative default — better small than OOM.


class OllamaAdapter(BaseLLMAdapter):

    def __init__(self, base_url: str = OLLAMA_BASE, timeout: float = 120.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop_id: Optional[int] = None
        # Cache of (model_name -> max context length) discovered from /api/show.
        self._ctx_cache: dict[str, int] = {}

    @staticmethod
    def _current_loop_id() -> Optional[int]:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    def _get_client(self) -> httpx.AsyncClient:
        current_loop_id = self._current_loop_id()
        if (
            self._client is None
            or self._client.is_closed
            or (current_loop_id is not None and self._client_loop_id not in (None, current_loop_id))
        ):
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            self._client_loop_id = current_loop_id
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client_loop_id = None

    async def _raise_for_status(self, response):
        """Handle both normal httpx responses and AsyncMock-based test doubles."""
        maybe = response.raise_for_status()
        if asyncio.iscoroutine(maybe):
            await maybe

    async def _resolve_num_ctx(self, model: str) -> int:
        """Discover the model's true context window once, cache, then reuse.

        Falls back to a tiered family default if /api/show is unreachable.
        Capping at 131072 keeps RAM finite even when a model claims more.
        """
        if not model:
            return 8192
        cached = self._ctx_cache.get(model)
        if cached:
            return cached
        try:
            client = self._get_client()
            resp = await client.post("/api/show", json={"name": model})
            await self._raise_for_status(resp)
            data = resp.json()
            # /api/show returns model_info — the relevant key is
            # "<arch>.context_length" e.g. "llama.context_length": 131072.
            info = data.get("model_info") or {}
            for key, val in info.items():
                if key.endswith(".context_length") and isinstance(val, int):
                    ctx = min(int(val), 131072)
                    self._ctx_cache[model] = ctx
                    return ctx
        except Exception:
            pass
        ctx = _default_ctx_for_model(model)
        self._ctx_cache[model] = ctx
        return ctx

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
        payload = await self._build_payload(
            model, messages, temperature, max_tokens, stream=False, images=images, tools=tools
        )
        client = self._get_client()
        last_err: Exception = RuntimeError("no attempts made")

        for i, delay in enumerate(RETRY_DELAYS):
            try:
                resp = await client.post("/api/chat", json=payload)
                await self._raise_for_status(resp)
                data = resp.json()
                msg = data.get("message", {})
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    normalized = []
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        normalized.append({
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": json.dumps(args) if not isinstance(args, str) else args,
                            },
                        })
                    if normalized:
                        return json.dumps({"tool_calls": normalized})
                return msg.get("content", "")
            except httpx.HTTPStatusError as e:
                # 404 from /api/chat means the model isn't pulled — surface a
                # clear actionable error rather than retrying.
                if e.response.status_code == 404:
                    raise LLMError(
                        f"Ollama model '{model}' is not installed. Run: ollama pull {model}"
                    ) from e
                if e.response.status_code < 500:
                    raise LLMError(f"Ollama rejected: {e.response.status_code} - {e.response.text}") from e
                last_err = e
            except Exception as e:
                last_err = e

            if i < len(RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)

        raise LLMError(f"Ollama failed after retries: {last_err}")

    async def stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        interrupt_check: Optional[object] = None,
        reasoning_enabled: Optional[bool] = None,
    ) -> AsyncIterator[str]:
        """Streaming completion with reasoning extraction.

        Yields content tokens. Reasoning is wrapped in <THOUGHT>...</THOUGHT>.
        Tool calls (when present) arrive on the final chunk and are yielded
        once as a single normalized JSON object.

        ``reasoning_enabled`` overrides Ollama's `think` flag: True forces
        reasoning on, False forces it off (saves tokens on thinking models),
        None falls back to model-name auto-detection.
        """
        payload = await self._build_payload(
            model, messages, temperature, max_tokens, stream=True,
            images=images, tools=tools, reasoning_enabled=reasoning_enabled,
        )
        client = self._get_client()

        # Cross-chunk-safe inline <think> handling. See _drain_thought_buffer
        # above for why a single-chunk regex was wrong.
        pending = ""
        in_thought = False

        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                await self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if interrupt_check and callable(interrupt_check) and interrupt_check():
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    done = bool(chunk.get("done"))

                    # Native thinking field (Ollama 0.7+ for thinking models).
                    thought = msg.get("thinking") or chunk.get("thought")
                    if thought:
                        yield f"<THOUGHT>{thought}</THOUGHT>"

                    # Native tool calls — emitted on the final chunk.
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        normalized = []
                        for tc in tool_calls:
                            fn = tc.get("function") or {}
                            args = fn.get("arguments", {})
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            normalized.append({
                                "type": "function",
                                "function": {
                                    "name": fn.get("name", ""),
                                    "arguments": args or "{}",
                                },
                            })
                        if normalized:
                            yield json.dumps({"tool_calls": normalized})

                    if content:
                        pending += content
                        emit, pending, in_thought = _drain_thought_buffer(
                            pending, in_thought, force_flush=done
                        )
                        for piece in emit:
                            if piece:
                                yield piece

                    if done:
                        # Final flush: drain remaining buffer and close any
                        # unterminated thought span so downstream parsers stay sane.
                        emit, pending, in_thought = _drain_thought_buffer(
                            pending, in_thought, force_flush=True
                        )
                        for piece in emit:
                            if piece:
                                yield piece
                        if in_thought:
                            yield "</THOUGHT>"
                            in_thought = False
                        break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise LLMError(
                    f"Ollama model '{model}' is not installed. Run: ollama pull {model}"
                ) from e
            if e.response.status_code < 500:
                raise LLMError(f"Ollama rejected stream: {e.response.status_code}") from e
            raise LLMError(f"Ollama stream failed: {e}") from e
        except Exception as e:
            raise LLMError(f"Ollama stream connection failed: {e}") from e

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

    async def _build_payload(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        reasoning_enabled: Optional[bool] = None,
    ) -> dict:
        num_ctx = await self._resolve_num_ctx(model)

        options: dict = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        if max_tokens:
            options["num_predict"] = int(max_tokens)

        # NOTE on prefix-cache reuse:
        #   Ollama keeps a per-model KV cache while the model is resident
        #   (controlled by keep_alive). Consecutive requests that share an
        #   identical message *prefix* skip prompt re-tokenization for those
        #   tokens — the practical win for multi-turn chat.
        #   Callers should pass the SAME message history (no rewording, no
        #   reordering of historical turns) to keep this hit-rate high.
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }

        # Reasoning channel control (Ollama 0.7+).
        # Explicit override wins over model-name auto-detection so callers can
        # silence reasoning on thinking models (saves tokens) or force it on
        # for models that aren't in the auto-detect regex.
        if reasoning_enabled is not None:
            payload["think"] = bool(reasoning_enabled)
        elif _is_thinking_model(model):
            payload["think"] = True

        if tools:
            payload["tools"] = tools

        # Vision: attach base64 images to the last user turn.
        if images:
            for msg in reversed(payload["messages"]):
                if msg["role"] == "user":
                    msg["images"] = images
                    break

        return payload
