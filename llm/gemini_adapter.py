"""
Google Gemini LLM Adapter
-------------------------
Requires: pip install google-genai
Env vars: GEMINI_API_KEY

API notes (verified against current google-genai SDK):
  - Use `client.aio.models.*` for native async (not asyncio.to_thread on sync iterators —
    that batches all chunks before yielding, defeating streaming).
  - System messages go in `config.system_instruction`, NOT folded into a user role —
    the model treats them very differently.
  - Gemini roles are `user` and `model`; consecutive same-role turns must be merged.
  - Function calls arrive on `chunk.function_calls` already parsed; no delta accumulation.
  - Images are attached as `Part.from_bytes(data=..., mime_type=...)` on the latest user turn.
"""

import asyncio
import base64
import json
import os
from typing import AsyncIterator, Optional, Any

from llm.base_adapter import BaseLLMAdapter, LLMError, RateLimitError, ProviderError

RETRY_DELAYS = [0.5, 1.5, 3.0]


class GeminiAdapter(BaseLLMAdapter):

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._client = None
        self._patched_async_wrapper = False

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMError("GEMINI_API_KEY not found in environment.")
            try:
                self._patch_async_httpx_wrapper_close()
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise LLMError("google-genai not installed. Run: pip install google-genai")
        return self._client

    def _patch_async_httpx_wrapper_close(self) -> None:
        """Patch a google-genai cleanup edge case seen in long-running servers.

        Some google-genai versions wrap ``httpx.AsyncClient`` with an object
        whose ``__del__`` schedules ``aclose()`` even when httpx internals were
        not initialized. That produces noisy background task exceptions:
        ``AsyncHttpxClientWrapper object has no attribute '_state'``. The
        request already completed; the cleanup error should be swallowed.
        """
        if self._patched_async_wrapper:
            return
        self._patched_async_wrapper = True
        try:
            from google.genai._interactions._base_client import AsyncHttpxClientWrapper
        except Exception:
            return
        if getattr(AsyncHttpxClientWrapper, "_shovs_safe_aclose", False):
            return
        original_aclose = AsyncHttpxClientWrapper.aclose

        async def safe_aclose(wrapper, *args, **kwargs):
            try:
                return await original_aclose(wrapper, *args, **kwargs)
            except AttributeError as exc:
                if "_state" in str(exc):
                    return None
                raise

        AsyncHttpxClientWrapper.aclose = safe_aclose
        AsyncHttpxClientWrapper._shovs_safe_aclose = True

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            aio_client = getattr(client, "aio", None)
            aclose = getattr(aio_client, "aclose", None)
            if callable(aclose):
                await aclose()
                return
        except AttributeError as exc:
            if "_state" not in str(exc):
                raise
        except Exception:
            pass
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    @staticmethod
    def _build_tool_config(tools: Optional[list[dict]]) -> Optional[list[dict]]:
        if not tools:
            return None
        declarations = []
        for t in tools:
            if "function" in t:
                fn = t["function"]
                declarations.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        if not declarations:
            return None
        return [{"function_declarations": declarations}]

    def _build_config(
        self,
        temperature: float,
        max_tokens: Optional[int],
        system_instruction: Optional[str],
        tools: Optional[list[dict]],
        reasoning_enabled: Optional[bool] = None,
        model: str = "",
    ) -> dict:
        config: dict = {"temperature": temperature}
        if max_tokens:
            config["max_output_tokens"] = int(max_tokens)
        if system_instruction:
            config["system_instruction"] = system_instruction
        tool_cfg = self._build_tool_config(tools)
        if tool_cfg:
            config["tools"] = tool_cfg
        # Gemini 2.5+ exposes a ``thinking_config`` knob:
        #   thinking_budget = -1 → dynamic (provider default),
        #   thinking_budget =  0 → disable thinking,
        #   thinking_budget >  0 → cap thinking tokens.
        # We only set it when reasoning_enabled is explicit AND the model is
        # known to support reasoning. Gating on supports_reasoning keeps us
        # safe across SDK versions: older SDKs silently dropped unknown keys
        # but newer SDKs reject them.
        if reasoning_enabled is not None:
            from llm.model_capabilities import supports_reasoning
            if supports_reasoning(model):
                config["thinking_config"] = {
                    "thinking_budget": -1 if reasoning_enabled else 0,
                }
        return config

    @staticmethod
    def _normalize_function_calls(function_calls) -> Optional[str]:
        if not function_calls:
            return None
        tool_calls = []
        for fc in function_calls:
            args = fc.args
            if args is None:
                args = {}
            elif not isinstance(args, dict):
                try:
                    args = dict(args)
                except Exception:
                    args = {}
            tool_calls.append({
                "id": "call_" + (fc.name or ""),
                "type": "function",
                "function": {
                    "name": fc.name or "",
                    "arguments": json.dumps(args),
                },
            })
        if not tool_calls:
            return None
        return json.dumps({"tool_calls": tool_calls})

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
        client = self._get_client()
        contents, system_instruction = self._convert_messages(messages, images)
        config = self._build_config(
            temperature, max_tokens, system_instruction, tools,
            reasoning_enabled=reasoning_enabled, model=model,
        )

        last_err: Exception = RuntimeError("no attempts made")
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                tool_payload = self._normalize_function_calls(getattr(response, "function_calls", None))
                if tool_payload:
                    return tool_payload

                return response.text or ""
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
        reasoning_enabled: Optional[bool] = None,
        **_extra_kwargs,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        contents, system_instruction = self._convert_messages(messages, images)
        config = self._build_config(
            temperature, max_tokens, system_instruction, tools,
            reasoning_enabled=reasoning_enabled, model=model,
        )

        try:
            # Native async streaming — yields chunks as they arrive.
            stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                if interrupt_check and callable(interrupt_check) and interrupt_check():
                    break

                tool_payload = self._normalize_function_calls(getattr(chunk, "function_calls", None))
                if tool_payload:
                    yield tool_payload
                    continue

                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            raise self._wrap_error(e) from e

    def _wrap_error(self, e: Exception) -> LLMError:
        err_str = str(e).lower()
        if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
            return RateLimitError(f"Gemini Rate Limit: {e}")
        if "500" in err_str or "503" in err_str or "service_unavailable" in err_str:
            return ProviderError(f"Gemini Provider Error: {e}")
        return LLMError(f"Gemini Error: {e}")

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            models_resp = await client.aio.models.list()
            ids: list[str] = []
            async for m in models_resp:
                # Names come back as "models/gemini-2.5-flash" — strip prefix for clarity.
                name = getattr(m, "name", "") or ""
                ids.append(name.split("/", 1)[-1] if name.startswith("models/") else name)
            return ids or self._default_models()
        except Exception:
            return self._default_models()

    async def health(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            resp = await client.aio.models.list(config={"page_size": 1})
            async for _ in resp:
                return True
            return True
        except Exception:
            return False

    @staticmethod
    def _default_models() -> list[str]:
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]

    def _convert_messages(
        self,
        messages: list[dict],
        images: Optional[list[str]] = None,
    ) -> tuple[list[Any], Optional[str]]:
        """Convert internal {role, content} to google-genai Contents.

        - System messages → joined into `system_instruction` (returned separately).
        - Roles: user/assistant → user/model.
        - Consecutive same-role turns are merged.
        - Images are attached as Parts on the last user turn.
        """
        from google.genai import types

        system_parts: list[str] = []
        raw_messages: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            mapped_role = "model" if role == "assistant" else "user"
            raw_messages.append({"role": mapped_role, "content": content})

        merged: list[dict] = []
        for msg in raw_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append({"role": msg["role"], "content": msg["content"]})

        # Backfill leading user turn — Gemini expects the first content to be 'user'.
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "[Continue from previous context]"})

        genai_contents: list[Any] = []
        for msg in merged:
            genai_contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[types.Part(text=msg["content"])],
                )
            )

        if images and genai_contents:
            for content in reversed(genai_contents):
                if content.role == "user":
                    for img_b64 in images:
                        try:
                            data = base64.b64decode(img_b64)
                        except Exception:
                            continue
                        content.parts.append(
                            types.Part.from_bytes(data=data, mime_type="image/jpeg")
                        )
                    break

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return genai_contents, system_instruction
