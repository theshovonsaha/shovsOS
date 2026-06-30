from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


Transport = Callable[[str, str, dict[str, Any] | None, dict[str, str], float], dict[str, Any]]


class LlamaCppError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlamaCppConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "llama.cpp"
    timeout: float = 60.0
    retries: int = 1


class LlamaCppClient:
    """Tiny OpenAI-compatible llama.cpp client.

    It only implements what the harness needs: chat completion text, model
    listing, and a health probe. The adapter is intentionally standalone so the
    extraction remains installable without the main ShovsOS provider stack.
    """

    def __init__(self, config: LlamaCppConfig | None = None, *, transport: Transport | None = None):
        self.config = config or LlamaCppConfig()
        self._transport = transport or _http_json

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 700,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        response = await self._request("POST", "/chat/completions", payload)
        return _completion_text(response)

    async def models(self) -> list[str]:
        response = await self._request("GET", "/models", None)
        data = response.get("data")
        if not isinstance(data, list):
            return []
        return [str(item.get("id") or item.get("name") or "").strip() for item in data if isinstance(item, dict) and (item.get("id") or item.get("name"))]

    async def health(self) -> bool:
        try:
            await self.models()
            return True
        except Exception:
            return False

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        attempts = max(1, self.config.retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await asyncio.to_thread(
                    self._transport,
                    method,
                    self.config.base_url.rstrip("/") + path,
                    payload,
                    headers,
                    self.config.timeout,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                await asyncio.sleep(min(0.25 * (attempt + 1), 1.0))
        raise LlamaCppError(f"llama.cpp request failed after {attempts} attempt(s): {last_error}") from last_error


def _http_json(method: str, url: str, payload: dict[str, Any] | None, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LlamaCppError(f"HTTP {exc.code} from {url}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise LlamaCppError(f"could not reach llama.cpp at {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        elapsed = time.monotonic() - started
        raise LlamaCppError(f"invalid JSON from llama.cpp after {elapsed:.2f}s") from exc
    if not isinstance(parsed, dict):
        raise LlamaCppError("llama.cpp returned non-object JSON")
    return parsed


def _completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaCppError("llama.cpp response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LlamaCppError("llama.cpp choice was not an object")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise LlamaCppError("llama.cpp choice did not include text content")
