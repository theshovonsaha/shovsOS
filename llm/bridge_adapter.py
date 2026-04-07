"""
BridgeAdapter — File-based IPC adapter for external agent control
------------------------------------------------------------------
Instead of calling a remote LLM API, this adapter writes the prompt
to a handoff file and polls for a response file.  An external agent
(e.g. a human, Copilot, or another process) reads the handoff,
produces a response, and writes it back — becoming the model.

Handoff directory defaults to BRIDGE_DIR env var or ./agent_sandbox/bridge/.
"""

import asyncio
import json
import os
import time
import uuid
from typing import AsyncIterator, Optional

from llm.base_adapter import BaseLLMAdapter


_DEFAULT_BRIDGE_DIR = os.path.join("agent_sandbox", "bridge")
_POLL_INTERVAL = float(os.getenv("BRIDGE_POLL_INTERVAL", "1.0"))
_TIMEOUT = float(os.getenv("BRIDGE_TIMEOUT", "300"))  # 5 min default


class BridgeAdapter(BaseLLMAdapter):
    """
    Adapter that delegates to an external agent via the filesystem.

    Protocol:
      1. Engine calls complete()/stream()
      2. Adapter writes  <bridge_dir>/handoff_<id>.json
      3. Adapter polls for <bridge_dir>/response_<id>.json
      4. External agent reads the handoff, writes the response
      5. Adapter returns the response text to the engine
    """

    def __init__(
        self,
        bridge_dir: Optional[str] = None,
        poll_interval: float = _POLL_INTERVAL,
        timeout: float = _TIMEOUT,
    ):
        self.bridge_dir = bridge_dir or os.getenv("BRIDGE_DIR", _DEFAULT_BRIDGE_DIR)
        self.poll_interval = poll_interval
        self.timeout = timeout
        os.makedirs(self.bridge_dir, exist_ok=True)

    # ── core handoff ────────────────────────────────────────────────────

    def _write_handoff(
        self,
        request_id: str,
        model: str,
        messages: list[dict],
        temperature: float,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Write handoff file and return its path."""
        payload = {
            "request_id": request_id,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools or [],
            "timestamp": time.time(),
        }
        handoff_path = os.path.join(self.bridge_dir, f"handoff_{request_id}.json")
        with open(handoff_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return handoff_path

    async def _poll_response(self, request_id: str) -> str:
        """Block until response file appears or timeout."""
        response_path = os.path.join(self.bridge_dir, f"response_{request_id}.json")
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            if os.path.exists(response_path):
                with open(response_path, "r") as f:
                    data = json.load(f)
                # clean up both files after reading
                handoff_path = os.path.join(self.bridge_dir, f"handoff_{request_id}.json")
                for p in (response_path, handoff_path):
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
                # support either {"content": "..."} or {"response": "..."}
                if isinstance(data, dict):
                    return str(data.get("content") or data.get("response") or "")
                return str(data)
            await asyncio.sleep(self.poll_interval)

        raise TimeoutError(
            f"Bridge response timed out after {self.timeout}s. "
            f"Expected file: {response_path}"
        )

    # ── BaseLLMAdapter interface ────────────────────────────────────────

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
    ) -> str:
        request_id = uuid.uuid4().hex[:12]
        handoff = self._write_handoff(request_id, model, messages, temperature, tools)
        print(f"\n[BRIDGE] Handoff written: {handoff}")
        print(f"[BRIDGE] Waiting for response: response_{request_id}.json")
        return await self._poll_response(request_id)

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
        # For streaming, use the same handoff mechanism but yield in chunks
        response = await self.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images,
            tools=tools,
        )
        # Yield word-by-word to simulate streaming
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"

    async def list_models(self) -> list[str]:
        return ["bridge"]

    async def health(self) -> bool:
        return os.path.isdir(self.bridge_dir)
