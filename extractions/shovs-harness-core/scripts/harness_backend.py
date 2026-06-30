from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shovs_harness_core import HarnessExtension, LlamaCppClient, LlamaCppConfig, enforce_proposed_actions, infer_source_contract
from shovs_harness_core.proposers import LLMProposer


DEFAULT_WORLD = {
    "entities": ["ROKU", "TBN", "SENEA", "EPAM"],
    "urls": {
        "ROKU": ["https://src.test/ROKU/0", "https://src.test/ROKU/1", "https://src.test/ROKU/2"],
        "TBN": ["https://src.test/TBN/0", "https://src.test/TBN/1", "https://src.test/TBN/2"],
        "SENEA": ["https://src.test/SENEA/0", "https://src.test/SENEA/1", "https://src.test/SENEA/2"],
        "EPAM": ["https://src.test/EPAM/0"],
    },
}


class HarnessHandler(BaseHTTPRequestHandler):
    server_version = "ShovsHarnessCore/0.1"

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/health":
            self._send_json({"ok": True, "service": "shovs-harness-core", "version": HarnessExtension.version})
            return
        if self.path == "/api/manifest":
            self._send_json(HarnessExtension().manifest())
            return
        if self.path == "/api/llamacpp/health":
            self._send_json(asyncio.run(_llamacpp_health()))
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        payload = self._read_json()
        if self.path == "/api/run":
            self._send_json(HarnessExtension().run(payload))
            return
        if self.path == "/api/llamacpp/probe":
            self._send_json(asyncio.run(_llamacpp_probe(payload)))
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[harness-backend] " + fmt % args + "\n")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)


async def _llamacpp_health() -> dict[str, Any]:
    client = _client()
    try:
        models = await client.models()
        return {"ok": True, "base_url": client.config.base_url, "models": models}
    except Exception as exc:
        return {"ok": False, "base_url": client.config.base_url, "error": f"{type(exc).__name__}: {exc}"}


async def _llamacpp_probe(payload: dict[str, Any]) -> dict[str, Any]:
    objective = str(payload.get("objective") or "Search top 3 stocks today, search each, fetch 3 URLs each.")
    model = str(payload.get("model") or os.getenv("LLAMACPP_DEFAULT_MODEL") or "local-model")
    world = payload.get("world") if isinstance(payload.get("world"), dict) else DEFAULT_WORLD
    contract = infer_source_contract(objective)
    proposer = LLMProposer(_client(), model)
    try:
        actions = await proposer.propose(contract, world)
        report = enforce_proposed_actions(contract, actions, candidate_urls=world.get("urls") if isinstance(world.get("urls"), dict) else None)
        return {
            "ok": True,
            "model": model,
            "objective": objective,
            "proposed_actions": [
                {"action": item.action, "entity": item.entity, "url": item.url, "entities": list(item.entities)}
                for item in actions
            ],
            "enforcement": report.to_dict(),
        }
    except Exception as exc:
        return {"ok": False, "model": model, "objective": objective, "error": f"{type(exc).__name__}: {exc}"}


def _client() -> LlamaCppClient:
    return LlamaCppClient(
        LlamaCppConfig(
            base_url=os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8081/v1"),
            api_key=os.getenv("LLAMACPP_API_KEY", "llama.cpp"),
            timeout=float(os.getenv("LLAMACPP_TIMEOUT", "90")),
            retries=int(os.getenv("LLAMACPP_RETRIES", "1")),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Shovs Harness Core JSON backend.")
    parser.add_argument("--host", default=os.getenv("HARNESS_BACKEND_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("HARNESS_BACKEND_PORT", "8091")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), HarnessHandler)
    print(f"Shovs Harness Core backend: http://{args.host}:{args.port}")
    print("Endpoints: /health, /api/manifest, POST /api/run, /api/llamacpp/health, POST /api/llamacpp/probe")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping harness backend.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
