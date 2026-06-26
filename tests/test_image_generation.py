import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as main_mod
import plugins.tools as tools_mod
from api.main import app
from memory.semantic_graph import SemanticGraph
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolRegistry
from plugins.tools import IMAGE_GENERATE_TOOL, register_tools
from run_engine.engine import RunEngine
from run_engine.tool_selection import fallback_tool_call
from run_engine.types import RunEngineRequest


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


def test_image_generate_tool_is_registered_in_subset():
    registry = ToolRegistry()

    register_tools(registry, "image_generate")

    tool = registry.get("image_generate")
    assert tool is not None
    assert tool.response_format == "json"
    assert "prompt" in tool.parameters["required"]


def test_image_generation_is_discoverable_from_platform_api(monkeypatch):
    class FakeAdapter:
        async def health(self):
            return True

        async def list_models(self):
            return ["fake-model"]

    monkeypatch.setattr(main_mod, "create_adapter", lambda provider=None: FakeAdapter())
    client = TestClient(app)

    tools_response = client.get("/tools")
    models_response = client.get("/models")

    assert tools_response.status_code == 200
    assert any(item["name"] == "image_generate" for item in tools_response.json()["tools"])
    assert models_response.status_code == 200
    assert models_response.json()["image_generation_model"]


def test_image_generation_intent_ranks_generation_before_search():
    ranked = RunEngine._rank_fallback_tool_candidates(
        "Generate an image of a calm product mockup",
        ["image_search", "image_generate", "web_search"],
    )
    fallback = fallback_tool_call("image_generate", "Generate an image of a calm product mockup")

    assert ranked[:2] == ["image_generate", "image_search"]
    assert fallback is not None
    assert fallback.arguments == {"prompt": "Generate an image of a calm product mockup"}


def test_image_generate_tool_returns_typed_error_without_provider(monkeypatch):
    async def fake_generate_image(**kwargs):
        raise RuntimeError("OPENAI_API_KEY is required for image generation")

    monkeypatch.setattr(tools_mod, "generate_image", fake_generate_image)

    result = tools_mod.asyncio.run(IMAGE_GENERATE_TOOL.handler(prompt="a quiet dashboard"))
    payload = json.loads(result)

    assert payload["type"] == "image_generation_error"
    assert payload["success"] is False
    assert "OPENAI_API_KEY" in payload["error"]


def test_image_generate_tool_returns_generation_payload(monkeypatch):
    async def fake_generate_image(**kwargs):
        return {
            "type": "image_generation_result",
            "url": "/sandbox/generated/images/demo.png",
            "path": "generated/images/demo.png",
            "prompt": kwargs["prompt"],
            "model": "gpt-image-1",
            "bytes": 12,
        }

    monkeypatch.setattr(tools_mod, "generate_image", fake_generate_image)

    result = tools_mod.asyncio.run(IMAGE_GENERATE_TOOL.handler(prompt="a quiet dashboard"))
    payload = json.loads(result)

    assert payload["type"] == "image_generation_result"
    assert payload["url"] == "/sandbox/generated/images/demo.png"
    assert payload["prompt"] == "a quiet dashboard"


@pytest.mark.asyncio
async def test_run_engine_small_model_dry_run_calls_image_generate(tmp_path, monkeypatch):
    async def fake_generate_image(**kwargs):
        return {
            "type": "image_generation_result",
            "url": "/sandbox/generated/images/dry.png",
            "path": "generated/images/dry.png",
            "prompt": kwargs["prompt"],
            "model": "gpt-image-1",
            "bytes": 12,
        }

    monkeypatch.setattr(tools_mod, "generate_image", fake_generate_image)
    registry = ToolRegistry()
    register_tools(registry, "image_generate")

    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value=json.dumps(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "image_generate",
                            "arguments": json.dumps({"prompt": "a calm product mockup"}),
                        }
                    }
                ]
            }
        )
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Generated image ready."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=MagicMock(append_event=MagicMock()),
        orchestrator=None,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="image-generate-dry-run",
                owner_id="owner-image-dry-run",
                agent_id="default",
                user_message="Generate an image of a calm product mockup",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("image_generate",),
                use_planner=False,
            )
        )
    ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "image_generate" for event in events)
    image_result = next(event for event in events if event.get("type") == "tool_result")
    assert image_result["success"] is True
    assert json.loads(image_result["content"])["type"] == "image_generation_result"
    assert any(event.get("type") == "done" for event in events)


def test_image_generation_endpoint_uses_shared_service(monkeypatch):
    async def fake_generate_image(**kwargs):
        return {
            "type": "image_generation_result",
            "url": "/sandbox/generated/images/api.png",
            "path": "generated/images/api.png",
            "prompt": kwargs["prompt"],
            "model": kwargs["model"] or "gpt-image-1",
            "bytes": 10,
        }

    monkeypatch.setattr(main_mod, "generate_image", fake_generate_image)
    client = TestClient(app)

    response = client.post(
        "/images/generate",
        json={"prompt": "minimal product mockup", "model": "gpt-image-1"},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "image_generation_result"
    assert response.json()["url"] == "/sandbox/generated/images/api.png"

    prefixed_response = client.post(
        "/api/images/generate",
        json={"prompt": "minimal product mockup", "model": "gpt-image-1"},
    )
    assert prefixed_response.status_code == 200
    assert prefixed_response.json()["url"] == "/sandbox/generated/images/api.png"
