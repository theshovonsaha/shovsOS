import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api.consumer_routes as consumer_routes_mod
import api.main as main_mod
from api.main import app
from llm.adapter_factory import _ADAPTER_CACHE, create_adapter, get_default_model
from llm.bridge_adapter import BridgeAdapter


def test_openai_compatible_adapters_keep_provider_identity(monkeypatch):
    _ADAPTER_CACHE.clear()
    monkeypatch.setenv("LMSTUDIO_DEFAULT_MODEL", "local/lmstudio-default")
    monkeypatch.setenv("LLAMACPP_DEFAULT_MODEL", "llama-cpp-default")
    monkeypatch.setenv("LOCAL_OPENAI_DEFAULT_MODEL", "local-openai-default")

    lmstudio = create_adapter("lmstudio:openai/gpt-oss-20b")
    llamacpp = create_adapter("llamacpp:llama-3.1-8b-instruct")
    local_openai = create_adapter("local_openai:qwen2.5")

    assert getattr(lmstudio, "provider_name") == "lmstudio"
    assert getattr(llamacpp, "provider_name") == "llamacpp"
    assert getattr(local_openai, "provider_name") == "local_openai"
    assert get_default_model(lmstudio) == "local/lmstudio-default"
    assert get_default_model(llamacpp) == "llama-cpp-default"
    assert get_default_model(local_openai) == "local-openai-default"


def test_bridge_handoff_includes_runtime_knobs(tmp_path, monkeypatch):
    adapter = BridgeAdapter(bridge_dir=str(tmp_path), timeout=0.01, poll_interval=0.01)
    monkeypatch.setattr("llm.bridge_adapter.uuid.uuid4", lambda: type("U", (), {"hex": "abc1234567890000"})())

    path = adapter._write_handoff(
        "abc123456789",
        "bridge",
        [{"role": "user", "content": "hello"}],
        0.2,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        max_tokens=123,
        images=["base64-image"],
        reasoning_enabled=False,
    )

    payload = json.loads(open(path).read())
    assert payload["max_tokens"] == 123
    assert payload["image_count"] == 1
    assert payload["images"] == ["base64-image"]
    assert payload["reasoning_enabled"] is False
    assert payload["tools"][0]["function"]["name"] == "lookup"


@pytest.mark.asyncio
async def test_bridge_stream_forwards_runtime_knobs_to_complete(tmp_path):
    adapter = BridgeAdapter(bridge_dir=str(tmp_path), timeout=0.01, poll_interval=0.01)
    adapter.complete = AsyncMock(return_value="hello world")
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    chunks = [
        chunk
        async for chunk in adapter.stream(
            model="bridge",
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            max_tokens=50,
            images=["image"],
            reasoning_enabled=True,
        )
    ]

    assert "".join(chunks) == "hello world"
    adapter.complete.assert_awaited_once()
    assert adapter.complete.await_args.kwargs["tools"] == tools
    assert adapter.complete.await_args.kwargs["max_tokens"] == 50
    assert adapter.complete.await_args.kwargs["images"] == ["image"]
    assert adapter.complete.await_args.kwargs["reasoning_enabled"] is True


class _FakeAdapter:
    def __init__(self, provider: str):
        self.provider = provider

    async def health(self):
        return self.provider == "nvidia"

    async def list_models(self):
        return [f"{self.provider}-model"]


def test_api_provider_surfaces_include_nvidia(monkeypatch):
    def fake_create_adapter(provider=None):
        return _FakeAdapter(str(provider or "auto"))

    monkeypatch.setattr(main_mod, "create_adapter", fake_create_adapter)
    monkeypatch.setattr(consumer_routes_mod, "create_adapter", fake_create_adapter)

    client = TestClient(app)
    health = client.get("/health")
    models = client.get("/models")
    consumer_health = client.get("/consumer/health")
    consumer_models = client.get("/consumer/models")

    assert health.status_code == 200
    assert health.json()["providers"]["nvidia"] is True
    assert models.status_code == 200
    assert models.json()["models"]["nvidia"] == ["nvidia-model"]
    assert models.json()["capabilities"]["nvidia:nvidia-model"]["chat"] is True
    assert "vision_model" in models.json()
    assert consumer_health.status_code == 200
    assert consumer_health.json()["providers"]["nvidia"] is True
    assert consumer_models.status_code == 200
    assert consumer_models.json()["models"]["nvidia"] == ["nvidia-model"]
    assert consumer_models.json()["capabilities"]["nvidia:nvidia-model"]["chat"] is True
    assert "vision_model" in consumer_models.json()
