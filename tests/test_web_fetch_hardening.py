import json

import pytest

from plugins import tools_web


@pytest.mark.asyncio
async def test_web_fetch_blocks_non_http_scheme():
    result = await tools_web._web_fetch("file:///etc/passwd")
    data = json.loads(result)
    assert data["type"] == "web_fetch_error"
    assert "http/https" in data["error"]


@pytest.mark.asyncio
async def test_web_fetch_blocks_private_host_by_default():
    assert tools_web.WEB_FETCH_ALLOW_PRIVATE is False
    result = await tools_web._web_fetch("http://127.0.0.1:8000/health")
    data = json.loads(result)
    assert data["type"] == "web_fetch_error"
    assert "127.0.0.1" in data["error"]


@pytest.mark.asyncio
async def test_web_fetch_groq_path_includes_metadata(monkeypatch):
    async def fake_fetch_groq(_url: str, _max_chars: int):
        return "Sample fetched body from model tool."

    monkeypatch.setattr(tools_web, "_fetch_groq", fake_fetch_groq)
    monkeypatch.setattr(tools_web, "GROQ_KEY", "test-key")

    raw = await tools_web._web_fetch(
        "https://example.com/path?utm_source=test&x=1",
        max_chars=2000,
        use_jina=False,
    )
    data = json.loads(raw)

    assert data["type"] == "web_fetch_result"
    assert data["backend"] == "groq"
    assert data["url"] == "https://example.com/path?x=1"
    assert data["final_url"] == "https://example.com/path?x=1"
    assert data["host"] == "example.com"
    assert data["status_code"] == 200
    assert "Sample fetched body" in data["content"]
