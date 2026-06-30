import pytest

from shovs_harness_core import LlamaCppClient, LlamaCppConfig, LlamaCppError


@pytest.mark.asyncio
async def test_llamacpp_client_completes_against_openai_compatible_shape():
    calls = []

    def transport(method, url, payload, headers, timeout):
        calls.append((method, url, payload, headers, timeout))
        assert method == "POST"
        assert url == "http://local.test/v1/chat/completions"
        assert headers["Authorization"] == "Bearer local-key"
        assert payload["model"] == "qwen3-8b"
        return {"choices": [{"message": {"content": "[{\"action\":\"respond\"}]"}}]}

    client = LlamaCppClient(
        LlamaCppConfig(base_url="http://local.test/v1", api_key="local-key", timeout=7.0),
        transport=transport,
    )

    text = await client.complete(model="qwen3-8b", messages=[{"role": "user", "content": "hi"}])

    assert text == '[{"action":"respond"}]'
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_llamacpp_client_lists_models():
    def transport(method, url, payload, headers, timeout):
        assert method == "GET"
        assert url == "http://local.test/v1/models"
        return {"data": [{"id": "qwen3-8b"}, {"id": "llama-3.2-3b"}]}

    client = LlamaCppClient(LlamaCppConfig(base_url="http://local.test/v1"), transport=transport)

    assert await client.models() == ["qwen3-8b", "llama-3.2-3b"]
    assert await client.health() is True


@pytest.mark.asyncio
async def test_llamacpp_client_reports_bad_completion_shape():
    def transport(method, url, payload, headers, timeout):
        return {"choices": []}

    client = LlamaCppClient(LlamaCppConfig(base_url="http://local.test/v1"), transport=transport)

    with pytest.raises(LlamaCppError):
        await client.complete(model="qwen3-8b", messages=[])
