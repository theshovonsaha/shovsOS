import pytest
import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

# We will implement this class in vector_engine.py
# from memory.vector_engine import VectorEngine 

@pytest.mark.asyncio
async def test_vector_engine_anchored_retrieval():
    """
    RED TEST: Should retrieve the raw 3-turn anchor when a semantic key is matched.
    """
    # This will fail because VectorEngine is not yet defined
    from memory.vector_engine import VectorEngine

    # Keep this as a deterministic unit test by mocking embeddings.
    with patch.object(VectorEngine, "_get_embedding", new=AsyncMock(return_value=[0.1] * 768)):
        ve = VectorEngine(session_id="test_session")
        await ve.clear()

        # 1. Index a "Key" anchored to a "Raw Turn"
        key = "User preference: Strictly uses dark mode for all UI components"
        anchor = "User: I only ever want to see dark mode. Assistant: Understood, I will apply dark mode."

        await ve.index(key=key, anchor=anchor, metadata={"importance": 1.0})

        # 2. Query with a different but semantically similar string
        results = await ve.query("What are the UI theme preferences?", limit=1)

        assert len(results) > 0
        assert results[0]["key"] == key
        assert results[0]["anchor"] == anchor

@pytest.mark.asyncio
async def test_vector_engine_multi_key_same_anchor():
    """
    Should preserve key→anchor relationships for multiple keys sharing one anchor.
    """
    from memory.vector_engine import VectorEngine

    with patch.object(VectorEngine, "_get_embedding", new=AsyncMock(return_value=[0.1] * 768)):
        ve = VectorEngine(session_id="test_session_unique")
        await ve.clear()

        anchor = "Static conversation turn"
        await ve.index(key="key1", anchor=anchor)
        await ve.index(key="key2", anchor=anchor)  # Same anchor, different key

        count = await ve.count()
        assert count == 2


@pytest.mark.asyncio
async def test_vector_engine_uniqueness():
    """
    Should prevent duplicate indexing of identical key+anchor pairs.
    """
    from memory.vector_engine import VectorEngine

    with patch.object(VectorEngine, "_get_embedding", new=AsyncMock(return_value=[0.1] * 768)):
        ve = VectorEngine(session_id="test_session_unique_exact")
        await ve.clear()

        anchor = "Static conversation turn"
        await ve.index(key="key1", anchor=anchor)
        await ve.index(key="key1", anchor=anchor)

        count = await ve.count()
        assert count == 1


def test_vector_engine_reuses_chroma_client_for_same_path():
    from memory.vector_engine import VectorEngine

    VectorEngine._chroma_clients.clear()
    shared_client = MagicMock()

    with patch.object(VectorEngine, "_ensure_collection", return_value=None):
        with patch("memory.vector_engine.chromadb.PersistentClient", side_effect=lambda path: shared_client) as mock_client:
            first = VectorEngine(session_id="test_session_one")
            second = VectorEngine(session_id="test_session_two")

    assert mock_client.call_count == 1
    assert first.client is second.client


def test_vector_engine_resolves_lmstudio_embeddings_transport(monkeypatch):
    from memory.vector_engine import VectorEngine

    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "lm-studio")

    with patch.object(VectorEngine, "_ensure_collection", return_value=None):
        ve = VectorEngine(session_id="embed_test", model="lmstudio:text-embedding-nomic-embed-text-v1.5")

    provider, base_url, endpoint, headers, payload = ve._resolve_embedding_transport("Some text to embed")
    assert provider == "lmstudio"
    assert base_url == "http://127.0.0.1:1234/v1"
    assert endpoint == "/v1/embeddings"
    assert payload["model"] == "text-embedding-nomic-embed-text-v1.5"
    assert payload["input"] == "Some text to embed"
    assert headers["Authorization"] == "Bearer lm-studio"


@pytest.mark.asyncio
async def test_vector_engine_ollama_embed_endpoint_falls_back_to_legacy_embeddings():
    from memory.vector_engine import VectorEngine

    with patch.object(VectorEngine, "_ensure_collection", return_value=None):
        ve = VectorEngine(session_id="embed_fallback_test", model="nomic-embed-text")

    class FakeResponse:
        def __init__(self, status_code: int, data: dict):
            self.status_code = status_code
            self._data = data

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "http://localhost:11434/api/embed")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError("404", request=request, response=response)

        def json(self):
            return self._data

    import httpx

    client = MagicMock()
    client.post = AsyncMock(side_effect=[
        FakeResponse(404, {}),
        FakeResponse(200, {"embedding": [0.1, 0.2, 0.3]}),
    ])

    with patch.object(VectorEngine, "_get_http_client", return_value=client):
        embedding = await ve._get_embedding("fallback check")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.post.await_args_list[0].args[0] == "/api/embed"
    assert client.post.await_args_list[1].args[0] == "/api/embeddings"
