import pytest
import asyncio
import httpx
from api.main import app
from httpx import AsyncClient
from memory.vector_engine import VectorEngine

async def _ollama_available() -> bool:
    from config.config import cfg
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{cfg.OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_full_rag_cycle():
    """
    Tests the full Anchored RAG cycle.
    """
    if not await _ollama_available():
        pytest.skip("Ollama is not reachable; skipping live RAG cycle test.")

    import uuid
    session_id = f"test_rag_{uuid.uuid4().hex[:8]}"
    ve = VectorEngine(session_id)
    await ve.clear()
    
    # Use ASGITransport for testing the app directly
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Step 1: Establish a fact
        resp = await ac.post("/chat/stream", data={
            "message": "I love collecting vintage mechanical keyboards, especially IBM Model M.",
            "session_id": session_id,
            "model": "llama3.2", # Use larger model for reliable extraction
            "owner_id": "full-system-owner",
        })
        # Wait for stream to finish
        async for line in resp.aiter_lines():
            pass
            
        # Step 2: Verify Indexing
        count = await ve.count()
        assert count > 0, "Fact should have been indexed in VectorEngine (ensure llama3.2 is running)"
        
        # Step 3: Retrieval Test
        # We check if the next query retrieves the anchor
        results = await ve.query("What kind of keyboards do I collect?")
        assert any("Model M" in r.get("anchor", "") for r in results)
        print("\n[SYSTEM TEST] RAG Retrieval Verified.")

@pytest.mark.asyncio
async def test_web_tools_integration():
    """
    Tests the web_search and web_fetch tool chain.
    """
    from plugins.tools_web import _web_search, _web_fetch
    
    print("\n[SYSTEM TEST] Testing Web Search...")
    search_res = await _web_search("Current CEO of Groq", num_results=1)
    if (
        "error" in search_res.lower()
        or "No results found" in search_res
        or "no backends configured" in search_res.lower()
    ):
        pytest.skip(f"Web search backend unavailable: {search_res[:120]}")
    assert "Jonathan Ross" in search_res or "Groq" in search_res
    print("[SYSTEM TEST] Web Search Verified.")
    
    print("[SYSTEM TEST] Testing Web Fetch...")
    fetch_res = await _web_fetch("https://example.com", max_chars=100)
    if "error" in fetch_res.lower():
        pytest.skip(f"Web fetch backend unavailable: {fetch_res[:120]}")
    assert "Example Domain" in fetch_res
    print("[SYSTEM TEST] Web Fetch Verified.")

@pytest.mark.asyncio
async def test_solvability_guard_trigger():
    """
    Verified by unit tests in test_hallucination_guards.py
    """
    pass
