import pytest
import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

from engine.context_engine import ContextEngine
from memory.semantic_graph import SemanticGraph
from orchestration.session_manager import SessionManager


def deterministic_mock_embedding(text: str) -> list[float]:
    """
    Returns explicit vectors where 'database' and 'PostgreSQL' are highly correlated,
    while 'Berlin' and 'Acme' are orthogonal.
    """
    if "postgresql" in text.lower():
        return [0.9, 0.1, 0.0] * 256
    if "database systems sql" in text.lower():
        return [0.8, 0.2, 0.0] * 256
    if "berlin" in text.lower():
        return [0.0, 0.9, 0.1] * 256
    if "acme" in text.lower():
        return [0.0, 0.0, 0.9] * 256
    return [0.1] * 768


@pytest.mark.asyncio
async def test_realistic_noisy_llm_extraction():
    """
    Test 1: Tests how the parser handles a realistically messy LLM output.
    LLMs often add conversational filler, mess up whitespace, or add partial tags.
    """
    adapter = MagicMock()
    # Noisy, conversational output from a misbehaving model
    adapter.complete = AsyncMock(return_value='''
Here is a summary of the exchange.
- The user mentioned they are moving
[FACT: User | location | Berlin, Germany ]
Wait, I should probably void the old one.
[ VOIDS : User| location ]
[FACT: User | likes ] <-- malformed, should be ignored
''')

    ctx_engine = ContextEngine(adapter=adapter, compression_model="mock-model")

    compressed_ctx, new_facts, voids = await ctx_engine.compress_exchange(
        user_message="I'm actually moving to Berlin, Germany now.",
        assistant_response="Got it, updated your location to Berlin, Germany.",
        current_context="- User location: Toronto",
        is_first_exchange=False,
        model="mock-model",
    )

    # Clean parsing of VOIDS despite whitespace and conversational noise
    assert any(v["subject"] == "User" and v["predicate"] == "location" for v in voids), "Failed to isolate VOIDS from noise"
    
    # Grounded extraction of the FACT
    assert any(f["subject"] == "User" and f["object"] == "Berlin, Germany" for f in new_facts), "Failed to isolate FACT from noise"
    
    # Malformed facts should be ignored safely
    assert len(new_facts) == 1, "Malformed fact was incorrectly parsed"


@pytest.mark.asyncio
async def test_is_grounded_fact_record_guardrail():
    """
    Test 2: Tests the fact guardrail. If an LLM hallucinates a fact that the user
    never said (and isn't in grounding text), the system MUST reject it.
    """
    adapter = MagicMock()
    # The LLM hallucinates that the user likes apples, even though they never said it.
    adapter.complete = AsyncMock(return_value='''
[FACT: User | likes | apples]
[FACT: User | likes | red cars]
''')

    ctx_engine = ContextEngine(adapter=adapter, compression_model="mock-model")

    # In the exchange, the user only mentions red cars.
    _, new_facts, _ = await ctx_engine.compress_exchange(
        user_message="I love red cars.",
        assistant_response="Got it.",
        current_context="- User likes blue cars",
        is_first_exchange=False,
        model="mock-model",
    )

    # 'apples' should be completely blocked by the guardrail.
    # 'red cars' should pass because it's in the user_message.
    assert any(f["object"] == "red cars" for f in new_facts), "Valid grounded fact was incorrectly blocked"
    assert not any(f["object"] == "apples" for f in new_facts), "Hallucinated fact bypassed the fact_guard"


def test_true_semantic_vector_retrieval():
    """
    Test 3: Tests the actual SemanticGraph using distinct mock embeddings to ensure 
    cosine similarity correctly ranks and filters irrelevant facts across sessions.
    """
    owner_id = "test_semantic_owner"
    graph = SemanticGraph()
    
    # Injecting our deterministic hasher to test true retrieval distance
    graph._get_embedding = AsyncMock(side_effect=deterministic_mock_embedding)
    graph.clear(owner_id=owner_id)

    # Write multiple distinct facts
    asyncio.run(graph.add_triplet("User", "likes", "PostgreSQL database", owner_id=owner_id))
    asyncio.run(graph.add_triplet("Company", "name", "Acme Corp", owner_id=owner_id))
    asyncio.run(graph.add_triplet("User", "lives_in", "Berlin", owner_id=owner_id))

    # We semantic traverse for database preferences
    nodes = asyncio.run(graph.traverse("database systems sql", top_k=1, threshold=0.1, owner_id=owner_id))
    
    # It must return PostgreSQL, and NOT Berlin or Acme Corp, because of distinct vector hashes
    assert len(nodes) == 1, "Semantic search returned too many or zero values"
    assert nodes[0]["object"] == "PostgreSQL database", "Semantic search failed nearest-neighbor logic"

    graph.clear(owner_id=owner_id)
