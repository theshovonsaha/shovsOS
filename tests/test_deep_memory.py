"""
Deep Memory Tests
-----------------
Tests the functionality of the SemanticGraph SQLite+Vector database
and the tool abstraction for storing/querying memories.
"""

import pytest
import asyncio
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Force the semantic graph to use a test DB
TEST_DB = "test_memory_graph.db"

@pytest.fixture(autouse=True)
def setup_teardown():
    # Cleanup before
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    # Cleanup after
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.mark.asyncio
async def test_semantic_graph_storage_and_retrieval():
    """Test that we can store a triplet and retrieve it via fuzzy conceptual match."""
    from memory.semantic_graph import SemanticGraph
    
    # Mock the embedding call to return deterministic vectors
    # "cilantro" ~ [0.9, 0.1]
    # "spicy food" ~ [0.1, 0.9]
    # "dietary restriction" query ~ [0.8, 0.2] (closer to cilantro)
    async def mock_get_embedding(text: str):
        if "cilantro" in text.lower(): return [0.9, 0.1]
        if "spicy" in text.lower(): return [0.1, 0.9]
        if "dietary" in text.lower(): return [0.8, 0.2]
        return [0.5, 0.5]
    
    with patch.object(SemanticGraph, '_get_embedding', side_effect=mock_get_embedding):
        graph = SemanticGraph(db_path=TEST_DB)
        
        # Store some memories
        await graph.add_triplet("User", "dislikes", "cilantro")
        await graph.add_triplet("User", "loves", "spicy food")
        
        # Query for something conceptually related to dietary/dislikes
        results = await graph.traverse("user dietary restrictions", top_k=5, threshold=0.1)
        
        assert len(results) == 2, "Should return both since threshold is low by default"
        
        # The cilantro memory should be ranked HIGHER because [0.8, 0.2] is closer to [0.9, 0.1]
        assert results[0]['object'] == "cilantro"
        assert results[1]['object'] == "spicy food"
        assert results[0]['similarity'] > results[1]['similarity']


@pytest.mark.asyncio
async def test_store_and_query_tools():
    """Test the tool wrappers for the Semantic Graph."""
    from plugins.tools import _store_memory, _query_memory
    
    # We patch the SemanticGraph class at the source
    with patch("memory.semantic_graph.SemanticGraph") as MockGraph:
        mock_instance = MagicMock()
        mock_instance.add_triplet = AsyncMock()
        
        # Mocking traverse to return a simulated DB hit
        mock_instance.traverse = AsyncMock(return_value=[
            {"subject": "User", "predicate": "is allergic to", "object": "peanuts", "similarity": 0.95}
        ])
        MockGraph.return_value = mock_instance
        
        # Test Store
        res_store = await _store_memory("User", "is allergic to", "peanuts")
        assert "Successfully stored" in res_store
        assert "peanuts" in res_store
        mock_instance.add_triplet.assert_called_once_with("User", "is allergic to", "peanuts")
        
        # Test Query
        res_query = await _query_memory("allergies")
        assert "Found 1" in res_query
        assert "peanuts" in res_query
        mock_instance.traverse.assert_called_once_with("allergies", top_k=5)


@pytest.mark.asyncio
async def test_store_memory_uses_runtime_embed_model():
    from plugins.tools import _store_memory

    with patch("memory.semantic_graph.SemanticGraph") as MockGraph:
        mock_instance = MagicMock()
        mock_instance.add_triplet = AsyncMock()
        MockGraph.return_value = mock_instance

        res_store = await _store_memory(
            "User",
            "prefers",
            "privacy-first budgeting",
            _embed_model="lmstudio:text-embedding-nomic-embed-text-v1.5",
        )

        assert "Successfully stored" in res_store
        MockGraph.assert_called_once_with(embedding_model="lmstudio:text-embedding-nomic-embed-text-v1.5")
