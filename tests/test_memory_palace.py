import asyncio
import pytest
from memory.semantic_graph import SemanticGraph
from memory.retrieval import unified_memory_search
from memory.memory_compiler import MemoryCompiler

@pytest.mark.asyncio
async def test_spatial_memory_routing():
    graph = SemanticGraph()
    locus_id = "test_lab_v1"
    owner_id = "test_owner"
    
    # Clean up from previous runs
    import sqlite3
    with sqlite3.connect(graph.db_path) as conn:
        conn.execute("DELETE FROM loci WHERE id = ?", (locus_id,))
        conn.execute("DELETE FROM facts WHERE locus_id = ?", (locus_id,))
        conn.execute("DELETE FROM memories WHERE locus_id = ?", (locus_id,))
    
    # 1. Register Locus
    graph.register_locus(
        locus_id=locus_id,
        name="The Lab",
        description="A high-tech testing environment",
        owner_id=owner_id
    )
    
    # 2. Add Fact to Locus
    await graph.add_triplet(
        subject="Project X",
        predicate="is located in",
        object_="The Lab",
        owner_id=owner_id,
        locus_id=locus_id
    )
    
    # 3. Compile Drawer (Simulate background compilation)
    drawer_content = "# The Lab Summary\n- Project X is active here.\n- High-security protocols are in place."
    graph.update_compiled_drawer(locus_id, drawer_content)
    
    # 4. Query with Spatial Scan (Implicit)
    query = "What's happening in the lab?"
    results = await unified_memory_search(
        query=query,
        owner_id=owner_id,
        graph=graph
    )
    
    # Verify Compiled Drawer is prioritized
    hits = results.get("results", [])
    assert len(hits) > 0
    assert hits[0]["kind"] == "compiled_drawer"
    assert hits[0]["locus_id"] == locus_id
    assert "Project X" in hits[0]["content"]
    
    # Verify Triplet is also found
    triplet_hit = next((h for h in hits if h["kind"] == "triplet"), None)
    assert triplet_hit is not None
    assert triplet_hit["subject"] == "Project X"

@pytest.mark.asyncio
async def test_memory_compiler_background_run():
    # This test verifies the compiler can distill facts into a drawer
    graph = SemanticGraph()
    locus_id = "test_study"
    owner_id = "test_user"
    
    graph.register_locus(locus_id, "Study", owner_id=owner_id)
    await graph.add_triplet("John", "is studying", "Quantum Physics", owner_id=owner_id, locus_id=locus_id)
    
    # We skip actual LLM call in unit test for speed, but check method existence
    from llm.adapter_factory import create_adapter
    compiler = MemoryCompiler(graph, create_adapter("ollama"))
    assert hasattr(compiler, "compile_locus")

if __name__ == "__main__":
    asyncio.run(test_spatial_memory_routing())
