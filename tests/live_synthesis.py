import asyncio
import os
from memory.semantic_graph import SemanticGraph
from memory.memory_compiler import MemoryCompiler
from llm.adapter_factory import create_adapter
from config.config import cfg

async def main():
    print(f"--- Live Spatial Synthesis Test ---")
    print(f"Default Model: {cfg.DEFAULT_MODEL}")
    print(f"Embed Model:   {cfg.EMBED_MODEL}")
    
    graph = SemanticGraph()
    adapter = create_adapter("ollama")
    compiler = MemoryCompiler(graph, adapter)
    
    locus_id = "project_alpha_room"
    owner_id = "shovon"
    
    # 1. Setup Locus
    print(f"\n1. Registering Locus: {locus_id}...")
    graph.register_locus(locus_id, "Project Alpha Ops", "Strategic planning for Alpha project", owner_id)
    
    # 2. Add raw fragments
    print("2. Adding raw memory fragments...")
    await graph.add_triplet("Deadline", "is set for", "Dec 20th", owner_id=owner_id, locus_id=locus_id)
    await graph.add_triplet("Budget", "exceeds", "$250k", owner_id=owner_id, locus_id=locus_id)
    await graph.add_triplet("Lead Engineer", "is", "Sarah Chen", owner_id=owner_id, locus_id=locus_id)
    
    # 3. RUN SYNTHESIS (Live LLM call)
    print("\n3. RUNNING LIVE SYNTHESIS (Memory Compiler)...")
    try:
        success = await compiler.compile_locus(locus_id, model=cfg.DEFAULT_MODEL)
        if success:
            print("✅ Synthesis Successful!")
            
            # 4. Verify Drawer
            drawer = graph.get_compiled_drawer(locus_id)
            print("\n--- Compiled Executive Drawer ---")
            print(drawer)
        else:
            print("❌ Synthesis failed (check logs)")
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
