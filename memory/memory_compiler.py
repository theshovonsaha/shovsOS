"""
Memory Compiler
---------------
Implements the Karpathy "LLM as a Compiler" pattern.
Takes raw relational triplets, temporal facts, and session evidence,
and "compiles" them into a high-density, structured Markdown "Drawer"
for a specific spatial Locus.
"""

import asyncio
from typing import Optional, List, Dict
from datetime import datetime

from memory.semantic_graph import SemanticGraph
from llm.base_adapter import BaseLLMAdapter
from config.config import cfg

COMPILER_PROMPT = """
You are the Shovs Memory Compiler.
Your task is to take the provided "Raw Relational Evidence" and "Temporal Facts" for a specific spatial Locus (Location/Room) and compile them into a high-density, structured Markdown "Drawer".

--- COMPILER DIRECTIVES ---
1. STRUCTURE: Use hierarchical headers (##, ###).
2. DENSITY: Eliminate conversational filler. Use bullet points and tables.
3. PRESERVATION: Keep exact names, versions, and technical details.
4. LINKING: Use [[WikiLinks]] to connect entities.
5. NO SLOP: If evidence is contradictory, note the contradiction rather than picking a side.
6. TARGET: This "Drawer" will be injected as the primary context for the agent when it enters this Locus.

Locus Name: {locus_name}
Locus Description: {locus_description}

RAW EVIDENCE:
{raw_evidence}

TEMPORAL FACTS:
{temporal_facts}

### COMPILED DRAWER OUTPUT ###
"""

class MemoryCompiler:
    def __init__(self, graph: SemanticGraph, adapter: BaseLLMAdapter):
        self.graph = graph
        self.adapter = adapter

    async def compile_locus(self, locus_id: str, owner_id: Optional[str] = None, model: Optional[str] = None) -> bool:
        """
        Gathers all raw data associated with a locus and uses the LLM to compile it.
        """
        locus = self.graph.get_locus(locus_id, owner_id=owner_id)
        if not locus:
            return False

        # Gather relational triplets
        triplets = await self.graph.traverse("", top_k=500, threshold=0.0, owner_id=owner_id, locus_id=locus_id)
        raw_evidence = "\n".join([f"- {t['subject']} {t['predicate']} {t['object']}" for t in triplets])

        # Gather temporal facts (we might need a method to filter facts by LocusId in SemanticGraph)
        # For now, we use the ones that matched locus_id in add_temporal_fact
        facts = self.graph.list_temporal_facts_by_locus(locus_id, owner_id=owner_id)
        fact_text = "\n".join([f"- {f['subject']} {f['predicate']} {f['object']} (Turn {f['valid_from']})" for f in facts])

        prompt = COMPILER_PROMPT.format(
            locus_name=locus['name'],
            locus_description=locus.get('description', ''),
            raw_evidence=raw_evidence or "No raw evidence found.",
            temporal_facts=fact_text or "No temporal facts found."
        )

        try:
            from llm.adapter_factory import get_default_model
            model = model or get_default_model(self.adapter)
            messages = [
                {"role": "system", "content": "You are a precise technical archiver."},
                {"role": "user", "content": prompt}
            ]
            compiled_markdown = await self.adapter.complete(model=model, messages=messages)
            self.graph.update_compiled_drawer(locus_id, compiled_markdown)
            return True
        except Exception as e:
            print(f"[MemoryCompiler] Error compiling locus {locus_id}: {e}")
            return False

    async def batch_compile_all(self, owner_id: Optional[str] = None):
        """Finds all loci and compiles them."""
        # We need a list_loci method in SemanticGraph
        loci = self.graph.list_loci(owner_id=owner_id)
        for locus in loci:
            await self.compile_locus(locus['id'], owner_id=owner_id)
