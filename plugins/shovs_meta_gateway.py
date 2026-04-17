"""
Shovs Meta Gateway
------------------
A reflection SDK exposed as tools for external agents (like OpenClaw).
This allows an autonomous agent to discover the Shovs platform,
interact with its memory, and delegate tasks back to the orchestrator.
"""

import json
from typing import Optional, List, Dict
from plugins.tool_registry import Tool, ToolRegistry
from memory.semantic_graph import SemanticGraph
from memory.retrieval import unified_memory_search
from config.logger import log

_tool_registry: Optional[ToolRegistry] = None
_graph: Optional[SemanticGraph] = None

def inject_gateway_dependencies(registry: ToolRegistry, graph: SemanticGraph):
    global _tool_registry, _graph
    _tool_registry = registry
    _graph = graph

async def _get_platform_manifest() -> str:
    """Returns a full description of Shovs OS capabilities."""
    if not _tool_registry:
        return "Gateway not initialized."
    
    tools = _tool_registry.list_tools()
    manifest = {
        "platform": "Shovs OS",
        "version": "2026.4",
        "capabilities": [
            "Spatial Memory (Memory Palace)",
            "Relational Graph Memory",
            "Managed Tool Execution",
            "Multi-Agent Delegation"
        ],
        "available_tools": tools
    }
    return json.dumps(manifest, indent=2)

async def _spatial_query(query: str, locus_id: Optional[str] = None, _owner_id: Optional[str] = None) -> str:
    """Query the Shovs memory system (Relational + Spatial)."""
    if not _graph:
        return "Memory graph not available."
    
    results = await unified_memory_search(
        query=query,
        owner_id=_owner_id,
        locus_id=locus_id,
        graph=_graph
    )
    return json.dumps(results, indent=2)

async def _spatial_store(subject: str, predicate: str, object: str, locus_id: Optional[str] = None, _owner_id: Optional[str] = None, _run_id: Optional[str] = None) -> str:
    """Store a fact into the Shovs semantic graph with optional spatial anchoring."""
    if not _graph:
        return "Memory graph not available."

    try:
        # Auto-register the locus if a locus_id is given but doesn't exist yet.
        if locus_id:
            existing = _graph.get_locus(locus_id, owner_id=_owner_id)
            if existing is None:
                _graph.register_locus(locus_id, name=locus_id, owner_id=_owner_id)
                log("memory", "gateway", f"Auto-registered locus '{locus_id}' during store")

        await _graph.add_triplet(
            subject=subject,
            predicate=predicate,
            object_=object,
            owner_id=_owner_id,
            run_id=_run_id,
            locus_id=locus_id
        )
        return f"Stored fact: {subject} {predicate} {object} (Locus: {locus_id or 'Global'})"
    except Exception as e:
        return f"Error storing fact: {e}"


async def _create_locus(locus_id: str, name: Optional[str] = None, description: str = "", _owner_id: Optional[str] = None) -> str:
    """Register a new named spatial room (Locus) in the Memory Palace."""
    if not _graph:
        return "Memory graph not available."
    try:
        locus_name = name or locus_id
        _graph.register_locus(locus_id, name=locus_name, description=description, owner_id=_owner_id)
        return json.dumps({
            "status": "created",
            "locus_id": locus_id,
            "name": locus_name,
            "description": description,
        })
    except Exception as e:
        return f"Error creating locus: {e}"


async def _list_spatial_loci(_owner_id: Optional[str] = None) -> str:
    """List all available Rooms/Rooms (Loci) in the Memory Palace."""
    if not _graph:
        return "Memory graph not available."
    
    loci = _graph.list_loci(owner_id=_owner_id)
    return json.dumps(loci, indent=2)

# Tool Definitions
GET_MANIFEST_TOOL = Tool(
    name="shovs_get_manifest",
    description="Reflect on Shovs OS capabilities and discover all available platform tools.",
    parameters={"type": "object", "properties": {}},
    handler=_get_platform_manifest,
    tags=["meta", "discovery"]
)

SPATIAL_QUERY_TOOL = Tool(
    name="shovs_memory_query",
    description="Perform a unified semantic and spatial search across Shovs OS memory.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "locus_id": {"type": "string", "description": "Optional: Focus search within a specific spatial room ID"}
        },
        "required": ["query"]
    },
    handler=_spatial_query,
    tags=["meta", "memory"]
)

SPATIAL_STORE_TOOL = Tool(
    name="shovs_memory_store",
    description="Store a new relational fact in Shovs OS memory, optionally anchored to a spatial room.",
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "locus_id": {"type": "string", "description": "Optional: Anchor fact to this room ID"}
        },
        "required": ["subject", "predicate", "object"]
    },
    handler=_spatial_store,
    tags=["meta", "memory"]
)

LIST_LOCI_TOOL = Tool(
    name="shovs_list_loci",
    description="List all available spatial rooms (Memory Palace Loci) for context anchoring.",
    parameters={"type": "object", "properties": {}},
    handler=_list_spatial_loci,
    tags=["meta", "memory", "spatial"]
)

def register_gateway_tools(registry: ToolRegistry):
    registry.register(GET_MANIFEST_TOOL)
    registry.register(SPATIAL_QUERY_TOOL)
    registry.register(SPATIAL_STORE_TOOL)
    registry.register(LIST_LOCI_TOOL)
