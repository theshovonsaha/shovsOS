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


def _resolve_turn(session_id: Optional[str], owner_id: Optional[str], graph: SemanticGraph) -> int:
    """Resolve the current turn number for temporal fact operations."""
    try:
        from orchestration.session_manager import SessionManager
        sm = SessionManager()
        session = sm.get(session_id, owner_id=owner_id) if session_id else None
        if session:
            return len(session.full_history)
    except Exception:
        pass
    return 0

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

async def _spatial_query(query: Optional[str] = None, topic: Optional[str] = None, locus_id: Optional[str] = None, _owner_id: Optional[str] = None) -> str:
    """Query the Shovs memory system (Relational + Spatial). Accepts 'query' or 'topic' interchangeably."""
    if not _graph:
        return "Memory graph not available."

    # Accept either 'query' or 'topic' — LLMs sometimes use the wrong alias.
    search_term = (query or topic or "").strip()
    if not search_term:
        return "shovs_memory_query requires a 'query' (or 'topic') argument."

    results = await unified_memory_search(
        query=search_term,
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


async def _spatial_update(
    subject: str,
    predicate: str,
    object: str,
    locus_id: Optional[str] = None,
    _owner_id: Optional[str] = None,
    _session_id: Optional[str] = None,
    _run_id: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Correct or update a fact in Shovs memory.

    Voids any existing fact for (subject, predicate) in this session's
    deterministic timeline, then writes the new value. Also updates the
    semantic graph triplet so future retrieval prefers the new value.
    Pass locus_id to anchor the corrected fact to a specific Memory Palace room.
    """
    if not _graph:
        return "Memory graph not available."

    session_id = _session_id or kwargs.get("session_id")
    try:
        if locus_id:
            existing = _graph.get_locus(locus_id, owner_id=_owner_id)
            if existing is None:
                _graph.register_locus(locus_id, name=locus_id, owner_id=_owner_id)

        turn = _resolve_turn(session_id, _owner_id, _graph)

        if session_id:
            # Void the previous fact for this (subject, predicate) pair
            _graph.void_temporal_fact(
                session_id, subject.strip(), predicate.strip(), turn, owner_id=_owner_id
            )
            # Write the corrected fact into the deterministic timeline
            _graph.add_temporal_fact(
                session_id, subject.strip(), predicate.strip(), object.strip(),
                turn, owner_id=_owner_id, run_id=_run_id, locus_id=locus_id,
            )

        # Hard-filter: drop superseded vector triplets so semantic retrieval
        # cannot resurface the old value alongside the corrected one.
        _graph.delete_triplets(subject.strip(), predicate.strip(), owner_id=_owner_id)

        # Update the semantic graph triplet
        await _graph.add_triplet(
            subject=subject.strip(),
            predicate=predicate.strip(),
            object_=object.strip(),
            owner_id=_owner_id,
            run_id=_run_id,
            locus_id=locus_id,
        )
        locus_note = f" (Locus: {locus_id})" if locus_id else ""
        lane_note = "session + semantic memory" if session_id else "semantic memory only"
        return (
            f"Updated memory in {lane_note}: [{subject}] --[{predicate}]--> [{object}]{locus_note}"
        )
    except Exception as e:
        return f"Error updating fact: {e}"


async def _spatial_void(
    subject: str,
    predicate: str,
    _owner_id: Optional[str] = None,
    _session_id: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Void (delete) a fact from the current session's deterministic memory.

    Marks the most recent fact for (subject, predicate) as superseded so it
    will no longer surface in retrieval. Use when the user explicitly says
    "forget" or "that's no longer true" and there is no replacement value.
    """
    if not _graph:
        return "Memory graph not available."

    session_id = _session_id or kwargs.get("session_id")
    if not session_id:
        return "shovs_memory_void requires an active session (cannot void without session context)."

    try:
        turn = _resolve_turn(session_id, _owner_id, _graph)
        _graph.void_temporal_fact(
            session_id, subject.strip(), predicate.strip(), turn, owner_id=_owner_id
        )
        # Hard-filter: drop matching vector triplets so semantic retrieval
        # honors the void rather than ranking the stale embedding alongside.
        removed = _graph.delete_triplets(subject.strip(), predicate.strip(), owner_id=_owner_id)
        suffix = f" — removed {removed} vector triplet(s)" if removed else ""
        return f"Voided fact: [{subject}] --[{predicate}]--> (no longer current){suffix}"
    except Exception as e:
        return f"Error voiding fact: {e}"


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
    description="Perform a unified semantic and spatial search across Shovs OS memory. Use 'query' (or alias 'topic') for the search term.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query (also accepted as 'topic')"},
            "topic": {"type": "string", "description": "Alias for 'query' — use either one"},
            "locus_id": {"type": "string", "description": "Optional: Focus search within a specific spatial room ID"}
        },
        "required": []
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

SPATIAL_UPDATE_TOOL = Tool(
    name="shovs_memory_update",
    description=(
        "Correct or update an existing fact in Shovs memory. "
        "Voids the previous value for (subject, predicate) and writes the new one. "
        "Use when the user corrects a fact: 'actually my name is James not John', "
        "'I moved to Vancouver', 'forget the old price — it's $49 now'. "
        "Pass locus_id to anchor the correction to a specific Memory Palace room."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity the fact is about (e.g. 'User', 'Project')"},
            "predicate": {"type": "string", "description": "The relationship being corrected (e.g. 'name', 'location', 'price')"},
            "object": {"type": "string", "description": "The new correct value"},
            "locus_id": {"type": "string", "description": "Optional: Anchor to this Memory Palace room ID"},
        },
        "required": ["subject", "predicate", "object"],
    },
    handler=_spatial_update,
    tags=["meta", "memory"],
)

SPATIAL_VOID_TOOL = Tool(
    name="shovs_memory_void",
    description=(
        "Remove (void) a fact from memory when the user says it is no longer true "
        "and provides no replacement. Examples: 'forget my editor preference', "
        "'I no longer work at Google', 'ignore the old budget'. "
        "Use shovs_memory_update instead when a replacement value is provided."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity the fact is about (e.g. 'User')"},
            "predicate": {"type": "string", "description": "The relationship to void (e.g. 'preferred_editor', 'employer')"},
        },
        "required": ["subject", "predicate"],
    },
    handler=_spatial_void,
    tags=["meta", "memory"],
)

LIST_LOCI_TOOL = Tool(
    name="shovs_list_loci",
    description="List all available spatial rooms (Memory Palace Loci) for context anchoring.",
    parameters={"type": "object", "properties": {}},
    handler=_list_spatial_loci,
    tags=["meta", "memory", "spatial"]
)

CREATE_LOCUS_TOOL = Tool(
    name="shovs_create_locus",
    description=(
        "Explicitly create a named spatial room (Locus) in the Memory Palace. "
        "Use this when the user requests a dedicated locus for a topic, project, or research area. "
        "After creation, anchor future facts to this locus via shovs_memory_store."
    ),
    parameters={
        "type": "object",
        "properties": {
            "locus_id": {"type": "string", "description": "Unique identifier for this locus (e.g. 'MUMBAI_RESEARCH_2026')"},
            "name": {"type": "string", "description": "Human-readable name (defaults to locus_id)"},
            "description": {"type": "string", "description": "What this locus stores / its purpose"},
        },
        "required": ["locus_id"]
    },
    handler=_create_locus,
    tags=["meta", "memory", "spatial"]
)

def register_gateway_tools(registry: ToolRegistry):
    registry.register(GET_MANIFEST_TOOL)
    registry.register(SPATIAL_QUERY_TOOL)
    registry.register(SPATIAL_STORE_TOOL)
    registry.register(SPATIAL_UPDATE_TOOL)
    registry.register(SPATIAL_VOID_TOOL)
    registry.register(LIST_LOCI_TOOL)
    registry.register(CREATE_LOCUS_TOOL)
