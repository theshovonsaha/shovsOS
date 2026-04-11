"""
OpenClaw Autonomous Bridge Bootstrap
------------------------------------
Version: 1.0 (Spatial-Aware)

This script provides the 'reflection manifest' that allows an OpenClaw agent 
to autonomously discover and bridge itself into the Shovs OS platform.

ARCHITECTURE:
1. Discovery: Call `shovs_get_manifest` to see available Shovs tools.
2. Memory: Use `shovs_memory_query` and `shovs_memory_store` for spatial-relational persistence.
3. Loci: Use `shovs_list_loci` to navigate the Memory Palace.

AUTONOMOUS BRIDGE HANDSHAKE:
----------------------------
If you are an OpenClaw agent, follow these steps:
1. Load the Shovs Meta Gateway skills.
2. Query Locus ID 'openclaw_bridge_v1' to retrieve your last persistent state.
3. Align your objective with the Shovs Orchestrator via the meta-skill manifest.
"""

import json
from plugins.shovs_meta_gateway import (
    GET_MANIFEST_TOOL,
    SPATIAL_QUERY_TOOL,
    SPATIAL_STORE_TOOL,
    LIST_LOCI_TOOL
)

def get_openclaw_skill_manifest():
    """Returns the JSON manifest for OpenClaw skill ingestion."""
    return {
        "namespace": "shovs",
        "description": "Native bridge for Shovs OS platform capabilities.",
        "skills": [
            {
                "name": GET_MANIFEST_TOOL.name,
                "description": GET_MANIFEST_TOOL.description,
                "parameters": GET_MANIFEST_TOOL.parameters
            },
            {
                "name": SPATIAL_QUERY_TOOL.name,
                "description": SPATIAL_QUERY_TOOL.description,
                "parameters": SPATIAL_QUERY_TOOL.parameters
            },
            {
                "name": SPATIAL_STORE_TOOL.name,
                "description": SPATIAL_STORE_TOOL.description,
                "parameters": SPATIAL_STORE_TOOL.parameters
            },
            {
                "name": LIST_LOCI_TOOL.name,
                "description": LIST_LOCI_TOOL.description,
                "parameters": LIST_LOCI_TOOL.parameters
            }
        ],
        "system_guidance": (
            "You are operating as a bridged-in agent within Shovs OS. "
            "Use the 'shovs_' prefixed tools to interact with the host platform's "
            "spatial memory and tool registry. "
            "Prioritize 'Compiled Drawers' returned by memory queries as they represent "
            "high-density synthesized context from the Shovs compiler."
        )
    }

if __name__ == "__main__":
    # When run directly, output the manifest for the external agent to swallow
    print(json.dumps(get_openclaw_skill_manifest(), indent=2))
