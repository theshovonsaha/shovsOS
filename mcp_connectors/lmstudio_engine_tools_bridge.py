#!/usr/bin/env python3
"""
LM Studio MCP Connector for Shovs Agent Platform tools.

Purpose:
- Expose existing in-repo ToolRegistry tools over MCP (stdio) so LM Studio
  can use them as callable tools.
- Keep your main system unchanged: this runs as a separate adapter process.

Usage examples:
  python mcp_connectors/lmstudio_engine_tools_bridge.py --self-check
  python mcp_connectors/lmstudio_engine_tools_bridge.py --profile full

LM Studio command:
  /path/to/venv/bin/python /path/to/repo/mcp_connectors/lmstudio_engine_tools_bridge.py --profile safe
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure imports work even when launched from outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp import types
from mcp.server.lowlevel.server import (
    InitializationOptions,
    NotificationOptions,
    Server,
)
from mcp.server.stdio import stdio_server

from plugins.tool_registry import ToolCall, ToolRegistry
from plugins.tools import register_all_tools
from plugins.tools_web import register_web_tools

SERVER_NAME = "shovs-engine-tools-bridge"
SERVER_VERSION = "0.1.0"

SAFE_ALLOWLIST = {
    "web_search",
    "web_fetch",
    "image_search",
    "weather_fetch",
    "places_search",
    "places_map",
    "query_memory",
    "rag_search",
    "file_view",
}


def _csv_to_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # MCP stdio requires clean stdout. Some bootstrap functions print status text,
    # so we silence stdout here to avoid protocol corruption.
    with contextlib.redirect_stdout(io.StringIO()):
        # Keep this adapter isolated from orchestration: no agent manager injection.
        register_all_tools(registry, agent_manager=None)
        register_web_tools(registry)

    return registry


def _select_tools(
    tool_defs: list[dict],
    profile: str,
    include: set[str],
    exclude: set[str],
) -> list[dict]:
    selected = list(tool_defs)

    if include:
        selected = [tool for tool in selected if tool.get("name") in include]
    elif profile == "safe":
        selected = [tool for tool in selected if tool.get("name") in SAFE_ALLOWLIST]

    if exclude:
        selected = [tool for tool in selected if tool.get("name") not in exclude]

    return selected


def _to_mcp_tool(tool_def: dict) -> types.Tool:
    name = tool_def.get("name", "unknown_tool")
    description = tool_def.get("description") or "No description provided."
    schema = tool_def.get("parameters") or {"type": "object", "properties": {}}

    # MCP expects object schema for tool input.
    if not isinstance(schema, dict) or schema.get("type") != "object":
        schema = {"type": "object", "properties": {}}

    return types.Tool(name=name, description=description, inputSchema=schema)


async def _run_stdio_server(args: argparse.Namespace) -> None:
    registry = _make_registry()
    selected_defs = _select_tools(
        tool_defs=registry.list_tools(),
        profile=args.profile,
        include=_csv_to_set(args.include),
        exclude=_csv_to_set(args.exclude),
    )

    selected_names = {tool["name"] for tool in selected_defs if "name" in tool}
    mcp_tools = [_to_mcp_tool(tool) for tool in selected_defs]

    if args.self_check:
        print(f"Selected tools: {len(mcp_tools)}")
        for tool in mcp_tools:
            print(f"- {tool.name}")
        return

    session_id = args.session_id or os.getenv("MCP_BRIDGE_SESSION_ID", "lmstudio-bridge")
    agent_id = args.agent_id or os.getenv("MCP_BRIDGE_AGENT_ID", "default")

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return mcp_tools

    @server.call_tool(validate_input=True)
    async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
        if name not in selected_names:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=(
                            f"Tool '{name}' is not exposed by this bridge. "
                            "Adjust --profile/--include/--exclude settings."
                        ),
                    )
                ],
                isError=True,
            )

        call = ToolCall(
            tool_name=name,
            arguments=arguments or {},
            raw_json=json.dumps({"tool": name, "arguments": arguments or {}}),
        )

        result = await registry.execute(
            call,
            context={
                "_session_id": session_id,
                "_agent_id": agent_id,
            },
        )

        if result.success:
            return [types.TextContent(type="text", text=result.content)]

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=True,
        )

    instructions = (
        "Bridge MCP server exposing Shovs engine tools to external MCP clients. "
        "By default it runs in safe profile; use --profile full for all tools."
    )

    init_options = InitializationOptions(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
        instructions=instructions,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expose Shovs engine tools as an MCP stdio server for LM Studio."
    )
    parser.add_argument(
        "--profile",
        choices=["safe", "full"],
        default=os.getenv("MCP_BRIDGE_TOOL_PROFILE", "safe"),
        help="safe = read-first allowlist; full = all registered tools",
    )
    parser.add_argument(
        "--include",
        default=os.getenv("MCP_BRIDGE_INCLUDE", ""),
        help="Comma-separated explicit tool allowlist (overrides profile selection)",
    )
    parser.add_argument(
        "--exclude",
        default=os.getenv("MCP_BRIDGE_EXCLUDE", ""),
        help="Comma-separated tool denylist",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("MCP_BRIDGE_SESSION_ID", ""),
        help="Optional session id injected into tool context",
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("MCP_BRIDGE_AGENT_ID", ""),
        help="Optional agent id injected into tool context",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Print selected tools and exit (no MCP server start)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run_stdio_server(args))


if __name__ == "__main__":
    main()
