# LM Studio MCP Bridge

This folder contains a standalone MCP adapter that exposes the existing engine tools to MCP clients such as LM Studio, without changing your core backend code.

## File

- `mcp_connectors/lmstudio_engine_tools_bridge.py`

## Why this adapter

- Reuses your existing tool implementations from `plugins/tools.py` and `plugins/tools_web.py`.
- Runs as a separate process (stdio MCP server), so your current FastAPI system remains unchanged.
- Supports safety profiles:
  - `safe`: read-first allowlist
  - `full`: all registered tools

## Prerequisites

Use your project venv and ensure MCP SDK is installed:

```bash
/Users/theshovonsaha/Developer/Github/agent/venv/bin/python -m pip install mcp
```

## Quick self-check

```bash
cd /Users/theshovonsaha/Developer/Github/agent
/Users/theshovonsaha/Developer/Github/agent/venv/bin/python mcp_connectors/lmstudio_engine_tools_bridge.py --self-check
```

## Run as MCP server (stdio)

Safe profile:

```bash
cd /Users/theshovonsaha/Developer/Github/agent
/Users/theshovonsaha/Developer/Github/agent/venv/bin/python mcp_connectors/lmstudio_engine_tools_bridge.py --profile safe
```

Full profile:

```bash
cd /Users/theshovonsaha/Developer/Github/agent
/Users/theshovonsaha/Developer/Github/agent/venv/bin/python mcp_connectors/lmstudio_engine_tools_bridge.py --profile full
```

## Tool filtering

Include only specific tools:

```bash
/Users/theshovonsaha/Developer/Github/agent/venv/bin/python mcp_connectors/lmstudio_engine_tools_bridge.py --include web_search,web_fetch,query_memory
```

Exclude tools:

```bash
/Users/theshovonsaha/Developer/Github/agent/venv/bin/python mcp_connectors/lmstudio_engine_tools_bridge.py --profile full --exclude bash,file_str_replace
```

## LM Studio setup

In LM Studio MCP server configuration, use:

- Command: `/Users/theshovonsaha/Developer/Github/agent/venv/bin/python`
- Args:
  - `/Users/theshovonsaha/Developer/Github/agent/mcp_connectors/lmstudio_engine_tools_bridge.py`
  - `--profile`
  - `safe`

Optional args for custom session/agent context:

- `--session-id lmstudio-main`
- `--agent-id default`

## Notes

- This bridge is intentionally isolated and does not require edits to your existing API routes.
- If you want full write/exec capability, use `--profile full` carefully.
