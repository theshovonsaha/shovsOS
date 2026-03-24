# Guardrails Integration Guide
## How to wire guardrails into your existing agent system

---

## 1. Install the package (copy the folder)

Place the `guardrails/` folder at your project root alongside
`plugins/`, `engine/`, `orchestration/` etc.

```
agent/
├── guardrails/          ← drop this here
│   ├── __init__.py
│   ├── risk_classifier.py
│   ├── confirmation_gate.py
│   ├── tool_logger.py
│   ├── middleware.py
│   └── api_routes.py
├── plugins/
├── engine/
├── orchestration/
└── api/
```

---

## 2. Wire into main.py / app startup

```python
# api/main.py  (or wherever you build your FastAPI app)

from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools
from guardrails import GuardrailMiddleware
from guardrails.api_routes import make_guardrail_router

# Build your existing stack as normal
registry = ToolRegistry()
register_all_tools(registry, agent_manager=manager)

# Wrap registry with guardrails — one line
middleware = GuardrailMiddleware(
    registry                  = registry,
    require_confirmation_for  = "all",   # every tool call pauses
    confirmation_timeout      = 120.0,   # seconds before auto-deny
    log_path                  = "./logs/tool_audit.jsonl",
)

# Mount the guardrails API routes
app.include_router(
    make_guardrail_router(middleware),
    prefix="/guardrails"
)
```

---

## 3. Replace registry.execute() in AgentCore

In `engine/core.py`, find wherever you call `registry.execute(call)` and
replace it with `middleware.execute_stream()`.

### Streaming version (recommended — plugs into your existing event loop):

```python
# engine/core.py
# Inject middleware via constructor or as a module-level singleton

async def _run_tool(self, call: ToolCall, session_id: str):
    async for event in self.middleware.execute_stream(
        call,
        session_id = session_id,
        agent_id   = self.agent_id,
        context    = {"_session_id": session_id},
    ):
        yield event   # forward directly to chat_stream caller
        
        # Stop processing if blocked/denied — return error to LLM
        if event["type"] in ("tool_blocked", "confirmation_denied", "confirmation_timeout"):
            return ToolResult(
                tool_name = call.tool_name,
                success   = False,
                content   = event.get("reason", event["type"]),
            )
        if event["type"] == "tool_result":
            return ToolResult(
                tool_name = call.tool_name,
                success   = event["success"],
                content   = event["content"],
            )
```

### Simple non-streaming version (if you don't need SSE events):

```python
result = await middleware.execute(
    call,
    session_id = session_id,
    agent_id   = self.agent_id,
)
```

---

## 4. Frontend integration

### Subscribe to confirmation events (SSE):

```javascript
// Connect once per session
const es = new EventSource(`/guardrails/stream/${sessionId}`);

es.addEventListener('confirmation_required', (e) => {
    const data = JSON.parse(e.data);
    showConfirmDialog({
        callId:   data.call_id,
        tool:     data.tool,
        preview:  data.preview,
        reason:   data.reason,
        args:     data.arguments,
    });
});
```

### Approve / Deny buttons:

```javascript
async function approve(callId) {
    await fetch(`/guardrails/approve/${callId}`, { method: 'POST' });
}

async function deny(callId, reason = 'User denied') {
    await fetch(`/guardrails/deny/${callId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
    });
}
```

### Show audit log:

```javascript
const res = await fetch('/guardrails/log?n=20');
const { entries } = await res.json();
// entries is array of {ts, event, tool, call_id, session_id, ...}
```

---

## 5. Configuration options

```python
middleware = GuardrailMiddleware(
    registry = registry,

    # Confirmation scope:
    # "all"               → every single tool pauses for approval
    # "confirm_and_above" → only bash/file_create/delegate pause
    # "block_only"        → only hard-blocked commands are stopped
    require_confirmation_for = "all",

    # How long to wait for user response before auto-denying
    confirmation_timeout = 120.0,

    # Where to write the audit log
    log_path = "./logs/tool_audit.jsonl",

    # Add your own blocked bash patterns
    extra_block_patterns = [
        "dropdb", "mongodrop", "redis-cli flushall"
    ],

    # Add your own confirm-required patterns
    extra_confirm_patterns = [
        "systemctl", "launchctl"
    ],

    # Override risk level for specific tools
    custom_tool_risks = {
        "web_search": RiskLevel.CONFIRM,   # make search require confirmation too
        "store_memory": RiskLevel.SAFE,    # trust memory writes
    },
)
```

---

## 6. Sub-agent safety rule

When creating agent profiles, give each sub-agent only the tools it needs:

```python
# Researcher: read-only, no confirmation friction
AgentProfile(id="researcher", tools=["web_search", "web_fetch", "query_memory"])

# Analyst: can write files in sandbox, confirm required
AgentProfile(id="analyst", tools=["file_create", "file_view"])

# Coder: bash only, always confirmed
AgentProfile(id="coder", tools=["bash", "file_create"])

# Never give sub-agents: delegate_to_agent (prevents recursive delegation)
```

---

## 7. Dry-run mode for tests

```bash
AGENT_DRY_RUN=true pytest tests/
```

In `middleware.py`, check this env var:

```python
import os
DRY_RUN = os.getenv("AGENT_DRY_RUN", "false").lower() == "true"

# In execute_stream, before calling registry.execute():
if DRY_RUN:
    yield {"type": "tool_result", "tool": call.tool_name,
           "success": True, "content": f"[DRY RUN] {classification.preview}"}
    return
```
