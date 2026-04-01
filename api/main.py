"""
Agent Platform — FastAPI Backend v0.6.0
---------------------------------------
Changes from v0.5.0:
  - Docker sandbox for bash execution
  - MCP client support (mcp_servers.json)
  - Per-session RAG with ChromaDB
  - rag_search as a callable agent tool
  - File upload endpoint for RAG ingestion
  - Researcher agent profile seeded at startup
  - Bug fixes: delegation 404, hallucinated URLs, context truncation
"""

import asyncio
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

# ── Ensure required directories exist (before any module creates DBs/logs) ────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _dir in ("logs", "chroma_db", "agent_sandbox"):
    (_PROJECT_ROOT / _dir).mkdir(parents=True, exist_ok=True)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine.core import AgentCore
from orchestration.agent_profiles import ProfileManager, AgentProfile
from orchestration.orchestrator import AgenticOrchestrator
from llm.llm_adapter import OllamaAdapter
from engine.context_engine import ContextEngine
from orchestration.session_manager import SessionManager
from llm.adapter_factory import create_adapter
from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools
from plugins.tools_web import register_web_tools
from engine.file_processor import FileProcessor
from orchestration.agent_manager import AgentManager
from config.logger import get_logger, log
from api.log_routes import setup_log_routes
from api.voice_endpoint import setup_voice_routes
from api.rag_routes import make_rag_router            # ← NEW
from api.consumer_routes import make_consumer_router
from api.memory_inspector import build_session_memory_payload
from services.storage_admin import StorageAdminService, StoreSelection
from services.consumer_store import ConsumerStoreService

from guardrails import GuardrailMiddleware
from guardrails.api_routes import make_guardrail_router

# ── Config ────────────────────────────────────────────────────────────────────
FALLBACK_CHAT_MODEL = "llama3.2"
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "30/minute")


def _require_owner_id(owner_id: Optional[str]) -> str:
    normalized = (owner_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="owner_id is required")
    return normalized


def _count_context_items(raw_context: str) -> int:
    if not raw_context:
        return 0
    try:
        payload = json.loads(raw_context)
        if isinstance(payload, dict) and payload.get("__v2__"):
            modules = payload.get("modules", {})
            return len(modules) if isinstance(modules, dict) else 0
    except Exception:
        pass
    return len([l for l in raw_context.split("\n") if l.strip()])


def _context_preview(raw_context: str) -> list[str]:
    if not raw_context:
        return []
    try:
        payload = json.loads(raw_context)
        if isinstance(payload, dict) and payload.get("__v2__"):
            goals = payload.get("active_goals", {})
            modules = payload.get("modules", {})
            lines: list[str] = []
            if isinstance(goals, dict) and goals:
                lines.append("Active Goals: " + ", ".join(list(goals.keys())[:5]))
            if isinstance(modules, dict):
                for key, mod in list(modules.items())[:30]:
                    content = mod.get("content", "") if isinstance(mod, dict) else ""
                    lines.append(f"- {key}: {content}")
            return lines
    except Exception:
        pass
    return [l for l in raw_context.split("\n") if l.strip()]

# ── Singletons ────────────────────────────────────────────────────────────────
adapter         = OllamaAdapter()
tool_registry   = ToolRegistry()
session_manager = SessionManager(max_sessions=200)
context_engine  = ContextEngine(adapter=adapter, compression_model=FALLBACK_CHAT_MODEL)
file_processor  = FileProcessor()
profile_manager = ProfileManager()
orchestrator    = AgenticOrchestrator(adapter=adapter)

# ── Guardrails ────────────────────────────────────────────────────────────────
guardrail_middleware = GuardrailMiddleware(
    registry                 = tool_registry,
    require_confirmation_for = "confirm_and_above",
    log_path                 = "./logs/tool_audit.jsonl",
)

agent_manager = AgentManager(
    profiles             = profile_manager,
    sessions             = session_manager,
    context_engine       = context_engine,
    adapter              = adapter,
    global_registry      = tool_registry,
    orchestrator         = orchestrator,
    guardrail_middleware = guardrail_middleware,
)
storage_admin = StorageAdminService(session_manager, profile_manager)
consumer_session_manager = SessionManager(max_sessions=200, db_path="consumer_sessions.db")
consumer_context_engine = ContextEngine(adapter=adapter, compression_model=FALLBACK_CHAT_MODEL)
consumer_agent_manager = AgentManager(
    profiles=profile_manager,
    sessions=consumer_session_manager,
    context_engine=consumer_context_engine,
    adapter=adapter,
    global_registry=tool_registry,
    orchestrator=orchestrator,
    guardrail_middleware=guardrail_middleware,
)
consumer_store = ConsumerStoreService(
    consumer_db_path="consumer.db",
    consumer_sessions_db_path="consumer_sessions.db",
)

# ── Register tools ────────────────────────────────────────────────────────────
register_all_tools(tool_registry, agent_manager=agent_manager)
register_web_tools(tool_registry)


# ── MCP Client (optional — loads mcp_servers.json if present) ─────────────────
mcp_manager = None

def _should_init_mcp() -> bool:
    enable_mcp = os.getenv("ENABLE_MCP", "true").lower() in ("1", "true", "yes", "on")
    enable_in_dev = os.getenv("ENABLE_MCP_IN_DEV", "false").lower() in ("1", "true", "yes", "on")
    is_reload_dev = os.getenv("UVICORN_RELOAD", "").lower() in ("1", "true", "yes", "on")
    if not enable_mcp:
        return False
    if is_reload_dev and not enable_in_dev:
        return False
    return True

async def _init_mcp():
    global mcp_manager
    if not _should_init_mcp():
        log("mcp", "startup", "MCP init skipped in local dev mode", level="info")
        return
    try:
        from plugins.mcp_client import MCPClientManager
        mcp_manager = MCPClientManager(tool_registry)
        count = await mcp_manager.load_from_config("mcp_servers.json")
        if count:
            log("mcp", "startup", f"Loaded {count} MCP tools", level="ok")
    except Exception as e:
        log("mcp", "startup", f"MCP init skipped: {e}", level="warn")


# ── Standard agent profiles (seeded at startup if missing) ────────────────────
def _seed_standard_profiles():
    standard = [
        AgentProfile(
            id="researcher",
            name="Research Specialist",
            model="groq:llama-3.3-70b-versatile",
            tools=["web_search", "web_fetch", "rag_search", "query_memory", "store_memory"],
            system_prompt=(
                "You are a meticulous research agent. Always verify claims across multiple sources. "
                "CRITICAL: Only call web_fetch with URLs returned by a prior web_search result. "
                "Never invent or guess URLs. Cite sources. Never fabricate data."
            ),
        ),
        AgentProfile(
            id="analyst",
            name="Data Analyst Agent",
            model="groq:llama-3.3-70b-versatile",
            tools=["file_create", "file_view", "file_str_replace", "rag_search", "bash"],
            system_prompt=(
                "You are a data analyst. Write clean, well-structured markdown reports and Python scripts. "
                "Save all outputs to files in the sandbox. Use rag_search to recall prior research."
            ),
        ),
        AgentProfile(
            id="coder",
            name="Coder Extraordinaire",
            model="groq:llama-3.3-70b-versatile",
            tools=["bash", "file_create", "file_view", "file_str_replace"],
            system_prompt=(
                "You are an expert programmer. Write clean, tested, working code. "
                "Always run code with bash after writing it to verify it works."
            ),
        ),
        AgentProfile(
            id="consumer",
            name="Consumer Assistant",
            description="Plain-language assistant for the consumer chat surface.",
            model="groq:moonshotai/kimi-k2-instruct",
            tools=["web_search", "web_fetch", "query_memory"],
            system_prompt=(
                "You are a clear, calm assistant for the consumer product surface. "
                "Prefer plain text, avoid internal execution chatter, and only use tools when they materially improve accuracy or complete the task."
            ),
            default_use_planner=False,
            default_loop_mode="auto",
            default_context_mode="v2",
            bootstrap_files=["IDENTITY.md", "SOUL.md"],
        ),
    ]
    for p in standard:
        if not profile_manager.get(p.id):
            profile_manager.create(p)
            log("startup", "profiles", f"Seeded agent profile: {p.id}")


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _seed_standard_profiles()
    await _init_mcp()
    yield
    # Shutdown
    if mcp_manager:
        await mcp_manager.disconnect_all()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="shovs", version="0.6.0", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
setup_log_routes(app)
setup_voice_routes(app, agent_manager)
app.include_router(make_guardrail_router(guardrail_middleware), prefix="/guardrails")
app.include_router(make_rag_router(), prefix="/rag")          # ← NEW
app.include_router(
    make_consumer_router(
        agent_manager=consumer_agent_manager,
        sessions=consumer_session_manager,
        file_processor=file_processor,
        consumer_store=consumer_store,
        tool_registry=tool_registry,
    )
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Static Sandbox Shell ──────────────────────────────────────────────────────
from plugins.tools import SANDBOX_DIR
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/sandbox", StaticFiles(directory=str(SANDBOX_DIR)), name="sandbox")


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:       str  = Field(..., min_length=1, max_length=32_000)
    session_id:    Optional[str] = None
    agent_id:      Optional[str] = "default"
    system_prompt: Optional[str] = None
    force_memory:  Optional[bool] = False
    forced_tools:  Optional[List[str]] = Field(default_factory=list)


class StorageSelectionPayload(BaseModel):
    sessions: bool = True
    agents: bool = False
    semantic_memory: bool = True
    tool_results: bool = True
    vector_memory: bool = True
    session_rag: bool = True


class StorageActionPayload(StorageSelectionPayload):
    backup_first: bool = True
    backup_label: str = ""
    preserve_default_agent: bool = True


class StorageBackupPayload(StorageSelectionPayload):
    backup_label: str = ""


# ── Core routes (unchanged from v0.5.0) ──────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "shovs", "version": "0.6.0",
        "docs": "/docs",
        "tools": len(tool_registry.list_tools()),
        "mcp_servers": mcp_manager.list_connected() if mcp_manager else [],
    }

@app.get("/health")
async def health():
    provider_status: dict[str, bool] = {}
    for provider in (
        "ollama",
        "lmstudio",
        "llamacpp",
        "local_openai",
        "openai",
        "groq",
        "gemini",
        "anthropic",
    ):
        try:
            provider_status[provider] = await create_adapter(provider).health()
        except Exception:
            provider_status[provider] = False

    return {
        "status": "ok" if any(provider_status.values()) else "degraded",
        "providers": provider_status,
        "tools": tool_registry.list_tools(),
    }

@app.get("/models")
async def list_models():
    grouped = {
        "ollama": [],
        "lmstudio": [],
        "llamacpp": [],
        "local_openai": [],
        "openai": [],
        "groq": [],
        "gemini": [],
        "anthropic": [],
    }
    embeddings = []
    for provider in grouped:
        try:
            a = create_adapter(provider)
            models = await a.list_models()
            if models:
                grouped[provider] = models
                if provider in {"ollama", "lmstudio", "llamacpp", "local_openai"}:
                    embeddings.extend([f"{provider}:{m}" for m in models])
        except Exception:
            pass
    
    # Static fallbacks for other providers if needed
    embeddings.extend([
        "openai:text-embedding-3-small", 
        "openai:text-embedding-3-large", 
        "openai:text-embedding-ada-002"
    ])
    
    return {"models": grouped, "embeddings": list(set(embeddings))}

@app.get("/tools")
async def list_tools():
    return {"tools": tool_registry.list_tools()}

@app.post("/chat/stream")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_stream(
    request:           Request,
    message:           str              = Form(...),
    session_id:        Optional[str]    = Form(None),
    agent_id:          Optional[str]    = Form("default"),
    model:             Optional[str]    = Form(None),
    system_prompt:     Optional[str]    = Form(None),
    search_backend:    Optional[str]    = Form(None),
    search_engine:     Optional[str]    = Form(None),
    force_memory:      Optional[bool]   = Form(False),
    use_planner:       Optional[bool]   = Form(True),
    loop_mode:         Optional[str]    = Form("auto"),
    context_mode:      Optional[str]    = Form(None),
    max_tool_calls:    Optional[int]    = Form(None),
    max_turns:         Optional[int]    = Form(None),
    planner_model:     Optional[str]    = Form(None),
    context_model:     Optional[str]    = Form(None),
    embed_model:       Optional[str]    = Form(None),
    forced_tools_json: Optional[str]    = Form(None),
    owner_id:          Optional[str]    = Form(None),
    files:             List[UploadFile] = File(default=[]),
):
    forced_tools = []
    if forced_tools_json:
        try:
            forced_tools = json.loads(forced_tools_json)
        except Exception:
            pass
    owner_id = _require_owner_id(owner_id)

    async def generate():
        try:
            effective_session_id = session_id
            processed_files, image_b64s = [], []
            for upload in files:
                raw = await upload.read()
                pf  = file_processor.process(upload.filename, raw, upload.content_type)
                processed_files.append(pf)
                ev = {
                    "type": "attachment", "filename": upload.filename,
                    "file_type": "image" if pf.is_image else "document", "ok": pf.ok,
                }
                if not pf.ok:
                    ev["error"] = pf.error
                yield f"data: {json.dumps(ev)}\n\n"
                if pf.ok and pf.is_image:
                    image_b64s.append(pf.base64_data)

                # ── NEW: auto-index non-image uploads into session RAG ──────
                if pf.ok and not pf.is_image and effective_session_id and pf.text_content:
                    try:
                        from memory.session_rag import get_session_rag
                        rag = get_session_rag(effective_session_id, owner_id=owner_id)
                        chunks = await rag.index_file(upload.filename, pf.text_content)
                        if chunks:
                            yield f"data: {json.dumps({'type': 'rag_indexed', 'filename': upload.filename, 'chunks': chunks})}\n\n"
                    except Exception:
                        pass

            text_injection = file_processor.build_text_injection([f for f in processed_files if f.ok])
            full_message   = f"{text_injection}\n\n{message}" if text_injection else message

            log("agent", "system", f"Incoming request: agent={agent_id} model={model or 'default'}")
            profile = profile_manager.get(agent_id or "default", owner_id=owner_id) or profile_manager.get("default", owner_id=owner_id)
            resolved_context_mode = context_mode or getattr(profile, "default_context_mode", "v2")
            if resolved_context_mode not in {"v1", "v2", "v3"}:
                resolved_context_mode = "v2"
            if not effective_session_id:
                created = session_manager.create(
                    model=model or (profile.model if profile else FALLBACK_CHAT_MODEL),
                    system_prompt=system_prompt or (profile.system_prompt if profile else ""),
                    agent_id=agent_id or "default",
                    owner_id=owner_id,
                )
                session_manager.set_context_mode(created.id, resolved_context_mode)
                effective_session_id = created.id
            agent_instance = agent_manager.get_agent_instance(agent_id or "default", owner_id=owner_id)
            
            if embed_model:
                agent_instance.embed_model = embed_model
                if getattr(agent_instance, "graph", None) is not None:
                    agent_instance.graph.embedding_model = embed_model

            resolved_use_planner = use_planner if use_planner is not None else bool(getattr(profile, "default_use_planner", True))
            resolved_loop_mode = loop_mode or getattr(profile, "default_loop_mode", "auto")
            resolved_planner_model = planner_model or None
            resolved_context_model = context_model or None

            async for event in agent_instance.chat_stream(
                user_message   = full_message,
                session_id     = effective_session_id,
                agent_id       = agent_id,
                model          = model,
                system_prompt  = system_prompt,
                search_backend = search_backend,
                search_engine  = search_engine,
                force_memory   = force_memory,
                use_planner    = resolved_use_planner,
                loop_mode      = resolved_loop_mode,
                max_tool_calls = max_tool_calls,
                max_turns      = max_turns,
                planner_model  = resolved_planner_model,
                context_model  = resolved_context_model,
                forced_tools   = forced_tools,
                images         = image_b64s or None,
                owner_id       = owner_id,
                agent_revision = getattr(profile, "revision", None),
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except asyncio.CancelledError:
            log("agent", "stream", "Stream cancelled (shutdown/interrupt)", level="info")
            raise
        except Exception as e:
            log("agent", "stream", f"Generator error: {e}", level="error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── All remaining routes identical to v0.5.0 ─────────────────────────────────
# (sessions, memory, tool-results, agents endpoints — no changes needed)

@app.post("/sessions/{session_id}/clear_context")
async def clear_session_context(session_id: str, owner_id: str):
    try:
        owner_id = _require_owner_id(owner_id)
        if not session_manager.get(session_id, owner_id=owner_id):
            raise HTTPException(status_code=404, detail="Session not found")
        session_manager.update_context(session_id, "")
        return {"status": "ok", "message": "Context purged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/context-mode")
async def set_context_mode(session_id: str, payload: dict):
    owner_id = _require_owner_id(payload.get("owner_id"))
    mode = payload.get("mode", "v1")
    if mode not in ("v1", "v2", "v3"):
        raise HTTPException(400, "mode must be 'v1', 'v2', or 'v3'")

    session = session_manager.get(session_id, owner_id=owner_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session_manager.set_context_mode(session_id, mode)

    # Keep the warm cached agent instance aligned with the selected mode.
    agent = agent_manager.get_agent_instance(session.agent_id or "default", owner_id=owner_id)
    agent.set_context_engine(mode)
    return {"session_id": session_id, "context_mode": mode}

@app.get("/sessions")
async def list_sessions(agent_id: Optional[str] = None, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    return {"sessions": session_manager.list_sessions(agent_id=agent_id, owner_id=owner_id)}

@app.post("/sessions")
async def create_session(payload: Optional[dict] = Body(default=None)):
    payload = payload or {}
    owner_id = _require_owner_id(payload.get("owner_id"))
    agent_id = payload.get("agent_id") or "default"
    model = payload.get("model") or FALLBACK_CHAT_MODEL
    requested_mode = payload.get("context_mode") or "v1"
    context_mode = requested_mode if requested_mode in ("v1", "v2", "v3") else "v1"

    profile = profile_manager.get(agent_id, owner_id=owner_id) or profile_manager.get("default", owner_id=owner_id)
    system_prompt = profile.system_prompt if profile else ""
    if "context_mode" not in payload and profile:
        context_mode = getattr(profile, "default_context_mode", "v2")

    s = session_manager.create(
        model=model,
        system_prompt=system_prompt,
        agent_id=agent_id,
        owner_id=owner_id,
    )
    session_manager.set_context_mode(s.id, context_mode)

    return {
        "id": s.id,
        "agent_id": s.agent_id,
        "model": s.model,
        "context_mode": context_mode,
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "message_count": s.message_count,
        "title": s.title or "New Chat",
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    s = session_manager.get(session_id, owner_id=owner_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return {
        "id": s.id, "title": s.title or "New Chat", "model": s.model,
        "created_at": s.created_at, "updated_at": s.updated_at,
        "message_count": s.message_count, "compressed_context": s.compressed_context,
        "context_lines": _count_context_items(s.compressed_context),
        "context_mode": getattr(s, "context_mode", "v1"),
        "history": s.full_history,
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    if not session_manager.delete(session_id, owner_id=owner_id):
        raise HTTPException(404, "Session not found")
    # Also clean up RAG collection on session delete
    try:
        from memory.session_rag import cleanup_session_rag
        cleanup_session_rag(session_id, owner_id=owner_id)
    except Exception:
        pass
    return {"deleted": session_id}

@app.post("/sessions/{session_id}/stop")
async def stop_session_execution(session_id: str, owner_id: str):
    owner_id = _require_owner_id(owner_id)
    if not session_manager.get(session_id, owner_id=owner_id):
        raise HTTPException(404, "Session not found")
    session_manager.interrupt(session_id)
    return {"status": "interrupt_sent", "session_id": session_id}

@app.get("/sessions/{session_id}/context")
async def get_context(session_id: str, owner_id: str):
    owner_id = _require_owner_id(owner_id)
    s = session_manager.get(session_id, owner_id=owner_id)
    if not s:
        raise HTTPException(404, "Session not found")
    lines = _context_preview(s.compressed_context)
    return {"session_id": session_id, "lines": len(lines), "context": lines, "raw": s.compressed_context}

@app.get("/sessions/{session_id}/memory-state")
async def get_session_memory_state(session_id: str, owner_id: str):
    owner_id = _require_owner_id(owner_id)
    s = session_manager.get(session_id, owner_id=owner_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return build_session_memory_payload(
        session=s,
        owner_id=owner_id,
        context_preview=_context_preview,
    )

@app.get("/memory")
async def list_memories(limit: int = 100, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    from memory.semantic_graph import SemanticGraph
    graph = SemanticGraph()
    return {"memories": graph.list_all(limit=limit, owner_id=owner_id), "total": graph.count(owner_id=owner_id), "limit": limit}

@app.post("/memory/search")
async def search_memory(payload: dict):
    from memory.semantic_graph import SemanticGraph
    query = payload.get("query", "")
    top_k = payload.get("top_k", 5)
    owner_id = _require_owner_id(payload.get("owner_id"))
    if not query:
        raise HTTPException(400, "query is required")
    results = await SemanticGraph().traverse(query, top_k=top_k, owner_id=owner_id)
    return {"query": query, "results": results}

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    from memory.semantic_graph import SemanticGraph
    if not SemanticGraph().delete_by_id(memory_id, owner_id=owner_id):
        raise HTTPException(404, f"Memory {memory_id} not found")
    return {"deleted": memory_id}

@app.delete("/memory")
async def clear_all_memories(owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    from memory.semantic_graph import SemanticGraph
    graph  = SemanticGraph()
    before = graph.count(owner_id=owner_id)
    graph.clear(owner_id=owner_id)
    return {"cleared": before}

@app.get("/storage/status")
async def get_storage_status():
    return storage_admin.status()


@app.get("/storage/backups")
async def list_storage_backups(limit: int = 20):
    return storage_admin.list_backups(limit=limit)


@app.post("/storage/backup")
async def create_storage_backup(payload: Optional[StorageBackupPayload] = Body(default=None)):
    selection = StoreSelection.from_payload(payload.model_dump() if payload else None)
    return storage_admin.backup(selection, label=payload.backup_label if payload else "")


@app.post("/storage/reset")
async def reset_storage(payload: StorageActionPayload):
    selection = StoreSelection.from_payload(payload.model_dump())
    return storage_admin.reset(
        selection,
        backup_first=payload.backup_first,
        backup_label=payload.backup_label,
        preserve_default_agent=payload.preserve_default_agent,
    )

@app.get("/tool-results/{session_id}")
async def get_tool_results(session_id: str, limit: int = 50):
    from memory.tool_results_db import ToolResultsDB
    results = ToolResultsDB().get_by_session(session_id, limit=limit)
    return {"session_id": session_id, "results": results, "count": len(results)}

@app.get("/apps")
async def list_generated_apps(limit: int = 100):
    from memory.tool_results_db import ToolResultsDB
    apps = ToolResultsDB().get_all_apps(limit=limit)
    return {"apps": apps, "count": len(apps)}

@app.get("/apps/{session_id}")
async def get_session_apps(session_id: str):
    from memory.tool_results_db import ToolResultsDB
    apps = ToolResultsDB().get_apps_by_session(session_id)
    return {"session_id": session_id, "apps": apps, "count": len(apps)}

@app.get("/agents")
async def list_agents(owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    return {"agents": profile_manager.list_all(owner_id=owner_id)}

@app.post("/agents")
async def create_agent(profile: AgentProfile):
    profile.owner_id = _require_owner_id(profile.owner_id)
    return profile_manager.create(profile)

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    p = profile_manager.get(agent_id, owner_id=owner_id)
    if not p:
        raise HTTPException(404, "Agent not found")
    return p

@app.patch("/agents/{agent_id}")
async def update_agent(agent_id: str, payload: dict):
    owner_id = _require_owner_id(payload.get("owner_id"))
    p = profile_manager.get(agent_id, owner_id=owner_id)
    if not p:
        raise HTTPException(404, "Agent not found")
    allowed = {
        "name",
        "description",
        "model",
        "embed_model",
        "system_prompt",
        "tools",
        "avatar_url",
        "workspace_path",
        "bootstrap_files",
        "bootstrap_max_chars",
        "default_use_planner",
        "default_loop_mode",
        "default_context_mode",
    }
    for key, val in payload.items():
        if key in allowed:
            setattr(p, key, val)
    from datetime import datetime, timezone
    p.updated_at = datetime.now(timezone.utc).isoformat()
    profile_manager.create(p)
    agent_manager.invalidate_cache(agent_id, owner_id=owner_id)
    return p

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    success = profile_manager.delete(agent_id, owner_id=owner_id)
    if not success:
        raise HTTPException(400, "Could not delete agent")
    agent_manager.invalidate_cache(agent_id, owner_id=owner_id)
    return {"status": "ok"}

# ── MCP management endpoints ──────────────────────────────────────────────────

@app.get("/mcp/servers")
async def list_mcp_servers():
    """List connected MCP servers."""
    connected = mcp_manager.list_connected() if mcp_manager else []
    return {"connected": connected, "count": len(connected)}

@app.post("/mcp/connect")
async def connect_mcp_server(payload: dict):
    """Dynamically connect a new MCP server at runtime."""
    if not mcp_manager:
        raise HTTPException(503, "MCP manager not initialized")
    server_id = payload.get("id")
    command   = payload.get("command", "npx")
    args      = payload.get("args", [])
    env       = payload.get("env", {})
    if not server_id:
        raise HTTPException(400, "id is required")
    count = await mcp_manager.connect_server(server_id, command, args, env)
    return {"server_id": server_id, "tools_registered": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
