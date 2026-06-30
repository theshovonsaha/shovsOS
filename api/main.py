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
import shutil
import socket
from dataclasses import asdict
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
from typing import Any, Optional, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from orchestration.agent_profiles import ProfileManager, AgentProfile
from orchestration.workflow_templates import list_workflow_templates
from orchestration.orchestrator import AgenticOrchestrator
from llm.llm_adapter import OllamaAdapter
from engine.context_engine import ContextEngine
from memory.semantic_graph import SemanticGraph
from orchestration.session_manager import SessionManager
from llm.adapter_factory import create_adapter
from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools
from engine.file_processor import FileProcessor
from orchestration.agent_manager import AgentManager
from config.logger import get_logger, log
from api.log_routes import setup_log_routes
from api.voice_endpoint import setup_voice_routes
from api.rag_routes import make_rag_router            # ← NEW
from api.consumer_routes import make_consumer_router
from api.harness_lab import make_harness_lab_router
from api.workflow_lab import make_workflow_lab_router
from api.memory_inspector import build_session_memory_payload
from api.owner import require_owner_id
from services.storage_admin import StorageAdminService, StoreSelection
from services.consumer_store import ConsumerStoreService
from services.image_generation import generate_image
from plugins.shovs_meta_gateway import inject_gateway_dependencies, register_gateway_tools

from guardrails import GuardrailMiddleware
from guardrails.api_routes import make_guardrail_router
from orchestration.run_store import get_run_store
from config.trace_store import get_trace_store
from run_engine import RunEngine, RunEngineRequest
from run_engine.kernel_engine import KernelRunEngine

from config.config import cfg
from engine.fact_guard import is_grounded_fact_record
from llm.model_capabilities import capability_flags
FALLBACK_CHAT_MODEL = cfg.DEFAULT_MODEL
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "30/minute")


def _require_owner_id(owner_id: Optional[str]) -> str:
    return require_owner_id(owner_id)


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


def _coerce_agent_patch_value(key: str, value: Any) -> Any:
    if key in {"tools", "skills", "bootstrap_files"}:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
        return []
    if key in {"default_use_planner", "unified_model_mode"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if key == "bootstrap_max_chars":
        try:
            return int(value)
        except Exception:
            return 8000
    return value


def _python_module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


async def _docker_runtime_status() -> dict[str, Any]:
    if os.getenv("DOCKER_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {
            "configured": False,
            "available": False,
            "message": "Docker sandbox disabled by DOCKER_DISABLED=true",
        }
    if not shutil.which("docker"):
        return {
            "configured": False,
            "available": False,
            "message": "Docker CLI not found; Docker-backed tools are unavailable",
        }

    def _ping() -> tuple[bool, str]:
        try:
            import docker

            client = docker.from_env()
            client.ping()
            return True, "Docker daemon reachable"
        except Exception as exc:
            return False, f"Docker daemon not reachable ({type(exc).__name__})"

    available, message = await asyncio.to_thread(_ping)
    return {
        "configured": True,
        "available": available,
        "message": message,
    }


async def _local_service_status(name: str, base_url: str, path: str = "/") -> dict[str, Any]:
    raw_url = (base_url or "").strip()
    if not raw_url:
        return {
            "configured": False,
            "available": False,
            "base_url": "",
            "message": f"{name} URL is not configured",
        }

    def _probe() -> dict[str, Any]:
        import urllib.error
        import urllib.parse
        import urllib.request

        parsed = urllib.parse.urlparse(raw_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        is_port_open = _port_open(host, port)
        probe_url = raw_url.rstrip("/") + (path if path.startswith("/") else f"/{path}")
        try:
            request = urllib.request.Request(probe_url, headers={"User-Agent": "shovs-runtime-health/1.0"})
            with urllib.request.urlopen(request, timeout=0.75) as response:
                return {
                    "configured": True,
                    "available": 200 <= int(response.status) < 500,
                    "base_url": raw_url,
                    "port_open": is_port_open,
                    "message": f"{name} returned HTTP {response.status}",
                }
        except urllib.error.HTTPError as exc:
            return {
                "configured": True,
                "available": int(exc.code) < 500,
                "base_url": raw_url,
                "port_open": is_port_open,
                "message": f"{name} returned HTTP {exc.code}",
            }
        except Exception as exc:
            return {
                "configured": True,
                "available": False,
                "base_url": raw_url,
                "port_open": is_port_open,
                "message": f"{name} not reachable ({type(exc).__name__})",
            }

    return await asyncio.to_thread(_probe)


def _voice_runtime_status() -> dict[str, Any]:
    stt_backends: list[str] = []
    tts_backends: list[str] = []
    if os.getenv("DEEPGRAM_API_KEY"):
        stt_backends.append("deepgram")
        tts_backends.append("deepgram-aura")
    if os.getenv("GROQ_API_KEY"):
        stt_backends.append("groq-whisper")
    if _python_module_available("faster_whisper"):
        stt_backends.append("faster-whisper")
    if _python_module_available("edge_tts"):
        tts_backends.append("edge-tts")
    if _python_module_available("kokoro"):
        tts_backends.append("kokoro")
    return {
        "websocket_paths": ["/ws/voice", "/api/ws/voice"],
        "stt_available": bool(stt_backends),
        "tts_available": bool(tts_backends),
        "stt_backends": stt_backends,
        "tts_backends": tts_backends,
        "message": "configured" if stt_backends and tts_backends else "voice requires at least one STT backend and one TTS backend",
    }


def _session_runtime_state(session, *, owner_id: str) -> dict[str, Any]:
    store = get_run_store()
    latest_run = store.latest_for_session(session.id, owner_id=owner_id)
    latest_run_payload: dict[str, Any] | None = None
    latest_pass_payload: dict[str, Any] | None = None
    latest_checkpoint_payload: dict[str, Any] | None = None
    if latest_run:
        latest_run_payload = asdict(latest_run)
        passes = store.list_passes(latest_run.run_id)
        checkpoints = store.list_checkpoints(latest_run.run_id)
        if passes:
            latest_pass_payload = asdict(passes[-1])
        if checkpoints:
            latest_checkpoint_payload = asdict(checkpoints[-1])

    trace_store = get_trace_store()
    latest_ledger: dict[str, Any] | None = None
    for event in trace_store.list_events(
        limit=30,
        session_id=session.id,
        owner_id=owner_id,
        event_type="run_ledger",
    ):
        hydrated = trace_store.get_event(str(event.get("id") or "")) or event
        data = hydrated.get("data") if isinstance(hydrated.get("data"), dict) else {}
        if data:
            latest_ledger = {
                "event_id": hydrated.get("id"),
                "run_id": hydrated.get("run_id") or data.get("run_id"),
                "created_at": hydrated.get("iso_ts"),
                "reason": data.get("reason"),
                "ledger_mode": data.get("ledger_mode"),
                "phase": data.get("phase") or data.get("current_phase"),
                "objective": data.get("objective"),
                "summary": data.get("summary") or {},
                "next_required_action": data.get("next_required_action") or {},
                "completion_gate": data.get("completion_gate") or {},
                "missing_requirements": data.get("missing_requirements") or [],
                "locked_entities": data.get("locked_entities") or [],
                "policy_violations": data.get("policy_violations") or [],
            }
            break

    compressed_context = str(getattr(session, "compressed_context", "") or "")
    continuation_state = dict(getattr(session, "continuation_state", {}) or {})
    full_history = list(getattr(session, "full_history", []) or [])
    candidate_signals = list(getattr(session, "candidate_signals", []) or [])
    context_status = "ready"
    stale_reasons: list[str] = []
    if not compressed_context and len(full_history) >= 4:
        context_status = "needs_compression"
        stale_reasons.append("session has history but no compressed context")
    if continuation_state:
        context_status = "continuation_pending"
        stale_reasons.append("session has pending continuation state")
    if latest_ledger and latest_ledger.get("missing_requirements"):
        context_status = "evidence_or_state_missing"
        stale_reasons.append("latest ledger reports missing requirements")

    return {
        "context_status": context_status,
        "stale_reasons": stale_reasons,
        "db_paths": {
            "sessions": session_manager.db_path,
            "agents": profile_manager.db_path,
            "runs": store.db_path,
            "semantic_memory": semantic_graph.db_path,
        },
        "session": {
            "id": session.id,
            "agent_id": getattr(session, "agent_id", "default"),
            "message_count": int(getattr(session, "message_count", 0) or 0),
            "history_count": len(full_history),
            "sliding_window_count": len(getattr(session, "sliding_window", []) or []),
            "compressed_context_chars": len(compressed_context),
            "candidate_signal_count": len(candidate_signals),
            "continuation_pending": bool(continuation_state),
            "continuation_state": continuation_state,
        },
        "latest_run": latest_run_payload,
        "latest_pass": latest_pass_payload,
        "latest_checkpoint": latest_checkpoint_payload,
        "latest_ledger": latest_ledger,
    }

# ── Singletons ────────────────────────────────────────────────────────────────
adapter         = OllamaAdapter()
tool_registry   = ToolRegistry()
session_manager = SessionManager(max_sessions=200)
context_engine  = ContextEngine(adapter=adapter, compression_model=FALLBACK_CHAT_MODEL)
file_processor  = FileProcessor()
profile_manager = ProfileManager()
orchestrator    = AgenticOrchestrator(adapter=adapter)
semantic_graph  = SemanticGraph()

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
run_engine = RunEngine(
    adapter=adapter,
    sessions=session_manager,
    tool_registry=tool_registry,
    run_store=get_run_store(),
    trace_store=get_trace_store(),
    orchestrator=orchestrator,
    context_engine=context_engine,
    graph=semantic_graph,
)
# Kernel-driven runtime (the deterministic control plane). Same construction
# surface and the same `.stream(request)` signature as RunEngine, so it is a
# drop-in A/B path: selected when control_policy == "kernel". The model is a
# slot-filler (lock entities + synthesize = 2 LLM calls), not the orchestrator.
kernel_run_engine = KernelRunEngine(
    adapter=adapter,
    sessions=session_manager,
    tool_registry=tool_registry,
    run_store=get_run_store(),
    trace_store=get_trace_store(),
    orchestrator=orchestrator,
    context_engine=context_engine,
    graph=semantic_graph,
)
storage_admin = StorageAdminService(session_manager, profile_manager)
consumer_session_manager = SessionManager(max_sessions=200, db_path="consumer_sessions.db")
consumer_context_engine = ContextEngine(adapter=adapter, compression_model=FALLBACK_CHAT_MODEL)
consumer_run_engine = RunEngine(
    adapter=adapter,
    sessions=consumer_session_manager,
    tool_registry=tool_registry,
    run_store=get_run_store(),
    trace_store=get_trace_store(),
    orchestrator=orchestrator,
    context_engine=consumer_context_engine,
    graph=semantic_graph,
)
consumer_store = ConsumerStoreService(
    consumer_db_path="consumer.db",
    consumer_sessions_db_path="consumer_sessions.db",
)

# ── Register tools ────────────────────────────────────────────────────────────
inject_gateway_dependencies(tool_registry, semantic_graph)
register_gateway_tools(tool_registry)
register_all_tools(tool_registry, agent_manager=agent_manager)


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
            model=cfg.DEFAULT_MODEL,
            tools=[
                "source_collect",
                "source_contract", "source_select", "source_next_action",
                "web_fetch_batch", "source_coverage",
                "finance_snapshot", "alpha_vantage_movers", "alpha_vantage_quote", "alpha_vantage_overview", "alpha_vantage_news",
                "web_search", "web_fetch", "image_search", "image_generate", "rag_search", "query_memory", "store_memory",
            ],
            system_prompt=(
                "You are a meticulous research agent. Always verify claims across multiple sources. "
                "CRITICAL: Only call web_fetch with URLs returned by a prior web_search result. "
                "Never invent or guess URLs. Cite sources. Never fabricate data."
            ),
        ),
        AgentProfile(
            id="finance-analyst",
            name="Finance Analyst",
            description="Ticker and market-mover analyst using deterministic Alpha Vantage data before web expansion.",
            model=cfg.DEFAULT_MODEL,
            tools=[
                "finance_snapshot",
                "alpha_vantage_movers",
                "alpha_vantage_quote",
                "alpha_vantage_overview",
                "alpha_vantage_news",
                "source_collect",
                "source_contract", "source_select", "source_next_action",
                "web_fetch_batch", "source_coverage",
                "web_search", "web_fetch", "query_memory",
            ],
            system_prompt=(
                "You are a finance research analyst. Use Alpha Vantage tools first for quotes, movers, fundamentals, "
                "and news sentiment when a ticker or stock-mover task is present. Lock ticker symbols from deterministic "
                "data before web expansion. Keep reports structured, cite only URLs present in tool results, and avoid "
                "buy/sell advice unless the user explicitly asks for a risk-framed opinion."
            ),
            default_use_planner=True,
            default_loop_mode="managed",
            default_context_mode="v3",
            workflow_template="finance_analyst_v1",
            prompt_version="finance_snapshot_v1",
            risk_policy="finance_read_only",
            ledger_mode="shadow",
        ),
        AgentProfile(
            id="analyst",
            name="Data Analyst",
            model=cfg.DEFAULT_MODEL,
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
            tools=[
                "source_collect",
                "source_contract", "source_select", "source_next_action",
                "web_fetch_batch", "source_coverage",
                "web_search", "web_fetch", "image_search", "image_generate", "query_memory",
            ],
            system_prompt=(
                "You are a clear, calm assistant for the consumer product surface. "
                "Prefer plain text, avoid internal execution chatter, and only use tools when they materially improve accuracy or complete the task."
            ),
            default_use_planner=True,
            default_loop_mode="auto",
            default_context_mode="v2",
            bootstrap_files=["IDENTITY.md", "SOUL.md"],
        ),
        AgentProfile(
            id="shopping-advisor",
            name="Shopping Advisor",
            description="Verifies product pages, prices, and tradeoffs before recommending.",
            model=cfg.DEFAULT_MODEL,
            tools=[
                "shopping_advice", "source_collect", "source_contract", "source_select", "source_next_action",
                "web_fetch_batch", "source_coverage",
                "web_search", "web_fetch", "query_memory", "store_memory",
            ],
            system_prompt=(
                "You are a practical shopping advisor for normal consumers. Use shopping_advice for buying questions. "
                "When location matters, compare relevant nearby Canadian retailers such as Costco, Canadian Tire, "
                "Shoppers, Metro, Dollarama, Walmart, and Best Buy. Give short recommendations based on verified URLs, "
                "observed prices, ratings, and concrete tradeoffs. If a price, product page, or availability was not "
                "verified, say that plainly."
            ),
            default_use_planner=True,
            default_loop_mode="managed",
            default_context_mode="v3",
            workflow_template="shopping_advisor_v1",
            prompt_version="shopping_patch_v1",
            risk_policy="consumer_verified",
            ledger_mode="shadow",
        ),
    ]
    for p in standard:
        existing = profile_manager.get(p.id)
        if not existing:
            profile_manager.create(p)
            log("startup", "profiles", f"Seeded agent profile: {p.id}")
        elif existing.default_use_planner != p.default_use_planner:
            profile_manager.create(p)
            log("startup", "profiles", f"Migrated planner default for profile: {p.id}")


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


@app.middleware("http")
async def api_prefix_compatibility(request: Request, call_next):
    """Allow frontend/static deployments to call /api/* without a dev proxy.

    Vite strips /api during local development, but a production/static backend
    may receive /api/models or /api/chat/stream directly. Rewriting here keeps
    both deployment shapes equivalent without duplicating every route.
    """
    path = request.scope.get("path", "")
    if path == "/api":
        request.scope["path"] = "/"
    elif isinstance(path, str) and path.startswith("/api/"):
        request.scope["path"] = path[4:]
    return await call_next(request)


setup_log_routes(app)
setup_voice_routes(app, run_engine=run_engine, profile_manager=profile_manager)
app.include_router(make_guardrail_router(guardrail_middleware), prefix="/guardrails")
app.include_router(make_rag_router(), prefix="/rag")          # ← NEW
app.include_router(make_harness_lab_router())
app.include_router(
    make_workflow_lab_router(
        run_engine=run_engine,
        profile_manager=profile_manager,
        default_model=FALLBACK_CHAT_MODEL,
    )
)
app.include_router(
    make_consumer_router(
        profile_manager=profile_manager,
        sessions=consumer_session_manager,
        file_processor=file_processor,
        consumer_store=consumer_store,
        tool_registry=tool_registry,
        run_engine=consumer_run_engine,
        graph=semantic_graph,
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
    force_memory:  Optional[bool] = False  # compatibility shim; managed runtime ignores this knob
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


class ImageGenerationPayload(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    model: str = ""
    size: str = "1024x1024"
    quality: str = "auto"
    background: str = "auto"
    output_format: str = "png"


class MemorySearchPayload(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=50)
    owner_id: Optional[str] = None
    session_id: Optional[str] = None


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
        "nvidia",
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
        "nvidia": [],
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
    
    capabilities: dict[str, dict[str, Any]] = {}
    for provider, provider_models in grouped.items():
        for model_name in provider_models:
            full_name = f"{provider}:{model_name}"
            capabilities[full_name] = capability_flags(full_name)
    for embedding_model in set(embeddings):
        capabilities[embedding_model] = capability_flags(embedding_model)
    return {
        "models": grouped,
        "embeddings": list(set(embeddings)),
        "capabilities": capabilities,
        "vision_model": cfg.VISION_MODEL,
        "image_generation_model": cfg.IMAGE_GENERATION_MODEL,
    }

@app.get("/tools")
async def list_tools():
    return {"tools": tool_registry.list_tools()}


@app.get("/runtime/health")
async def runtime_health():
    tools = tool_registry.list_tools()
    tool_names = [str(tool.get("name") or "") for tool in tools]
    docker_status, searxng_status, llamacpp_status = await asyncio.gather(
        _docker_runtime_status(),
        _local_service_status("SearXNG", os.getenv("SEARXNG_URL", ""), "/"),
        _local_service_status(
            "llama.cpp",
            os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1"),
            "/models",
        ),
    )
    voice_status = _voice_runtime_status()
    return {
        "status": "ok",
        "runtime": "run_engine",
        "local_mode": os.getenv("DOCKER_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"},
        "tool_count": len(tools),
        "tools": {
            "web_search": "web_search" in tool_names,
            "web_fetch": "web_fetch" in tool_names,
            "image_generate": "image_generate" in tool_names,
            "image_search": "image_search" in tool_names,
            "finance_snapshot": "finance_snapshot" in tool_names,
            "alpha_vantage_movers": "alpha_vantage_movers" in tool_names,
        },
        "models": {
            "default_model": cfg.DEFAULT_MODEL,
            "vision_model": cfg.VISION_MODEL,
            "image_generation_model": cfg.IMAGE_GENERATION_MODEL,
        },
        "persistence": {
            "project_root": cfg.PROJECT_ROOT,
            "sessions_db": session_manager.db_path,
            "agents_db": profile_manager.db_path,
            "runs_db": get_run_store().db_path,
            "semantic_memory_db": semantic_graph.db_path,
            "chroma_db_path": cfg.CHROMA_DB_PATH,
        },
        "features": {
            "harness_lab": True,
            "workflow_lab": True,
            "api_prefix_compatibility": True,
            "sandbox_static": True,
            "voice_io": True,
            "finance_data": "finance_snapshot" in tool_names,
        },
        "optional_services": {
            "docker_sandbox": docker_status,
            "searxng": searxng_status,
            "llamacpp": llamacpp_status,
            "voice": voice_status,
            "alpha_vantage": {
                "available": bool(os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")),
                "configured": bool(os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")),
                "message": "configured" if (os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")) else "ALPHA_VANTAGE_API_KEY not set",
            },
        },
        "requirements": {
            "image_generation": "OPENAI_API_KEY" if not cfg.OPENAI_API_KEY else "configured",
            "vision": "VISION_MODEL" if not cfg.VISION_MODEL else "configured",
            "voice": "configured" if voice_status.get("stt_available") and voice_status.get("tts_available") else voice_status.get("message"),
            "bash_tool": "configured" if docker_status.get("available") else docker_status.get("message"),
            "local_search": "configured" if searxng_status.get("available") else searxng_status.get("message"),
            "llamacpp": "configured" if llamacpp_status.get("available") else llamacpp_status.get("message"),
            "alpha_vantage": "configured" if (os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")) else "ALPHA_VANTAGE_API_KEY not set",
        },
    }


@app.post("/images/generate")
async def generate_image_endpoint(payload: ImageGenerationPayload):
    try:
        return await generate_image(
            prompt=payload.prompt,
            model=payload.model,
            size=payload.size,
            quality=payload.quality,
            background=payload.background,
            output_format=payload.output_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


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
    force_memory:      Optional[bool]   = Form(False),  # compatibility shim
    use_planner:       Optional[bool]   = Form(True),
    loop_mode:         Optional[str]    = Form("auto"),  # compatibility shim
    context_mode:      Optional[str]    = Form(None),
    max_tool_calls:    Optional[int]    = Form(None),
    max_turns:         Optional[int]    = Form(None),
    planner_model:     Optional[str]    = Form(None),
    context_model:     Optional[str]    = Form(None),
    embed_model:       Optional[str]    = Form(None),
    forced_tools_json: Optional[str]    = Form(None),
    owner_id:          Optional[str]    = Form(None),
    reasoning_enabled: Optional[bool]   = Form(None),
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

            log(
                "agent",
                "system",
                f"Incoming request: agent={agent_id} model={model or 'default'}",
                owner_id=owner_id,
            )
            profile = profile_manager.get(agent_id or "default", owner_id=owner_id) or profile_manager.get("default", owner_id=owner_id)
            resolved_context_mode = context_mode or getattr(profile, "default_context_mode", "v3")
            if resolved_context_mode not in {"v1", "v2", "v3"}:
                resolved_context_mode = "v3"
            if not effective_session_id:
                created = session_manager.create(
                    model=model or (profile.model if profile else FALLBACK_CHAT_MODEL),
                    system_prompt=system_prompt or (profile.system_prompt if profile else ""),
                    agent_id=agent_id or "default",
                    owner_id=owner_id,
                )
                session_manager.set_context_mode(created.id, resolved_context_mode)
                effective_session_id = created.id
                try:
                    from plugins.hook_registry import hooks
                    hooks.emit_sync(
                        "session_started",
                        {"agent_id": created.agent_id, "model": created.model, "owner_id": owner_id},
                        session_id=created.id,
                    )
                except Exception:
                    pass
            else:
                existing_session = session_manager.get(effective_session_id, owner_id=owner_id)
                if existing_session:
                    session_manager.set_context_mode(effective_session_id, resolved_context_mode)

            resolved_use_planner = use_planner if use_planner is not None else bool(getattr(profile, "default_use_planner", True))
            # Unified mode: one model drives chat/planner/context. Per-slot
            # overrides from the client are dropped so the agent stays coherent.
            unified = bool(getattr(profile, "unified_model_mode", True))
            if unified:
                resolved_planner_model = None
                resolved_context_model = None
            else:
                resolved_planner_model = planner_model or None
                resolved_context_model = context_model or None
            # Embed model is profile-bound and immutable; ignore any client value.
            resolved_embed_model = getattr(profile, "embed_model", None) or "nomic-embed-text"

            run_request = RunEngineRequest(
                session_id=effective_session_id,
                owner_id=owner_id,
                agent_id=agent_id or "default",
                user_message=full_message,
                model=model or (profile.model if profile else FALLBACK_CHAT_MODEL),
                system_prompt=system_prompt or (profile.system_prompt if profile else ""),
                context_mode=resolved_context_mode,
                allowed_tools=tuple(getattr(profile, "tools", []) or []),
                use_planner=resolved_use_planner,
                max_tool_calls=max_tool_calls,
                max_turns=max_turns,
                planner_model=resolved_planner_model,
                context_model=resolved_context_model,
                embed_model=resolved_embed_model,
                images=image_b64s or None,
                search_backend=search_backend,
                search_engine=search_engine,
                agent_revision=getattr(profile, "revision", None),
                workflow_template=getattr(profile, "workflow_template", "general_operator_v1"),
                prompt_version=getattr(profile, "prompt_version", "role_contracts_v1"),
                risk_policy=getattr(profile, "risk_policy", "standard"),
                ledger_mode=getattr(profile, "ledger_mode", "shadow"),
                control_policy=loop_mode or getattr(profile, "default_loop_mode", "auto"),
                forced_tools=tuple(str(item) for item in forced_tools if isinstance(item, str)),
                workspace_path=getattr(profile, "workspace_path", None),
                reasoning_enabled=reasoning_enabled,
                memory_commit_mode=os.getenv("SHOVSOS_MEMORY_COMMIT_MODE", "async"),
            )
            _engine = kernel_run_engine if run_request.control_policy == "kernel" else run_engine
            async for event in _engine.stream(run_request):
                yield f"data: {json.dumps(event)}\n\n"
            return

        except asyncio.CancelledError:
            log(
                "agent",
                "stream",
                "Stream cancelled (shutdown/interrupt)",
                level="info",
                owner_id=owner_id,
            )
            raise
        except Exception as e:
            log(
                "agent",
                "stream",
                f"Generator error: {e}",
                level="error",
                owner_id=owner_id,
            )
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
    mode = payload.get("mode", "v3")
    if mode not in ("v1", "v2", "v3"):
        raise HTTPException(400, "mode must be 'v1', 'v2', or 'v3'")

    session = session_manager.get(session_id, owner_id=owner_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session_manager.set_context_mode(session_id, mode)

    # Managed runtime reads context_mode from session state each turn.
    # Native compatibility instances may still support eager engine switching.
    agent = agent_manager.get_agent_instance(session.agent_id or "default", owner_id=owner_id)
    if hasattr(agent, "set_context_engine"):
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
    requested_mode = payload.get("context_mode") or "v3"
    context_mode = requested_mode if requested_mode in ("v1", "v2", "v3") else "v3"

    profile = profile_manager.get(agent_id, owner_id=owner_id) or profile_manager.get("default", owner_id=owner_id)
    model = payload.get("model") or (profile.model if profile else FALLBACK_CHAT_MODEL)
    system_prompt = profile.system_prompt if profile else ""
    if "context_mode" not in payload and profile:
        context_mode = getattr(profile, "default_context_mode", "v3")

    s = session_manager.create(
        model=model,
        system_prompt=system_prompt,
        agent_id=agent_id,
        owner_id=owner_id,
    )
    session_manager.set_context_mode(s.id, context_mode)
    try:
        from plugins.hook_registry import hooks
        hooks.emit_sync(
            "session_started",
            {"agent_id": s.agent_id, "model": s.model, "owner_id": owner_id},
            session_id=s.id,
        )
    except Exception:
        pass

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
        "agent_id": s.agent_id,
        "created_at": s.created_at, "updated_at": s.updated_at,
        "message_count": s.message_count, "compressed_context": s.compressed_context,
        "context_lines": _count_context_items(s.compressed_context),
        "context_mode": getattr(s, "context_mode", "v1"),
        "history": s.full_history,
    }


@app.patch("/sessions/{session_id}/messages/{message_index}")
async def edit_session_message(session_id: str, message_index: int, payload: dict = Body(...)):
    owner_id = _require_owner_id(payload.get("owner_id"))
    new_content = str(payload.get("content") or "")
    if not new_content.strip():
        raise HTTPException(400, "content is required")

    # truncate_downstream=True by default — removes messages after the edit
    # so ghost facts cannot re-teach superseded content on the next turn.
    # Pass truncate_downstream=false explicitly to preserve downstream messages
    # (use only for typo corrections in the most recent message).
    truncate_downstream = payload.get("truncate_downstream", True)

    session = session_manager.get(session_id, owner_id=owner_id)
    if not session:
        raise HTTPException(404, "Session not found")

    history_before = len(session.full_history)
    message = session_manager.edit_message(
        session_id,
        content=new_content,
        message_index=message_index,
        owner_id=owner_id,
        truncate_downstream=bool(truncate_downstream),
    )
    session_after = session_manager.get(session_id, owner_id=owner_id)
    history_after = len(session_after.full_history) if session_after else history_before
    messages_truncated = max(0, history_before - history_after)
    if message is None:
        raise HTTPException(404, "Message not found")

    try:
        semantic_graph.clear_session_facts(session_id, owner_id=owner_id)
    except Exception:
        pass

    try:
        from memory.vector_engine import VectorEngine
        from memory.bm25_engine import BM25Engine

        await VectorEngine(session_id, agent_id=session.agent_id or "default", owner_id=owner_id).clear()
        BM25Engine(session_id=session_id, agent_id=session.agent_id or "default", owner_id=owner_id).clear()
    except Exception:
        pass

    try:
        from memory.session_rag import clear_session_rag

        await clear_session_rag(session_id, owner_id=owner_id)
    except Exception:
        pass

    return {
        "session_id": session_id,
        "message": message,
        "cascade": {
            "messages_truncated": messages_truncated,
            "history_length_after": history_after,
            "truncate_downstream": bool(truncate_downstream),
        },
        "derived_state_reset": {
            "compressed_context": True,
            "candidate_signals": True,
            "deterministic_facts": True,
            "vector_memory": True,
            "session_rag": True,
        },
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
    payload = build_session_memory_payload(
        session=s,
        owner_id=owner_id,
        context_preview=_context_preview,
        graph=semantic_graph,
    )
    payload["runtime_state"] = _session_runtime_state(s, owner_id=owner_id)
    return payload

@app.get("/memory")
async def list_memories(limit: int = 100, owner_id: Optional[str] = None):
    owner_id = _require_owner_id(owner_id)
    from memory.semantic_graph import SemanticGraph
    graph = SemanticGraph()
    return {"memories": graph.list_all(limit=limit, owner_id=owner_id), "total": graph.count(owner_id=owner_id), "limit": limit}

@app.post("/memory/search")
async def search_memory(payload: MemorySearchPayload):
    owner_id = _require_owner_id(payload.owner_id)
    return await run_engine._context_governor.search_memory(
        payload.query,
        owner_id=owner_id,
        session_id=payload.session_id,
        top_k=payload.top_k,
    )


@app.post("/memory/benchmark/run")
async def run_memory_benchmark(payload: Optional[dict] = Body(default=None)):
    from memory.benchmark_harness import run_memory_benchmark as run_benchmark
    from memory.benchmark_store import save_latest

    owner_id = _require_owner_id((payload or {}).get("owner_id"))
    result = await run_benchmark(owner_id)
    save_latest(owner_id, result)
    return result


@app.get("/memory/benchmark/latest")
async def get_memory_benchmark_latest(owner_id: Optional[str] = None):
    from memory.benchmark_store import load_latest

    owner_id = _require_owner_id(owner_id)
    result = load_latest(owner_id)
    return {"result": result}

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

@app.get("/agent-templates")
async def list_agent_templates():
    return {"templates": list_workflow_templates()}

@app.post("/agents")
async def create_agent(profile: AgentProfile):
    profile.owner_id = _require_owner_id(profile.owner_id)
    created = profile_manager.create(profile)
    agent_manager.invalidate_cache(created.id, owner_id=created.owner_id)
    return created

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
    # NOTE: embed_model is intentionally NOT in this set — embedders are
    # immutable after agent creation because the vector store and existing
    # memory rows are bound to the embedder used at creation time.
    allowed = {
        "name",
        "description",
        "model",
        "system_prompt",
        "tools",
        "skills",
        "avatar_url",
        "workspace_path",
        "bootstrap_files",
        "bootstrap_max_chars",
        "default_use_planner",
        "default_loop_mode",
        "default_context_mode",
        "unified_model_mode",
        "workflow_template",
        "prompt_version",
        "risk_policy",
        "ledger_mode",
    }
    if "embed_model" in payload and payload["embed_model"] != p.embed_model:
        raise HTTPException(
            400,
            "embed_model is immutable after agent creation. Create a new agent to use a different embedder.",
        )
    for key, val in payload.items():
        if key in allowed:
            setattr(p, key, _coerce_agent_patch_value(key, val))
    from datetime import datetime, timezone
    p.updated_at = datetime.now(timezone.utc).isoformat()
    updated = profile_manager.create(p)
    agent_manager.invalidate_cache(agent_id, owner_id=owner_id)
    return updated

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
