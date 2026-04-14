from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orchestration.agent_profiles import ProfileManager
from orchestration.session_manager import SessionManager
from engine.file_processor import FileProcessor
from llm.adapter_factory import create_adapter
from plugins.tool_registry import ToolRegistry
from api.memory_inspector import build_session_memory_payload
from api.owner import require_owner_id
from run_engine.engine import RunEngine
from run_engine.types import RunEngineRequest
from services.consumer_store import ConsumerStoreService, ConsumerStoreSelection

CONSUMER_AGENT_ID = "consumer"


class ConsumerOptionsPayload(BaseModel):
    model: str


class ConsumerStoreSelectionPayload(BaseModel):
    consumer_db: bool = True
    consumer_sessions: bool = True


class ConsumerStoreActionPayload(ConsumerStoreSelectionPayload):
    backup_first: bool = True
    backup_label: str = ""


def make_consumer_router(
    *,
    profile_manager: ProfileManager,
    sessions: SessionManager,
    file_processor: FileProcessor,
    consumer_store: ConsumerStoreService,
    tool_registry: ToolRegistry,
    run_engine: RunEngine,
) -> APIRouter:
    router = APIRouter(prefix="/consumer", tags=["consumer"])
    _require_owner_id = require_owner_id

    @router.get("/health")
    async def consumer_health():
        ollama = create_adapter("ollama")
        groq = create_adapter("groq")
        openai = create_adapter("openai")
        gemini = create_adapter("gemini")
        anthropic = create_adapter("anthropic")
        return {
            "status": "ok",
            "providers": {
                "ollama": await ollama.health(),
                "groq": await groq.health(),
                "openai": await openai.health(),
                "gemini": await gemini.health(),
                "anthropic": await anthropic.health(),
            },
        }

    @router.get("/models")
    async def consumer_models():
        grouped = {"ollama": [], "groq": [], "openai": [], "gemini": [], "anthropic": []}
        for provider in grouped:
            try:
                adapter = create_adapter(provider)
                models = await adapter.list_models()
                if models:
                    grouped[provider] = models
            except Exception:
                pass
        return {"models": grouped}

    @router.get("/options")
    async def get_consumer_options():
        return consumer_store.get_options()

    @router.post("/options")
    async def set_consumer_options(payload: ConsumerOptionsPayload):
        return consumer_store.set_options(payload.model)

    @router.get("/storage/status")
    async def consumer_storage_status():
        return consumer_store.status()

    @router.get("/storage/backups")
    async def consumer_storage_backups(limit: int = 20):
        return consumer_store.list_backups(limit=limit)

    @router.post("/storage/backup")
    async def consumer_storage_backup(payload: Optional[ConsumerStoreSelectionPayload] = Body(default=None)):
        selection = ConsumerStoreSelection.from_payload(payload.model_dump() if payload else None)
        return consumer_store.backup(selection)

    @router.post("/storage/reset")
    async def consumer_storage_reset(payload: ConsumerStoreActionPayload):
        selection = ConsumerStoreSelection.from_payload(payload.model_dump())
        result = consumer_store.reset(
            selection,
            backup_first=payload.backup_first,
            backup_label=payload.backup_label,
        )
        if selection.consumer_sessions:
            sessions.reset_all()
        return result

    @router.post("/session")
    async def consumer_create_session(payload: Optional[dict] = Body(default=None)):
        payload = payload or {}
        owner_id = _require_owner_id(payload.get("owner_id"))
        options = consumer_store.get_options()
        profile = profile_manager.get(CONSUMER_AGENT_ID, owner_id=owner_id) or profile_manager.get(
            "default",
            owner_id=owner_id,
        )
        model = payload.get("model") or options.get("model") or getattr(profile, "model", None)
        context_mode = payload.get("context_mode") or getattr(profile, "default_context_mode", "v2")
        if context_mode not in {"v1", "v2", "v3"}:
            context_mode = "v2"
        s = sessions.create(
            model=model,
            system_prompt=getattr(profile, "system_prompt", "") or "",
            agent_id=CONSUMER_AGENT_ID,
            owner_id=owner_id,
        )
        sessions.set_context_mode(s.id, context_mode)
        return {
            "id": s.id,
            "model": s.model,
            "agent_id": s.agent_id,
            "created_at": s.created_at,
            "context_mode": context_mode,
        }

    @router.post("/chat/stream")
    async def consumer_chat_stream(
        message: str = Form(...),
        session_id: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        owner_id: Optional[str] = Form(None),
        files: list[UploadFile] = File(default=[]),
    ):
        owner_id = _require_owner_id(owner_id)
        async def generate():
            last_run_id: Optional[str] = None
            options = consumer_store.get_options()
            profile = profile_manager.get(CONSUMER_AGENT_ID, owner_id=owner_id) or profile_manager.get(
                "default",
                owner_id=owner_id,
            )
            active_model = model or options.get("model") or getattr(profile, "model", None) or "llama3.2"
            image_b64s: list[str] = []
            text_injection = ""
            if files:
                for upload in files:
                    raw = await upload.read()
                    processed = file_processor.process(upload.filename, raw, upload.content_type)
                    if processed.ok and processed.is_image and processed.base64_data:
                        image_b64s.append(processed.base64_data)
                    if processed.ok and not processed.is_image and processed.text_content:
                        text_injection += f"\n\n[ATTACHMENT: {upload.filename}]\n{processed.text_content}"
                    yield f"data: {json.dumps({'type': 'activity_short', 'text': f'Processed attachment: {upload.filename}'})}\n\n"

            full_message = f"{text_injection}\n\n{message}" if text_injection else message
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'thinking'})}\n\n"

            tool_events = 0
            sid = session_id
            resolved_context_mode = getattr(profile, "default_context_mode", "v2")
            if resolved_context_mode not in {"v1", "v2", "v3"}:
                resolved_context_mode = "v2"
            if not sid:
                created = sessions.create(
                    model=active_model,
                    system_prompt=getattr(profile, "system_prompt", "") or "",
                    agent_id=CONSUMER_AGENT_ID,
                    owner_id=owner_id,
                )
                sid = created.id
                sessions.set_context_mode(sid, resolved_context_mode)
            else:
                existing = sessions.get(sid, owner_id=owner_id)
                if existing and getattr(existing, "context_mode", "") != resolved_context_mode:
                    sessions.set_context_mode(sid, resolved_context_mode)

            run_request = RunEngineRequest(
                session_id=sid,
                owner_id=owner_id,
                agent_id=CONSUMER_AGENT_ID,
                user_message=full_message,
                model=active_model,
                system_prompt=getattr(profile, "system_prompt", "") or "",
                context_mode=resolved_context_mode,
                allowed_tools=tuple(getattr(profile, "tools", []) or []),
                use_planner=bool(getattr(profile, "default_use_planner", True)),
                embed_model=getattr(profile, "embed_model", None),
                images=image_b64s or None,
                agent_revision=getattr(profile, "revision", None),
                workspace_path=getattr(profile, "workspace_path", None),
            )
            async for event in run_engine.stream(run_request):
                event_type = event.get("type")
                if event_type == "session":
                    sid = event.get("session_id")
                    last_run_id = event.get("run_id")
                    yield f"data: {json.dumps({'type': 'session', 'session_id': sid, 'run_id': event.get('run_id')})}\n\n"
                elif event_type == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': event.get('content', '')})}\n\n"
                elif event_type == "tool_call":
                    tool_events += 1
                    if tool_events == 1:
                        yield f"data: {json.dumps({'type': 'phase', 'phase': 'working'})}\n\n"
                    tool_name = event.get("tool_name", "tool")
                    friendly_tool = str(tool_name).replace("_", " ")
                    yield f"data: {json.dumps({'type': 'activity_short', 'text': f'Using {friendly_tool}'})}\n\n"
                elif event_type == "tool_result":
                    tool_name = event.get("tool_name", "tool")
                    ok = event.get("success", False)
                    detail = event.get("content", "")
                    status = "completed" if ok else "failed"
                    yield f"data: {json.dumps({'type': 'activity_detail', 'text': f'{tool_name} {status}', 'detail': detail[:400]})}\n\n"
                elif event_type == "compressing":
                    yield f"data: {json.dumps({'type': 'phase', 'phase': 'finalizing'})}\n\n"
                elif event_type == "plan":
                    yield f"data: {json.dumps({'type': 'plan', 'strategy': event.get('strategy', ''), 'tools': event.get('tools', []), 'confidence': event.get('confidence', 0)})}\n\n"
                elif event_type == "verification_warning":
                    yield f"data: {json.dumps({'type': 'verification_warning', 'issues': event.get('issues', []), 'confidence': event.get('confidence', 0)})}\n\n"
                elif event_type == "conversation_tension":
                    yield f"data: {json.dumps({'type': 'tension', 'summary': event.get('summary', ''), 'challenge_level': event.get('challenge_level', '')})}\n\n"
                elif event_type == "logical_stall_alert":
                    yield f"data: {json.dumps({'type': 'activity_detail', 'text': 'logical stall alert', 'detail': event.get('message', '')[:400]})}\n\n"
                elif event_type == "done":
                    sid = event.get("session_id", sid)
                    last_run_id = event.get("run_id", last_run_id)
                    yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'run_id': last_run_id})}\n\n"
                    return
                elif event_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': event.get('message', 'Unknown error')})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'run_id': last_run_id})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @router.get("/sessions/{session_id}/memory-state")
    async def consumer_session_memory_state(session_id: str, owner_id: str):
        owner_id = _require_owner_id(owner_id)
        session = sessions.get(session_id, owner_id=owner_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return build_session_memory_payload(
            session=session,
            owner_id=owner_id,
            context_preview=lambda raw: [line for line in (raw or "").splitlines() if line.strip()],
        )

    @router.get("/sessions")
    async def consumer_list_sessions(owner_id: str):
        owner_id = _require_owner_id(owner_id)
        items = sessions.list_sessions(agent_id=CONSUMER_AGENT_ID, owner_id=owner_id)
        return {"sessions": items}

    @router.get("/sessions/{session_id}/messages")
    async def consumer_session_messages(session_id: str, owner_id: str):
        owner_id = _require_owner_id(owner_id)
        session = sessions.get(session_id, owner_id=owner_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = []
        for m in session.full_history:
            role = m.get("role", "")
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": m.get("content", "")})
        return {"messages": msgs, "session_id": session_id}

    @router.get("/tools")
    async def consumer_tools():
        return {"tools": tool_registry.list_tools()}

    return router
