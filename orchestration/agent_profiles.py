"""
Agent Profiles Persistence
--------------------------
Manages the storage and lifecycle of Agent configurations.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field

DB_PATH = "agents.db"
DEFAULT_RUNTIME_KIND = "managed"
RUNTIME_KIND_ALIASES = {
    "managed": "managed",
    "run_engine": "managed",
    "native": "native",
    "legacy": "native",
    "agent_core": "native",
    "agentcore": "native",
}
DEFAULT_AGENT_TOOLS = [
    "web_search",
    "web_fetch",
    "image_search",
    "weather_fetch",
    "places_search",
    "places_map",
    "store_memory",
    "query_memory",
    "delegate_to_agent",
]

PLATINUM_SYSTEM_PROMPT = (
    "You are the 'Shovs' Platinum AI. Your mission: Production-grade intelligence with a Luxury-Dark aesthetic.\n\n"
    "--- PLATINUM DIRECTIVES ---\n"
    "1. TRUE BLACK: All HTML/SVG output must use background: #000000. No exceptions.\n"
    "2. ACCENTS: Use electric cyan (#00d1ff) and deep violet (#8b5cf6) for highlights and glow.\n"
    "3. SPA ARCHITECTURE: Every app is a Single-Page Application (SPA). Use vanilla JS to manage sections/tabs dynamically.\n"
    "4. TYPOGRAPHY: Use 'Inter' or 'Roboto Mono' via Google Fonts. Pair with dramatic scale.\n"
    "5. NO SLOP: No placeholders. Use real data, Lucide icons (CDN), and high-quality assets.\n\n"
    "--- BEHAVIORAL STANDARDS ---\n"
    "- CONTEXT: Prioritize 'Historical Context' and 'Session Memory' for persona consistency.\n"
    "- MEMORY: Use `query_memory` to rediscover user preferences and past interactions.\n"
    "- TOOLS: Output JSON ONLY: {\"tool\": \"...\", \"arguments\": {...}}.\n"
    "- DELEGATION: Use `delegate_to_agent` when a task needs a specialized agent.\n"
    "- ACCURACY: Never fabricate tool results. Report limitations honestly.\n"
)

GENERAL_SYSTEM_PROMPT = (
    "You are Shovs, a capable general-purpose AI operating system assistant.\n\n"
    "Core behavior:\n"
    "- Default to clear, direct conversational help.\n"
    "- Use tools only when they materially improve accuracy or complete a concrete task.\n"
    "- If a tool is listed for this agent, treat it as available in the current runtime. Do not claim you cannot browse, search, fetch, or access a tool when that tool is available.\n"
    "- For current events, current prices, market moves, news, or anything phrased as latest/current/today, prefer web_search before giving a limitation-only answer.\n"
    "- Do not emit HTML, SVG, app fragments, faux system dashboards, or code blocks unless the user explicitly asks for a visual artifact, file, app, code, or markup.\n"
    "- For greetings, casual chat, and clarification turns, respond in normal plain text.\n"
    "- Do not create files, scripts, or reports unless the user explicitly asks for a file or artifact.\n"
    "- Never fabricate tool results. Be explicit about uncertainty and limitations.\n"
)

APP_KEYWORDS = ("app", "html", "ui", "design", "dashboard", "visual", "frontend", "website")
APP_TOOLS = {"generate_app"}


def _is_visual_profile(profile: "AgentProfile") -> bool:
    haystacks = [
        profile.name or "",
        profile.description or "",
    ]
    text = " ".join(haystacks).lower()
    if any(keyword in text for keyword in APP_KEYWORDS):
        return True
    return bool(APP_TOOLS & set(profile.tools or []))

class AgentProfile(BaseModel):
    id:            str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner_id:      Optional[str] = None
    name:          str
    description:   str = ""
    runtime_kind:  str = DEFAULT_RUNTIME_KIND
    model:         str = "llama3.2"  # Fallback only
    embed_model:   str = "nomic-embed-text"
    system_prompt: str = GENERAL_SYSTEM_PROMPT
    tools:         List[str] = Field(default_factory=lambda: ["web_search", "web_fetch"])
    avatar_url:    Optional[str] = None
    workspace_path: Optional[str] = None
    bootstrap_files: List[str] = Field(default_factory=lambda: ["AGENTS.md", "IDENTITY.md", "SOUL.md", "TOOLS.md"])
    bootstrap_max_chars: int = 8000
    default_use_planner: bool = True
    default_loop_mode: str = "auto"
    default_context_mode: str = "v2"
    revision:      int = 1
    created_at:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ProfileManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._ensure_default_agent()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    name TEXT,
                    description TEXT,
                    runtime_kind TEXT DEFAULT 'managed',
                    model TEXT,
                    embed_model TEXT,
                    system_prompt TEXT,
                    tools TEXT,
                    avatar_url TEXT,
                    workspace_path TEXT,
                    bootstrap_files TEXT,
                    bootstrap_max_chars INTEGER DEFAULT 8000,
                    default_use_planner INTEGER DEFAULT 1,
                    default_loop_mode TEXT DEFAULT 'auto',
                    default_context_mode TEXT DEFAULT 'v2',
                    revision INTEGER DEFAULT 1,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            # Migration
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN embed_model TEXT DEFAULT 'nomic-embed-text'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN owner_id TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN revision INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN runtime_kind TEXT DEFAULT 'managed'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN workspace_path TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN bootstrap_files TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN bootstrap_max_chars INTEGER DEFAULT 8000")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN default_use_planner INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN default_loop_mode TEXT DEFAULT 'auto'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE agent_profiles ADD COLUMN default_context_mode TEXT DEFAULT 'v2'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    "UPDATE agent_profiles SET runtime_kind = ? WHERE runtime_kind IS NULL OR TRIM(runtime_kind) = ''",
                    (DEFAULT_RUNTIME_KIND,),
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def _ensure_default_agent(self):
        """Ensure a sane default agent exists without overwriting unrelated agent personalities."""
        existing_default = self.get("default")
        if not existing_default:
            self.create(AgentProfile(
                id="default",
                name="shovs Assistant",
                description="The standard general-purpose agent.",
                system_prompt=GENERAL_SYSTEM_PROMPT,
                tools=DEFAULT_AGENT_TOOLS,
            ))
        else:
            # Keep existing custom order but guarantee required core tools are present.
            merged_tools = list(existing_default.tools)
            for tool in DEFAULT_AGENT_TOOLS:
                if tool not in merged_tools:
                    merged_tools.append(tool)
            tools_changed = merged_tools != existing_default.tools
            prompt_needs_reset = existing_default.system_prompt in {
                "",
                "You are a specialized AI assistant.",
                PLATINUM_SYSTEM_PROMPT,
            }
            if tools_changed:
                existing_default.tools = merged_tools
            if prompt_needs_reset:
                existing_default.system_prompt = GENERAL_SYSTEM_PROMPT
                if existing_default.name == "shovs Platinum Assistant":
                    existing_default.name = "shovs Assistant"
            runtime_kind_changed = existing_default.runtime_kind != DEFAULT_RUNTIME_KIND
            if runtime_kind_changed:
                existing_default.runtime_kind = DEFAULT_RUNTIME_KIND
            if (
                tools_changed
                or prompt_needs_reset
                or runtime_kind_changed
            ):
                self.create(existing_default)

        # One-time cleanup for profiles that were previously force-upgraded to the app-builder prompt.
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_profiles")
            for row in cursor.fetchall():
                p = self._row_to_profile(row)
                if p.id == "default":
                    continue
                if p.system_prompt == "You are a specialized AI assistant.":
                    p.system_prompt = GENERAL_SYSTEM_PROMPT
                    self.create(p)
                    print(f"[ProfileManager] Upgraded legacy agent '{p.name}' ({p.id}) to the general assistant prompt.")
                    continue
                if p.system_prompt == PLATINUM_SYSTEM_PROMPT and not _is_visual_profile(p):
                    p.system_prompt = GENERAL_SYSTEM_PROMPT
                    self.create(p)
                    print(f"[ProfileManager] Reset over-broad visual prompt for '{p.name}' ({p.id}).")

    def create(self, p: AgentProfile) -> AgentProfile:
        p = self._sanitize_profile(p)
        existing = self.get(p.id, owner_id=p.owner_id)
        if existing:
            changed = any(
                getattr(existing, field) != getattr(p, field)
                for field in (
                    "owner_id",
                    "name",
                    "description",
                    "runtime_kind",
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
                )
            )
            if changed and p.revision <= existing.revision:
                p.revision = existing.revision + 1
            elif not changed:
                p.revision = existing.revision
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO agent_profiles
                (id, owner_id, name, description, runtime_kind, model, embed_model, system_prompt, tools, avatar_url, workspace_path, bootstrap_files, bootstrap_max_chars, default_use_planner, default_loop_mode, default_context_mode, revision, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                p.id, p.owner_id, p.name, p.description, p.runtime_kind, p.model, p.embed_model, p.system_prompt,
                json.dumps(p.tools), p.avatar_url, p.workspace_path, json.dumps(p.bootstrap_files), p.bootstrap_max_chars,
                1 if p.default_use_planner else 0, p.default_loop_mode, p.default_context_mode,
                p.revision, p.created_at, p.updated_at
            ))
            conn.commit()
        return p

    def _sanitize_profile(self, p: AgentProfile) -> AgentProfile:
        raw_runtime_kind = str(getattr(p, "runtime_kind", "") or "").strip().lower()
        runtime_kind = RUNTIME_KIND_ALIASES.get(raw_runtime_kind, DEFAULT_RUNTIME_KIND)

        tools = []
        seen_tools = set()
        for tool in p.tools or []:
            normalized = str(tool or "").strip()
            if not normalized or normalized in seen_tools:
                continue
            seen_tools.add(normalized)
            tools.append(normalized)
        if not tools:
            tools = ["web_search", "web_fetch", "query_memory", "store_memory"]

        workspace_path = str(p.workspace_path or "").strip() or None
        if workspace_path:
            try:
                workspace_path = str(Path(workspace_path).expanduser())
            except Exception:
                workspace_path = str(p.workspace_path).strip()

        bootstrap_files: list[str] = []
        seen_files = set()
        for name in p.bootstrap_files or []:
            normalized = Path(str(name or "").strip()).name
            if not normalized or normalized in seen_files:
                continue
            seen_files.add(normalized)
            bootstrap_files.append(normalized)
        if not bootstrap_files:
            bootstrap_files = ["AGENTS.md", "IDENTITY.md", "SOUL.md", "TOOLS.md"]
        bootstrap_files = bootstrap_files[:8]

        bootstrap_max_chars = max(1000, min(20000, int(p.bootstrap_max_chars or 8000)))
        default_loop_mode = str(p.default_loop_mode or "auto").strip().lower()
        if default_loop_mode not in {"auto", "single", "managed"}:
            default_loop_mode = "auto"
        default_context_mode = str(p.default_context_mode or "v2").strip().lower()
        if default_context_mode not in {"v1", "v2", "v3"}:
            default_context_mode = "v2"

        raw_system_prompt = p.system_prompt if isinstance(p.system_prompt, str) else ""
        system_prompt = raw_system_prompt if raw_system_prompt.strip() else GENERAL_SYSTEM_PROMPT

        return p.model_copy(update={
            "name": str(p.name or "").strip(),
            "description": str(p.description or "").strip(),
            "runtime_kind": runtime_kind,
            "system_prompt": system_prompt,
            "tools": tools,
            "workspace_path": workspace_path,
            "bootstrap_files": bootstrap_files,
            "bootstrap_max_chars": bootstrap_max_chars,
            "default_use_planner": bool(p.default_use_planner),
            "default_loop_mode": default_loop_mode,
            "default_context_mode": default_context_mode,
        })

    def get(self, profile_id: str, owner_id: Optional[str] = None) -> Optional[AgentProfile]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is not None:
                cursor.execute(
                    "SELECT * FROM agent_profiles WHERE id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (profile_id, owner_id),
                )
                r = cursor.fetchone()
                if r:
                    return self._row_to_profile(r)
                cursor.execute(
                    "SELECT * FROM agent_profiles WHERE id = ? AND owner_id IS NULL",
                    (profile_id,),
                )
                r = cursor.fetchone()
                if r:
                    return self._row_to_profile(r)
            else:
                cursor.execute("SELECT * FROM agent_profiles WHERE id = ?", (profile_id,))
                r = cursor.fetchone()
                if r:
                    return self._row_to_profile(r)
        return None

    def list_all(self, owner_id: Optional[str] = None) -> List[AgentProfile]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT * FROM agent_profiles ORDER BY created_at DESC")
            else:
                cursor.execute(
                    "SELECT * FROM agent_profiles WHERE owner_id = ? OR owner_id IS NULL ORDER BY created_at DESC",
                    (owner_id,),
                )
            return [self._row_to_profile(r) for r in cursor.fetchall()]

    def delete(self, profile_id: str, owner_id: Optional[str] = None) -> bool:
        if profile_id == "default": return False # Don't delete the fallback
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("DELETE FROM agent_profiles WHERE id = ?", (profile_id,))
            else:
                cursor.execute(
                    "DELETE FROM agent_profiles WHERE id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (profile_id, owner_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def reset_all(self, preserve_default: bool = True) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if preserve_default:
                cursor.execute("DELETE FROM agent_profiles WHERE id != 'default'")
            else:
                cursor.execute("DELETE FROM agent_profiles")
            conn.commit()
            deleted = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else 0

        self._ensure_default_agent()
        return deleted

    def _row_to_profile(self, r) -> AgentProfile:
        return AgentProfile(
            id=r["id"],
            owner_id=r["owner_id"] if "owner_id" in r.keys() else None,
            name=r["name"],
            description=r["description"],
            runtime_kind=RUNTIME_KIND_ALIASES.get(
                str(r["runtime_kind"]).strip().lower(),
                DEFAULT_RUNTIME_KIND,
            ) if "runtime_kind" in r.keys() and r["runtime_kind"] else DEFAULT_RUNTIME_KIND,
            model=r["model"],
            embed_model=r["embed_model"] if "embed_model" in r.keys() else "nomic-embed-text",
            system_prompt=r["system_prompt"],
            tools=json.loads(r["tools"]),
            avatar_url=r["avatar_url"],
            workspace_path=r["workspace_path"] if "workspace_path" in r.keys() else None,
            bootstrap_files=json.loads(r["bootstrap_files"]) if "bootstrap_files" in r.keys() and r["bootstrap_files"] else ["AGENTS.md", "IDENTITY.md", "SOUL.md", "TOOLS.md"],
            bootstrap_max_chars=r["bootstrap_max_chars"] if "bootstrap_max_chars" in r.keys() and r["bootstrap_max_chars"] else 8000,
            default_use_planner=bool(r["default_use_planner"]) if "default_use_planner" in r.keys() else True,
            default_loop_mode=r["default_loop_mode"] if "default_loop_mode" in r.keys() and r["default_loop_mode"] else "auto",
            default_context_mode=r["default_context_mode"] if "default_context_mode" in r.keys() and r["default_context_mode"] else "v2",
            revision=r["revision"] if "revision" in r.keys() else 1,
            created_at=r["created_at"],
            updated_at=r["updated_at"]
        )
