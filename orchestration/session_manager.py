"""
Session Manager (FIXED)
-----------------------
BUG FIXED: SLIDING_WINDOW_SIZE was 6 (3 turns). After 4+ exchanges,
the first message fell completely out of the window AND was never
compressed (trivial skip). Model had zero access to early messages
and hallucinated answers to "what was my first message?".

FIXES:
  1. SLIDING_WINDOW_SIZE raised from 6 → 20 (10 turns in view at all times)
  2. Session stores first_message explicitly — always available regardless
     of window size or compression. Never lost.
  3. append_message now passes is_first_exchange=True signal via return value
     so AgentCore can tell context_engine to never skip the first compression.
"""

import json
import sqlite3
import uuid
import asyncio
from datetime import datetime, timezone
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


from config.config import cfg
from engine.candidate_signals import parse_candidate_context, render_candidate_signals

MAX_SESSIONS        = cfg.MAX_SESSIONS
SLIDING_WINDOW_SIZE = cfg.SLIDING_WINDOW_SIZE  # Centralized — no more mismatch
DB_PATH             = cfg.SESSIONS_DB


@dataclass
class Session:
    id:                  str
    agent_id:            str = "default"  # Scoping key
    owner_id:            Optional[str] = None
    created_at:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model:               str = cfg.DEFAULT_MODEL # Fallback
    system_prompt:       str = ""
    compressed_context:  str = ""
    candidate_context:   str = ""
    candidate_signals:   list[dict] = field(default_factory=list)
    sliding_window:      list[dict] = field(default_factory=list)
    full_history:        list[dict] = field(default_factory=list)
    title:               Optional[str] = None
    first_message:       Optional[str] = None
    parent_id:           Optional[str] = None
    lock:                asyncio.Lock = field(default_factory=asyncio.Lock)
    message_count:       int = 0
    context_mode:        str = "v1"
    _interrupted:        bool = False # Transient interrupt signal  # "v1" (linear) | "v2" (convergent) | "v3" (hybrid)


class SessionManager:

    def __init__(self, max_sessions: int = MAX_SESSIONS, db_path: str = DB_PATH):
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._max       = max_sessions
        self.db_path    = db_path
        self._init_db()
        self._load_from_db()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT DEFAULT 'default',
                    owner_id TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    model TEXT,
                    system_prompt TEXT,
                    compressed_context TEXT,
                    candidate_context TEXT,
                    candidate_signals_json TEXT,
                    sliding_window TEXT,
                    full_history TEXT,
                    title TEXT,
                    first_message TEXT,
                    parent_id TEXT,
                    message_count INTEGER
                )
            ''')
            # Migrations for existing DBs
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN first_message TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN agent_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN parent_id TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN context_mode TEXT DEFAULT 'v1'")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN owner_id TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN candidate_context TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN candidate_signals_json TEXT")
            except sqlite3.OperationalError: pass
            conn.commit()

    def _load_from_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (self._max,))
            for r in reversed(cursor.fetchall()):
                s = self._row_to_session(r)
                self._sessions[s.id] = s

    def _row_to_session(self, r) -> Session:
        candidate_signals = (
            json.loads(r["candidate_signals_json"])
            if "candidate_signals_json" in r.keys() and r["candidate_signals_json"]
            else parse_candidate_context(r["candidate_context"] if "candidate_context" in r.keys() else "")
        )
        full_history = self._normalize_history(json.loads(r["full_history"]))
        sliding_window = self._normalize_history(json.loads(r["sliding_window"]))
        return Session(
            id=r["id"],
            agent_id=r["agent_id"] if "agent_id" in r.keys() else "default",
            owner_id=r["owner_id"] if "owner_id" in r.keys() else None,
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            model=r["model"],
            system_prompt=r["system_prompt"],
            compressed_context=r["compressed_context"] or "",
            candidate_context=render_candidate_signals(candidate_signals) if candidate_signals else "",
            candidate_signals=candidate_signals,
            sliding_window=sliding_window,
            full_history=full_history,
            title=r["title"],
            first_message=r["first_message"] if "first_message" in r.keys() else None,
            parent_id=r["parent_id"] if "parent_id" in r.keys() else None,
            message_count=r["message_count"],
            context_mode=r["context_mode"] if "context_mode" in r.keys() else "v1",
        )

    def _save_to_db(self, s: Session):
        candidate_context = render_candidate_signals(s.candidate_signals) if s.candidate_signals else ""
        s.candidate_context = candidate_context
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO sessions
                (id, agent_id, owner_id, created_at, updated_at, model, system_prompt, compressed_context,
                 candidate_context, candidate_signals_json, sliding_window, full_history, title, first_message, parent_id, message_count, context_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                s.id, s.agent_id, s.owner_id, s.created_at, s.updated_at, s.model, s.system_prompt,
                s.compressed_context,
                candidate_context,
                json.dumps(s.candidate_signals),
                json.dumps(s.sliding_window),
                json.dumps(s.full_history),
                s.title, s.first_message, s.parent_id, s.message_count, s.context_mode,
            ))
            conn.commit()

    def _delete_from_db(self, session_id: str, owner_id: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            if owner_id is None:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            else:
                conn.execute(
                    "DELETE FROM sessions WHERE id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (session_id, owner_id),
                )
            conn.commit()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def create(
        self,
        model: str,
        system_prompt: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Session:
        now = datetime.now(timezone.utc).isoformat()
        sid = session_id or str(uuid.uuid4())
        session = Session(id=sid, agent_id=agent_id, created_at=now, updated_at=now,
                          model=model, system_prompt=system_prompt, parent_id=parent_id, owner_id=owner_id)
        self._sessions[sid] = session
        self._save_to_db(session)
        self._evict_if_needed()
        return session

    def get(self, session_id: str, owner_id: Optional[str] = None) -> Optional[Session]:
        if session_id in self._sessions:
            cached = self._sessions[session_id]
            if owner_id is None or cached.owner_id == owner_id:
                self._sessions.move_to_end(session_id)
                return cached
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            else:
                cursor.execute(
                    "SELECT * FROM sessions WHERE id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (session_id, owner_id),
                )
            r = cursor.fetchone()
            if r:
                s = self._row_to_session(r)
                self._sessions[s.id] = s
                self._evict_if_needed()
                return s
        return None

    def get_or_create(
        self,
        session_id: Optional[str],
        model: str,
        system_prompt: str,
        agent_id: str = "default",
        parent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Session:
        if session_id:
            s = self.get(session_id, owner_id=owner_id)
            if s:
                return s
        return self.create(
            model=model,
            system_prompt=system_prompt,
            agent_id=agent_id,
            session_id=session_id,
            parent_id=parent_id,
            owner_id=owner_id,
        )

    def delete(self, session_id: str, owner_id: Optional[str] = None) -> bool:
        existed = session_id in self._sessions
        if existed:
            cached = self._sessions[session_id]
            if owner_id is None or cached.owner_id == owner_id:
                self._sessions.pop(session_id, None)
            else:
                existed = False
        self._delete_from_db(session_id, owner_id=owner_id)
        return existed

    def reset_all(self, owner_id: Optional[str] = None) -> int:
        count = len(self._sessions)
        if owner_id is None:
            self._sessions.clear()
        else:
            self._sessions = OrderedDict(
                (sid, session)
                for sid, session in self._sessions.items()
                if session.owner_id != owner_id
            )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("DELETE FROM sessions")
            else:
                cursor.execute(
                    "DELETE FROM sessions WHERE COALESCE(owner_id, '') = COALESCE(?, '')",
                    (owner_id,),
                )
            conn.commit()
            return cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else count

    def update_model(self, session_id: str, model: str):
        """Update the model for an existing session (when user switches models mid-chat)."""
        s = self._sessions.get(session_id)
        if s:
            s.model = model
            self._save_to_db(s)

    def set_context_mode(self, session_id: str, mode: str):
        s = self.get(session_id)
        if s and mode in ("v1", "v2", "v3"):
            s.context_mode = mode
            self._save_to_db(s)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def interrupt(self, session_id: str):
        if s := self.get(session_id):
            s._interrupted = True
            print(f"[SessionManager] Interrupt signal set for {session_id}")

    def is_interrupted(self, session_id: str) -> bool:
        if s := self.get(session_id):
            return s._interrupted
        return False

    def clear_interrupt(self, session_id: str):
        if s := self.get(session_id):
            s._interrupted = False

    def update_context(self, session_id: str, context: str):
        if s := self.get(session_id):
            s.compressed_context = context
            s.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_to_db(s)

    def update_candidate_context(self, session_id: str, candidate_context: str):
        if s := self.get(session_id):
            s.candidate_signals = parse_candidate_context(candidate_context)
            s.candidate_context = render_candidate_signals(s.candidate_signals) if s.candidate_signals else ""
            s.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_to_db(s)

    def update_candidate_signals(self, session_id: str, candidate_signals: list[dict]):
        if s := self.get(session_id):
            s.candidate_signals = list(candidate_signals or [])
            s.candidate_context = render_candidate_signals(s.candidate_signals)
            s.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_to_db(s)

    @staticmethod
    def _normalize_history(history: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for index, item in enumerate(history or []):
            role = str((item or {}).get("role") or "user")
            content = str((item or {}).get("content") or "")
            normalized.append(
                {
                    "id": str((item or {}).get("id") or uuid.uuid4()),
                    "role": role,
                    "content": content,
                    "created_at": (item or {}).get("created_at"),
                    "edited": bool((item or {}).get("edited", False)),
                    "edited_at": (item or {}).get("edited_at"),
                    "sequence": int((item or {}).get("sequence") or index),
                }
            )
        return normalized

    def clear_derived_state(self, session_id: str, owner_id: Optional[str] = None) -> bool:
        s = self.get(session_id, owner_id=owner_id)
        if not s:
            return False
        s.compressed_context = ""
        s.candidate_signals = []
        s.candidate_context = ""
        s.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_to_db(s)
        return True

    def edit_message(
        self,
        session_id: str,
        *,
        content: str,
        message_id: Optional[str] = None,
        message_index: Optional[int] = None,
        owner_id: Optional[str] = None,
    ) -> Optional[dict]:
        s = self.get(session_id, owner_id=owner_id)
        if not s:
            return None

        target_index: Optional[int] = None
        if message_id is not None:
            for index, item in enumerate(s.full_history):
                if str(item.get("id") or "") == str(message_id):
                    target_index = index
                    break
        elif message_index is not None and 0 <= int(message_index) < len(s.full_history):
            target_index = int(message_index)

        if target_index is None:
            return None

        target = dict(s.full_history[target_index])
        target["content"] = str(content or "")
        target["edited"] = True
        target["edited_at"] = datetime.now(timezone.utc).isoformat()
        target.setdefault("created_at", target["edited_at"])
        target.setdefault("id", str(uuid.uuid4()))
        target["sequence"] = target_index
        s.full_history[target_index] = target
        s.sliding_window = s.full_history[-SLIDING_WINDOW_SIZE:]
        s.message_count = len(s.full_history)

        first_user = next(
            (str(item.get("content") or "") for item in s.full_history if str(item.get("role") or "") == "user"),
            None,
        )
        s.first_message = first_user
        s.title = (first_user or "New Chat")[:60] + ("…" if first_user and len(first_user) > 60 else "")
        s.compressed_context = ""
        s.candidate_signals = []
        s.candidate_context = ""
        s.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_to_db(s)
        return target

    def append_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Append message. Returns True if this was the FIRST user message
        so AgentCore knows to pass is_first_exchange=True to ContextEngine.
        """
        s = self.get(session_id)
        if not s:
            return False

        is_first = False
        now = datetime.now(timezone.utc).isoformat()
        msg = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "created_at": now,
            "edited": False,
            "edited_at": None,
            "sequence": len(s.full_history),
        }
        s.full_history.append(msg)
        s.sliding_window.append(msg)

        if len(s.sliding_window) > SLIDING_WINDOW_SIZE:
            s.sliding_window = s.sliding_window[-SLIDING_WINDOW_SIZE:]

        s.message_count += 1
        s.updated_at = datetime.now(timezone.utc).isoformat()

        # Store first user message permanently
        if role == "user" and s.first_message is None:
            s.first_message = content
            is_first = True

        # Auto-title
        if role == "user" and not s.title:
            s.title = content[:60] + ("…" if len(content) > 60 else "")

        self._save_to_db(s)
        return is_first

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_sessions(self, agent_id: Optional[str] = None, owner_id: Optional[str] = None) -> list[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if agent_id and owner_id is not None:
                    cursor.execute(
                        "SELECT id, title, created_at, updated_at, model, message_count, context_mode "
                        "FROM sessions WHERE agent_id = ? AND parent_id IS NULL AND COALESCE(owner_id, '') = COALESCE(?, '') "
                        "ORDER BY updated_at DESC",
                        (agent_id, owner_id),
                    )
                elif agent_id:
                    cursor.execute(
                        "SELECT id, title, created_at, updated_at, model, message_count, context_mode "
                        "FROM sessions WHERE agent_id = ? AND parent_id IS NULL ORDER BY updated_at DESC", (agent_id,)
                    )
                elif owner_id is not None:
                    cursor.execute(
                        "SELECT id, title, created_at, updated_at, model, message_count, context_mode "
                        "FROM sessions WHERE parent_id IS NULL AND COALESCE(owner_id, '') = COALESCE(?, '') "
                        "ORDER BY updated_at DESC",
                        (owner_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT id, title, created_at, updated_at, model, message_count, context_mode "
                        "FROM sessions WHERE parent_id IS NULL ORDER BY updated_at DESC"
                    )
                return [
                    {"id": r["id"], "title": r["title"] or "New Chat",
                     "created_at": r["created_at"], "updated_at": r["updated_at"],
                     "model": r["model"], "message_count": r["message_count"],
                     "context_mode": r["context_mode"] if "context_mode" in r.keys() else "v1"}
                    for r in cursor.fetchall()
                ]
        except Exception as e:
            print(f"[SessionManager] DB read failed: {e}")
            return [
                {"id": s.id, "title": s.title or "New Chat", "created_at": s.created_at,
                 "updated_at": s.updated_at, "model": s.model, "message_count": s.message_count,
                 "context_mode": s.context_mode}
                for s in reversed(list(self._sessions.values()))
            ]

    def _evict_if_needed(self):
        while len(self._sessions) > self._max:
            evicted_id, _ = self._sessions.popitem(last=False)
            print(f"[SessionManager] Evicted: {evicted_id}")
