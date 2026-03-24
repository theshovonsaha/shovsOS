"""
Tool Results DB
-----------------
Lightweight persistence for tool execution results.
Keeps tool outputs, generated apps, and artifacts linked to sessions
so they survive page reloads.

Separate from sessions.db to keep concerns clean.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict
from config.config import cfg


DB_PATH = "tool_results.db"


class ToolResultsDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tool_results (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT DEFAULT 'default',
                    tool_name TEXT NOT NULL,
                    arguments TEXT,
                    result TEXT,
                    success INTEGER DEFAULT 1,
                    result_type TEXT DEFAULT 'text',
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tool_results_session
                ON tool_results(session_id)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tool_results_type
                ON tool_results(result_type)
            ''')
            conn.commit()

    def store(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict,
        result: str,
        success: bool = True,
        result_type: str = "text",
        agent_id: str = "default",
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a tool result. Returns the result ID."""
        result_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO tool_results
                (id, session_id, agent_id, tool_name, arguments, result, success, result_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_id, session_id, agent_id, tool_name,
                json.dumps(arguments), result, int(success),
                result_type, json.dumps(metadata or {}), now
            ))
            conn.commit()
        return result_id

    def get_by_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve all tool results for a session, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM tool_results
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (session_id, limit))
            return [dict(r) for r in cursor.fetchall()]

    def get_apps_by_session(self, session_id: str) -> List[Dict]:
        """Get only generate_app results for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM tool_results
                WHERE session_id = ? AND result_type = 'app_view'
                ORDER BY created_at DESC
            ''', (session_id,))
            return [dict(r) for r in cursor.fetchall()]

    def get_all_apps(self, limit: int = 100) -> List[Dict]:
        """Get all generated apps across all sessions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM tool_results
                WHERE result_type = 'app_view'
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def get_by_id(self, result_id: str) -> Optional[Dict]:
        """Retrieve a single tool result by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM tool_results WHERE id = ?', (result_id,))
            r = cursor.fetchone()
            return dict(r) if r else None

    def delete_by_session(self, session_id: str) -> int:
        """Delete all tool results for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM tool_results WHERE session_id = ?', (session_id,))
            conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM tool_results')
            return cursor.fetchone()[0]

    def reset_all(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM tool_results')
            conn.commit()
            return cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else 0
