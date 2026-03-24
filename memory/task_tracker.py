"""Session-scoped task tracker for TodoWrite-style continuity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Literal

TaskStatus = Literal["pending", "in_progress", "completed"]
TaskPriority = Literal["low", "medium", "high"]


@dataclass
class Task:
    id: str
    content: str
    status: TaskStatus = "pending"
    priority: TaskPriority = "medium"


class SessionTaskTracker:
    """Keeps lightweight task state per session id."""

    def __init__(self):
        self._tasks_by_session: dict[str, list[Task]] = {}
        self._meta_by_session: dict[str, dict] = {}
        self._lock = RLock()

    def write(self, session_id: str, tasks: list[dict], topic: str | None = None) -> str:
        if not session_id:
            return "todo_write failed: missing session_id"

        parsed: list[Task] = []
        for idx, raw in enumerate(tasks or [], 1):
            task_id = str(raw.get("id") or f"task-{idx}").strip()
            content = str(raw.get("content") or "").strip()
            status = str(raw.get("status") or "pending").strip().lower()
            priority = str(raw.get("priority") or "medium").strip().lower()

            if not content:
                return f"todo_write failed: task '{task_id}' has empty content"
            if status not in {"pending", "in_progress", "completed"}:
                return f"todo_write failed: invalid status '{status}' for task '{task_id}'"
            if priority not in {"low", "medium", "high"}:
                return f"todo_write failed: invalid priority '{priority}' for task '{task_id}'"

            parsed.append(Task(id=task_id, content=content, status=status, priority=priority))

        with self._lock:
            self._tasks_by_session[session_id] = parsed
            self._meta_by_session[session_id] = {
                "updated_at": self._now(),
                "version": int(self._meta_by_session.get(session_id, {}).get("version", 0)) + 1,
                "topic": (topic or "").strip(),
            }
        return self.render(session_id)

    def update(self, session_id: str, task_id: str, status: str) -> str:
        if not session_id:
            return "todo_update failed: missing session_id"
        if not task_id:
            return "todo_update failed: missing task_id"

        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"pending", "in_progress", "completed"}:
            return f"todo_update failed: invalid status '{status}'"

        with self._lock:
            tasks = self._tasks_by_session.get(session_id, [])
            for task in tasks:
                if task.id == task_id:
                    task.status = normalized_status
                    self._touch(session_id)
                    return self._render_locked(tasks)

        return f"todo_update failed: task '{task_id}' not found"

    def has_tasks(self, session_id: str) -> bool:
        with self._lock:
            return bool(self._tasks_by_session.get(session_id))

    def has_active_tasks(self, session_id: str) -> bool:
        with self._lock:
            return any(task.status != "completed" for task in self._tasks_by_session.get(session_id, []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._tasks_by_session.pop(session_id, None)
            self._meta_by_session.pop(session_id, None)

    def is_stale(self, session_id: str, timeout_seconds: int) -> bool:
        with self._lock:
            meta = self._meta_by_session.get(session_id)
            if not meta:
                return False
            updated_at = meta.get("updated_at")
            if not isinstance(updated_at, str):
                return False
            try:
                age = self._now_ts() - datetime.fromisoformat(updated_at).timestamp()
            except Exception:
                return False
            return age >= max(1, timeout_seconds)

    def summary(self, session_id: str) -> dict:
        with self._lock:
            tasks = self._tasks_by_session.get(session_id, [])
            meta = dict(self._meta_by_session.get(session_id, {}))
            return {
                "task_count": len(tasks),
                "active_count": sum(1 for task in tasks if task.status != "completed"),
                "completed_count": sum(1 for task in tasks if task.status == "completed"),
                "topic": meta.get("topic", ""),
                "updated_at": meta.get("updated_at"),
                "version": meta.get("version", 0),
            }

    def render(self, session_id: str) -> str:
        with self._lock:
            tasks = self._tasks_by_session.get(session_id, [])
            return self._render_locked(tasks, self._meta_by_session.get(session_id))

    @staticmethod
    def _render_locked(tasks: list[Task], meta: dict | None = None) -> str:
        if not tasks:
            return "Current tasks: none"

        lines = ["Current tasks:"]
        topic = (meta or {}).get("topic", "")
        if topic:
            lines.append(f"Workflow topic: {topic}")
        for task in tasks:
            lines.append(
                f"- [{task.status}] {task.id}: {task.content} (priority={task.priority})"
            )
        return "\n".join(lines)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _now_ts() -> float:
        return datetime.now(timezone.utc).timestamp()

    def _touch(self, session_id: str) -> None:
        meta = self._meta_by_session.setdefault(session_id, {"version": 1, "topic": ""})
        meta["updated_at"] = self._now()


_TRACKER: SessionTaskTracker | None = None


def get_session_task_tracker() -> SessionTaskTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = SessionTaskTracker()
    return _TRACKER
