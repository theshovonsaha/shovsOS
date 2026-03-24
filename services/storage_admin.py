from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.config import cfg
from memory.semantic_graph import SemanticGraph
from memory.tool_results_db import ToolResultsDB
from memory.vector_engine import VectorEngine
from memory.session_rag import CHROMA_DIR as SESSION_RAG_DIR, reset_session_rag_storage
from orchestration.agent_profiles import ProfileManager
from orchestration.session_manager import SessionManager


@dataclass(frozen=True)
class StoreSelection:
    sessions: bool = True
    agents: bool = False
    semantic_memory: bool = True
    tool_results: bool = True
    vector_memory: bool = True
    session_rag: bool = True

    @classmethod
    def from_payload(cls, payload: Optional[dict[str, Any]] = None) -> "StoreSelection":
        payload = payload or {}
        return cls(
            sessions=bool(payload.get("sessions", True)),
            agents=bool(payload.get("agents", False)),
            semantic_memory=bool(payload.get("semantic_memory", True)),
            tool_results=bool(payload.get("tool_results", True)),
            vector_memory=bool(payload.get("vector_memory", True)),
            session_rag=bool(payload.get("session_rag", True)),
        )

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.sessions:
            names.append("sessions")
        if self.agents:
            names.append("agents")
        if self.semantic_memory:
            names.append("semantic_memory")
        if self.tool_results:
            names.append("tool_results")
        if self.vector_memory:
            names.append("vector_memory")
        if self.session_rag:
            names.append("session_rag")
        return names


class StorageAdminService:
    def __init__(
        self,
        sessions: SessionManager,
        profiles: ProfileManager,
        backup_root: Optional[Path] = None,
        path_overrides: Optional[dict[str, str | Path]] = None,
    ):
        self.sessions = sessions
        self.profiles = profiles
        self.project_root = Path(__file__).resolve().parent.parent
        self.backup_root = Path(backup_root) if backup_root else self.project_root / "backups" / "storage"
        overrides = {k: Path(v) for k, v in (path_overrides or {}).items()}
        self.paths = {
            "sessions": Path(overrides.get("sessions", sessions.db_path)).resolve(),
            "agents": Path(overrides.get("agents", profiles.db_path)).resolve(),
            "semantic_memory": Path(overrides.get("semantic_memory", "memory_graph.db")).resolve(),
            "tool_results": Path(overrides.get("tool_results", "tool_results.db")).resolve(),
            "vector_memory": Path(overrides.get("vector_memory", cfg.CHROMA_DB_PATH)).resolve(),
            "session_rag": Path(overrides.get("session_rag", SESSION_RAG_DIR)).resolve(),
        }
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _store_info(self, key: str) -> dict[str, Any]:
        path = self.paths[key]
        exists = path.exists()
        size_bytes = self._path_size(path) if exists else 0
        return {
            "key": key,
            "path": str(path),
            "exists": exists,
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": size_bytes,
        }

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def status(self) -> dict[str, Any]:
        stores = {name: self._store_info(name) for name in self.paths}
        stores["sessions"]["records"] = len(self.sessions.list_sessions())
        stores["agents"]["records"] = len(self.profiles.list_all())
        stores["semantic_memory"]["records"] = SemanticGraph(db_path=str(self.paths["semantic_memory"])).count()
        stores["tool_results"]["records"] = ToolResultsDB(db_path=str(self.paths["tool_results"])).count()
        stores["vector_memory"]["records"] = None
        stores["session_rag"]["records"] = None
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backup_root": str(self.backup_root.resolve()),
            "stores": stores,
        }

    def list_backups(self, limit: int = 20) -> dict[str, Any]:
        backups: list[dict[str, Any]] = []
        for path in sorted(self.backup_root.glob("*"), reverse=True):
            if not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            manifest: dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except Exception:
                    manifest = {}
            backups.append(
                {
                    "name": path.name,
                    "path": str(path.resolve()),
                    "created_at": manifest.get("created_at"),
                    "stores": manifest.get("stores", []),
                    "size_bytes": self._path_size(path),
                }
            )
            if len(backups) >= limit:
                break
        return {"backups": backups, "count": len(backups)}

    def backup(self, selection: StoreSelection, label: str = "") -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_label = "_".join((label or "").strip().split())[:40]
        backup_name = f"{stamp}_{safe_label}" if safe_label else stamp
        target_dir = self.backup_root / backup_name
        target_dir.mkdir(parents=True, exist_ok=False)

        copied: list[dict[str, Any]] = []
        for store_name in selection.enabled_names():
            source = self.paths[store_name]
            if not source.exists():
                continue
            destination = target_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            copied.append(
                {
                    "store": store_name,
                    "source": str(source),
                    "backup": str(destination),
                    "size_bytes": self._path_size(source),
                }
            )

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stores": [item["store"] for item in copied],
            "label": label,
            "items": copied,
        }
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return {"backup_name": backup_name, "path": str(target_dir.resolve()), "items": copied}

    def reset(
        self,
        selection: StoreSelection,
        backup_first: bool = True,
        backup_label: str = "",
        preserve_default_agent: bool = True,
    ) -> dict[str, Any]:
        backup_info = self.backup(selection, backup_label) if backup_first else None
        cleared: dict[str, Any] = {}

        if selection.sessions:
            cleared["sessions"] = {"deleted_rows": self.sessions.reset_all()}

        if selection.agents:
            cleared["agents"] = {"deleted_rows": self.profiles.reset_all(preserve_default=preserve_default_agent)}

        if selection.semantic_memory:
            graph = SemanticGraph(db_path=str(self.paths["semantic_memory"]))
            before = graph.count()
            graph.clear()
            cleared["semantic_memory"] = {"deleted_rows": before}

        if selection.tool_results:
            results_db = ToolResultsDB(db_path=str(self.paths["tool_results"]))
            cleared["tool_results"] = {"deleted_rows": results_db.reset_all()}

        if selection.vector_memory:
            VectorEngine.reset_storage(str(self.paths["vector_memory"]))
            cleared["vector_memory"] = {"reset_path": str(self.paths["vector_memory"])}

        if selection.session_rag:
            reset_session_rag_storage(self.paths["session_rag"])
            cleared["session_rag"] = {"reset_path": str(self.paths["session_rag"])}

        return {
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "backup": backup_info,
            "cleared": cleared,
            "status": self.status(),
        }
