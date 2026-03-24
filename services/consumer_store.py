from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ConsumerStoreSelection:
    consumer_db: bool = True
    consumer_sessions: bool = True

    @classmethod
    def from_payload(cls, payload: Optional[dict[str, Any]] = None) -> "ConsumerStoreSelection":
        payload = payload or {}
        return cls(
            consumer_db=bool(payload.get("consumer_db", True)),
            consumer_sessions=bool(payload.get("consumer_sessions", True)),
        )

    def enabled_names(self) -> list[str]:
        names: list[str] = []
        if self.consumer_db:
            names.append("consumer_db")
        if self.consumer_sessions:
            names.append("consumer_sessions")
        return names


class ConsumerStoreService:
    def __init__(
        self,
        *,
        consumer_db_path: str = "consumer.db",
        consumer_sessions_db_path: str = "consumer_sessions.db",
        backup_root: Optional[Path] = None,
    ):
        self.project_root = Path(__file__).resolve().parent.parent
        self.consumer_db_path = Path(consumer_db_path).resolve()
        self.consumer_sessions_db_path = Path(consumer_sessions_db_path).resolve()
        self.backup_root = Path(backup_root) if backup_root else self.project_root / "backups" / "consumer_storage"
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self._init_consumer_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.consumer_db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_consumer_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consumer_options (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    model TEXT NOT NULL DEFAULT 'groq:moonshotai/kimi-k2-instruct',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO consumer_options (id, model, created_at, updated_at)
                VALUES (1, ?, ?, ?)
                """,
                ("groq:moonshotai/kimi-k2-instruct", now, now),
            )
            conn.commit()

    @staticmethod
    def _path_size(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _store_info(self, key: str, path: Path) -> dict[str, Any]:
        return {
            "key": key,
            "path": str(path),
            "exists": path.exists(),
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": self._path_size(path),
        }

    def get_options(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT model, updated_at FROM consumer_options WHERE id = 1").fetchone()
            return {"model": row["model"], "updated_at": row["updated_at"]} if row else {"model": "groq:moonshotai/kimi-k2-instruct"}

    def set_options(self, model: str) -> dict[str, Any]:
        model = (model or "").strip() or "groq:moonshotai/kimi-k2-instruct"
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE consumer_options
                SET model = ?, updated_at = ?
                WHERE id = 1
                """,
                (model, now),
            )
            conn.commit()
        return self.get_options()

    def status(self) -> dict[str, Any]:
        stores = {
            "consumer_db": self._store_info("consumer_db", self.consumer_db_path),
            "consumer_sessions": self._store_info("consumer_sessions", self.consumer_sessions_db_path),
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backup_root": str(self.backup_root.resolve()),
            "stores": stores,
            "options": self.get_options(),
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

    def backup(self, selection: ConsumerStoreSelection, label: str = "") -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_label = "_".join((label or "").strip().split())[:40]
        backup_name = f"{stamp}_{safe_label}" if safe_label else stamp
        target_dir = self.backup_root / backup_name
        target_dir.mkdir(parents=True, exist_ok=False)

        sources = {
            "consumer_db": self.consumer_db_path,
            "consumer_sessions": self.consumer_sessions_db_path,
        }

        copied: list[dict[str, Any]] = []
        for store_name in selection.enabled_names():
            source = sources[store_name]
            if not source.exists():
                continue
            destination = target_dir / source.name
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

    def reset(self, selection: ConsumerStoreSelection, backup_first: bool = True, backup_label: str = "") -> dict[str, Any]:
        backup_info = self.backup(selection, backup_label) if backup_first else None
        cleared: dict[str, Any] = {}

        if selection.consumer_sessions and self.consumer_sessions_db_path.exists():
            self.consumer_sessions_db_path.unlink(missing_ok=True)
            cleared["consumer_sessions"] = {"deleted": True}

        if selection.consumer_db:
            if self.consumer_db_path.exists():
                self.consumer_db_path.unlink(missing_ok=True)
            self._init_consumer_db()
            cleared["consumer_db"] = {"reset": True}

        return {
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "backup": backup_info,
            "cleared": cleared,
            "status": self.status(),
        }
