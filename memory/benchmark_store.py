from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional


_STORE_PATH = Path("logs/memory_benchmark_latest.json")
_LOCK = threading.RLock()


def _read_all() -> dict:
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_all(payload: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_latest(owner_id: str, result: dict) -> None:
    safe_owner = str(owner_id or "").strip()
    if not safe_owner:
        return
    with _LOCK:
        payload = _read_all()
        payload[safe_owner] = result
        _write_all(payload)


def load_latest(owner_id: str) -> Optional[dict]:
    safe_owner = str(owner_id or "").strip()
    if not safe_owner:
        return None
    with _LOCK:
        payload = _read_all()
        value = payload.get(safe_owner)
        return value if isinstance(value, dict) else None

