"""
Structured Trace Store
----------------------
Persist high-volume agent traces with a compact index + optional payload blobs.
Designed for prompt/pass observability without blowing up memory usage.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterator, Optional


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Never break trace writing due to non-serializable payloads.
        return json.dumps({"unserializable": str(data)}, ensure_ascii=False)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(8, max_chars // 2)
    return f"{text[:keep]} ... {text[-keep:]}"


def _build_preview(event_type: str, data: Any, max_chars: int) -> str:
    if isinstance(data, dict):
        if event_type == "llm_prompt":
            msgs = data.get("messages")
            if isinstance(msgs, list):
                roles = [str(m.get("role", "?")) for m in msgs if isinstance(m, dict)]
                total_chars = sum(len(str(m.get("content", ""))) for m in msgs if isinstance(m, dict))
                turn = data.get("turn")
                model = data.get("model")
                return _clip(
                    f"turn={turn} model={model} messages={len(msgs)} roles={roles} total_chars={total_chars}",
                    max_chars,
                )

        if event_type in {"tool_call", "tool_result"}:
            tool_name = data.get("tool_name")
            success = data.get("success")
            turn = data.get("turn")
            content_preview = str(data.get("content_preview", ""))
            return _clip(
                f"turn={turn} tool={tool_name} success={success} {content_preview}",
                max_chars,
            )

        top_keys = list(data.keys())[:10]
        return _clip(f"keys={top_keys}", max_chars)

    return _clip(str(data), max_chars)


def _payload_summary(data: Any) -> dict:
    if isinstance(data, dict):
        summary: dict[str, Any] = {"type": "object", "keys": list(data.keys())[:20]}
        messages = data.get("messages")
        if isinstance(messages, list):
            summary["messages"] = {
                "count": len(messages),
                "roles": [
                    str(item.get("role", "?"))
                    for item in messages
                    if isinstance(item, dict)
                ][:20],
                "total_chars": sum(
                    len(str(item.get("content", "")))
                    for item in messages
                    if isinstance(item, dict)
                ),
            }
        for field in (
            "turn",
            "model",
            "message_count",
            "estimated_tokens",
            "tool_name",
            "success",
            "content_length",
        ):
            if field in data:
                summary[field] = data[field]
        return summary

    if isinstance(data, list):
        return {"type": "list", "count": len(data)}

    text = str(data)
    return {"type": type(data).__name__, "preview": _clip(text, 180)}


class TraceStore:
    def __init__(self):
        trace_root = os.getenv("TRACE_DIR", "./logs")
        self._store_dir = os.getenv("TRACE_STORE_DIR", os.path.join(trace_root, "traces"))
        self._index_path = os.path.join(self._store_dir, "trace_index.jsonl")
        self._payload_root = os.path.join(self._store_dir, "payloads")

        self._max_inline_bytes = _env_int("TRACE_INLINE_MAX_BYTES", 4096)
        self._max_recent = _env_int("TRACE_RECENT_CACHE", 2000)
        self._max_preview_chars = _env_int("TRACE_PREVIEW_CHARS", 220)

        os.makedirs(self._store_dir, exist_ok=True)
        os.makedirs(self._payload_root, exist_ok=True)

        self._lock = threading.RLock()
        self._recent: deque[dict] = deque(maxlen=self._max_recent)

        self._load_recent_cache()

    def _load_recent_cache(self) -> None:
        if not os.path.exists(self._index_path):
            return

        cached: list[dict] = []
        for line in self._iter_index_reverse():
            if len(cached) >= self._max_recent:
                break
            try:
                item = json.loads(line)
                # Keep cache lightweight.
                if item.get("payload_ref"):
                    item["data"] = None
                cached.append(item)
            except Exception:
                continue

        for item in reversed(cached):
            self._recent.append(item)

    def _iter_index_reverse(self) -> Iterator[str]:
        if not os.path.exists(self._index_path):
            return

        block_size = 64 * 1024
        with open(self._index_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            buffer = b""

            while pos > 0:
                read_size = min(block_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                buffer = chunk + buffer

                lines = buffer.split(b"\n")
                buffer = lines[0]

                for raw in reversed(lines[1:]):
                    if raw.strip():
                        yield raw.decode("utf-8", errors="replace")

            if buffer.strip():
                yield buffer.decode("utf-8", errors="replace")

    def _write_payload_blob(self, event_id: str, payload_json: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = os.path.join(self._payload_root, day)
        os.makedirs(day_dir, exist_ok=True)

        filename = f"{event_id}.json"
        abs_path = os.path.join(day_dir, filename)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(payload_json)

        return os.path.join("payloads", day, filename)

    def append_event(
        self,
        agent_id: str,
        session_id: str,
        event_type: str,
        data: Any,
        *,
        pass_index: Optional[int] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> dict:
        now = time.time()
        event_id = f"{int(now * 1000)}-{uuid.uuid4().hex[:10]}"

        payload_json = _safe_json_dumps(data)
        payload_size = len(payload_json.encode("utf-8"))

        # Large payloads (especially prompts) go to blob files.
        force_blob_events = {"llm_prompt", "prompt_chain", "assistant_response"}
        to_blob = payload_size > self._max_inline_bytes or event_type in force_blob_events

        inline_data: Any = data
        payload_ref: Optional[str] = None
        if to_blob:
            payload_ref = self._write_payload_blob(event_id, payload_json)
            inline_data = {"summary": _payload_summary(data)}

        entry = {
            "id": event_id,
            "ts": now,
            "iso_ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "session_id": session_id,
            "run_id": run_id,
            "owner_id": owner_id,
            "event_type": event_type,
            "pass_index": pass_index,
            "size_bytes": payload_size,
            "preview": _build_preview(event_type, data, self._max_preview_chars),
            "payload_ref": payload_ref,
            "data": inline_data,
        }

        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self._index_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            cache_item = dict(entry)
            if payload_ref:
                cache_item["data"] = None
            self._recent.append(cache_item)

        return entry

    def _matches(
        self,
        item: dict,
        *,
        session_id: Optional[str],
        run_id: Optional[str],
        owner_id: Optional[str],
        event_type: Optional[str],
        before_ts: Optional[float],
    ) -> bool:
        if session_id and item.get("session_id") != session_id:
            return False
        if run_id and item.get("run_id") != run_id:
            return False
        if owner_id is not None and item.get("owner_id") != owner_id:
            return False
        if event_type and item.get("event_type") != event_type:
            return False
        if before_ts is not None and float(item.get("ts", 0.0)) >= before_ts:
            return False
        return True

    def list_events(
        self,
        *,
        limit: int = 120,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        event_type: Optional[str] = None,
        before_ts: Optional[float] = None,
    ) -> list[dict]:
        limit = max(1, min(500, int(limit)))
        results: list[dict] = []
        seen: set[str] = set()

        with self._lock:
            for item in reversed(self._recent):
                if not self._matches(
                    item,
                    session_id=session_id,
                    run_id=run_id,
                    owner_id=owner_id,
                    event_type=event_type,
                    before_ts=before_ts,
                ):
                    continue
                item_id = str(item.get("id"))
                if item_id in seen:
                    continue
                seen.add(item_id)
                results.append(dict(item))
                if len(results) >= limit:
                    return results

            for line in self._iter_index_reverse():
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                item_id = str(item.get("id"))
                if item_id in seen:
                    continue
                if not self._matches(
                    item,
                    session_id=session_id,
                    run_id=run_id,
                    owner_id=owner_id,
                    event_type=event_type,
                    before_ts=before_ts,
                ):
                    continue

                if item.get("payload_ref"):
                    item["data"] = None

                results.append(item)
                seen.add(item_id)
                if len(results) >= limit:
                    break

        return results

    def _read_payload_blob(self, payload_ref: str) -> Any:
        abs_path = os.path.join(self._store_dir, payload_ref)
        if not os.path.exists(abs_path):
            return None
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_event(self, event_id: str) -> Optional[dict]:
        with self._lock:
            for item in reversed(self._recent):
                if item.get("id") == event_id:
                    found = dict(item)
                    break
            else:
                found = None

            if found is None:
                for line in self._iter_index_reverse():
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if item.get("id") == event_id:
                        found = item
                        break

        if found is None:
            return None

        payload_ref = found.get("payload_ref")
        if payload_ref:
            found["data"] = self._read_payload_blob(payload_ref)

        return found

    def stats(
        self,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        window: int = 400,
    ) -> dict:
        window = max(10, min(2000, int(window)))
        events = self.list_events(limit=window, session_id=session_id, run_id=run_id, owner_id=owner_id)

        by_type: dict[str, int] = {}
        by_session: dict[str, int] = {}
        by_owner: dict[str, int] = {}
        by_run: dict[str, int] = {}
        payload_backed = 0
        total_size = 0
        max_pass = -1

        for e in events:
            et = str(e.get("event_type", "unknown"))
            sid = str(e.get("session_id", "unknown"))
            rid = str(e.get("run_id", "unknown"))
            oid = str(e.get("owner_id", "unknown"))
            by_type[et] = by_type.get(et, 0) + 1
            by_session[sid] = by_session.get(sid, 0) + 1
            by_run[rid] = by_run.get(rid, 0) + 1
            by_owner[oid] = by_owner.get(oid, 0) + 1
            if e.get("payload_ref"):
                payload_backed += 1
            total_size += int(e.get("size_bytes", 0) or 0)

            pass_index = e.get("pass_index")
            if isinstance(pass_index, int):
                max_pass = max(max_pass, pass_index)

        latest_ts = events[0].get("ts") if events else None

        return {
            "window": window,
            "event_count": len(events),
            "payload_backed_events": payload_backed,
            "inline_events": len(events) - payload_backed,
            "total_size_bytes": total_size,
            "avg_size_bytes": (total_size // len(events)) if events else 0,
            "max_pass_index": max_pass,
            "event_types": by_type,
            "sessions": by_session,
            "runs": by_run,
            "owners": by_owner,
            "latest_ts": latest_ts,
        }


_TRACE_STORE_SINGLETON = TraceStore()


def get_trace_store() -> TraceStore:
    return _TRACE_STORE_SINGLETON
