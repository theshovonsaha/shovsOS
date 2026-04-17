from __future__ import annotations

import sqlite3
import uuid
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_RUNS_DB = "runs.db"


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    agent_id: str
    model: str
    status: str
    owner_id: Optional[str] = None
    agent_revision: Optional[int] = None
    parent_run_id: Optional[str] = None
    started_at: str = ""
    ended_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class LoopCheckpoint:
    checkpoint_id: int
    run_id: str
    phase: str
    tool_turn: int = 0
    status: str = ""
    strategy: str = ""
    notes: str = ""
    confidence: Optional[float] = None
    tools: list[str] | None = None
    tool_results: list[dict[str, Any]] | None = None
    candidate_facts: list[str] | None = None
    created_at: str = ""


@dataclass
class RunArtifact:
    artifact_id: str
    run_id: str
    session_id: str
    artifact_type: str
    label: str
    owner_id: Optional[str] = None
    tool_name: Optional[str] = None
    storage_path: Optional[str] = None
    content_hash: str = ""
    size_bytes: int = 0
    preview: str = ""
    metadata: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class RunEval:
    eval_id: str
    run_id: str
    session_id: str
    eval_type: str
    phase: str
    passed: bool
    owner_id: Optional[str] = None
    score: Optional[float] = None
    detail: str = ""
    metadata: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class RunPassRecord:
    pass_id: int
    run_id: str
    phase: str
    tool_turn: int = 0
    status: str = ""
    objective: str = ""
    strategy: str = ""
    notes: str = ""
    selected_tools: list[str] | None = None
    tool_results: list[dict[str, Any]] | None = None
    compiled_context: dict[str, Any] | None = None
    response_preview: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    created_at: str = ""


class RunStore:
    def __init__(self, db_path: str = DEFAULT_RUNS_DB):
        self.db_path = str(Path(db_path))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    owner_id TEXT,
                    agent_revision INTEGER,
                    parent_run_id TEXT,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    tool_turn INTEGER DEFAULT 0,
                    status TEXT,
                    strategy TEXT,
                    notes TEXT,
                    confidence REAL,
                    tools_json TEXT,
                    tool_results_json TEXT,
                    candidate_facts_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    owner_id TEXT,
                    tool_name TEXT,
                    artifact_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    storage_path TEXT,
                    content_hash TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    preview TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_evals (
                    eval_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    owner_id TEXT,
                    eval_type TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    score REAL,
                    detail TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_passes (
                    pass_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    tool_turn INTEGER DEFAULT 0,
                    status TEXT,
                    objective TEXT,
                    strategy TEXT,
                    notes TEXT,
                    selected_tools_json TEXT,
                    tool_results_json TEXT,
                    compiled_context_json TEXT,
                    response_preview TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    estimated_cost_usd REAL DEFAULT 0,
                    cumulative_cost_usd REAL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            for statement in (
                "ALTER TABLE run_passes ADD COLUMN input_tokens INTEGER DEFAULT 0",
                "ALTER TABLE run_passes ADD COLUMN output_tokens INTEGER DEFAULT 0",
                "ALTER TABLE run_passes ADD COLUMN total_tokens INTEGER DEFAULT 0",
                "ALTER TABLE run_passes ADD COLUMN estimated_cost_usd REAL DEFAULT 0",
                "ALTER TABLE run_passes ADD COLUMN cumulative_cost_usd REAL DEFAULT 0",
            ):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_parent_run_id ON runs(parent_run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_owner_id ON runs(owner_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_checkpoints_run_id ON run_checkpoints(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_artifacts_run_id ON run_artifacts(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_artifacts_owner_id ON run_artifacts(owner_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_evals_run_id ON run_evals(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_evals_owner_id ON run_evals(owner_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_passes_run_id ON run_passes(run_id)")
            conn.commit()

    def start_run(
        self,
        *,
        session_id: str,
        agent_id: str,
        model: str,
        owner_id: Optional[str] = None,
        agent_revision: Optional[int] = None,
        parent_run_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> RunRecord:
        now = datetime.now(timezone.utc).isoformat()
        record = RunRecord(
            run_id=run_id or str(uuid.uuid4()),
            session_id=session_id,
            agent_id=agent_id,
            owner_id=owner_id,
            agent_revision=agent_revision,
            parent_run_id=parent_run_id,
            model=model,
            status="running",
            started_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, session_id, agent_id, owner_id, agent_revision, parent_run_id, model, status, started_at, ended_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.session_id,
                    record.agent_id,
                    record.owner_id,
                    record.agent_revision,
                    record.parent_run_id,
                    record.model,
                    record.status,
                    record.started_at,
                    record.ended_at,
                    record.error,
                ),
            )
            conn.commit()
        return record

    def finish_run(self, run_id: str, *, status: str, error: Optional[str] = None) -> None:
        ended_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, ended_at = ?, error = ?
                WHERE run_id = ?
                """,
                (status, ended_at, error, run_id),
            )
            conn.commit()

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            return RunRecord(**dict(row))

    def save_checkpoint(
        self,
        *,
        run_id: str,
        phase: str,
        tool_turn: int = 0,
        status: str = "",
        strategy: str = "",
        notes: str = "",
        confidence: Optional[float] = None,
        tools: Optional[list[str]] = None,
        tool_results: Optional[list[dict[str, Any]]] = None,
        candidate_facts: Optional[list[str]] = None,
    ) -> LoopCheckpoint:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO run_checkpoints
                (run_id, phase, tool_turn, status, strategy, notes, confidence, tools_json, tool_results_json, candidate_facts_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    phase,
                    tool_turn,
                    status,
                    strategy,
                    notes,
                    confidence,
                    json.dumps(tools or []),
                    json.dumps(tool_results or []),
                    json.dumps(candidate_facts or []),
                    created_at,
                ),
            )
            conn.commit()
            checkpoint_id = int(cursor.lastrowid)
        return LoopCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            phase=phase,
            tool_turn=tool_turn,
            status=status,
            strategy=strategy,
            notes=notes,
            confidence=confidence,
            tools=list(tools or []),
            tool_results=list(tool_results or []),
            candidate_facts=list(candidate_facts or []),
            created_at=created_at,
        )

    def latest_checkpoint(self, run_id: str) -> Optional[LoopCheckpoint]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM run_checkpoints
                WHERE run_id = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_checkpoint(row)

    def list_checkpoints(self, run_id: str) -> list[LoopCheckpoint]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_checkpoints
                WHERE run_id = ?
                ORDER BY checkpoint_id ASC
                """,
                (run_id,),
            ).fetchall()
            return [self._row_to_checkpoint(row) for row in rows]

    def save_pass(
        self,
        *,
        run_id: str,
        phase: str,
        tool_turn: int = 0,
        status: str = "",
        objective: str = "",
        strategy: str = "",
        notes: str = "",
        selected_tools: Optional[list[str]] = None,
        tool_results: Optional[list[dict[str, Any]]] = None,
        compiled_context: Optional[dict[str, Any]] = None,
        response_preview: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: Optional[int] = None,
        estimated_cost_usd: float = 0.0,
    ) -> RunPassRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        safe_input_tokens = max(0, int(input_tokens or 0))
        safe_output_tokens = max(0, int(output_tokens or 0))
        safe_total_tokens = max(
            0,
            int(total_tokens if total_tokens is not None else safe_input_tokens + safe_output_tokens),
        )
        safe_cost = round(max(0.0, float(estimated_cost_usd or 0.0)), 8)
        with self._connect() as conn:
            cumulative_row = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM run_passes WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            cumulative_cost_usd = round(float(cumulative_row[0] or 0.0) + safe_cost, 8)
            cursor = conn.execute(
                """
                INSERT INTO run_passes
                (run_id, phase, tool_turn, status, objective, strategy, notes, selected_tools_json, tool_results_json, compiled_context_json, response_preview, input_tokens, output_tokens, total_tokens, estimated_cost_usd, cumulative_cost_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    phase,
                    tool_turn,
                    status,
                    objective,
                    strategy,
                    notes,
                    json.dumps(selected_tools or []),
                    json.dumps(tool_results or []),
                    json.dumps(compiled_context or {}),
                    response_preview,
                    safe_input_tokens,
                    safe_output_tokens,
                    safe_total_tokens,
                    safe_cost,
                    cumulative_cost_usd,
                    created_at,
                ),
            )
            conn.commit()
            pass_id = int(cursor.lastrowid)
        return RunPassRecord(
            pass_id=pass_id,
            run_id=run_id,
            phase=phase,
            tool_turn=tool_turn,
            status=status,
            objective=objective,
            strategy=strategy,
            notes=notes,
            selected_tools=list(selected_tools or []),
            tool_results=list(tool_results or []),
            compiled_context=dict(compiled_context or {}),
            response_preview=response_preview,
            input_tokens=safe_input_tokens,
            output_tokens=safe_output_tokens,
            total_tokens=safe_total_tokens,
            estimated_cost_usd=safe_cost,
            cumulative_cost_usd=cumulative_cost_usd,
            created_at=created_at,
        )

    def list_passes(self, run_id: str) -> list[RunPassRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_passes
                WHERE run_id = ?
                ORDER BY pass_id ASC
                """,
                (run_id,),
            ).fetchall()
            return [self._row_to_pass(row) for row in rows]

    def summarize_usage(self, run_id: str) -> dict[str, float | int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM run_passes
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return {
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "estimated_cost_usd": round(float(row["estimated_cost_usd"] or 0.0), 8),
        }

    def save_artifact(
        self,
        *,
        run_id: str,
        session_id: str,
        artifact_type: str,
        label: str,
        owner_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        content_hash: str = "",
        size_bytes: int = 0,
        preview: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> RunArtifact:
        created_at = datetime.now(timezone.utc).isoformat()
        artifact = RunArtifact(
            artifact_id=str(uuid.uuid4()),
            run_id=run_id,
            session_id=session_id,
            artifact_type=artifact_type,
            label=label,
            owner_id=owner_id,
            tool_name=tool_name,
            storage_path=storage_path,
            content_hash=content_hash,
            size_bytes=int(size_bytes or 0),
            preview=preview,
            metadata=dict(metadata or {}),
            created_at=created_at,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_artifacts
                (artifact_id, run_id, session_id, owner_id, tool_name, artifact_type, label, storage_path, content_hash, size_bytes, preview, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.session_id,
                    artifact.owner_id,
                    artifact.tool_name,
                    artifact.artifact_type,
                    artifact.label,
                    artifact.storage_path,
                    artifact.content_hash,
                    artifact.size_bytes,
                    artifact.preview,
                    json.dumps(artifact.metadata or {}),
                    artifact.created_at,
                ),
            )
            conn.commit()
        return artifact

    def list_artifacts(self, run_id: str) -> list[RunArtifact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_artifacts
                WHERE run_id = ?
                ORDER BY created_at ASC, artifact_id ASC
                """,
                (run_id,),
            ).fetchall()
            return [self._row_to_artifact(row) for row in rows]

    def save_eval(
        self,
        *,
        run_id: str,
        session_id: str,
        eval_type: str,
        phase: str,
        passed: bool,
        owner_id: Optional[str] = None,
        score: Optional[float] = None,
        detail: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> RunEval:
        created_at = datetime.now(timezone.utc).isoformat()
        evaluation = RunEval(
            eval_id=str(uuid.uuid4()),
            run_id=run_id,
            session_id=session_id,
            eval_type=eval_type,
            phase=phase,
            passed=bool(passed),
            owner_id=owner_id,
            score=float(score) if score is not None else None,
            detail=detail,
            metadata=dict(metadata or {}),
            created_at=created_at,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_evals
                (eval_id, run_id, session_id, owner_id, eval_type, phase, passed, score, detail, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation.eval_id,
                    evaluation.run_id,
                    evaluation.session_id,
                    evaluation.owner_id,
                    evaluation.eval_type,
                    evaluation.phase,
                    1 if evaluation.passed else 0,
                    evaluation.score,
                    evaluation.detail,
                    json.dumps(evaluation.metadata or {}),
                    evaluation.created_at,
                ),
            )
            conn.commit()
        return evaluation

    def list_evals(self, run_id: str) -> list[RunEval]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_evals
                WHERE run_id = ?
                ORDER BY created_at ASC, eval_id ASC
                """,
                (run_id,),
            ).fetchall()
            return [self._row_to_eval(row) for row in rows]

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> LoopCheckpoint:
        return LoopCheckpoint(
            checkpoint_id=int(row["checkpoint_id"]),
            run_id=row["run_id"],
            phase=row["phase"],
            tool_turn=int(row["tool_turn"] or 0),
            status=row["status"] or "",
            strategy=row["strategy"] or "",
            notes=row["notes"] or "",
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            tools=json.loads(row["tools_json"] or "[]"),
            tool_results=json.loads(row["tool_results_json"] or "[]"),
            candidate_facts=json.loads(row["candidate_facts_json"] or "[]"),
            created_at=row["created_at"] or "",
        )

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row) -> RunArtifact:
        return RunArtifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            owner_id=row["owner_id"],
            tool_name=row["tool_name"],
            artifact_type=row["artifact_type"],
            label=row["label"],
            storage_path=row["storage_path"],
            content_hash=row["content_hash"] or "",
            size_bytes=int(row["size_bytes"] or 0),
            preview=row["preview"] or "",
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"] or "",
        )

    @staticmethod
    def _row_to_eval(row: sqlite3.Row) -> RunEval:
        return RunEval(
            eval_id=row["eval_id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            owner_id=row["owner_id"],
            eval_type=row["eval_type"],
            phase=row["phase"],
            passed=bool(row["passed"]),
            score=float(row["score"]) if row["score"] is not None else None,
            detail=row["detail"] or "",
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"] or "",
        )

    @staticmethod
    def _row_to_pass(row: sqlite3.Row) -> RunPassRecord:
        return RunPassRecord(
            pass_id=int(row["pass_id"]),
            run_id=row["run_id"],
            phase=row["phase"],
            tool_turn=int(row["tool_turn"] or 0),
            status=row["status"] or "",
            objective=row["objective"] or "",
            strategy=row["strategy"] or "",
            notes=row["notes"] or "",
            selected_tools=json.loads(row["selected_tools_json"] or "[]"),
            tool_results=json.loads(row["tool_results_json"] or "[]"),
            compiled_context=json.loads(row["compiled_context_json"] or "{}"),
            response_preview=row["response_preview"] or "",
            input_tokens=int(row["input_tokens"] or 0),
            output_tokens=int(row["output_tokens"] or 0),
            total_tokens=int(row["total_tokens"] or 0),
            estimated_cost_usd=float(row["estimated_cost_usd"] or 0.0),
            cumulative_cost_usd=float(row["cumulative_cost_usd"] or 0.0),
            created_at=row["created_at"] or "",
        )


_RUN_STORE = RunStore()


def get_run_store() -> RunStore:
    return _RUN_STORE
