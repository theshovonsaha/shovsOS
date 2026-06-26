from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException

from run_engine.types import RunEngineRequest


DEFAULT_WORKFLOW_RUNS_DB = "workflow_runs.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


@dataclass(frozen=True)
class WorkflowStepDefinition:
    id: str
    label: str
    kind: str
    description: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["consumes"] = list(self.consumes)
        data["produces"] = list(self.produces)
        data["gates"] = list(self.gates)
        return data


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    label: str
    description: str
    template: str
    policy: str
    ledger_mode: str
    context_mode: str
    prompt_version: str
    risk_policy: str
    tools: tuple[str, ...]
    memory_policy: dict[str, Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    steps: tuple[WorkflowStepDefinition, ...]
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "template": self.template,
            "runtime": {
                "policy": self.policy,
                "ledger_mode": self.ledger_mode,
                "context_mode": self.context_mode,
                "prompt_version": self.prompt_version,
                "risk_policy": self.risk_policy,
            },
            "tools": list(self.tools),
            "memory_policy": deepcopy(self.memory_policy),
            "input_schema": deepcopy(self.input_schema),
            "output_schema": deepcopy(self.output_schema),
            "steps": [step.to_dict() for step in self.steps],
            "tags": list(self.tags),
        }


COMMON_SOURCE_STEPS = (
    WorkflowStepDefinition(
        id="intake",
        label="Input",
        kind="input",
        description="Normalize the request into objective, location, entities, and constraints.",
        consumes=("external_input",),
        produces=("objective", "constraints"),
    ),
    WorkflowStepDefinition(
        id="context",
        label="Context Build",
        kind="context",
        description="Build the small context ladder: active objective, memory signal, contract, and evidence refs.",
        consumes=("objective", "memory_policy"),
        produces=("phase_packet", "context_ladder"),
        gates=("hide_stale_memory",),
    ),
    WorkflowStepDefinition(
        id="plan",
        label="Plan",
        kind="prompt",
        description="Commit a source contract before acting: locked entities, required searches, required fetches.",
        consumes=("phase_packet",),
        produces=("source_contract", "next_required_action"),
        gates=("entity_lock_required",),
    ),
    WorkflowStepDefinition(
        id="tools",
        label="Tools",
        kind="tool",
        description="Run only allowed tools and validate each call against the source contract.",
        consumes=("next_required_action",),
        produces=("tool_results", "evidence_refs"),
        gates=("reject_off_contract_fetch", "reject_unlocked_entity_search"),
    ),
    WorkflowStepDefinition(
        id="evidence",
        label="Evidence",
        kind="evidence",
        description="Select successful tool results as evidence and record missing coverage explicitly.",
        consumes=("tool_results",),
        produces=("selected_evidence", "coverage_report"),
    ),
    WorkflowStepDefinition(
        id="memory",
        label="Memory",
        kind="memory",
        description="Read useful preferences and write only provenance-backed facts or candidates.",
        consumes=("selected_evidence", "memory_policy"),
        produces=("memory_writes",),
        gates=("provenance_required",),
    ),
    WorkflowStepDefinition(
        id="verify",
        label="Verification",
        kind="verify",
        description="Block final response if required evidence or source coverage is missing.",
        consumes=("selected_evidence", "coverage_report"),
        produces=("verification", "completion_gate"),
        gates=("no_final_with_missing_evidence",),
    ),
    WorkflowStepDefinition(
        id="response",
        label="Response",
        kind="response",
        description="Return the configured output schema with source IDs and unresolved slots.",
        consumes=("verification", "selected_evidence"),
        produces=("result",),
    ),
)


WORKFLOW_DEFINITIONS: dict[str, WorkflowDefinition] = {
    "research_brief_v1": WorkflowDefinition(
        id="research_brief_v1",
        label="Research Brief",
        description="Evidence-first brief that searches, fetches, verifies, and reports source-backed findings.",
        template="research_agent_v1",
        policy="plan_execute",
        ledger_mode="ledger_enforced",
        context_mode="ladder",
        prompt_version="role_contracts_v1",
        risk_policy="evidence_first",
        tools=("web_search", "web_fetch", "source_collect", "source_contract", "source_select", "source_coverage", "query_memory"),
        memory_policy={"read": ["topic_preferences"], "write": "verified_facts_only", "provenance_required": True},
        input_schema={"query": "string", "source_count": "number", "freshness": "optional string"},
        output_schema={"type": "research_brief", "fields": ["summary", "findings", "sources", "confidence", "unresolved"]},
        steps=COMMON_SOURCE_STEPS,
        tags=("research", "sources", "verification"),
    ),
    "shopping_comparison_v1": WorkflowDefinition(
        id="shopping_comparison_v1",
        label="Shopping Comparison",
        description="Compare products or stores using locked entities, fetched source pages, and explicit tradeoffs.",
        template="shopping_advisor_v1",
        policy="plan_execute",
        ledger_mode="ledger_enforced",
        context_mode="ladder",
        prompt_version="shopping_patch_v1",
        risk_policy="consumer_verified",
        tools=("web_search", "web_fetch", "shopping_advice", "source_collect", "source_contract", "source_select", "source_coverage", "query_memory"),
        memory_policy={"read": ["location_preferences", "store_preferences"], "write": "explicit_user_preferences_only", "provenance_required": True},
        input_schema={"product": "string", "location": "optional string", "stores": "optional string[]", "budget": "optional string"},
        output_schema={"type": "comparison_table", "fields": ["option", "price", "match_quality", "rating", "source_urls", "recommendation"]},
        steps=COMMON_SOURCE_STEPS,
        tags=("shopping", "consumer", "comparison"),
    ),
    "local_recommendation_v1": WorkflowDefinition(
        id="local_recommendation_v1",
        label="Local Recommendation",
        description="Find local places, lock candidates, fetch review/menu/source pages, and summarize practical fit.",
        template="research_agent_v1",
        policy="plan_execute",
        ledger_mode="ledger_enforced",
        context_mode="ladder",
        prompt_version="role_contracts_v1",
        risk_policy="consumer_verified",
        tools=("web_search", "web_fetch", "source_collect", "source_contract", "source_select", "source_coverage", "query_memory"),
        memory_policy={"read": ["location_preferences", "taste_preferences"], "write": "explicit_user_preferences_only", "provenance_required": True},
        input_schema={"category": "string", "location": "string", "constraints": "optional string"},
        output_schema={"type": "ranked_local_options", "fields": ["place", "why", "best_for", "cautions", "source_urls"]},
        steps=COMMON_SOURCE_STEPS,
        tags=("local", "places", "recommendation"),
    ),
    "vision_inspection_v1": WorkflowDefinition(
        id="vision_inspection_v1",
        label="Vision Inspection",
        description="Analyze attached images with a vision-capable model, extract visible details, and answer a focused question.",
        template="research_agent_v1",
        policy="react",
        ledger_mode="ledger_enforced",
        context_mode="ladder",
        prompt_version="role_contracts_v1",
        risk_policy="evidence_first",
        tools=("query_memory",),
        memory_policy={"read": ["visual_preferences"], "write": "explicit_user_preferences_only", "provenance_required": True},
        input_schema={"question": "string", "images": "optional base64[]", "focus": "optional string"},
        output_schema={"type": "vision_inspection", "fields": ["answer", "visible_details", "visible_text", "uncertainties", "followups"]},
        steps=(
            COMMON_SOURCE_STEPS[0],
            COMMON_SOURCE_STEPS[1],
            WorkflowStepDefinition(
                id="inspect",
                label="Inspect Image",
                kind="model",
                description="Use a vision-capable model to inspect attached image content without inventing unseen details.",
                consumes=("objective", "images"),
                produces=("visual_observations",),
                gates=("vision_model_required", "no_unseen_detail_claims"),
            ),
            COMMON_SOURCE_STEPS[6],
            COMMON_SOURCE_STEPS[7],
        ),
        tags=("vision", "images", "inspection"),
    ),
    "memory_fact_guard_v1": WorkflowDefinition(
        id="memory_fact_guard_v1",
        label="Memory Fact Guard",
        description="Inspect, dispute, demote, and commit memory facts with provenance and contradiction traces.",
        template="memory_agent_v1",
        policy="graph_harness",
        ledger_mode="ledger_enforced",
        context_mode="ladder",
        prompt_version="role_contracts_v1",
        risk_policy="memory_strict",
        tools=("query_memory", "store_memory", "shovs_memory_query", "shovs_memory_store", "shovs_list_loci"),
        memory_policy={"read": ["current_facts", "candidates", "disputes"], "write": "provenance_required", "provenance_required": True},
        input_schema={"claim": "string", "source": "optional string", "owner_scope": "optional string"},
        output_schema={"type": "memory_audit", "fields": ["current", "candidate", "superseded", "conflicts", "decision"]},
        steps=(
            COMMON_SOURCE_STEPS[0],
            COMMON_SOURCE_STEPS[1],
            WorkflowStepDefinition(
                id="inspect",
                label="Inspect Memory",
                kind="memory",
                description="Read current facts, candidates, and dispute traces before deciding.",
                consumes=("claim", "memory_policy"),
                produces=("memory_state",),
            ),
            COMMON_SOURCE_STEPS[6],
            COMMON_SOURCE_STEPS[7],
        ),
        tags=("memory", "facts", "provenance"),
    ),
    "coding_patch_eval_v1": WorkflowDefinition(
        id="coding_patch_eval_v1",
        label="Coding Patch Eval",
        description="Inspect files, make scoped changes, run tests, and return patch evidence.",
        template="coding_agent_v1",
        policy="graph_harness",
        ledger_mode="ledger_enforced",
        context_mode="ladder",
        prompt_version="role_contracts_v1",
        risk_policy="change_controlled",
        tools=("file_view", "file_str_replace", "file_create", "bash", "query_memory"),
        memory_policy={"read": ["project_preferences"], "write": "explicit_project_lessons_only", "provenance_required": True},
        input_schema={"task": "string", "workspace_path": "optional string", "test_command": "optional string"},
        output_schema={"type": "patch_report", "fields": ["changed_files", "tests", "risks", "next_steps"]},
        steps=(
            COMMON_SOURCE_STEPS[0],
            COMMON_SOURCE_STEPS[1],
            WorkflowStepDefinition(
                id="inspect",
                label="Inspect Files",
                kind="tool",
                description="Read relevant files before editing.",
                consumes=("objective",),
                produces=("file_context",),
                gates=("no_edit_before_inspect",),
            ),
            WorkflowStepDefinition(
                id="patch",
                label="Patch",
                kind="tool",
                description="Apply scoped file changes.",
                consumes=("file_context",),
                produces=("patch_artifacts",),
            ),
            WorkflowStepDefinition(
                id="test",
                label="Test",
                kind="tool",
                description="Run relevant verification commands and store outputs.",
                consumes=("patch_artifacts",),
                produces=("test_results",),
            ),
            COMMON_SOURCE_STEPS[6],
            COMMON_SOURCE_STEPS[7],
        ),
        tags=("coding", "tests", "patch"),
    ),
}


class WorkflowRunStore:
    def __init__(self, db_path: str = DEFAULT_WORKFLOW_RUNS_DB):
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
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    owner_id TEXT,
                    callback_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    definition_json TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    phase TEXT,
                    status TEXT,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    event_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_events_run_id ON workflow_events(run_id, event_index)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_runs_owner_id ON workflow_runs(owner_id)")
            conn.commit()

    def create_run(
        self,
        *,
        definition: WorkflowDefinition,
        input_payload: dict[str, Any],
        mode: str,
        owner_id: str = "",
        callback_url: str = "",
        status: str = "created",
        run_id: str = "",
    ) -> dict[str, Any]:
        now = _now()
        record = {
            "run_id": run_id or _id("wfr"),
            "workflow_id": definition.id,
            "status": status,
            "mode": mode,
            "owner_id": owner_id,
            "callback_url": callback_url,
            "created_at": now,
            "updated_at": now,
            "error": "",
            "definition": definition.to_dict(),
            "input": deepcopy(input_payload),
            "events": [],
            "result": None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workflow_runs
                (run_id, workflow_id, status, mode, owner_id, callback_url, created_at, updated_at, error, definition_json, input_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["run_id"],
                    record["workflow_id"],
                    record["status"],
                    record["mode"],
                    record["owner_id"],
                    record["callback_url"],
                    record["created_at"],
                    record["updated_at"],
                    record["error"],
                    _json_dumps(record["definition"]),
                    _json_dumps(record["input"]),
                    None,
                ),
            )
            conn.commit()
        return record

    def update_status(self, run_id: str, status: str, *, error: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE workflow_runs SET status = ?, updated_at = ?, error = ? WHERE run_id = ?",
                (status, _now(), error, run_id),
            )
            conn.commit()

    def append_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        event_payload = dict(event)
        event_payload.setdefault("id", _id("wfe"))
        event_payload.setdefault("run_id", run_id)
        event_payload.setdefault("created_at", _now())
        event_payload.setdefault("status", "info")
        event_payload.setdefault("phase", "")
        event_payload.setdefault("event_type", "workflow_event")
        event_payload.setdefault("summary", "")
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(event_index), -1) + 1 AS idx FROM workflow_events WHERE run_id = ?", (run_id,)).fetchone()
            event_index = int(row["idx"] if row else 0)
            event_payload["index"] = event_index
            conn.execute(
                """
                INSERT INTO workflow_events
                (event_id, run_id, event_index, event_type, phase, status, summary, created_at, event_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_payload["id"]),
                    run_id,
                    event_index,
                    str(event_payload.get("event_type") or ""),
                    str(event_payload.get("phase") or ""),
                    str(event_payload.get("status") or ""),
                    str(event_payload.get("summary") or ""),
                    str(event_payload.get("created_at") or _now()),
                    _json_dumps(event_payload),
                ),
            )
            conn.execute("UPDATE workflow_runs SET updated_at = ? WHERE run_id = ?", (_now(), run_id))
            conn.commit()
        return event_payload

    def set_result(self, run_id: str, result: dict[str, Any], *, status: str = "completed") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE workflow_runs SET result_json = ?, status = ?, updated_at = ?, error = '' WHERE run_id = ?",
                (_json_dumps(result), status, _now(), run_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            record = dict(row)
        events = self.list_events(run_id)
        return {
            "run_id": record["run_id"],
            "workflow_id": record["workflow_id"],
            "status": record["status"],
            "mode": record["mode"],
            "owner_id": record.get("owner_id") or "",
            "callback_url": record.get("callback_url") or "",
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "error": record.get("error") or "",
            "definition": _json_loads(record.get("definition_json"), {}),
            "input": _json_loads(record.get("input_json"), {}),
            "events": events,
            "result": _json_loads(record.get("result_json"), None),
        }

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_json FROM workflow_events WHERE run_id = ? ORDER BY event_index ASC",
                (run_id,),
            ).fetchall()
        return [_json_loads(row["event_json"], {}) for row in rows]


DEFAULT_WORKFLOW_RUN_STORE = WorkflowRunStore()


def list_workflow_definitions() -> list[dict[str, Any]]:
    return [definition.to_dict() for definition in WORKFLOW_DEFINITIONS.values()]


def get_workflow_definition(workflow_id: str) -> WorkflowDefinition:
    definition = WORKFLOW_DEFINITIONS.get(str(workflow_id or "").strip())
    if definition is None:
        raise KeyError(workflow_id)
    return definition


def _compact_input(payload: dict[str, Any]) -> str:
    parts = []
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        rendered = ", ".join(str(item) for item in value[:4]) if isinstance(value, list) else str(value)
        parts.append(f"{key}: {rendered}")
    return "; ".join(parts) or "empty input"


def _input_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("input"), dict):
        data = dict(payload.get("input") or {})
    else:
        data = {
            key: value
            for key, value in payload.items()
            if key not in {"owner_id", "callback_url", "mode", "session_id", "agent_id", "model", "images", "image_base64"}
        }
    raw_images = _image_payload(payload, data)
    data.pop("images", None)
    data.pop("image_base64", None)
    if raw_images:
        data["image_count"] = len(raw_images)
    return data


def _image_payload(payload: dict[str, Any], input_payload: dict[str, Any]) -> list[str]:
    candidates: Any = payload.get("images")
    raw_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    if candidates is None:
        candidates = raw_input.get("images") or input_payload.get("images")
    if candidates is None and (raw_input.get("image_base64") or input_payload.get("image_base64")):
        candidates = [raw_input.get("image_base64") or input_payload.get("image_base64")]
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, list):
        return []
    return [str(item).strip() for item in candidates if str(item or "").strip()]


def _workflow_event(
    *,
    run_id: str,
    workflow_id: str,
    phase: str,
    event_type: str,
    status: str,
    summary: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "id": _id("wfe"),
        "run_id": run_id,
        "workflow_id": workflow_id,
        "phase": phase,
        "event_type": event_type,
        "status": status,
        "created_at": _now(),
        "summary": summary,
        "data": dict(data or {}),
    }


def _build_workflow_events(definition: WorkflowDefinition, run_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    objective = _compact_input(payload)
    for index, step in enumerate(definition.steps):
        status = "success"
        summary = step.description
        data: dict[str, Any] = {
            "step_id": step.id,
            "step_label": step.label,
            "kind": step.kind,
            "consumes": list(step.consumes),
            "produces": list(step.produces),
            "gates": list(step.gates),
        }
        if step.id == "intake":
            summary = f"Accepted input: {objective}"
            data["input"] = deepcopy(payload)
        elif step.id == "plan":
            summary = f"Selected {definition.policy} with {definition.ledger_mode}."
            data["policy"] = definition.policy
            data["ledger_mode"] = definition.ledger_mode
        elif step.id == "tools":
            summary = f"Allowed tools: {', '.join(definition.tools[:6])}."
            data["allowed_tools"] = list(definition.tools)
        elif step.id == "memory":
            summary = "Memory reads/writes are governed by the workflow memory policy."
            data["memory_policy"] = deepcopy(definition.memory_policy)
        elif step.id == "verify":
            summary = "Completion gate requires produced outputs and policy checks before response."
            data["completion_gate"] = {"final_answer_allowed": True, "reason": "deterministic_workflow_contract_satisfied"}
        event = _workflow_event(
            run_id=run_id,
            workflow_id=definition.id,
            phase=step.id,
            event_type=f"workflow_{step.kind}",
            status=status,
            summary=summary,
            data=data,
        )
        event["index"] = index
        events.append(event)
    return events


def _build_workflow_result(
    definition: WorkflowDefinition,
    run_id: str,
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    final_text: str = "",
    run_engine_run_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    output_fields = list(definition.output_schema.get("fields") or [])
    live = bool(run_engine_run_id or final_text)
    fields = {}
    for field in output_fields:
        if field in {"summary", "recommendation", "decision", "next_steps"} and final_text:
            fields[field] = {"status": "ready", "value": final_text}
        elif live:
            fields[field] = {"status": "not_recorded", "value": "not recorded by live run output parser"}
        else:
            fields[field] = {
                "status": "not_recorded" if field in {"price", "rating", "source_urls"} else "ready",
                "value": "not recorded in deterministic preview" if field in {"price", "rating", "source_urls"} else f"derived {field}",
            }
    return {
        "run_id": run_id,
        "workflow_id": definition.id,
        "status": "completed",
        "output_schema": deepcopy(definition.output_schema),
        "summary": final_text.strip() or f"{definition.label} completed through {len(events)} typed steps.",
        "fields": fields,
        "api_contract": {
            "start": f"POST /workflow-lab/workflows/{definition.id}/runs",
            "status": f"GET /workflow-lab/runs/{run_id}",
            "events": f"GET /workflow-lab/runs/{run_id}/events",
            "result": f"GET /workflow-lab/runs/{run_id}/result",
        },
        "input_echo": deepcopy(payload),
        "run_engine_run_id": run_engine_run_id,
        "session_id": session_id,
        "trace_summary": {
            "event_count": len(events),
            "policy": definition.policy,
            "ledger_mode": definition.ledger_mode,
            "tools": list(definition.tools),
            "live_run_engine": live,
        },
    }


def start_workflow_run(
    workflow_id: str,
    payload: dict[str, Any],
    *,
    store: WorkflowRunStore | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    definition = get_workflow_definition(workflow_id)
    store = store or DEFAULT_WORKFLOW_RUN_STORE
    input_payload = _input_payload(payload)
    mode = str(payload.get("mode") or "deterministic_contract")
    record = store.create_run(
        definition=definition,
        input_payload=input_payload,
        mode=mode,
        owner_id=str(payload.get("owner_id") or ""),
        callback_url=str(payload.get("callback_url") or ""),
        status="running" if status == "completed" else status,
    )
    events = _build_workflow_events(definition, record["run_id"], input_payload)
    for event in events:
        store.append_event(record["run_id"], event)
    if status == "completed":
        result = _build_workflow_result(definition, record["run_id"], input_payload, events)
        store.set_result(record["run_id"], result, status="completed")
    else:
        store.update_status(record["run_id"], status)
    return store.get_run(record["run_id"]) or record


def workflow_catalog() -> dict[str, Any]:
    return {
        "title": "Workflow Lab",
        "subtitle": "Compose ShovsOS features into typed agent workflows that external apps can run.",
        "workflows": list_workflow_definitions(),
        "lifecycle": ["created", "queued", "running", "waiting", "failed", "completed"],
        "run_modes": ["deterministic_contract", "live_run_engine"],
        "api": {
            "list": "GET /workflow-lab/catalog",
            "start": "POST /workflow-lab/workflows/{workflow_id}/runs",
            "status": "GET /workflow-lab/runs/{run_id}",
            "events": "GET /workflow-lab/runs/{run_id}/events",
            "result": "GET /workflow-lab/runs/{run_id}/result",
        },
        "legal_usage": [
            "Use workflows as automation and decision-support, not regulated final decisions without human review.",
            "Do not store sensitive personal data unless the workflow memory policy explicitly allows it.",
            "Third-party source content must keep attribution through source URLs or evidence IDs.",
            "If required evidence is not recorded, clients should display not recorded instead of guessing.",
        ],
    }


def _workflow_objective(definition: WorkflowDefinition, payload: dict[str, Any]) -> str:
    return (
        f"Run workflow {definition.id} ({definition.label}). "
        f"Use the workflow contract, allowed tools, memory policy, and completion gates. "
        f"Input JSON: {_json_dumps(payload)}"
    )


def _agent_id_for_definition(definition: WorkflowDefinition, requested: str = "") -> str:
    if requested:
        return requested
    return {
        "shopping_advisor_v1": "shopping-advisor",
        "research_agent_v1": "researcher",
        "coding_agent_v1": "coder",
        "memory_agent_v1": "default",
    }.get(definition.template, "default")


def _event_from_run_engine(
    *,
    run_id: str,
    workflow_id: str,
    raw: dict[str, Any],
    token_index: int,
) -> dict[str, Any]:
    raw_type = str(raw.get("type") or raw.get("event_type") or "event")
    status = "success"
    phase = str(raw.get("phase") or raw_type)
    summary = raw_type.replace("_", " ")
    if raw_type == "token":
        summary = "Received answer token."
        phase = "response"
    elif raw_type == "tool_call":
        summary = f"Tool call: {raw.get('tool') or raw.get('tool_name') or 'tool'}"
        phase = "tools"
    elif raw_type == "tool_result":
        summary = f"Tool result: {raw.get('tool') or raw.get('tool_name') or 'tool'}"
        phase = "tools"
    elif raw_type == "done":
        summary = "RunEngine completed."
        phase = "response"
    elif raw_type == "error":
        summary = str(raw.get("message") or "RunEngine error")
        phase = "error"
        status = "error"
    return _workflow_event(
        run_id=run_id,
        workflow_id=workflow_id,
        phase=phase,
        event_type=f"run_engine_{raw_type}",
        status=status,
        summary=summary if raw_type != "token" else f"Received answer token {token_index}.",
        data={"raw": raw},
    )


async def execute_live_workflow_run(
    *,
    run_id: str,
    store: WorkflowRunStore,
    run_engine: Any,
    definition: WorkflowDefinition,
    input_payload: dict[str, Any],
    owner_id: str = "",
    session_id: str = "",
    agent_id: str = "",
    model: str = "",
    system_prompt: str = "",
    max_tool_calls: Optional[int] = None,
    max_turns: Optional[int] = None,
    images: Optional[list[str]] = None,
) -> None:
    if run_engine is None:
        store.update_status(run_id, "failed", error="live_run_engine mode requires RunEngine")
        store.append_event(
            run_id,
            _workflow_event(
                run_id=run_id,
                workflow_id=definition.id,
                phase="runtime",
                event_type="workflow_runtime_missing",
                status="error",
                summary="RunEngine dependency was not available.",
            ),
        )
        return
    store.update_status(run_id, "running")
    store.append_event(
        run_id,
        _workflow_event(
            run_id=run_id,
            workflow_id=definition.id,
            phase="runtime",
            event_type="workflow_live_run_started",
            status="running",
            summary="Live RunEngine execution started.",
            data={"policy": definition.policy, "ledger_mode": definition.ledger_mode},
        ),
    )
    final_text = ""
    run_engine_run_id = ""
    effective_session_id = session_id or f"workflow-{run_id}"
    try:
        request = RunEngineRequest(
            session_id=effective_session_id,
            owner_id=owner_id,
            agent_id=_agent_id_for_definition(definition, agent_id),
            user_message=_workflow_objective(definition, input_payload),
            model=model or "llama3.2",
            system_prompt=system_prompt,
            context_mode="v3" if definition.context_mode == "ladder" else definition.context_mode,
            allowed_tools=tuple(definition.tools),
            use_planner=True,
            max_tool_calls=max_tool_calls,
            max_turns=max_turns,
            workflow_template=definition.template,
            prompt_version=definition.prompt_version,
            risk_policy=definition.risk_policy,
            ledger_mode=definition.ledger_mode,
            control_policy=definition.policy,
            images=images or None,
        )
        token_index = 0
        async for raw in run_engine.stream(request):
            if not isinstance(raw, dict):
                continue
            raw_type = str(raw.get("type") or "")
            if raw_type == "token":
                token_index += 1
                final_text += str(raw.get("content") or "")
                if token_index > 25 and token_index % 20 != 0:
                    continue
            if raw_type == "session":
                effective_session_id = str(raw.get("session_id") or effective_session_id)
                run_engine_run_id = str(raw.get("run_id") or run_engine_run_id)
            if raw_type == "done":
                run_engine_run_id = str(raw.get("run_id") or run_engine_run_id)
                effective_session_id = str(raw.get("session_id") or effective_session_id)
            store.append_event(
                run_id,
                _event_from_run_engine(
                    run_id=run_id,
                    workflow_id=definition.id,
                    raw=raw,
                    token_index=token_index,
                ),
            )
        events = store.list_events(run_id)
        result = _build_workflow_result(
            definition,
            run_id,
            input_payload,
            events,
            final_text=final_text,
            run_engine_run_id=run_engine_run_id,
            session_id=effective_session_id,
        )
        store.set_result(run_id, result, status="completed")
        store.append_event(
            run_id,
            _workflow_event(
                run_id=run_id,
                workflow_id=definition.id,
                phase="runtime",
                event_type="workflow_live_run_completed",
                status="success",
                summary="Live RunEngine execution completed.",
                data={"run_engine_run_id": run_engine_run_id, "session_id": effective_session_id},
            ),
        )
    except asyncio.CancelledError:
        store.update_status(run_id, "failed", error="workflow run cancelled")
        raise
    except Exception as exc:
        store.update_status(run_id, "failed", error=str(exc))
        store.append_event(
            run_id,
            _workflow_event(
                run_id=run_id,
                workflow_id=definition.id,
                phase="runtime",
                event_type="workflow_live_run_failed",
                status="error",
                summary=str(exc),
            ),
        )


def _run_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "workflow_id": record["workflow_id"],
        "status": record["status"],
        "mode": record["mode"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "event_count": len(record.get("events") or []),
        "result_ready": bool(record.get("result")),
        "error": record.get("error") or "",
        "definition": record["definition"],
    }


def make_workflow_lab_router(
    *,
    run_engine: Any = None,
    profile_manager: Any = None,
    store: WorkflowRunStore | None = None,
    default_model: str = "llama3.2",
) -> APIRouter:
    store = store or DEFAULT_WORKFLOW_RUN_STORE
    router = APIRouter(prefix="/workflow-lab", tags=["workflow-lab"])

    @router.get("/catalog")
    async def catalog():
        return workflow_catalog()

    @router.get("/workflows/{workflow_id}")
    async def workflow_detail(workflow_id: str):
        try:
            return get_workflow_definition(workflow_id).to_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail="Workflow not found")

    @router.post("/workflows/{workflow_id}/runs")
    async def create_run(
        workflow_id: str,
        background_tasks: BackgroundTasks,
        payload: dict[str, Any] = Body(default_factory=dict),
    ):
        try:
            definition = get_workflow_definition(workflow_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Workflow not found")
        payload = payload or {}
        mode = str(payload.get("mode") or "deterministic_contract")
        owner_id = str(payload.get("owner_id") or "")
        callback_url = str(payload.get("callback_url") or "")
        input_payload = _input_payload(payload)
        images = _image_payload(payload, input_payload)
        if mode == "live_run_engine":
            record = store.create_run(
                definition=definition,
                input_payload=input_payload,
                mode=mode,
                owner_id=owner_id,
                callback_url=callback_url,
                status="queued",
            )
            profile = None
            agent_id = _agent_id_for_definition(definition, str(payload.get("agent_id") or ""))
            if profile_manager is not None:
                try:
                    profile = profile_manager.get(agent_id, owner_id=owner_id) or profile_manager.get("default", owner_id=owner_id)
                except Exception:
                    profile = None
            background_tasks.add_task(
                execute_live_workflow_run,
                run_id=record["run_id"],
                store=store,
                run_engine=run_engine,
                definition=definition,
                input_payload=input_payload,
                owner_id=owner_id,
                session_id=str(payload.get("session_id") or ""),
                agent_id=agent_id,
                model=str(payload.get("model") or getattr(profile, "model", None) or default_model or "llama3.2"),
                system_prompt=str(payload.get("system_prompt") or getattr(profile, "system_prompt", "") or definition.description),
                max_tool_calls=payload.get("max_tool_calls") if isinstance(payload.get("max_tool_calls"), int) else None,
                max_turns=payload.get("max_turns") if isinstance(payload.get("max_turns"), int) else None,
                images=images,
            )
            record = store.get_run(record["run_id"]) or record
        else:
            record = start_workflow_run(workflow_id, payload, store=store)
        return {
            "run_id": record["run_id"],
            "workflow_id": record["workflow_id"],
            "status": record["status"],
            "mode": record["mode"],
            "created_at": record["created_at"],
            "status_url": f"/workflow-lab/runs/{record['run_id']}",
            "events_url": f"/workflow-lab/runs/{record['run_id']}/events",
            "result_url": f"/workflow-lab/runs/{record['run_id']}/result",
        }

    @router.get("/runs/{run_id}")
    async def run_status(run_id: str):
        record = store.get_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail="Workflow run not found")
        return _run_summary(record)

    @router.get("/runs/{run_id}/events")
    async def run_events(run_id: str):
        record = store.get_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail="Workflow run not found")
        return {"run_id": run_id, "events": record.get("events") or []}

    @router.get("/runs/{run_id}/result")
    async def run_result(run_id: str):
        record = store.get_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail="Workflow run not found")
        if not record.get("result"):
            raise HTTPException(status_code=409, detail="Workflow result is not ready")
        return record["result"]

    return router
