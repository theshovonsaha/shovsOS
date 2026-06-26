from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from run_engine.workflow_contracts import WorkflowContract


PASS_FRAMEWORK_VERSION = "agent-pass-framework-v1"
PASS_GRAPH_EXECUTION_VERSION = "pass-graph-execution-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentPassSpec:
    id: str
    role: str
    purpose: str
    context_strategy: str
    allowed_inputs: list[str] = field(default_factory=list)
    required_outputs: list[str] = field(default_factory=list)
    model_policy: str = "deterministic_or_small"
    temperature: float = 0.0
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PassEdge:
    source: str
    target: str
    condition: str = "always"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PassGraph:
    id: str
    version: str
    workflow_shape: str
    context_strategy: str
    passes: list[AgentPassSpec]
    edges: list[PassEdge]
    stop_condition: str
    max_rounds: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "workflow_shape": self.workflow_shape,
            "context_strategy": self.context_strategy,
            "passes": [item.to_dict() for item in self.passes],
            "edges": [item.to_dict() for item in self.edges],
            "stop_condition": self.stop_condition,
            "max_rounds": self.max_rounds,
            "metadata": dict(self.metadata),
        }

    def render(self) -> str:
        lines = [
            "Agent Pass Framework:",
            f"- graph: {self.id}",
            f"- workflow shape: {self.workflow_shape}",
            f"- context strategy: {self.context_strategy}",
            f"- stop condition: {self.stop_condition}",
            f"- max rounds: {self.max_rounds}",
            "Passes:",
        ]
        for item in self.passes[:10]:
            outputs = ", ".join(item.required_outputs) or "none"
            lines.append(
                f"- {item.id} [{item.role}] {item.context_strategy}: {item.purpose} -> {outputs}"
            )
        return "\n".join(lines)


@dataclass
class PassNodeState:
    id: str
    role: str
    status: str = "pending"
    attempts: int = 0
    depends_on: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PassGraphExecution:
    graph_id: str
    version: str
    status: str
    current_node_id: str = ""
    nodes: list[PassNodeState] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "version": self.version,
            "status": self.status,
            "current_node_id": self.current_node_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": [node.to_dict() for node in self.nodes],
            "summary": {
                "pending": len([node for node in self.nodes if node.status == "pending"]),
                "running": len([node for node in self.nodes if node.status == "running"]),
                "completed": len([node for node in self.nodes if node.status == "completed"]),
                "failed": len([node for node in self.nodes if node.status == "failed"]),
                "blocked": len([node for node in self.nodes if node.status == "blocked"]),
            },
        }

    def node(self, node_id: str) -> PassNodeState | None:
        return next((node for node in self.nodes if node.id == node_id), None)

    def next_pending(self) -> PassNodeState | None:
        completed = {node.id for node in self.nodes if node.status == "completed"}
        for node in self.nodes:
            if node.status != "pending":
                continue
            if all(dep in completed for dep in node.depends_on):
                return node
        return None


def initialize_pass_graph_execution(graph: PassGraph) -> PassGraphExecution:
    dependencies: dict[str, list[str]] = {item.id: [] for item in graph.passes}
    for edge in graph.edges:
        dependencies.setdefault(edge.target, []).append(edge.source)
    return PassGraphExecution(
        graph_id=graph.id,
        version=PASS_GRAPH_EXECUTION_VERSION,
        status="pending",
        nodes=[
            PassNodeState(
                id=item.id,
                role=item.role,
                depends_on=list(dependencies.get(item.id, [])),
            )
            for item in graph.passes
        ],
    )


def start_next_pass_node(execution: PassGraphExecution) -> PassNodeState | None:
    running = next((node for node in execution.nodes if node.status == "running"), None)
    if running is not None:
        return running
    node = execution.next_pending()
    if node is None:
        execution.status = "completed" if all(item.status == "completed" for item in execution.nodes) else execution.status
        execution.current_node_id = ""
        execution.updated_at = _now()
        return None
    node.status = "running"
    node.attempts += 1
    node.started_at = node.started_at or _now()
    execution.status = "running"
    execution.current_node_id = node.id
    execution.updated_at = _now()
    return node


def complete_pass_node(
    execution: PassGraphExecution,
    node_id: str,
    *,
    outputs: dict[str, Any] | None = None,
) -> PassNodeState:
    node = execution.node(node_id)
    if node is None:
        raise ValueError(f"Unknown pass node: {node_id}")
    node.status = "completed"
    node.outputs.update(dict(outputs or {}))
    node.completed_at = _now()
    execution.updated_at = _now()
    next_node = execution.next_pending()
    execution.current_node_id = next_node.id if next_node else ""
    if next_node is None and all(item.status == "completed" for item in execution.nodes):
        execution.status = "completed"
    return node


def fail_pass_node(
    execution: PassGraphExecution,
    node_id: str,
    *,
    issue: str,
    blocked: bool = False,
) -> PassNodeState:
    node = execution.node(node_id)
    if node is None:
        raise ValueError(f"Unknown pass node: {node_id}")
    node.status = "blocked" if blocked else "failed"
    if issue:
        node.issues.append(str(issue))
    node.completed_at = _now()
    execution.status = node.status
    execution.current_node_id = node.id
    execution.updated_at = _now()
    return node


def build_pass_graph(contract: WorkflowContract | None, *, workflow_template: str = "") -> PassGraph:
    shape = getattr(contract, "workflow_shape", "") if contract is not None else ""
    if shape == "source_collection":
        return _source_collection_graph(contract)
    if shape == "research_report":
        return _research_report_graph(contract)
    if shape == "coding_change":
        return _coding_change_graph(contract)
    if shape == "memory_correction":
        return _memory_correction_graph(contract)
    if shape == "simple_chat":
        return _simple_chat_graph(contract)
    return _general_graph(contract, workflow_template=workflow_template)


def _source_collection_graph(contract: WorkflowContract | None) -> PassGraph:
    metadata = dict(getattr(contract, "metadata", {}) or {})
    passes = [
        AgentPassSpec(
            id="contract",
            role="contract_agent",
            purpose="Convert the user request into entity, source, fetch, and answer requirements.",
            context_strategy="structured_contract",
            allowed_inputs=["objective", "allowed_tools"],
            required_outputs=["workflow_contract"],
            model_policy="deterministic",
        ),
        AgentPassSpec(
            id="retrieve",
            role="retrieval_agent",
            purpose="Find or verify locked entities, then search each entity with the required query shape.",
            context_strategy="local_retrieval",
            allowed_inputs=["objective", "workflow_contract", "tool_results"],
            required_outputs=["locked_entities", "search_results"],
        ),
        AgentPassSpec(
            id="select",
            role="scoring_agent",
            purpose="Select source URLs from successful searches without inventing URLs.",
            context_strategy="deterministic_ranking",
            allowed_inputs=["search_results", "locked_entities", "workflow_contract"],
            required_outputs=["selected_urls"],
            model_policy="deterministic",
        ),
        AgentPassSpec(
            id="fetch",
            role="retrieval_agent",
            purpose="Fetch selected URLs until source coverage reaches the contract quota.",
            context_strategy="tool_execution",
            allowed_inputs=["selected_urls", "fetched_urls"],
            required_outputs=["fetched_sources"],
            model_policy="deterministic",
        ),
        AgentPassSpec(
            id="evaluate",
            role="evaluation_agent",
            purpose="Check entity locks, per-entity quotas, fetched source coverage, and drift.",
            context_strategy="scenario_state_eval",
            allowed_inputs=["workflow_contract", "tool_calls", "tool_results"],
            required_outputs=["coverage_eval", "missing_slots"],
            model_policy="deterministic",
        ),
        AgentPassSpec(
            id="orchestrate",
            role="orchestration_agent",
            purpose="Finalize only when the completion gate is satisfied; otherwise persist continuation.",
            context_strategy="completion_gate",
            allowed_inputs=["coverage_eval", "missing_slots", "draft_response"],
            required_outputs=["final_or_continuation"],
            model_policy="deterministic_or_small",
        ),
    ]
    return PassGraph(
        id="pass_graph_source_collection_v1",
        version=PASS_FRAMEWORK_VERSION,
        workflow_shape="source_collection",
        context_strategy="local_retrieval_plus_contract_eval",
        passes=passes,
        edges=_chain_edges([item.id for item in passes]),
        stop_condition="workflow_contract.completion_gate.final_answer_allowed == true",
        max_rounds=max(1, min(24, 3 + int(metadata.get("entity_count") or 0) + int(metadata.get("total_fetches") or 0))),
        metadata=metadata,
    )


def _research_report_graph(contract: WorkflowContract | None) -> PassGraph:
    passes = [
        AgentPassSpec("retrieve", "retrieval_agent", "Retrieve locally relevant source chunks.", "local_retrieval", ["objective"], ["candidate_chunks"]),
        AgentPassSpec("analyze", "reasoning_agent", "Analyze chunks independently for claims, gaps, and useful signals.", "chunk_wise", ["candidate_chunks"], ["chunk_observations"]),
        AgentPassSpec("aggregate", "summary_agent", "Aggregate chunk observations into a global representation.", "global_reasoning", ["chunk_observations"], ["global_summary"]),
        AgentPassSpec("evaluate", "evaluation_agent", "Check completeness, contradictions, and citation coverage.", "global_eval", ["global_summary"], ["quality_eval", "missing_slots"]),
        AgentPassSpec("orchestrate", "orchestration_agent", "Answer or continue based on quality evaluation.", "completion_gate", ["quality_eval"], ["final_or_continuation"]),
    ]
    return PassGraph(
        id="pass_graph_research_report_v1",
        version=PASS_FRAMEWORK_VERSION,
        workflow_shape="research_report",
        context_strategy="local_plus_global_reasoning",
        passes=passes,
        edges=_chain_edges([item.id for item in passes]),
        stop_condition="quality_eval.complete == true or max_rounds reached",
        max_rounds=3,
    )


def _coding_change_graph(contract: WorkflowContract | None) -> PassGraph:
    passes = [
        AgentPassSpec("inspect", "retrieval_agent", "Read relevant files, tests, and contracts.", "local_retrieval", ["objective", "workspace"], ["code_context"]),
        AgentPassSpec("plan", "reasoning_agent", "Create a scoped implementation plan.", "structured_plan", ["code_context"], ["plan_steps"]),
        AgentPassSpec("edit", "orchestration_agent", "Apply the smallest code change that satisfies the plan.", "tool_execution", ["plan_steps"], ["patches"]),
        AgentPassSpec("test", "evaluation_agent", "Run focused tests and classify failures.", "test_eval", ["patches"], ["test_results"]),
        AgentPassSpec("summarize", "summary_agent", "Report changed files, tests, and residual risk.", "final_summary", ["test_results"], ["final_response"]),
    ]
    return PassGraph(
        id="pass_graph_coding_change_v1",
        version=PASS_FRAMEWORK_VERSION,
        workflow_shape="coding_change",
        context_strategy="inspect_plan_edit_test",
        passes=passes,
        edges=_chain_edges([item.id for item in passes]),
        stop_condition="tests pass or blocking failure recorded",
        max_rounds=2,
    )


def _memory_correction_graph(contract: WorkflowContract | None) -> PassGraph:
    passes = [
        AgentPassSpec("extract", "retrieval_agent", "Extract explicit user-stated facts and correction signals.", "deterministic_extraction", ["user_message"], ["fact_candidates"]),
        AgentPassSpec("evaluate", "evaluation_agent", "Check supersession and contradiction lanes.", "fact_guard", ["fact_candidates", "current_facts"], ["memory_decision"]),
        AgentPassSpec("commit", "orchestration_agent", "Commit eligible facts transactionally and quarantine candidates.", "memory_commit", ["memory_decision"], ["memory_write"]),
    ]
    return PassGraph(
        id="pass_graph_memory_correction_v1",
        version=PASS_FRAMEWORK_VERSION,
        workflow_shape="memory_correction",
        context_strategy="deterministic_fact_guard",
        passes=passes,
        edges=_chain_edges([item.id for item in passes]),
        stop_condition="memory decision recorded",
        max_rounds=1,
    )


def _simple_chat_graph(contract: WorkflowContract | None) -> PassGraph:
    passes = [
        AgentPassSpec("respond", "orchestration_agent", "Return a direct conversational response without tool or memory expansion.", "direct_response", ["user_message"], ["final_response"], model_policy="deterministic_or_primary"),
    ]
    return PassGraph(
        id="pass_graph_simple_chat_v1",
        version=PASS_FRAMEWORK_VERSION,
        workflow_shape="simple_chat",
        context_strategy="direct_response",
        passes=passes,
        edges=[],
        stop_condition="response emitted",
        max_rounds=1,
    )


def _general_graph(contract: WorkflowContract | None, *, workflow_template: str = "") -> PassGraph:
    shape = getattr(contract, "workflow_shape", "open_ended_chat") if contract is not None else "open_ended_chat"
    passes = [
        AgentPassSpec("plan", "reasoning_agent", "Plan the next smallest useful step.", "structured_plan", ["objective", "ledger"], ["plan_steps"]),
        AgentPassSpec("act", "orchestration_agent", "Use allowed tools or answer directly from available evidence.", "phase_packet", ["plan_steps", "allowed_tools"], ["tool_call_or_response"]),
        AgentPassSpec("evaluate", "evaluation_agent", "Check whether the answer is supported or more work is needed.", "ledger_eval", ["tool_results", "draft_response"], ["verification"]),
    ]
    return PassGraph(
        id="pass_graph_general_v1",
        version=PASS_FRAMEWORK_VERSION,
        workflow_shape=shape,
        context_strategy="phase_packet_loop",
        passes=passes,
        edges=_chain_edges([item.id for item in passes]),
        stop_condition="verification supported or continuation persisted",
        max_rounds=3,
        metadata={"workflow_template": workflow_template},
    )


def _chain_edges(ids: list[str]) -> list[PassEdge]:
    return [
        PassEdge(source=ids[index], target=ids[index + 1])
        for index in range(len(ids) - 1)
    ]
