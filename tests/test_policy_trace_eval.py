from run_engine.scenario_eval import evaluate_policy_trace


def test_policy_trace_eval_requires_recovery_for_policy_violations():
    result = evaluate_policy_trace(
        trace_events=[
            {"phase": "policy", "control_policy": "plan_execute", "ledger_mode": "ledger_enforced"},
            {"phase": "policy_violation", "type": "policy_violation", "issue": "unlocked_entity_search"},
            {"phase": "tool", "tool": "web_search", "status": "success"},
            {"phase": "completion_gate", "status": "blocked"},
        ],
        expected_policy="plan_execute",
        expected_ledger_mode="ledger_enforced",
        require_completion_gate=True,
        require_recovery_for_violations=True,
    )

    assert result.passed is False
    assert "violation_without_recovery" in result.issues


def test_policy_trace_eval_accepts_enforced_recovery_path():
    result = evaluate_policy_trace(
        trace_events=[
            {"phase": "policy", "control_policy": "plan_execute", "ledger_mode": "ledger_enforced"},
            {"phase": "policy_violation", "type": "policy_violation", "issue": "unlocked_entity_search"},
            {"phase": "recovery_started", "type": "recovery_started", "recovered_tool": "web_search"},
            {"phase": "tool", "tool": "web_search", "status": "success"},
            {"phase": "completion_gate", "status": "passed"},
        ],
        expected_policy="plan_execute",
        expected_ledger_mode="ledger_enforced",
        require_completion_gate=True,
        require_recovery_for_violations=True,
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.state["recovery_count"] == 1


def test_policy_trace_eval_requires_graph_node_events_for_graph_harness():
    missing = evaluate_policy_trace(
        trace_events=[
            {"phase": "policy", "control_policy": "graph_harness", "ledger_mode": "ledger_enforced"},
            {"phase": "completion_gate", "status": "passed"},
        ],
        expected_policy="graph_harness",
        expected_ledger_mode="ledger_enforced",
        require_completion_gate=True,
        require_graph_nodes=True,
    )
    present = evaluate_policy_trace(
        trace_events=[
            {"phase": "policy", "control_policy": "graph_harness", "ledger_mode": "ledger_enforced"},
            {"phase": "pass_graph_execution", "event_type": "pass_node_started", "status": "success"},
            {"phase": "tool", "tool": "web_search", "status": "success"},
            {"phase": "completion_gate", "status": "passed"},
        ],
        expected_policy="graph_harness",
        expected_ledger_mode="ledger_enforced",
        require_completion_gate=True,
        require_graph_nodes=True,
    )

    assert missing.passed is False
    assert "missing_graph_node_events" in missing.issues
    assert present.passed is True
