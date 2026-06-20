"""
Contract tests for the convergence patch set.

These pin the ENFORCEMENT behavior — not just detection. Each test asserts a
handoff that previously leaked: a signal that was computed and then dropped.
If any of these regress, a guarantee the platform claims has silently broken.

Run:  pytest tests/test_convergence_contracts.py -v
"""

from __future__ import annotations

import pytest

from engine.conversation_tension import (
    analyze_conversation_tension,
    detect_evidence_fact_conflicts,
    ConversationTension,
)
from engine.planner_contract import (
    filter_tools_for_planner,
    is_small_model,
    build_planner_prompt,
    parse_planner_output,
    PlannerStatus,
)
from engine.circuit_breaker import CircuitBreaker, CircuitState
from run_engine.memory_pipeline import derive_tension_enforcement


# ════════════════════════════════════════════════════════════════════════════
# 1. Predicate normalization closes the conflict-detection gap
# ════════════════════════════════════════════════════════════════════════════
def test_predicate_normalization_detects_cross_surface_conflict():
    """'location: Toronto' vs 'lives in: Berlin' must register as a conflict.
    Before the fix these were different predicate strings and slipped through."""
    tension = analyze_conversation_tension(
        user_message="actually I moved to Berlin",
        current_facts=[("user", "location", "Toronto")],
        deterministic_keyed_facts=[
            {"subject": "user", "predicate": "lives in", "object": "Berlin"}
        ],
        session_history=[],
    )
    assert tension.conflicting_facts, "cross-surface predicate conflict not detected"
    assert tension.storage_action == "void_previous_store_current"


def test_same_predicate_same_value_is_not_a_conflict():
    tension = analyze_conversation_tension(
        user_message="I live in Toronto",
        current_facts=[("user", "location", "Toronto")],
        deterministic_keyed_facts=[
            {"subject": "user", "predicate": "location", "object": "Toronto"}
        ],
        session_history=[],
    )
    assert not tension.conflicting_facts


# ════════════════════════════════════════════════════════════════════════════
# 2. storage_action is ENFORCED, not just rendered
# ════════════════════════════════════════════════════════════════════════════
def test_void_action_produces_deterministic_voids():
    """A voiding storage_action must yield concrete void records derived from
    the exact conflicts — no dependency on a compression model emitting [VOIDS:]."""
    tension = ConversationTension(
        storage_action="void_previous_store_current",
        conflicting_facts=(
            {"subject": "user", "predicate": "location",
             "previous": "Toronto", "current": "Berlin"},
        ),
    )
    voids, demotions, action = derive_tension_enforcement(tension, [])
    assert action == "void_previous_store_current"
    assert len(voids) == 1
    assert voids[0]["subject"] == "user"
    assert voids[0]["predicate"] == "location"
    assert voids[0]["void_source"] == "tension_policy"


def test_trace_action_stamps_conflict_provenance_without_voiding():
    tension = ConversationTension(
        storage_action="store_current_with_conflict_trace",
        conflicting_facts=(
            {"subject": "user", "predicate": "role",
             "previous": "engineer", "current": "manager"},
        ),
    )
    facts = [{"subject": "user", "predicate": "role", "object": "manager"}]
    voids, demotions, action = derive_tension_enforcement(tension, facts)
    assert voids == []                       # trace policy does not void
    assert facts[0]["conflict_trace"] is True
    assert facts[0]["prior_value_disputed"] is True


def test_none_action_is_inert():
    voids, demotions, action = derive_tension_enforcement(None, [])
    assert voids == [] and demotions == [] and action == "none"


# ════════════════════════════════════════════════════════════════════════════
# 3. Evidence-vs-memory lane (the new contradiction class)
# ════════════════════════════════════════════════════════════════════════════
def test_evidence_disputes_stored_fact():
    """Stored 'Acme CEO is Smith' but fresh evidence names Jones → dispute."""
    conflicts = detect_evidence_fact_conflicts(
        current_facts=[("acme", "ceo", "smith")],
        evidence_items=[
            {"success": True,
             "content": "Acme Corporation today announced that Jones will lead "
                        "the company as its new chief executive officer."}
        ],
    )
    assert conflicts, "evidence-vs-memory conflict not detected"
    assert conflicts[0]["lane"] == "evidence_vs_memory"
    assert conflicts[0]["stored"] == "smith"


def test_evidence_demotes_not_voids_immediately():
    """Evidence conflicts demote to candidate (asymmetric vs user corrections)."""
    tension = analyze_conversation_tension(
        user_message="who is the acme ceo",
        current_facts=[("acme", "ceo", "smith")],
        deterministic_keyed_facts=[],
        session_history=[],
        evidence_items=[
            {"success": True,
             "content": "Acme named Jones as chief executive officer this morning."}
        ],
    )
    assert tension.evidence_conflicts
    assert tension.storage_action == "demote_to_candidate_pending_verification"
    voids, demotions, _ = derive_tension_enforcement(tension, [])
    assert demotions, "evidence conflict should produce a candidate demotion"


def test_evidence_no_conflict_when_object_present():
    conflicts = detect_evidence_fact_conflicts(
        current_facts=[("acme", "ceo", "smith")],
        evidence_items=[
            {"success": True, "content": "Acme's CEO Smith spoke at the conference."}
        ],
    )
    assert not conflicts


def test_evidence_ignores_unsuccessful_results():
    conflicts = detect_evidence_fact_conflicts(
        current_facts=[("acme", "ceo", "smith")],
        evidence_items=[
            {"success": False, "content": "Acme named Jones as CEO."}
        ],
    )
    assert not conflicts


# ════════════════════════════════════════════════════════════════════════════
# 4. Planner contract — loud failure, tiered prompts, tool filtering
# ════════════════════════════════════════════════════════════════════════════
def test_small_model_detection():
    assert is_small_model("ollama:gemma4")
    assert is_small_model("qwen2.5-coder-3b")
    assert not is_small_model("anthropic:claude-sonnet-4-6")


def test_tool_filter_caps_and_prioritizes():
    all_tools = [f"tool_{i}" for i in range(27)] + ["web_search", "web_fetch"]
    picked = filter_tools_for_planner(all_tools, "fetch the url and read the page")
    assert "web_fetch" in picked
    assert len(picked) <= 6


def test_small_model_gets_minimal_prompt():
    prompt = build_planner_prompt(
        model="ollama:gemma4", objective="find top stock", tools=["web_search"]
    )
    assert "ONLY one JSON object" in prompt
    assert "argument_clues" in prompt


def test_planner_parse_rejects_toolless_tool_loop():
    out = parse_planner_output('{"route": "tool_loop", "tools": []}')
    assert out.status == PlannerStatus.FAILED
    assert out.reason == "tool_loop_without_tools"


def test_planner_parse_accepts_direct_answer_without_tools():
    out = parse_planner_output('{"route": "direct_answer", "tools": []}')
    assert out.status == PlannerStatus.OK
    assert out.actionable is False     # no tools, but a valid plan
    assert out.route == "direct_answer"


def test_planner_parse_recovers_from_markdown_fence():
    raw = '```json\n{"route":"tool_loop","tools":["web_search"]}\n```'
    out = parse_planner_output(raw)
    assert out.status == PlannerStatus.OK
    assert out.tools == ("web_search",)


def test_planner_parse_empty_is_failure_not_silent_ok():
    out = parse_planner_output("")
    assert out.status == PlannerStatus.FAILED
    assert not out.actionable


# ════════════════════════════════════════════════════════════════════════════
# 5. Circuit breaker — half-open recovery + LRU
# ════════════════════════════════════════════════════════════════════════════
def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(threshold=3)
    for _ in range(2):
        assert cb.record_failure("s1", "web_fetch", current_turn=1) is False
    assert cb.record_failure("s1", "web_fetch", current_turn=1) is True
    assert cb.is_open("s1", "web_fetch", current_turn=1)


def test_circuit_half_opens_after_cooldown():
    cb = CircuitBreaker(threshold=2, cooldown_turns=3)
    cb.record_failure("s1", "web_fetch", current_turn=1)
    cb.record_failure("s1", "web_fetch", current_turn=1)   # opens at turn 1
    assert cb.is_open("s1", "web_fetch", current_turn=2)   # still open
    # cooldown elapsed → half-open → trial allowed (is_open False)
    assert cb.is_open("s1", "web_fetch", current_turn=4) is False
    assert cb.state_of("s1", "web_fetch") == CircuitState.HALF_OPEN


def test_half_open_success_closes_circuit():
    cb = CircuitBreaker(threshold=2, cooldown_turns=1)
    cb.record_failure("s1", "t", current_turn=1)
    cb.record_failure("s1", "t", current_turn=1)
    cb.is_open("s1", "t", current_turn=3)        # transitions to half-open
    cb.record_success("s1", "t")
    assert cb.state_of("s1", "t") == CircuitState.CLOSED
    assert not cb.is_open("s1", "t", current_turn=4)


def test_half_open_failure_reopens():
    cb = CircuitBreaker(threshold=2, cooldown_turns=1)
    cb.record_failure("s1", "t", current_turn=1)
    cb.record_failure("s1", "t", current_turn=1)
    cb.is_open("s1", "t", current_turn=3)        # half-open
    reopened = cb.record_failure("s1", "t", current_turn=3)
    assert reopened is True
    assert cb.state_of("s1", "t") == CircuitState.OPEN


def test_lru_eviction_caps_sessions():
    cb = CircuitBreaker(threshold=2, max_sessions=3)
    for i in range(5):
        cb.record_failure(f"s{i}", "t", current_turn=1)
    # only the 3 most recent sessions retained
    assert len(cb._sessions) == 3
    assert "s0" not in cb._sessions
    assert "s4" in cb._sessions


def test_persistence_roundtrip():
    cb = CircuitBreaker(threshold=2)
    cb.record_failure("s1", "web_fetch", current_turn=1)
    cb.record_failure("s1", "web_fetch", current_turn=1)
    dumped = cb.dump_session("s1")
    cb2 = CircuitBreaker(threshold=2)
    cb2.load_session("s1", dumped)
    assert cb2.is_open("s1", "web_fetch", current_turn=1)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))