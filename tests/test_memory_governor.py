from run_engine.memory_governor import govern_memory_commit
from run_engine.memory_pipeline import plan_memory_commit


def test_memory_governor_adds_missing_void_for_changed_current_fact():
    governed = govern_memory_commit(
        facts=[{"subject": "User", "predicate": "location", "object": "Berlin"}],
        voids=[],
        current_facts=[("User", "location", "Toronto")],
    )

    assert governed.facts == [{"subject": "User", "predicate": "location", "object": "Berlin"}]
    assert governed.voids == [{"subject": "User", "predicate": "location", "void_source": "memory_governor"}]
    assert governed.decisions[0].action == "void_and_replace"


def test_memory_governor_ignores_duplicate_current_fact():
    governed = govern_memory_commit(
        facts=[{"subject": "User", "predicate": "location", "object": "Toronto"}],
        voids=[],
        current_facts=[("User", "location", "Toronto")],
    )

    assert governed.facts == []
    assert governed.voids == []
    assert governed.decisions[0].action == "ignore"
    assert governed.decisions[0].reason == "current_fact_already_matches"


def test_memory_governor_rejects_malformed_or_general_facts():
    governed = govern_memory_commit(
        facts=[
            {"subject": "General", "predicate": "working_on", "object": "Secret roadmap"},
            {"subject": "User", "predicate": "location", "object": object()},
        ],
        voids=[],
        current_facts=[],
    )

    assert governed.facts == []
    assert governed.voids == []
    assert [decision.reason for decision in governed.decisions] == [
        "general_subject_not_current_memory",
        "non_scalar_object",
    ]


def test_memory_governor_keeps_conflict_traced_fact_auditable_without_auto_void():
    governed = govern_memory_commit(
        facts=[
            {
                "subject": "User",
                "predicate": "location",
                "object": "Berlin",
                "conflict_trace": True,
                "prior_value_disputed": True,
                "conflict_reason": "unresolved correction",
            }
        ],
        voids=[],
        current_facts=[("User", "location", "Toronto")],
    )

    assert len(governed.facts) == 1
    assert governed.voids == []
    assert governed.decisions[0].action == "store_current_with_conflict_trace"


def test_plan_memory_commit_includes_governor_decisions_and_missing_voids():
    plan = plan_memory_commit(
        context_result=("", [], []),
        user_message="Actually I moved to Berlin.",
        tool_results=[],
        deterministic_keyed_facts=[
            {"subject": "User", "predicate": "location", "object": "Berlin", "fact": "User location Berlin"}
        ],
        deterministic_voids=[],
        current_facts=[("User", "location", "Toronto")],
        existing_candidate_signals=[],
        existing_candidate_context="",
    )

    assert plan.merged_facts[0]["subject"] == "User"
    assert plan.merged_facts[0]["predicate"] == "location"
    assert plan.merged_facts[0]["object"] == "Berlin"
    assert plan.merged_voids == [
        {"subject": "User", "predicate": "location", "void_source": "memory_governor"}
    ]
    assert plan.memory_decisions[0]["action"] == "void_and_replace"
