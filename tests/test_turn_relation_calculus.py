from engine.context_schema import ContextPhase
from run_engine.language_kernel import build_kernel_snapshot
from run_engine.ledger import RunLedger
from run_engine.turn_relation import classify_turn_relation, simulate_turn_relation_cases


def test_turn_relation_simulation_covers_expected_cases():
    results = simulate_turn_relation_cases()
    by_name = {item["name"]: item for item in results}

    assert by_name["fresh_topic"]["relation"] == "fresh_topic"
    assert "previous_plan" in by_name["fresh_topic"]["blocked_context"]

    assert by_name["direct_continuation"]["relation"] == "direct_continuation"
    assert "continuation_state" in by_name["direct_continuation"]["required_context"]
    assert by_name["direct_continuation"]["tool_policy"] == "resume_next_required_action"

    assert by_name["distant_resumption"]["relation"] == "distant_resumption"
    assert "raw_refs" in by_name["distant_resumption"]["anchors"]

    assert by_name["correction"]["relation"] == "correction"
    assert "older_conflicting_fact_as_current" in by_name["correction"]["blocked_context"]

    assert by_name["refinement"]["relation"] == "refinement"
    assert "current_turn_patch" in by_name["refinement"]["anchors"]

    assert by_name["short_followup_frame"]["relation"] == "refinement"
    assert "active_constraints" in by_name["short_followup_frame"]["carried_context"]
    assert "Current turn update: what about razr" in by_name["short_followup_frame"]["resolved_objective"]

    assert by_name["meta_instruction"]["relation"] == "meta_instruction"
    assert by_name["meta_instruction"]["tool_policy"] == "update_policy_before_tools"


def test_explicit_new_topic_blocks_old_workflow_state():
    relation = classify_turn_relation(
        user_message="new topic: write a Python script to rename files",
        continuation_state={"objective": "Find a storage bin under $20 near Toronto."},
    )

    assert relation.relation == "fresh_topic"
    assert "previous_plan" in relation.blocked_context
    assert relation.tool_policy == "clear_continuation_then_plan"
    assert relation.proof_obligations == ["actor_context excludes stale workflow state"]


def test_direct_continuation_requires_next_action_context():
    relation = classify_turn_relation(
        user_message="continue",
        continuation_state={
            "objective": "Fetch 9 source URLs for ROKU, TBN, SENEA.",
            "pending_steps": [{"id": "fetch_roku", "tool": "web_fetch"}],
        },
        ambiguous_followup=True,
    )

    assert relation.relation == "direct_continuation"
    assert "next_required_action" in relation.required_context
    assert "unrelated_memory" in relation.blocked_context


def test_distant_resumption_uses_refs_not_full_raw_history():
    relation = classify_turn_relation(
        user_message="Let's return to the Toronto sushi workflow",
        distant_memory_signals=["top 3 sushi places in Toronto, search each, fetch 3 URLs each"],
    )

    assert relation.relation == "distant_resumption"
    assert "semantic_capsules" in relation.anchors
    assert "full_raw_history_by_default" in relation.blocked_context


def test_short_followup_carries_latest_product_frame():
    relation = classify_turn_relation(
        user_message="what about razr",
        recent_turns=[
            {"role": "user", "content": "i just want to buy the razr one give me the direct cheapest link"},
            {"role": "assistant", "content": "Here are Motorola Razr phone options."},
            {"role": "user", "content": "i want to buy an ergonomic gaming race chair under 500 cad"},
            {"role": "assistant", "content": "Here are gaming chair options under 500 CAD."},
            {"role": "user", "content": "what about razr"},
        ],
    )

    assert relation.relation == "refinement"
    assert relation.reason == "short follow-up needs the latest active task frame"
    assert relation.resolved_objective == (
        "i want to buy an ergonomic gaming race chair under 500 cad\n"
        "Current turn update: what about razr"
    )
    assert "older_unrelated_entities" in relation.blocked_context


def test_bare_entity_refinement_preserves_latest_person_frame():
    relation = classify_turn_relation(
        user_message="york university one",
        recent_turns=[
            {"role": "user", "content": "who is shawon saha"},
            {"role": "assistant", "content": "There may be multiple people named Shawon Saha."},
            {"role": "user", "content": "york university one"},
        ],
    )

    assert relation.relation == "refinement"
    assert relation.reason == "short follow-up needs the latest active task frame"
    assert relation.resolved_objective == (
        "who is shawon saha\n"
        "Current turn update: york university one"
    )
    assert "older_unrelated_entities" in relation.blocked_context


def test_fresh_short_question_is_not_forced_into_previous_frame():
    relation = classify_turn_relation(
        user_message="who is shawon saha",
        recent_turns=[
            {"role": "user", "content": "find top 3 sushi places in Toronto"},
            {"role": "assistant", "content": "Here are three sushi places."},
        ],
    )

    assert relation.relation == "fresh_topic"
    assert relation.resolved_objective == "who is shawon saha"


def test_turn_relation_enters_language_kernel_prompt_contract():
    relation = classify_turn_relation(
        user_message="No, make the buttons pink not purple.",
        recent_turns=[{"role": "user", "content": "Make the buttons purple."}],
    )
    ledger = RunLedger(
        run_id="run-relation-kernel",
        session_id="session-relation-kernel",
        turn_id="turn-1",
        objective="No, make the buttons pink not purple.",
        allowed_tools=[],
    )
    ledger.set_turn_relation(relation.to_dict())

    snapshot = build_kernel_snapshot(ledger, ContextPhase.ACTING)
    contract = snapshot.prompt_contract
    rendered = contract.render()

    assert contract.turn_relation["relation"] == "correction"
    assert "turn relation: correction" in rendered
    assert "latest user correction" in " ".join(contract.rules)
    assert "older_conflicting_fact_as_current" in rendered
