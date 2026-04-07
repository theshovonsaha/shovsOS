from run_engine.engine import RunEngine


def test_resolve_effective_objective_reuses_previous_user_goal_for_retry():
    history = [
        {"role": "user", "content": "research wigglebudget.com"},
        {"role": "assistant", "content": "I will gather evidence."},
        {"role": "user", "content": "try again"},
    ]

    resolved = RunEngine._resolve_effective_objective("try again", history)

    assert resolved == "research wigglebudget.com"


def test_bootstrap_tools_use_resolved_objective_for_ambiguous_retry():
    tools = [{"name": "web_search"}]
    history = [
        {"role": "user", "content": "research wigglebudget.com"},
        {"role": "assistant", "content": "I will gather evidence."},
        {"role": "user", "content": "try again"},
    ]

    selected = RunEngine._bootstrap_tools_for_turn(
        user_message="try again",
        allowed_tools=tools,
        planner_enabled=True,
        session_history=history,
    )

    assert selected == ["web_search"]


def test_bootstrap_tools_do_not_blindly_default_to_search_for_save_it():
    tools = [{"name": "web_search"}]
    history = [
        {"role": "user", "content": "remember this in memory"},
        {"role": "assistant", "content": "I can do that."},
        {"role": "user", "content": "save it"},
    ]

    selected = RunEngine._bootstrap_tools_for_turn(
        user_message="save it",
        allowed_tools=tools,
        planner_enabled=True,
        session_history=history,
    )

    assert selected == []
