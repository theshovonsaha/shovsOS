from engine.response_guard import (
    guard_final_response,
    is_small_or_local_model,
    looks_like_tool_json,
)


def test_response_guard_detects_unregistered_tool_json():
    text = '{"tool": "name_suggest", "arguments": {"preferred_name": "Shovon"}}'

    result = guard_final_response(
        text,
        user_message="What should you call me?",
        model="ollama:llama3.2:3b",
    )

    assert looks_like_tool_json(text)
    assert result.changed is True
    assert result.replacement_used is True
    assert "tool_json_final_response" in result.issues
    assert result.text == "You should be called Shovon."


def test_response_guard_does_not_rewrite_normal_frontier_answer():
    text = "You should be called Shovon."

    result = guard_final_response(
        text,
        user_message="What should you call me?",
        model="openai:gpt-5",
    )

    assert result.changed is False
    assert result.text == text


def test_response_guard_extracts_name_from_query_arg_tool_json():
    result = guard_final_response(
        '{"tool": "name_lookup", "arguments": {"q": "preferred name: Shovon"}}',
        user_message="What should you call me?",
        model="ollama:llama3.2:3b",
    )

    assert result.text == "You should be called Shovon."


def test_response_guard_replaces_greeting_tool_json_with_plain_greeting():
    result = guard_final_response(
        '{"tool": "greet", "arguments": {"name": "Shovs"}}',
        user_message="hi",
        model="ollama:llama3.2:3b",
    )

    assert result.text == "Hi."


def test_response_guard_strips_hidden_runtime_language_for_user_answer():
    text = (
        "I will update the canonical run ledger later. "
        "You should be called Shovon from now on."
    )

    result = guard_final_response(
        text,
        user_message="What should you call me?",
        model="ollama:llama3.2:3b",
    )

    assert result.changed is True
    assert "hidden_runtime_language" in result.issues
    assert result.text == "You should be called Shovon from now on."


def test_response_guard_allows_architecture_terms_when_user_asks():
    text = "The run ledger stores tool calls and verification state."

    result = guard_final_response(
        text,
        user_message="Explain the run ledger architecture.",
        model="ollama:llama3.2:3b",
    )

    assert result.changed is False
    assert result.text == text


def test_small_local_model_detection_is_profile_aware():
    assert is_small_or_local_model("ollama:llama3.2:3b")
    assert is_small_or_local_model("openai:gpt-5", profile="small_local")
    assert not is_small_or_local_model("openai:gpt-5", profile="frontier_standard")
