"""The converged turn router: intent -> {tool palette, planner, answer-from-memory}."""

import pytest

from run_engine.turn_policy import resolve_turn_policy

ALL = [
    "web_search", "web_fetch", "web_fetch_batch",
    "query_memory", "store_memory", "update_memory", "rag_search", "source_collect",
]


def _p(message, **kw):
    return resolve_turn_policy(message, user_message=message, allowed_tools=ALL, **kw)


@pytest.mark.parametrize("msg", ["whats my name?", "what do you remember about me", "who am i", "do you know my name"])
def test_identity_recall_never_web(msg):
    p = _p(msg)
    assert p.intent == "memory_recall"
    assert p.forbid_web and not p.use_planner
    assert "web_search" not in p.constrain(ALL)


@pytest.mark.parametrize("msg", ["remember my name", "remember my name is Drake", "update my preferred name to Von"])
def test_memory_store_never_web(msg):
    p = _p(msg)
    assert p.intent == "memory_store"
    assert "web_search" not in p.constrain(ALL)
    assert "store_memory" in p.constrain(ALL)


def test_read_recent_chat_is_conversation_recall_not_web():
    p = _p("read recent chat")
    assert p.intent == "conversation_recall"
    assert "web_search" not in p.constrain(ALL)


@pytest.mark.parametrize("msg", [
    "but you forgot what we were chating about",
    "you forgot what we were talking about",
    "what were we chatting about",
    "you lost the conversation context",
])
def test_context_loss_complaints_are_conversation_recall_not_web(msg):
    p = _p(msg)
    assert p.intent == "conversation_recall"
    assert p.forbid_web and not p.use_planner
    assert "web_search" not in p.constrain(ALL)


@pytest.mark.parametrize("msg", ["I am a photographer", "I like blue color", "my favorite color is blue", "I'm a designer"])
def test_personal_disclosure_stores_not_searches(msg):
    p = _p(msg)
    assert p.intent == "disclosure"
    assert "web_search" not in p.constrain(ALL)


def test_direct_fact_answers_from_memory_with_no_tools_or_planner():
    p = _p("what's my name?", direct_fact_answerable=True)
    assert p.answer_from_memory and not p.use_planner


@pytest.mark.parametrize("msg", [
    "find top 3 stocks and fetch sources",
    "what's the latest on NVDA",
    "I'm looking for the best DAW",
    "remember to search the web for the weather",  # external signal -> not a memory turn
    "whats the price of NVDA",
])
def test_research_and_mixed_intents_keep_web(msg):
    p = _p(msg, workflow_shape="research_report")
    assert p.tool_whitelist is None or "web_search" in p.constrain(ALL)


def test_source_collection_keeps_source_palette():
    p = _p("top 3 stocks, fetch top 3 urls each", workflow_shape="source_collection")
    assert "web_search" in p.constrain(ALL)


def test_simple_chat_takes_no_tools():
    p = _p("hey", workflow_shape="simple_chat")
    assert p.constrain(ALL) == [] and not p.use_planner
