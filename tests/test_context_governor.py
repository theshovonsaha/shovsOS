from unittest.mock import MagicMock

from engine.context_engine import ContextEngine
from engine.context_engine_v2 import ContextEngineV2
from engine.context_engine_v3 import ContextEngineV3
from engine.context_memory_items import build_context_engine_memory_items
from engine.context_governor import ContextGovernor


def test_context_governor_resolves_policy_modes_and_bootstraps_v1():
    governor = ContextGovernor(adapter=MagicMock(), v1_engine=None, semantic_graph=None)

    v1 = governor.resolve("v1", compression_model="llama3.2")
    v2 = governor.resolve("v2", compression_model="llama3.2")
    v3 = governor.resolve("v3", compression_model="llama3.2")

    assert isinstance(v1, ContextEngine)
    assert isinstance(v2, ContextEngineV2)
    assert isinstance(v3, ContextEngineV3)
    assert governor.resolve("v2", compression_model="llama3.2") is v2
    assert governor.resolve("v3", compression_model="llama3.2") is v3


def test_context_v2_prioritizes_recent_goal_overlap_over_stale_goal_overlap():
    ctx = ContextEngineV2(adapter=MagicMock())
    ctx._turn = 10
    ctx._active_goals = {
        "legacy cleanup": {"first_seen_turn": 1, "last_seen_turn": 5},
        "current packaging": {"first_seen_turn": 10, "last_seen_turn": 10},
    }
    ctx._modules = {
        "Legacy Module": {
            "content": "Old cleanup details",
            "goals": {"legacy cleanup"},
            "hit_count": 4,
            "created_turn": 1,
            "last_seen_turn": 6,
            "protected": False,
        },
        "Current Module": {
            "content": "Packaging details",
            "goals": {"current packaging"},
            "hit_count": 1,
            "created_turn": 10,
            "last_seen_turn": 10,
            "protected": False,
        },
        "Shared Module": {
            "content": "Shared instructions",
            "goals": {"legacy cleanup", "current packaging"},
            "hit_count": 2,
            "created_turn": 5,
            "last_seen_turn": 10,
            "protected": False,
        },
    }

    ranked = ctx._rank_by_convergence()

    assert ranked[0][0] == "Shared Module"
    assert ranked[1][0] == "Current Module"
    assert ranked[2][0] == "Legacy Module"
    assert ctx._ordered_active_goal_labels()[:2] == ["current packaging", "legacy cleanup"]


def test_context_v3_selects_recent_high_signal_durable_lines():
    ctx = ContextEngineV3(adapter=MagicMock())
    durable_lines = [
        "- First message: \"Help me build a memory runtime.\"",
        "- filler detail one",
        "- filler detail two",
        "- filler detail three",
        "- filler detail four",
        "- filler detail five",
        "- filler detail six",
        "- filler detail seven",
        "- Actually, I moved to Berlin.",
        "- Do not use web_search for this task.",
    ]

    selected = ctx._select_durable_lines(durable_lines, max_items=4)

    assert durable_lines[0] in selected
    assert "- Actually, I moved to Berlin." in selected
    assert "- Do not use web_search for this task." in selected
    assert len(selected) == 4


def test_context_governor_builds_standardized_memory_items_with_pattern_cues_for_v1():
    governor = ContextGovernor(adapter=MagicMock(), v1_engine=ContextEngine(adapter=MagicMock()), semantic_graph=None)
    engine = governor.resolve("v1", compression_model="llama3.2")

    items = governor.build_memory_items(
        engine=engine,
        context="- First message: \"Help me build this.\"\n- Actually, I moved to Toronto.\n- Keep scope to engine/.\n",
        current_facts=[
            ("User", "location", "Toronto"),
            ("User", "preferred_editor", "VS Code"),
            ("Task", "scope_boundary", "engine/"),
        ],
        trace_prefix="memory:test",
    )

    assert any(item.item_id == "context_governor_v1_memory" for item in items)
    pattern_item = next(item for item in items if item.item_id == "context_governor_pattern_cues")
    assert "Identity anchors" in pattern_item.content
    assert "Task frame" in pattern_item.content
    assert "Correction lineage" in pattern_item.content


def test_context_memory_items_prefers_governor_rendering_over_engine_local_items():
    governor = ContextGovernor(adapter=MagicMock(), v1_engine=ContextEngine(adapter=MagicMock()), semantic_graph=None)
    engine = governor.resolve("v2", compression_model="llama3.2")
    engine._turn = 3
    engine._active_goals = {"current packaging": {"first_seen_turn": 3, "last_seen_turn": 3}}
    engine._modules = {
        "Packaging Module": {
            "content": "Package and publish the memory facade.",
            "goals": {"current packaging"},
            "hit_count": 1,
            "created_turn": 3,
            "last_seen_turn": 3,
            "protected": False,
        }
    }
    serialized = engine._serialize_context()

    items = build_context_engine_memory_items(
        engine,
        serialized,
        context_governor=governor,
        current_facts=[("Task", "budget_limit", "two hours")],
    )

    assert any(item.item_id == "context_governor_v2_memory" for item in items)
    pattern_item = next(item for item in items if item.item_id == "context_governor_pattern_cues")
    assert "Goal convergence" in pattern_item.content
    assert "Memory profile" in pattern_item.content
