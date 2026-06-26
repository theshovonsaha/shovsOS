import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.context_engine import ContextEngine
from engine.context_engine_v2 import ContextEngineV2
from engine.context_engine_v3 import ContextEngineV3
from engine.context_hygiene import is_low_value_social_turn, should_skip_memory_compression


def test_low_value_social_turn_detection_keeps_action_turns():
    assert is_low_value_social_turn("hi again")
    assert is_low_value_social_turn("thanks that helps")
    assert should_skip_memory_compression("hello there", "Verbose but friendly assistant greeting.")
    assert not is_low_value_social_turn("hello search Toronto sushi")
    assert not is_low_value_social_turn("continue")
    assert not is_low_value_social_turn("call me Shovon")


@pytest.mark.asyncio
async def test_context_engine_v1_skips_verbose_greeting_without_memory_write():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value="- User has said hi multiple times")
    engine = ContextEngine(adapter=adapter)

    updated, facts, voids = await engine.compress_exchange(
        user_message="hi again",
        assistant_response="Hello. I can help with research, coding, and workflow analysis. What would you like to do next?",
        current_context="- Existing durable fact",
        is_first_exchange=True,
    )

    assert updated == "- Existing durable fact"
    assert facts == []
    assert voids == []
    adapter.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_engine_v2_trivial_turn_preserves_existing_serialized_context():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value="goal: noisy greeting")
    engine = ContextEngineV2(adapter=adapter)
    existing = json.dumps({
        "__v2__": True,
        "turn": 3,
        "active_goals": {"shopping comparison": {"first_seen_turn": 1, "last_seen_turn": 3}},
        "modules": {
            "Budget Stores": {
                "content": "User is comparing stores by price and quality.",
                "goals": ["shopping comparison"],
                "hit_count": 2,
                "created_turn": 1,
                "last_seen_turn": 3,
                "protected": False,
            }
        },
    })

    updated, facts, voids = await engine.compress_exchange(
        user_message="thanks that helps",
        assistant_response="You are welcome. I can keep helping with the comparison.",
        current_context=existing,
        is_first_exchange=False,
    )

    assert json.loads(updated)["modules"]["Budget Stores"]["content"].startswith("User is comparing")
    assert facts == []
    assert voids == []
    adapter.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_engine_v3_skips_social_turn_without_mutating_context():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value="- User has said hello repeatedly")
    engine = ContextEngineV3(adapter=adapter)
    existing = json.dumps({
        "__v3__": True,
        "durable_context": "- User prefers compact tables",
        "convergent_context": json.dumps({"__v2__": True, "turn": 1, "active_goals": {}, "modules": {}}),
    })

    updated, facts, voids = await engine.compress_exchange(
        user_message="ok got it",
        assistant_response="Acknowledged. I will proceed when you ask for the next task.",
        current_context=existing,
        is_first_exchange=False,
    )

    assert updated == existing
    assert facts == []
    assert voids == []
    adapter.complete.assert_not_awaited()
