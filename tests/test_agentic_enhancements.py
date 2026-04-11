import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestration.orchestrator import (
    AgenticOrchestrator,
    OBSERVATION_PROMPT,
    PLANNING_PROMPT,
    VERIFICATION_PROMPT,
)
from plugins.tool_registry import ToolRegistry, Tool, ToolCall
from plugins.tools import _query_memory


@pytest.mark.asyncio
async def test_orchestrator_plan_with_context_structured_output():
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value=(
            '{"strategy":"Need tools","tools":['
            '{"name":"web_search","priority":"high","reason":"fresh data"},'
            '{"name":"web_fetch","priority":"low","reason":"optional"},'
            '{"name":"not_real","priority":"high","reason":"invalid"}],'
            '"confidence":0.9}'
        )
    )
    orch = AgenticOrchestrator(adapter)
    tools = [
        {"name": "web_search", "description": "Search"},
        {"name": "web_fetch", "description": "Fetch URL"},
        {"name": "query_memory", "description": "Recall memory"},
    ]

    plan = await orch.plan_with_context(
        query="What did we discuss before and what's new?",
        tools_list=tools,
        model="llama3",
        session_has_history=True,
        current_fact_count=1,
        failed_tools=["web_fetch"],
    )

    assert plan["tools"][0]["name"] == "query_memory"
    assert plan["tools"][1]["name"] == "web_search"
    assert all(t["name"] != "web_fetch" for t in plan["tools"])
    assert all(t["name"] != "not_real" for t in plan["tools"])
    assert plan["force_memory"] is True


@pytest.mark.asyncio
async def test_orchestrator_observation_prefers_first_party_pricing_gap_closure():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{"status":"finalize","tools":[],"notes":"","confidence":0.4}')
    orch = AgenticOrchestrator(adapter)
    tools = [
        {"name": "web_search", "description": "Search"},
        {"name": "web_fetch", "description": "Fetch URL"},
    ]

    observation = await orch.observe_with_context(
        query="Investigate wigglebudget.com, what it costs, whether it is trustworthy, and whether you would recommend it.",
        tools_list=tools,
        tool_results=[
            {
                "tool_name": "web_fetch",
                "success": True,
                "content": "URL: https://wigglebudget.com/ Wiggle Budget is a personal finance web app.",
                "arguments": {"url": "https://wigglebudget.com/"},
            }
        ],
        model="llama3",
    )

    assert observation["status"] == "continue"
    assert observation["tools"][0]["name"] == "web_fetch"
    assert "pricing" in observation["notes"].lower()


def test_orchestrator_prompts_enforce_exact_entities_and_no_admin_bloat():
    assert "Preserve exact user entities" in PLANNING_PROMPT
    assert "Do not broaden or weaken the user's query" in PLANNING_PROMPT
    assert "pricing page" in PLANNING_PROMPT
    assert "Do not ask for `todo_write` again" in OBSERVATION_PROMPT
    assert "pricing" in OBSERVATION_PROMPT
    assert "hidden packets" in OBSERVATION_PROMPT
    assert "mutated domains" in VERIFICATION_PROMPT
    assert "internal execution chatter" in VERIFICATION_PROMPT


def test_tool_registry_validate_tool_call():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="dummy",
            description="dummy",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name"],
            },
            handler=lambda **_: "ok",
        )
    )

    missing_req = registry.validate_tool_call(ToolCall(tool_name="dummy", arguments={}, raw_json="{}"))
    assert "Missing required argument 'name'" in missing_req

    wrong_type = registry.validate_tool_call(
        ToolCall(tool_name="dummy", arguments={"name": "x", "count": "bad"}, raw_json="{}")
    )
    assert "must be of type 'integer'" in wrong_type

    valid = registry.validate_tool_call(
        ToolCall(tool_name="dummy", arguments={"name": "ok", "count": 2}, raw_json="{}")
    )
    assert valid is None


@pytest.mark.asyncio
async def test_query_memory_uses_session_facts(monkeypatch):
    class FakeGraph:
        async def traverse(self, topic, top_k=5, threshold=0.5, owner_id=None, locus_id=None):
            return []

        def get_current_facts(self, session_id, owner_id=None):
            return [("User", "likes", "blue")]

        def list_loci(self, owner_id=None):
            return []

        def get_compiled_drawer(self, locus_id):
            return None

    monkeypatch.setattr("memory.semantic_graph.SemanticGraph", FakeGraph)
    result = await _query_memory("blue", _session_id="session-1")
    assert "Unified memory results" in result
    assert "likes" in result
