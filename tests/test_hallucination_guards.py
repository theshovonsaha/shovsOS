import pytest
import asyncio
from engine.core import AgentCore, _ev
from unittest.mock import AsyncMock, MagicMock

class AsyncIter:
    def __init__(self, items):
        self.items = items
    async def __aiter__(self):
        for item in self.items:
            yield item

@pytest.mark.asyncio
async def test_solvability_guard_pivot_logic():
    """
    GREEN TEST: Verifies that AgentCore triggers a Hard Pivot after 3 tool failures.
    The pivot message is injected as: [SYSTEM: Tool '...' failed 3 times. ...]
    """
    # 1. Setup Mocks
    adapter = MagicMock()
    # Simulate a tool call in the first 3 turns, then a final response
    adapter.stream.side_effect = [
        AsyncIter(['{"tool":"test_tool", "arguments":{}}']), # Turn 1
        AsyncIter(['{"tool":"test_tool", "arguments":{}}']), # Turn 2
        AsyncIter(['{"tool":"test_tool", "arguments":{}}']), # Turn 3
        AsyncIter(['Final answer']),                        # Turn 4
    ]
    
    tools = MagicMock()
    tools.has_tools.return_value = True
    tools.build_tools_block.return_value = ""
    # validate_tool_call must return None (= valid) so the loop reaches execute()
    tools.validate_tool_call.return_value = None
    # detect_tool_calls (plural) should return a list of call objects
    from plugins.tool_registry import ToolCall
    call = ToolCall(tool_name="test_tool", raw_json='{"tool":"test_tool", "arguments":{}}', arguments={})
    tools.detect_tool_calls.side_effect = [[call], [call], [call], []]

    # execute should return failure for all 3
    result = MagicMock(tool_name="test_tool", success=False, content="Error 404")
    tools.execute = AsyncMock(return_value=result)

    # middleware=None forces the loop to call tools.execute directly (no middleware path)
    agent = AgentCore(adapter, MagicMock(), MagicMock(), tools, embed_model="test-embed")
    agent.middleware = None
    
    # 2. Run the loop
    events = []
    messages = []
    full_output = ""
    async for ev in agent._agent_loop("model", messages, adapter=adapter):
        events.append(ev)
        if ev["type"] == "token":
            full_output += ev["content"]
        
    # 3. Verify — the pivot message uses "[SYSTEM: Tool '...' failed 3 times. ...]"
    last_user_msg = messages[-1]["content"]
    assert "failed 3 times" in last_user_msg
    assert "test_tool" in last_user_msg
    assert "Stop attempting it" in last_user_msg


@pytest.mark.asyncio
async def test_citation_guard_verification():
    """
    GREEN TEST: Verifies that _verify_citations only catches explicit negations.
    """
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # Case 1: Explicit negation against a positive preference should be unfaithful
    response = "I hate that and I avoid using it."
    anchors = [{"key": "User Preference", "anchor": "User: I prefer dark themes. Assistant: Understood."}]
    
    is_faithful = agent._verify_citations(response, anchors)
    assert is_faithful is False
 
    # Case 2: Topic overlap without negation should be treated as faithful
    response_ok = "I can help with dark and light theme options."
    is_faithful_ok = agent._verify_citations(response_ok, anchors)
    assert is_faithful_ok is True


@pytest.mark.asyncio
async def test_citation_guard_empty_anchors():
    """
    GREEN TEST: No anchors means nothing to contradict — always faithful.
    """
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    assert agent._verify_citations("Anything at all", []) is True
    assert agent._verify_citations("Anything at all", None) is True
