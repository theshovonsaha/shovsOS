import pytest
from unittest.mock import AsyncMock, MagicMock

from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools


@pytest.mark.asyncio
async def test_delegation():
    """
    Delegation tool should call AgentManager.run_agent_task with correct parent_id wiring.
    Kept hermetic: no live LLM/network dependency.
    """
    registry = ToolRegistry()
    manager = MagicMock()
    manager.run_agent_task = AsyncMock(return_value="The delegation was successful!")

    register_all_tools(registry, agent_manager=manager)
    delegate_tool = registry.get("delegate_to_agent")
    assert delegate_tool is not None

    result = await delegate_tool.handler(target_agent_id="default", task="Say success")
    assert "successful" in result.lower()
    manager.run_agent_task.assert_awaited_once_with("default", "Say success", parent_id=None)

    manager.run_agent_task.reset_mock()
    await delegate_tool.handler(target_agent_id="coder", task="Write script", _session_id="parent_123")
    manager.run_agent_task.assert_awaited_once_with("coder", "Write script", parent_id="parent_123")
