import pytest

from memory.task_tracker import SessionTaskTracker
from plugins.tool_registry import HookResult, Tool, ToolCall, ToolRegistry


@pytest.mark.asyncio
async def test_before_hook_can_block_tool_execution():
    registry = ToolRegistry()

    async def _echo(message: str) -> str:
        return message

    registry.register(
        Tool(
            name="echo",
            description="Echo text",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=_echo,
        )
    )

    async def deny_echo(tool_name: str, arguments: dict):
        return HookResult(decision="deny", reason=f"{tool_name} blocked")

    registry.register_before_hook(r"^echo$", deny_echo)

    result = await registry.execute(
        ToolCall(
            tool_name="echo",
            arguments={"message": "hello"},
            raw_json='{"tool":"echo","arguments":{"message":"hello"}}',
        )
    )

    assert result.success is False
    assert "blocked" in result.content


@pytest.mark.asyncio
async def test_after_hook_can_transform_tool_result():
    registry = ToolRegistry()

    async def _echo(message: str) -> str:
        return message

    registry.register(
        Tool(
            name="echo",
            description="Echo text",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=_echo,
        )
    )

    async def uppercase_output(tool_name: str, arguments: dict, result):
        return HookResult(decision="transform", transformed_content=result.content.upper())

    registry.register_after_hook(r"^echo$", uppercase_output)

    result = await registry.execute(
        ToolCall(
            tool_name="echo",
            arguments={"message": "hello"},
            raw_json='{"tool":"echo","arguments":{"message":"hello"}}',
        )
    )

    assert result.success is True
    assert result.content == "HELLO"


def test_session_task_tracker_write_and_update():
    tracker = SessionTaskTracker()

    initial = tracker.write(
        "session-1",
        [
            {"id": "1", "content": "Collect requirements", "status": "pending", "priority": "high"},
            {"id": "2", "content": "Implement feature", "status": "in_progress", "priority": "medium"},
        ],
    )
    assert "Current tasks:" in initial
    assert "[pending] 1: Collect requirements" in initial
    assert "[in_progress] 2: Implement feature" in initial

    updated = tracker.update("session-1", "2", "completed")
    assert "[completed] 2: Implement feature" in updated
    assert tracker.has_tasks("session-1") is True
