from plugins.tool_registry import ToolCall, ToolRegistry
from plugins.tools import FILE_STR_REPLACE_TOOL


def test_file_str_replace_empty_old_str_has_actionable_guidance():
    registry = ToolRegistry()
    registry.register(FILE_STR_REPLACE_TOOL)
    call = ToolCall(
        tool_name="file_str_replace",
        arguments={"path": "memory/2026-03-23.md", "old_str": "", "new_str": "x"},
        raw_json='{"tool":"file_str_replace","arguments":{"path":"memory/2026-03-23.md","old_str":"","new_str":"x"}}',
    )
    error = registry.validate_tool_call(call)
    assert error is not None
    assert "cannot be empty" in error
    assert "file_view" in error
    assert "file_create" in error
