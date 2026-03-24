from engine.core import (
    AgentCore,
    _adaptive_tool_turn_budget,
    _build_task_progress_instruction,
    _compact_tool_result_for_followup,
    _select_followup_tool_results,
    _sanitize_followup_messages,
    _should_include_task_reminder,
    _normalize_forced_tools_for_task_state,
    _tool_turn_budget,
)
from plugins.tool_registry import Tool, ToolCall, ToolRegistry


def test_adaptive_tool_budget_expands_for_file_inspection():
    base = _tool_turn_budget("groq:llama-3.3-70b-versatile", None)
    adaptive = _adaptive_tool_turn_budget(
        "groq:llama-3.3-70b-versatile",
        None,
        route_type="open_ended",
        forced_tools=["file_view"],
        user_message="check the md files you have",
    )
    assert adaptive > base


def test_adaptive_tool_budget_caps_growth():
    adaptive = _adaptive_tool_turn_budget(
        "anthropic:claude-sonnet-4-5",
        None,
        route_type="multi_step",
        forced_tools=["file_view", "web_search", "web_fetch"],
        user_message="research deeply and inspect markdown files",
    )
    assert adaptive <= 8


def test_strip_detected_tool_json_removes_trailing_call_block():
    raw = 'I checked two files already.\n{"tool": "file_view", "arguments": {"path": "soul.md"}}'
    calls = [ToolCall(tool_name="file_view", arguments={"path": "soul.md"}, raw_json='{"tool": "file_view", "arguments": {"path": "soul.md"}}')]
    clean = AgentCore._strip_detected_tool_json(raw, calls)
    assert clean == "I checked two files already."


def test_strip_detected_tool_json_removes_refusal_preamble_when_tool_call_exists():
    raw = (
        "I'm sorry, but I can't assist with that.\n"
        '{"tool": "web_search", "arguments": {"query": "wigglebudget.com"}}'
    )
    calls = [ToolCall(tool_name="web_search", arguments={"query": "wigglebudget.com"}, raw_json='{"tool": "web_search", "arguments": {"query": "wigglebudget.com"}}')]
    clean = AgentCore._strip_detected_tool_json(raw, calls)
    assert clean == ""


def test_detect_tool_calls_accepts_single_key_tool_shape():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_search",
            description="Search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=lambda **_: None,
        )
    )
    calls = registry.detect_tool_calls('```json\n{"web_search": {"query": "wigglebudget.com"}}\n```')
    assert len(calls) == 1
    assert calls[0].tool_name == "web_search"
    assert calls[0].arguments == {"query": "wigglebudget.com"}


def test_detect_tool_calls_accepts_tool_name_with_params_shape():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="store_memory",
            description="Memory",
            parameters={"type": "object", "properties": {"subject": {"type": "string"}}},
            handler=lambda **_: None,
        )
    )
    calls = registry.detect_tool_calls('{"tool_name":"store_memory","params":{"subject":"User"}}')
    assert len(calls) == 1
    assert calls[0].tool_name == "store_memory"
    assert calls[0].arguments == {"subject": "User"}


def test_normalize_forced_tools_drops_todo_write_after_task_bootstrap():
    normalized = _normalize_forced_tools_for_task_state(
        ["todo_write", "web_search", "todo_write"],
        has_tasks=True,
    )
    assert normalized == ["web_search"]


def test_build_task_progress_instruction_biases_away_from_repeating_todo_write():
    instruction = _build_task_progress_instruction(
        route_type="multi_step",
        user_message="research wigglebudget.com and create a tldr summary",
        tool_results=[
            {"tool_name": "todo_write", "success": True, "content": "Current tasks: ..."},
        ],
        has_active_tasks=True,
    )
    assert "Do not call todo_write again" in instruction
    assert "next substantive tool" in instruction


def test_build_task_progress_instruction_prefers_summary_after_evidence():
    instruction = _build_task_progress_instruction(
        route_type="multi_step",
        user_message="find competing budgeting apps and create a tldr summary",
        tool_results=[
            {"tool_name": "web_search", "success": True, "content": '{"results":[{"title":"A"}]}'},
        ],
        has_active_tasks=True,
    )
    assert "summary/report/TLDR" in instruction
    assert "Do not call todo_write again" in instruction


def test_should_include_task_reminder_drops_after_summary_evidence():
    include = _should_include_task_reminder(
        route_type="multi_step",
        user_message="research wigglebudget.com and create a tldr summary",
        tool_results=[
            {"tool_name": "web_search", "success": True, "content": '{"results":[{"title":"A"}]}'},
        ],
        has_active_tasks=True,
    )
    assert include is False


def test_sanitize_followup_messages_prunes_synthetic_and_reduces_forced_admin_tools():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "research wigglebudget.com"},
        {"role": "assistant", "content": '{"tool":"todo_write","arguments":{"tasks":[]}}'},
        {"role": "user", "content": "<SYSTEM_EVIDENCE_PACKET>\nTool: web_search\nStatus: ok\n</SYSTEM_EVIDENCE_PACKET>"},
        {"role": "assistant", "content": "Found enough evidence to summarize."},
    ]
    sanitized_messages, sanitized_forced_tools, stats = _sanitize_followup_messages(
        messages,
        forced_tools=["todo_write", "todo_update", "web_search"],
        user_message="research wigglebudget.com and create a tldr summary",
        route_type="multi_step",
        latest_tool_results=[
            {"tool_name": "web_search", "success": True, "content": '{"results":[{"title":"A"}]}'},
        ],
        has_tasks=True,
    )

    assert len(sanitized_messages) == 3
    assert sanitized_messages[0]["role"] == "system"
    assert sanitized_forced_tools == ["web_search"]
    assert stats["removed_messages"] >= 2


def test_compact_tool_result_for_followup_sanitizes_web_search_payload():
    raw = (
        '{"type":"web_search_results","query":"wigglebudget.com","backend":"tavily","engine":"tavily",'
        '"context_summary":{"requested_results":8,"curated_results":4,"unique_domains":["a.com","b.com"]},'
        '"results":[{"title":"A","url":"https://a.com","snippet":"x"*10},{"title":"B","url":"https://b.com","snippet":"y"*10}]}'
    ).replace('"x"*10', '"' + ("x" * 10) + '"').replace('"y"*10', '"' + ("y" * 10) + '"')
    compact = _compact_tool_result_for_followup("web_search", raw)
    assert '"query": "wigglebudget.com"' in compact
    assert '"unique_domains": ["a.com", "b.com"]' in compact or '"unique_domains": ["a.com","b.com"]' in compact
    assert '"results":' in compact


def test_compact_tool_result_for_followup_sanitizes_web_fetch_payload():
    raw = (
        '{"type":"web_fetch_result","url":"https://example.com","title":"Example","backend":"jina",'
        '"status_code":200,"truncated":false,"total_length":5000,"content":"' + ("a" * 1400) + '"}'
    )
    compact = _compact_tool_result_for_followup("web_fetch", raw)
    assert '"content_preview":' in compact
    assert '"total_length": 5000' in compact or '"total_length":5000' in compact
    assert '"url": "https://example.com"' in compact


def test_compact_tool_result_for_followup_sanitizes_task_state_text():
    raw = (
        "Current tasks:\n"
        "Workflow topic: budgeting apps\n"
        "- [pending] 1: Search wigglebudget.com (priority=medium)\n"
        "- [pending] 2: Compare alternatives (priority=medium)\n"
    )
    compact = _compact_tool_result_for_followup("todo_write", raw)
    assert '"type": "task_state_update"' in compact
    assert '"task_count": 2' in compact or '"task_count":2' in compact
    assert '"topic": "budgeting apps"' in compact


def test_select_followup_tool_results_prioritizes_exact_domain_fetch_over_noisy_later_search():
    tool_results = [
        {
            "tool_name": "web_search",
            "success": True,
            "content": '{"type":"web_search_results","query":"wigglebudget.com budget app review","results":[{"title":"Result A"}]}',
            "arguments": {"query": "wigglebudget.com budget app review"},
        },
        {
            "tool_name": "web_fetch",
            "success": True,
            "content": '{"type":"web_fetch_result","url":"https://wigglebudget.com/","title":"wigglebudget.com","content":"Wiggle Budget is a personal finance web app."}',
            "arguments": {"url": "https://wigglebudget.com/"},
        },
        {
            "tool_name": "web_search",
            "success": True,
            "content": '{"type":"web_search_results","query":"wigglebudget.com review Medium","results":[{"title":"Guest posting site"}]}',
            "arguments": {"query": "wigglebudget.com review Medium"},
        },
    ]

    selected = _select_followup_tool_results(
        tool_results,
        user_message="Investigate wigglebudget.com and compare competitors",
        max_results=2,
    )

    assert selected[0]["tool_name"] == "web_fetch"
    assert "wigglebudget.com" in str(selected[0]["content"]).lower()
