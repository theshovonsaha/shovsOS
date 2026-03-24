import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.trace_store import get_trace_store
from engine.core import AgentCore, DEFAULT_PLANNER_FALLBACK_MODEL, _normalize_loop_mode
from orchestration.run_store import get_run_store
from orchestration.session_manager import SessionManager
from plugins.tool_registry import Tool, ToolRegistry


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


def _unique_session(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _session_events(session_id: str, limit: int = 80) -> list[dict]:
    store = get_trace_store()
    index_events = store.list_events(limit=limit, session_id=session_id)
    return [store.get_event(item["id"]) or item for item in reversed(index_events)]


def _event_data(events: list[dict], event_type: str) -> list[dict]:
    matches = []
    for event in events:
        if event.get("event_type") == event_type:
            matches.append(event.get("data") or {})
    return matches


def test_auto_loop_mode_prefers_single_for_local_runtime():
    class FakeOllamaAdapter:
        pass

    requested, effective = _normalize_loop_mode(
        "auto",
        use_planner=True,
        has_orchestrator=True,
        is_trivial_turn=False,
        adapter=FakeOllamaAdapter(),
        model="ollama:llama3.2",
    )

    assert requested == "auto"
    assert effective == "single"


@pytest.mark.asyncio
async def test_trivial_turn_trace_captures_route_and_prompt_layers():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["Hello there."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=ToolRegistry(),
        orchestrator=None,
    )

    session_id = _unique_session("trace_trivial")
    _ = [event async for event in core.chat_stream("hi", session_id=session_id, use_planner=False)]
    events = _session_events(session_id)

    route = _event_data(events, "route_decision")[-1]
    retrieval = _event_data(events, "retrieval_policy")[-1]
    task_policy = _event_data(events, "task_policy")[-1]
    compiled = _event_data(events, "compiled_context")[-1]
    prompt = _event_data(events, "llm_prompt")[-1]
    response = _event_data(events, "assistant_response")[-1]

    assert route["route_type"] == "trivial_chat"
    assert retrieval["should_retrieve"] is False
    assert task_policy["task_state_injected"] is False
    assert compiled["phase"] == "acting"
    assert compiled["summary"]["included_count"] >= 2
    assert "instruction" in compiled["summary"]["included_kinds"]
    assert isinstance(prompt["messages"], list)
    assert prompt["messages"][-1]["role"] == "user"
    assert prompt["messages"][-1]["content"] == "hi"
    assert "Hello there." in response["content"]


@pytest.mark.asyncio
async def test_multistep_turn_trace_shows_todo_bias_and_task_policy():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["I will break this into steps."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(Tool(
        name="todo_write",
        description="Initialize session task list.",
        parameters={
            "type": "object",
            "properties": {
                "tasks": {"type": "array", "items": {"type": "object"}},
                "topic": {"type": "string"},
            },
            "required": ["tasks"],
        },
        handler=AsyncMock(return_value="Current tasks:\n- [pending] task-1: gather notes"),
    ))

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=None,
    )

    session_id = _unique_session("trace_multistep")
    events = [event async for event in core.chat_stream(
        "research local ai tools then summarize and save notes",
        session_id=session_id,
        use_planner=False,
    )]
    trace_events = _session_events(session_id)

    route = _event_data(trace_events, "route_decision")[-1]
    task_policy = _event_data(trace_events, "task_policy")[-1]
    plan_events = [event for event in events if event.get("type") == "plan"]

    assert route["route_type"] == "multi_step"
    assert task_policy["has_tasks"] is False
    assert any("todo_write" in (event.get("tools") or []) for event in plan_events)


@pytest.mark.asyncio
async def test_tool_execution_trace_captures_prompt_tool_call_and_clean_response():
    class TwoPassAdapter:
        def __init__(self):
            self.pass_index = 0

        async def stream(self, *args, **kwargs):
            if self.pass_index == 0:
                self.pass_index += 1
                for token in [
                    '{"tool":"file_create","arguments":{"path":"demo.txt","content":"hello world"}}'
                ]:
                    yield token
                return
            for token in ["Created the file successfully."]:
                yield token

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()

    async def _file_create(path: str, content: str, **kwargs) -> str:
        return f"created {path} with {len(content)} chars"

    registry.register(Tool(
        name="file_create",
        description="Create a file in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=_file_create,
    ))

    core = AgentCore(
        adapter=TwoPassAdapter(),
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=None,
    )

    session_id = _unique_session("trace_tool")
    with patch("engine.core.VectorEngine") as mock_ve:
        ve = MagicMock()
        ve.query = AsyncMock(return_value=[])
        ve.index = AsyncMock(return_value=None)
        mock_ve.return_value = ve

        events = [event async for event in core.chat_stream(
            "create a demo file",
            session_id=session_id,
            use_planner=False,
        )]

    trace_events = _session_events(session_id, limit=120)
    tool_calls = _event_data(trace_events, "tool_call")
    tool_results = _event_data(trace_events, "tool_result")
    prompts = _event_data(trace_events, "llm_prompt")
    response = _event_data(trace_events, "assistant_response")[-1]
    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    artifacts = get_run_store().list_artifacts(run_id)

    assert any(call["tool_name"] == "file_create" for call in tool_calls)
    assert any(result["tool_name"] == "file_create" for result in tool_results)
    assert len(prompts) >= 2
    assert '{"tool":"file_create"' not in response["content"]
    assert "Created the file successfully." in response["content"]
    assert any(item.artifact_type == "file" and item.label == "demo.txt" for item in artifacts)


@pytest.mark.asyncio
async def test_forced_tool_retry_recovers_when_small_model_ignores_tool_on_first_pass():
    class RetryAdapter:
        def __init__(self):
            self.pass_index = 0

        async def stream(self, *args, **kwargs):
            if self.pass_index == 0:
                self.pass_index += 1
                yield "I can help with that research."
                return
            if self.pass_index == 1:
                self.pass_index += 1
                yield '{"tool":"web_search","arguments":{"query":"wigglebudget.com"}}'
                return
            yield "I found relevant information."

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=AsyncMock(return_value='{"ok": true}'),
        )
    )

    core = AgentCore(
        adapter=RetryAdapter(),
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=None,
    )

    session_id = _unique_session("forced_retry")
    events = [event async for event in core.chat_stream(
        "research wigglebudget.com",
        session_id=session_id,
        use_planner=False,
        loop_mode="single",
        forced_tools=["web_search"],
        max_tool_calls=1,
        max_turns=3,
    )]
    trace_events = _session_events(session_id, limit=120)

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in events)
    retries = _event_data(trace_events, "forced_tool_retry")
    assert retries
    assert retries[-1]["tools"] == ["web_search"]


@pytest.mark.asyncio
async def test_todo_write_is_suppressed_after_task_bootstrap_within_same_run():
    class RepeatTodoAdapter:
        def __init__(self):
            self.pass_index = 0

        async def stream(self, *args, **kwargs):
            if self.pass_index == 0:
                self.pass_index += 1
                yield '{"tool":"todo_write","arguments":{"tasks":[{"id":"1","content":"Research wigglebudget.com","status":"pending","priority":"medium"}],"topic":"budgeting apps"}}'
                return
            if self.pass_index == 1:
                self.pass_index += 1
                yield '{"tool":"todo_write","arguments":{"tasks":[{"id":"1","content":"Research wigglebudget.com","status":"in_progress","priority":"medium"}],"topic":"budgeting apps"}}'
                return
            yield "I will continue with the actual research next."

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="todo_write",
            description="Initialize task list.",
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "object"}},
                    "topic": {"type": "string"},
                },
                "required": ["tasks"],
            },
            handler=AsyncMock(return_value="Current tasks:\n- [pending] 1: Research wigglebudget.com"),
        )
    )

    core = AgentCore(
        adapter=RepeatTodoAdapter(),
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=None,
    )

    session_id = _unique_session("todo_bootstrap")
    events = [event async for event in core.chat_stream(
        "research wigglebudget.com and summarize it",
        session_id=session_id,
        use_planner=False,
        loop_mode="single",
        forced_tools=["todo_write"],
        max_tool_calls=2,
        max_turns=4,
    )]
    trace_events = _session_events(session_id, limit=160)

    suppressed = _event_data(trace_events, "task_bootstrap_suppressed")
    todo_results = [
        event for event in events
        if event.get("type") == "tool_result" and event.get("tool_name") == "todo_write"
    ]

    assert suppressed
    assert len(todo_results) == 2
    assert todo_results[0]["success"] is True
    assert todo_results[1]["success"] is False
    assert "already initialized task state" in todo_results[1]["content"]


@pytest.mark.asyncio
async def test_terminal_tool_call_is_recovered_into_final_answer_turn():
    class TerminalToolAdapter:
        def __init__(self):
            self.pass_index = 0

        async def stream(self, *args, **kwargs):
            if self.pass_index == 0:
                self.pass_index += 1
                yield '{"tool":"web_search","arguments":{"query":"wigglebudget.com"}}'
                return
            if self.pass_index == 1:
                self.pass_index += 1
                yield '{"tool":"web_search","arguments":{"query":"best budgeting apps canada"}}'
                return
            yield "Final answer from gathered evidence."

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=AsyncMock(return_value='{"type":"web_search_results","query":"wigglebudget.com","results":[{"title":"WiggleBudget","url":"https://wigglebudget.com","snippet":"Budgeting tool"}]}'),
        )
    )

    core = AgentCore(
        adapter=TerminalToolAdapter(),
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=None,
    )

    session_id = _unique_session("terminal_tool_recovery")
    events = [event async for event in core.chat_stream(
        "research wigglebudget.com and summarize it",
        session_id=session_id,
        use_planner=False,
        loop_mode="single",
        forced_tools=["web_search"],
        max_tool_calls=2,
        max_turns=2,
    )]
    trace_events = _session_events(session_id, limit=160)

    recoveries = _event_data(trace_events, "terminal_tool_call_recovered")
    responses = _event_data(trace_events, "assistant_response")

    assert recoveries
    assert responses
    assert responses[-1]["content"] == "Final answer from gathered evidence."


@pytest.mark.asyncio
async def test_planner_defaults_to_large_fallback_model():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["Done."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "No tools needed.",
        "tools": [],
        "force_memory": False,
        "confidence": 0.4,
    })
    orchestrator.set_adapter = MagicMock()

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=ToolRegistry(),
        orchestrator=orchestrator,
    )

    _ = [event async for event in core.chat_stream(
        "research the architecture and summarize it",
        session_id=_unique_session("planner_fallback"),
        model="llama3.2",
        use_planner=True,
    )]

    assert orchestrator.plan_with_context.await_args.kwargs["model"] == DEFAULT_PLANNER_FALLBACK_MODEL
    assert orchestrator.plan_with_context.await_args.kwargs["compiled_context"]


@pytest.mark.asyncio
async def test_phase_contexts_and_run_phases_are_traced():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["Done."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "Search first.",
        "tools": [{"name": "web_search", "priority": "high", "reason": "Need current data"}],
        "force_memory": False,
        "confidence": 0.6,
    })
    orchestrator.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=AsyncMock(return_value="search results"),
    ))

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=orchestrator,
    )

    session_id = _unique_session("phase_trace")
    _ = [event async for event in core.chat_stream(
        "research the architecture and summarize it",
        session_id=session_id,
        model="llama3.2",
        use_planner=True,
    )]

    events = _session_events(session_id, limit=160)
    phase_contexts = _event_data(events, "phase_context")
    run_phases = _event_data(events, "run_phase")

    assert {item["phase"] for item in phase_contexts} >= {"planning", "response", "memory_commit"}
    assert any(item["phase"] == "planning" and item["status"] == "start" for item in run_phases)
    assert any(item["phase"] == "acting" and item["status"] == "complete" for item in run_phases)
    assert any(item["phase"] == "memory_commit" and item["status"] == "complete" for item in run_phases)


@pytest.mark.asyncio
async def test_manager_observation_and_verification_are_traced_inside_single_loop():
    class TwoPassAdapter:
        def __init__(self):
            self.pass_index = 0

        async def stream(self, *args, **kwargs):
            if self.pass_index == 0:
                self.pass_index += 1
                yield '{"tool":"file_create","arguments":{"path":"demo.txt","content":"hello world"}}'
                return
            yield "Created the file successfully."

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()

    async def _file_create(path: str, content: str, **kwargs) -> str:
        return f"created {path} with {len(content)} chars"

    registry.register(Tool(
        name="file_create",
        description="Create a file in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=_file_create,
    ))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "Create the file first.",
        "tools": [{"name": "file_create", "priority": "high", "reason": "Need the file artifact"}],
        "force_memory": False,
        "confidence": 0.9,
    })
    orchestrator.observe_with_context = AsyncMock(return_value={
        "status": "finalize",
        "strategy": "Enough evidence gathered from the file operation.",
        "tools": [],
        "notes": "Answer directly now.",
        "confidence": 0.85,
    })
    orchestrator.verify_with_context = AsyncMock(return_value={
        "supported": True,
        "issues": [],
        "confidence": 0.92,
    })
    orchestrator.set_adapter = MagicMock()

    core = AgentCore(
        adapter=TwoPassAdapter(),
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=orchestrator,
    )

    session_id = _unique_session("manager_loop")
    events = [event async for event in core.chat_stream(
        "create a demo file",
        session_id=session_id,
        use_planner=True,
    )]

    trace_events = _session_events(session_id, limit=180)
    run_phases = _event_data(trace_events, "run_phase")
    manager_observations = [event for event in events if event.get("type") == "manager_observation"]
    verification_results = _event_data(trace_events, "verification_result")
    loop_checkpoints = _event_data(trace_events, "loop_checkpoint")
    loop_modes = _event_data(trace_events, "runtime_loop_mode")
    evidence_packets = _event_data(trace_events, "managed_evidence_packet")
    minimizations = _event_data(trace_events, "managed_prompt_minimization")
    prompts = _event_data(trace_events, "llm_prompt")
    phase_contexts = _event_data(trace_events, "phase_context")
    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    stored_checkpoints = get_run_store().list_checkpoints(run_id)
    stored_evals = get_run_store().list_evals(run_id)
    stored_artifacts = get_run_store().list_artifacts(run_id)

    assert manager_observations
    assert loop_modes[-1]["effective"] == "managed"
    assert evidence_packets
    assert evidence_packets[-1]["compact"] is True
    assert minimizations
    assert minimizations[-1]["removed_messages"] >= 0
    assert orchestrator.observe_with_context.await_count == 1
    assert orchestrator.verify_with_context.await_count == 1
    assert any(item["phase"] == "observation" and item["status"] == "complete" for item in run_phases)
    assert any(item["phase"] == "verification" and item["status"] == "complete" for item in run_phases)
    assert verification_results[-1]["supported"] is True
    assert any(item["phase"] == "planning" for item in loop_checkpoints)
    assert any(item["phase"] == "acting" for item in loop_checkpoints)
    assert any(item["phase"] == "observation" for item in loop_checkpoints)
    assert any(item["phase"] == "verification" for item in loop_checkpoints)
    assert any(item["phase"] == "commit" for item in loop_checkpoints)
    assert any(checkpoint.phase == "verification" and checkpoint.status == "supported" for checkpoint in stored_checkpoints)
    assert any(item.eval_type == "response_support" and item.passed for item in stored_evals)
    assert any(item.artifact_type == "file" and item.label == "demo.txt" for item in stored_artifacts)
    assert any("<SYSTEM_OBSERVATION>" in prompt["messages"][-1]["content"] for prompt in prompts if len(prompt.get("messages", [])) >= 2)
    assert any("<SYSTEM_EVIDENCE_PACKET>" in prompt["messages"][-1]["content"] for prompt in prompts if len(prompt.get("messages", [])) >= 2)
    assert any(
        not any(
            message.get("role") == "assistant" and '{"tool":"file_create"' in message.get("content", "")
            for message in prompt.get("messages", [])
        )
        for prompt in prompts[1:]
    )
    assert any(
        item["phase"] == "verification"
        and any(entry.get("item_id") == "observation_state" for entry in item.get("included", []))
        for item in phase_contexts
    )


@pytest.mark.asyncio
async def test_single_loop_mode_bypasses_manager_controller_even_if_planner_enabled():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["Plain response."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "Should not run.",
        "tools": [{"name": "web_search", "priority": "high", "reason": "Should not run."}],
        "force_memory": False,
        "confidence": 0.9,
    })
    orchestrator.observe_with_context = AsyncMock()
    orchestrator.verify_with_context = AsyncMock()
    orchestrator.set_adapter = MagicMock()

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=ToolRegistry(),
        orchestrator=orchestrator,
    )

    session_id = _unique_session("single_mode")
    _ = [event async for event in core.chat_stream(
        "research the architecture and summarize it",
        session_id=session_id,
        use_planner=True,
        loop_mode="single",
    )]

    trace_events = _session_events(session_id, limit=120)
    loop_modes = _event_data(trace_events, "runtime_loop_mode")
    loop_checkpoints = _event_data(trace_events, "loop_checkpoint")
    verification_results = _event_data(trace_events, "verification_result")

    assert loop_modes[-1]["requested"] == "single"
    assert loop_modes[-1]["effective"] == "single"
    assert orchestrator.plan_with_context.await_count == 0
    assert orchestrator.observe_with_context.await_count == 0
    assert orchestrator.verify_with_context.await_count == 0
    assert not verification_results
    assert not any(item["phase"] == "planning" for item in loop_checkpoints)
    assert any(item["phase"] == "acting" for item in loop_checkpoints)
    assert any(item["phase"] == "commit" for item in loop_checkpoints)


@pytest.mark.asyncio
async def test_verification_warning_blocks_memory_commit():
    class TwoPassAdapter:
        def __init__(self):
            self.pass_index = 0

        async def stream(self, *args, **kwargs):
            if self.pass_index == 0:
                self.pass_index += 1
                yield '{"tool":"file_create","arguments":{"path":"demo.txt","content":"hello world"}}'
                return
            yield "The file was created and also uploaded to production."

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()

    async def _file_create(path: str, content: str, **kwargs) -> str:
        return f"created {path} with {len(content)} chars"

    registry.register(Tool(
        name="file_create",
        description="Create a file in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=_file_create,
    ))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "Create the file first.",
        "tools": [{"name": "file_create", "priority": "high", "reason": "Need the file artifact"}],
        "force_memory": False,
        "confidence": 0.9,
    })
    orchestrator.observe_with_context = AsyncMock(return_value={
        "status": "finalize",
        "strategy": "Enough evidence gathered from the file operation.",
        "tools": [],
        "notes": "",
        "confidence": 0.85,
    })
    orchestrator.verify_with_context = AsyncMock(return_value={
        "supported": False,
        "issues": ["Answer claims an upload that never happened."],
        "confidence": 0.93,
    })
    orchestrator.set_adapter = MagicMock()

    core = AgentCore(
        adapter=TwoPassAdapter(),
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=orchestrator,
    )

    session_id = _unique_session("verify_block")
    events = [event async for event in core.chat_stream(
        "create a demo file",
        session_id=session_id,
        use_planner=True,
    )]

    assert any(event["type"] == "verification_warning" for event in events)
    context_engine.compress_exchange.assert_not_awaited()


@pytest.mark.asyncio
async def test_thought_tokens_do_not_enter_stored_response():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter([
        "<think>",
        "private chain of thought",
        "</think>",
        "Public answer only.",
    ]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    session_manager = SessionManager()
    session_id = _unique_session("thought_filter")
    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=session_manager,
        tool_registry=ToolRegistry(),
        orchestrator=None,
    )

    _ = [event async for event in core.chat_stream("answer briefly", session_id=session_id, use_planner=False)]
    session = session_manager.get_or_create(session_id=session_id, model="llama3.2", system_prompt="test", agent_id="default")

    assert session.sliding_window[-1]["role"] == "assistant"
    assert session.sliding_window[-1]["content"] == "Public answer only."
