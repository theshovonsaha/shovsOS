import pytest
import asyncio
import os
import json
import uuid
from unittest.mock import patch
from typing import AsyncGenerator

from api.main import app
from llm.adapter_factory import create_adapter
from orchestration.agent_profiles import ProfileManager, AgentProfile
from orchestration.session_manager import SessionManager
from engine.context_engine import ContextEngine
from orchestration.orchestrator import AgenticOrchestrator
from orchestration.agent_manager import AgentManager
from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools

from dotenv import load_dotenv

load_dotenv()

# The specific trace log we will capture data into
TRACE_FILE = "agent_trace.jsonl"

def append_trace(event_type: str, data: dict):
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps({"event": event_type, "data": data}) + "\n")

@pytest.mark.asyncio
@patch('memory.vector_engine.VectorEngine.query')
@patch('memory.vector_engine.VectorEngine.index')
async def test_deep_groq_simulation(mock_index, mock_query):
    """
    Simulates a complex real-world prompt using Groq's API models.
    Tracks all internal tool calls, planner phases, and delegation flows.
    """
    # Mock RAG to return empty to avoid local Ollama requirements
    mock_query.return_value = []
    
    print("\n--- Starting Deep Simulation with Groq API ---")
    
    # 1. Setup Infrastructure targeting Groq directly
    adapter = create_adapter("groq")
    if not await adapter.health():
        pytest.skip("Groq API unavailable (missing credentials or network).")
    
    # Strongest reasoning model for the Main Agents
    mother_model = "llama-3.3-70b-versatile"
    
    # Fast model for the Planner orchestration and Context Compression
    planner_model = "llama-3.2-3b-preview"
    
    registry = ToolRegistry()
    session_mgr = SessionManager(max_sessions=10)
    context_eng = ContextEngine(adapter=adapter, compression_model=planner_model)
    profile_mgr = ProfileManager()
    
    # Explicitly instantiate the orchestrator
    orchestrator = AgenticOrchestrator(adapter=adapter)
    
    # Ensure default agent uses the massive model
    p_default = profile_mgr.get("default")
    if p_default:
        p_default.model = mother_model
        p_default.system_prompt = "You are the Mother Agent. You have full access to tools. Plan carefully."
        profile_mgr.create(p_default)
        
    # Create the specialized child agent for downstream delegation
    profile_mgr.create(AgentProfile(
        id="analyst",
        name="Data Analyst Agent",
        model=mother_model, # Analyst also needs strong reasoning to create good files
        tools=["file_create", "bash"] # Give it sandbox access only
    ))
    
    manager = AgentManager(
        profiles=profile_mgr,
        sessions=session_mgr,
        context_engine=context_eng,
        adapter=adapter,
        global_registry=registry,
        orchestrator=orchestrator
    )
    
    register_all_tools(registry, agent_manager=manager)
    
    # The session we will track
    simulation_session_id = f"sim_{uuid.uuid4().hex[:8]}"
    
    # The complex prompt demanding Search + Data Extraction + Delegation + File IO
    query = (
        "Search the web for the current stock price of Apple (AAPL). Extract the exact price. "
        "Then delegate to the 'analyst' agent to write a file called `aapl_report.md` in the sandbox with that price and a short summary."
    )
    
    mother_agent = manager.get_agent_instance("default", model_override=mother_model)
    
    print(f"\nUser Query: {query}")
    append_trace("USER_QUERY", {"session_id": simulation_session_id, "query": query})
    
    full_response = ""
    
    # We yield from the main loop to capture SSE events
    async for event in mother_agent.chat_stream(query, session_id=simulation_session_id):
        etype = event.get("type")
        content = event.get("content", "")
        
        if etype == "token":
            print(content, end="", flush=True)
            full_response += content
            
        elif etype == "plan":
            strategy = event.get("strategy")
            print(f"\n[PLAN STAGE] {strategy}")
            append_trace("PLANNER", {"strategy": strategy})
            
        elif etype == "tool_call":
            tool_name = event.get("tool")
            args = event.get("arguments")
            print(f"\n[TOOL CALLED] {tool_name} with args: {args}")
            append_trace("TOOL_CALL", {"tool": tool_name, "args": args})
            
        elif etype == "tool_result":
            # We truncate large web search results for the terminal output
            res = str(content)
            trunc_res = res[:150] + "..." if len(res) > 150 else res
            print(f"\n[TOOL RESULT INJECTED] {trunc_res}")
            append_trace("TOOL_RESULT", {"result": res})
            
        elif etype == "context_updated":
            print(f"\n[CONTEXT COMPRESSED] Background Memory updated.")
            append_trace("CONTEXT", {"session_id": simulation_session_id})
            
    print("\n\n--- Simulation Complete ---")
    append_trace("FINAL_RESPONSE", {"content": full_response})
    
    # Assertions for the simulation correctness
    assert "aapl_report.md" in full_response or "created" in full_response.lower(), "Mother agent failed to confirm file creation."
    
    # Verify the file was actually written to the sandbox by the child agent
    from plugins.tools import SANDBOX_DIR
    expected_file = SANDBOX_DIR / "aapl_report.md"
    assert expected_file.exists(), "The analyst agent failed to create the markdown file in the sandbox."
    
    # Clean up the test file
    expected_file.unlink(missing_ok=True)
    print("\n✅ Deep Simulation Test Passed! Data flushed to agent_trace.jsonl")
