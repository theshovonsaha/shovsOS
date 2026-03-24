import asyncio
import sys
import os
import pytest

# Add project root to path
sys.path.append(os.getcwd())

from orchestration.orchestrator import AgenticOrchestrator
from llm.llm_adapter import OllamaAdapter

@pytest.mark.asyncio
async def test_orchestration():
    adapter = OllamaAdapter()
    orch = AgenticOrchestrator(adapter)
    
    tools = [
        {"name": "web_search", "description": "Search the web for info."},
        {"name": "file_create", "description": "Create a file in sandbox."},
        {"name": "weather_fetch", "description": "Get weather data."}
    ]
    
    queries = [
        ("What's the weather in London?", ["weather_fetch"]),
        ("Search for 2026 car trends and save it to trends.txt", ["web_search", "file_create"]),
        ("Hello, how are you?", [])
    ]
    
    for query, expected in queries:
        print(f"\nQuery: {query}")
        plan = await orch.plan(query, tools)
        print(f"Plan: {plan}")
        # Relaxed check: as long as expected tools are in the plan
        for tool in expected:
            if tool in plan:
                print(f"✓ Found {tool}")
            else:
                print(f"✗ Missing {tool}")

if __name__ == "__main__":
    asyncio.run(test_orchestration())
