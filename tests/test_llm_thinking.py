import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from llm.llm_adapter import OllamaAdapter

@pytest.mark.asyncio
async def test_ollama_reasoning_extraction():
    """Verify that OllamaAdapter correctly extracts and yields reasoning tokens."""
    adapter = OllamaAdapter(base_url="http://localhost:11434")
    
    # Mock stream response content
    chunks = [
        {"message": {"content": "Here is <think>"}},
        {"message": {"content": "my reasoning"}},
        {"message": {"content": "</think> and my answer."}},
        {"done": True}
    ]
    
    async def mock_aiter():
        for chunk in chunks:
            yield json.dumps(chunk)
            
    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_aiter
    mock_response.__aenter__.return_value = mock_response
    
    # Patch the _get_client to return a mock client
    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    
    with patch.object(adapter, "_get_client", return_value=mock_client):
        tokens = []
        async for token in adapter.stream(model="deepseek-r1", messages=[{"role":"user", "content":"hi"}]):
            tokens.append(token)
            
    # Expected: <THOUGHT>, my reasoning, </THOUGHT>,  and my answer.
    # Note: "Here is " before <think> should be yielded before <THOUGHT>
    # In our impl: 
    # Chunk 1: "Here is <think>" -> yields "Here is ", then yields "<THOUGHT>"
    # Chunk 2: "my reasoning" -> yields "my reasoning"
    # Chunk 3: "</think> and my answer." -> yields "</THOUGHT>", then " and my answer."
    
    assert "<THOUGHT>" in tokens
    assert "</THOUGHT>" in tokens
    assert "my reasoning" in tokens
    assert " and my answer." in tokens
    
    # Verify sequence
    thought_idx = tokens.index("<THOUGHT>")
    end_thought_idx = tokens.index("</THOUGHT>")
    assert thought_idx < tokens.index("my reasoning") < end_thought_idx

@pytest.mark.asyncio
async def test_ollama_native_tool_call_extraction():
    """Verify that OllamaAdapter correctly extracts native tool calls."""
    adapter = OllamaAdapter(base_url="http://localhost:11434")
    
    tool_call_chunk = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "weather in Paris"}
                    }
                }
            ]
        },
        "done": True
    }
    
    async def mock_aiter():
        yield json.dumps(tool_call_chunk)
        
    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_aiter
    mock_response.__aenter__.return_value = mock_response
    
    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    
    with patch.object(adapter, "_get_client", return_value=mock_client):
        tokens = []
        async for token in adapter.stream(model="llama3.2", messages=[{"role":"user", "content":"weather?"}]):
            tokens.append(token)
            
    assert len(tokens) == 1
    data = json.loads(tokens[0])
    assert "tool_calls" in data
    assert data["tool_calls"][0]["function"]["name"] == "web_search"
