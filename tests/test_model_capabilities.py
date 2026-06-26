from llm.model_capabilities import capability_flags, is_vision_capable


def test_vision_capability_detection_for_cloud_and_local_models():
    assert is_vision_capable("openai:gpt-4o-mini") is True
    assert is_vision_capable("gemini:gemini-1.5-flash") is True
    assert is_vision_capable("anthropic:claude-3-5-sonnet-latest") is True
    assert is_vision_capable("ollama:llava") is True
    assert is_vision_capable("llava:latest") is True
    assert is_vision_capable("ollama:qwen2.5-coder") is False
    assert is_vision_capable("nomic-embed-text:latest") is False


def test_capability_flags_include_vision_without_changing_chat_class():
    flags = capability_flags("openai:gpt-4o-mini")

    assert flags["chat"] is True
    assert flags["vision"] is True
    assert flags["embed"] is False
    assert flags["class"] == "chat"
