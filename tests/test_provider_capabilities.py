from llm.provider_capabilities import (
    infer_provider_id,
    provider_runtime_capabilities,
    render_provider_capabilities,
)


def test_infer_provider_id_prefers_model_prefix():
    assert infer_provider_id("gemini:gemini-2.5-flash", adapter_name="OpenAIAdapter") == "gemini"


def test_provider_capabilities_known_native_tool_provider():
    caps = provider_runtime_capabilities("openai:gpt-5")

    assert caps.provider == "openai"
    assert caps.native_tools is True
    assert caps.structured_json is True
    assert caps.reasoning_control is True
    assert caps.fallback_tool_protocol == "native_tools"


def test_provider_capabilities_unknown_provider_falls_back_to_json_draft():
    caps = provider_runtime_capabilities("custom-model-7b")

    assert caps.provider == "unknown"
    assert caps.native_tools is False
    assert caps.structured_json is False
    assert caps.fallback_tool_protocol == "json_tool_draft"
    assert "use_json_tool_draft_fallback" in caps.notes


def test_provider_capabilities_vision_and_image_generation_are_separate():
    vision_caps = provider_runtime_capabilities("ollama:qwen2.5-vl")
    image_caps = provider_runtime_capabilities("openai:gpt-image-1")

    assert vision_caps.vision is True
    assert vision_caps.image_generation is False
    assert image_caps.image_generation is True


def test_render_provider_capabilities_is_plain_text_contract():
    caps = provider_runtime_capabilities("gemini:gemini-3-flash")
    rendered = render_provider_capabilities(caps)

    assert "Provider Runtime Capabilities:" in rendered
    assert "- provider: gemini" in rendered
    assert "- native_tools: true" in rendered
    assert "- fallback: native_tools" in rendered
