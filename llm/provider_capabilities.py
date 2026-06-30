from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm.model_capabilities import is_vision_capable, supports_reasoning


PROVIDER_CAPABILITY_VERSION = "provider-capabilities-v1"


@dataclass(frozen=True)
class ProviderRuntimeCapabilities:
    provider: str
    model: str
    native_tools: bool
    structured_json: bool
    vision: bool
    image_generation: bool
    reasoning_control: bool
    parallel_tools: bool
    tool_choice: bool
    fallback_tool_protocol: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NATIVE_TOOL_PROVIDERS = {"openai", "local_openai", "lmstudio", "llamacpp", "groq", "gemini", "anthropic", "ollama"}
_STRUCTURED_JSON_PROVIDERS = {"openai", "local_openai", "lmstudio", "llamacpp", "groq", "gemini", "anthropic", "ollama"}
_TOOL_CHOICE_PROVIDERS = {"openai", "local_openai", "lmstudio", "llamacpp", "groq", "anthropic"}
_PARALLEL_TOOL_PROVIDERS = {"openai", "local_openai", "lmstudio", "llamacpp", "groq"}
_IMAGE_GENERATION_MARKERS = ("image", "dall-e", "gpt-image", "imagen", "flux", "sdxl", "stable-diffusion")


def infer_provider_id(model: str | None, *, adapter_name: str = "") -> str:
    text = str(model or "").strip()
    if ":" in text:
        prefix = text.split(":", 1)[0].strip().lower()
        if prefix:
            return prefix
    adapter = str(adapter_name or "").lower()
    if "openai" in adapter:
        return "openai"
    if "gemini" in adapter or "google" in adapter:
        return "gemini"
    if "anthropic" in adapter or "claude" in adapter:
        return "anthropic"
    if "groq" in adapter:
        return "groq"
    if "ollama" in adapter:
        return "ollama"
    return "unknown"


def provider_runtime_capabilities(
    model: str | None,
    *,
    adapter_name: str = "",
    image_generation_enabled: bool | None = None,
) -> ProviderRuntimeCapabilities:
    provider = infer_provider_id(model, adapter_name=adapter_name)
    model_id = str(model or "").strip()
    native_tools = provider in _NATIVE_TOOL_PROVIDERS
    structured_json = provider in _STRUCTURED_JSON_PROVIDERS
    vision = is_vision_capable(model_id)
    reasoning = supports_reasoning(model_id)
    image_generation = (
        bool(image_generation_enabled)
        if image_generation_enabled is not None
        else any(marker in model_id.lower() for marker in _IMAGE_GENERATION_MARKERS)
    )
    notes: list[str] = []
    if not native_tools:
        notes.append("use_json_tool_draft_fallback")
    if not structured_json:
        notes.append("validate_freeform_output")
    if vision:
        notes.append("image_inputs_allowed")
    if image_generation:
        notes.append("image_generation_allowed")
    if reasoning:
        notes.append("reasoning_flag_supported")
    return ProviderRuntimeCapabilities(
        provider=provider,
        model=model_id,
        native_tools=native_tools,
        structured_json=structured_json,
        vision=vision,
        image_generation=image_generation,
        reasoning_control=reasoning,
        parallel_tools=provider in _PARALLEL_TOOL_PROVIDERS,
        tool_choice=provider in _TOOL_CHOICE_PROVIDERS,
        fallback_tool_protocol="native_tools" if native_tools else "json_tool_draft",
        notes=notes,
    )


def render_provider_capabilities(capabilities: ProviderRuntimeCapabilities) -> str:
    flags = capabilities.to_dict()
    lines = [
        "Provider Runtime Capabilities:",
        f"- provider: {capabilities.provider}",
        f"- model: {capabilities.model or 'not recorded'}",
    ]
    for key in (
        "native_tools",
        "structured_json",
        "vision",
        "image_generation",
        "reasoning_control",
        "parallel_tools",
        "tool_choice",
    ):
        lines.append(f"- {key}: {str(bool(flags[key])).lower()}")
    lines.append(f"- fallback: {capabilities.fallback_tool_protocol}")
    if capabilities.notes:
        lines.append("- notes: " + ", ".join(capabilities.notes))
    return "\n".join(lines)
