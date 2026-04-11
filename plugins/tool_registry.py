"""
Tool Registry
-------------
Modular plugin system. Register tools; AgentCore routes to them.

Protocol:
  LLM emits:    {"tool": "name", "arguments": {...}}
  System calls: handler(**arguments)
  Result injects back as tool_result in next user message.
  LLM continues reasoning with result in context.

FIX v2: Replaced fragile regex with proper JSON scanner that handles:
  - Nested objects in arguments (bash commands, complex queries)
  - Multiple JSON objects in the same output (takes first valid tool call)
  - Whitespace variations and model quirks

To add a tool:
  1. Write an async handler function
  2. Define its JSON Schema parameters
  3. registry.register(Tool(...))
"""

import json
import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class Tool:
    name:        str
    description: str
    parameters:  dict
    handler:     Callable
    tags:        list[str] = field(default_factory=list)
    response_format: str = "text"


@dataclass
class ToolCall:
    tool_name:  str
    arguments:  dict
    raw_json:   str


@dataclass
class ToolResult:
    tool_name: str
    success:   bool
    content:   str


@dataclass
class HookResult:
    decision: str = "allow"  # allow | deny | transform
    reason: Optional[str] = None
    system_message: Optional[str] = None
    transformed_arguments: Optional[dict] = None
    transformed_content: Optional[str] = None
    transformed_success: Optional[bool] = None


BeforeHook = Callable[[str, dict], Awaitable[Optional[HookResult]] | Optional[HookResult]]
AfterHook = Callable[[str, dict, ToolResult], Awaitable[Optional[HookResult]] | Optional[HookResult]]


def _extract_json_objects(text: str) -> list[tuple[dict, str]]:
    """
    Extract all valid JSON objects from text using a brace-counting scanner.
    Handles nested objects, escaped strings — things regex cannot.
    Returns list of (parsed_dict, original_substring) tuples.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        # Found a potential object start — scan to find matching close brace
        depth = 0
        in_string = False
        escape_next = False
        start = i
        j = i
        while j < len(text):
            ch = text[j]
            if escape_next:
                escape_next = False
                j += 1
                continue
            if ch == '\\' and in_string:
                escape_next = True
                j += 1
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:j+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                results.append((parsed, candidate))
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1

        # Fallback: if we reached the end of the string but depth > 0
        # (happens when LLM emits <|tool_call_end|> without closing braces)
        if depth > 0 and j == len(text):
            candidate = text[start:]
            # Strip instruction model stopping tokens if they leaked
            clean_cand = candidate.split("<|tool")[0].strip()
            # Guess missing closing braces
            close_braces = "}" * depth
            try:
                parsed = json.loads(clean_cand + close_braces)
                if isinstance(parsed, dict):
                    # For original_substring tracking, return the full raw candidate so core can strip it
                    results.append((parsed, candidate))
            except json.JSONDecodeError:
                pass

        i = j + 1
    return results


class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._before_hooks: list[tuple[re.Pattern[str], BeforeHook]] = []
        self._after_hooks: list[tuple[re.Pattern[str], AfterHook]] = []
        self._tools_block_cache: Optional[str] = None

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        self._tools_block_cache = None
        print(f"[ToolRegistry] Registered: {tool.name}")

    def unregister(self, name: str):
        self._tools.pop(name, None)
        self._tools_block_cache = None

    def register_before_hook(self, matcher: str, hook: BeforeHook):
        """Register a hook that runs before tool execution."""
        self._before_hooks.append((re.compile(matcher), hook))

    def register_after_hook(self, matcher: str, hook: AfterHook):
        """Register a hook that runs after tool execution."""
        self._after_hooks.append((re.compile(matcher), hook))

    def before_tool(self, matcher: str):
        """Decorator form of register_before_hook."""
        def decorator(fn: BeforeHook):
            self.register_before_hook(matcher, fn)
            return fn
        return decorator

    def after_tool(self, matcher: str):
        """Decorator form of register_after_hook."""
        def decorator(fn: AfterHook):
            self.register_after_hook(matcher, fn)
            return fn
        return decorator

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has_tools(self) -> bool:
        return bool(self._tools)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def get_schemas(self) -> list[dict]:
        """Return tools in standard OpenAPI/function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self._tools.values()
        ]

    def build_tools_block(self, allowed_names: Optional[set[str]] = None) -> str:
        """Inject tool descriptions into system prompt."""
        if not self._tools:
            return ""

        if allowed_names is None and self._tools_block_cache is not None:
            return self._tools_block_cache
            
        tool_docs = []
        for t in self._tools.values():
            if allowed_names is not None and t.name not in allowed_names:
                continue
            params = t.parameters.get("properties", {})
            required = t.parameters.get("required", [])
            
            param_str = []
            for name, prop in params.items():
                req_mark = "(REQUIRED)" if name in required else "(optional)"
                desc = prop.get("description", "")
                ptype = prop.get("type", "string")
                param_str.append(f'      "{name}": <{ptype}> {req_mark} - {desc}')
                
            args_block = "{\n" + ",\n".join(param_str) + "\n    }" if params else "{}"
            
            tool_docs.append(
                f"  Tool: {t.name}\n"
                f"  Description: {t.description}\n"
                f"  Arguments Schema:\n    {args_block}"
            )
            
        doc_string = "\n\n".join(tool_docs)

        block = (
            "--- Available Tools ---\n"
            "To use a tool, you MUST output ONLY the following JSON on its own line and then STOP:\n"
            '{"tool": "<tool_name>", "arguments": {<args>}}\n\n'
            "CRITICAL: Do NOT output the tool's schema definition. You must output the actual invocation with real values.\n"
            "CRITICAL: You MUST escape all double quotes (\") inside your JSON strings! For example, write \"<html lang=\\\"en\\\">\" instead of \"<html lang=\"en\">\". Unescaped quotes will break the parser.\n\n"
            f"{doc_string}\n"
            "--- End Tools ---"
        )
        if allowed_names is None:
            self._tools_block_cache = block
        return block

    def detect_tool_call(self, text: str) -> Optional[ToolCall]:
        """
        Parse LLM output for a tool call JSON block.
        Uses brace-counting JSON scanner — handles nested objects correctly.
        Returns first valid tool call found, or None.
        """
        calls = self.detect_tool_calls(text)
        return calls[0] if calls else None

    def detect_tool_calls(self, text: str) -> list[ToolCall]:
        """
        Parse LLM output for all tool call JSON blocks.
        Returns a list of all valid tool calls found.
        """
        calls = []
        candidates = _extract_json_objects(text)
        for obj, original_text in candidates:
            tool_name = obj.get("tool")
            arguments = obj.get("arguments")

            # Common small-model variations:
            # 1. {"tool_name": "web_search", "params": {...}}
            # 2. {"web_search": {...}}
            # 3. {"name": "web_search", "input": {...}}
            if not (isinstance(tool_name, str) and isinstance(arguments, dict)):
                alt_tool_name = obj.get("tool_name") or obj.get("name")
                alt_arguments = obj.get("arguments")
                if not isinstance(alt_arguments, dict):
                    alt_arguments = obj.get("params")
                if not isinstance(alt_arguments, dict):
                    alt_arguments = obj.get("parameters")
                if not isinstance(alt_arguments, dict):
                    alt_arguments = obj.get("input")

                if isinstance(alt_tool_name, str) and isinstance(alt_arguments, dict):
                    tool_name = alt_tool_name
                    arguments = alt_arguments
                elif len(obj) == 1:
                    only_key, only_value = next(iter(obj.items()))
                    if isinstance(only_key, str) and isinstance(only_value, dict) and only_key in self._tools:
                        tool_name = only_key
                        arguments = only_value
            if (
                tool_name
                and isinstance(tool_name, str)
                and isinstance(arguments, dict)
                and tool_name in self._tools
            ):
                calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_json=original_text,
                ))
        return calls

    def validate_tool_call(self, call: ToolCall) -> Optional[str]:
        """
        Perform lightweight schema validation before execution.
        Returns None when valid, otherwise a human-readable validation error.
        """
        tool = self._tools.get(call.tool_name)
        if not tool:
            return f"Tool '{call.tool_name}' is not registered."

        schema = tool.parameters or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = schema.get("required", []) if isinstance(schema, dict) else []
        args = call.arguments or {}
        self._normalize_common_argument_shapes(call.tool_name, args)

        if not isinstance(args, dict):
            return "Tool arguments must be a JSON object."

        for req_name in required:
            value = args.get(req_name, None)
            if value is None:
                return f"Missing required argument '{req_name}'."
            if isinstance(value, str) and not value.strip():
                if call.tool_name == "file_str_replace" and req_name == "old_str":
                    return (
                        "Required argument 'old_str' cannot be empty. "
                        "Use file_view first to copy an exact unique snippet, then retry file_str_replace. "
                        "If you intend to replace the entire file, use file_create instead."
                    )
                return f"Required argument '{req_name}' cannot be empty."

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for arg_name, value in args.items():
            if arg_name not in properties:
                continue
            expected_type_name = properties[arg_name].get("type")

            # Light coercion for common model output quirks (e.g. "1" for integer fields).
            if expected_type_name == "integer" and isinstance(value, str):
                raw = value.strip()
                if re.fullmatch(r"[-+]?\d+", raw):
                    args[arg_name] = int(raw)
                    continue
            if expected_type_name == "number" and isinstance(value, str):
                raw = value.strip()
                try:
                    args[arg_name] = float(raw)
                    continue
                except ValueError:
                    pass
            if expected_type_name == "boolean" and isinstance(value, str):
                raw = value.strip().lower()
                if raw in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                    args[arg_name] = raw in {"true", "1", "yes", "on"}
                    continue

            expected_type = type_map.get(expected_type_name)
            if expected_type and not isinstance(value, expected_type):
                return (
                    f"Argument '{arg_name}' must be of type '{expected_type_name}', "
                    f"got '{type(value).__name__}'."
                )

        # Tool-specific guardrails
        if call.tool_name == "bash":
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                return "Argument 'command' must be a non-empty string."

        if call.tool_name in {"file_view", "file_create", "file_str_replace"}:
            path_like = args.get("path") or args.get("filename")
            if path_like is not None and (not isinstance(path_like, str) or not path_like.strip()):
                return "File path argument must be a non-empty string."

        return None

    def _normalize_common_argument_shapes(self, tool_name: str, args: dict) -> None:
        if not isinstance(args, dict):
            return

        if tool_name == "todo_write":
            tasks = args.get("tasks")
            if not isinstance(tasks, list):
                return
            normalized_tasks = []
            for idx, item in enumerate(tasks, 1):
                if not isinstance(item, dict):
                    normalized_tasks.append(item)
                    continue
                patched = dict(item)
                if not str(patched.get("content") or "").strip() and str(patched.get("title") or "").strip():
                    patched["content"] = str(patched.get("title") or "").strip()
                if not str(patched.get("id") or "").strip():
                    patched["id"] = f"task_{idx}"
                raw_status = str(patched.get("status") or patched.get("state") or "").strip().lower()
                if raw_status:
                    status_map = {
                        "start": "in_progress",
                        "started": "in_progress",
                        "doing": "in_progress",
                        "done": "completed",
                        "complete": "completed",
                    }
                    patched["status"] = status_map.get(raw_status, raw_status)
                normalized_tasks.append(patched)
            args["tasks"] = normalized_tasks

        if tool_name == "todo_update":
            if "status" not in args and "state" in args:
                raw_status = str(args.get("state") or "").strip().lower()
                status_map = {
                    "start": "in_progress",
                    "started": "in_progress",
                    "doing": "in_progress",
                    "done": "completed",
                    "complete": "completed",
                }
                args["status"] = status_map.get(raw_status, raw_status)


    async def execute(self, call: ToolCall, context: Optional[dict] = None) -> ToolResult:
        """Execute a tool call and return a ToolResult."""
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=f"Tool '{call.tool_name}' not found.",
            )

        exec_args = dict(call.arguments)
        before_block = await self._run_before_hooks(call.tool_name, exec_args)
        if before_block is not None:
            return before_block

        try:
            # Merge context into arguments (e.g. _session_id)
            kwargs = {**exec_args}
            if context:
                # Only inject context keys (starting with _) if handler accepts **kwargs or named param
                sig = inspect.signature(tool.handler)
                has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                
                for k, v in context.items():
                    if has_kwargs or k in sig.parameters:
                        kwargs[k] = v

            if asyncio.iscoroutinefunction(tool.handler):
                raw = await tool.handler(**kwargs)
            else:
                raw = tool.handler(**kwargs)
            success, content = self._serialize_tool_output(tool, raw)
            result = ToolResult(tool_name=call.tool_name, success=success, content=content)
            return await self._run_after_hooks(call.tool_name, exec_args, result)
        except TypeError as e:
            # Argument mismatch — give model useful error to self-correct
            sig = inspect.signature(tool.handler)
            result = ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=(
                    f"Tool argument error: {e}\n"
                    f"Expected signature: {sig}\n"
                    f"Received arguments: {list(exec_args.keys())}"
                ),
            )
            return await self._run_after_hooks(call.tool_name, exec_args, result)
        except Exception as e:
            result = ToolResult(
                tool_name=call.tool_name,
                success=False,
                content=f"Tool error: {e}",
            )
            return await self._run_after_hooks(call.tool_name, exec_args, result)

    async def _run_before_hooks(self, tool_name: str, arguments: dict) -> Optional[ToolResult]:
        for pattern, hook in self._before_hooks:
            if not pattern.search(tool_name):
                continue

            hook_result = hook(tool_name, dict(arguments))
            if inspect.isawaitable(hook_result):
                hook_result = await hook_result

            if hook_result is None:
                continue
            if isinstance(hook_result, dict):
                hook_result = HookResult(**hook_result)
            if not isinstance(hook_result, HookResult):
                continue

            decision = (hook_result.decision or "allow").lower()
            if decision == "deny":
                reason = hook_result.system_message or hook_result.reason or "Blocked by before_tool hook."
                return ToolResult(tool_name=tool_name, success=False, content=reason)
            if decision == "transform" and isinstance(hook_result.transformed_arguments, dict):
                arguments.clear()
                arguments.update(hook_result.transformed_arguments)

        return None

    def _serialize_tool_output(self, tool: Tool, raw: Any) -> tuple[bool, str]:
        if tool.response_format == "json":
            if isinstance(raw, (dict, list)):
                success = True
                if isinstance(raw, dict) and isinstance(raw.get("success"), bool):
                    success = bool(raw.get("success"))
                return success, json.dumps(raw)

            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    error_payload = {
                        "type": "tool_output_validation_error",
                        "tool": tool.name,
                        "error": "Expected JSON output but received plain text.",
                        "raw_preview": raw[:500],
                    }
                    return False, json.dumps(error_payload)
                if not isinstance(parsed, (dict, list)):
                    error_payload = {
                        "type": "tool_output_validation_error",
                        "tool": tool.name,
                        "error": "Expected a JSON object or array.",
                        "raw_preview": raw[:500],
                    }
                    return False, json.dumps(error_payload)
                success = True
                if isinstance(parsed, dict) and isinstance(parsed.get("success"), bool):
                    success = bool(parsed.get("success"))
                return success, raw

            return True, json.dumps(raw)

        if isinstance(raw, dict) and isinstance(raw.get("success"), bool):
            return bool(raw.get("success")), json.dumps(raw)
        return True, raw if isinstance(raw, str) else json.dumps(raw)

    async def _run_after_hooks(self, tool_name: str, arguments: dict, result: ToolResult) -> ToolResult:
        current = result
        for pattern, hook in self._after_hooks:
            if not pattern.search(tool_name):
                continue

            hook_result = hook(tool_name, dict(arguments), current)
            if inspect.isawaitable(hook_result):
                hook_result = await hook_result

            if hook_result is None:
                continue
            if isinstance(hook_result, dict):
                hook_result = HookResult(**hook_result)
            if not isinstance(hook_result, HookResult):
                continue

            decision = (hook_result.decision or "allow").lower()
            if decision == "deny":
                reason = hook_result.system_message or hook_result.reason or "Blocked by after_tool hook."
                current = ToolResult(tool_name=tool_name, success=False, content=reason)
                continue
            if decision == "transform":
                content = hook_result.transformed_content
                success = hook_result.transformed_success
                current = ToolResult(
                    tool_name=tool_name,
                    success=current.success if success is None else bool(success),
                    content=current.content if content is None else str(content),
                )

        return current
