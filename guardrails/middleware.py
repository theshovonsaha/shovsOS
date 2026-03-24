"""
Guardrail Middleware
--------------------
Single entry point that wraps ToolRegistry.execute().
Plugs the classifier + gate + logger together in one pipeline.

Drop-in replacement:
  BEFORE: result = await registry.execute(call)
  AFTER:  result = await middleware.execute(call, session_id=sid, agent_id=aid)

The middleware emits async generator events so your existing
chat_stream loop can yield them to the frontend:

    async for event in middleware.execute_stream(call, ...):
        yield event   # type: "confirmation_required" | "tool_result" | "tool_blocked"
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, AsyncGenerator

from plugins.tool_registry import ToolRegistry, ToolCall, ToolResult
from guardrails.risk_classifier import RiskClassifier, RiskLevel, Classification
from guardrails.confirmation_gate import ConfirmationGate
from guardrails.tool_logger import ToolAuditLogger


class GuardrailMiddleware:
    """
    Wraps ToolRegistry with risk classification, confirmation gating,
    and audit logging.

    Args:
        registry:                   Your existing ToolRegistry instance.
        require_confirmation_for:   "all" | "confirm_and_above" | "block_only"
        confirmation_timeout:       Seconds to wait for user response (default 120).
        log_path:                   Path for audit log JSONL file.
        extra_block_patterns:       Additional bash patterns to hard-block.
        extra_confirm_patterns:     Additional bash patterns to flag for confirmation.
        custom_tool_risks:          Override default risk level per tool name.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        require_confirmation_for: str = "all",
        confirmation_timeout: float = 120.0,
        log_path: str = "./logs/tool_audit.jsonl",
        extra_block_patterns: Optional[list[str]] = None,
        extra_confirm_patterns: Optional[list[str]] = None,
        custom_tool_risks: Optional[dict] = None,
    ):
        self.registry   = registry
        self.classifier = RiskClassifier(
            require_confirmation_for = require_confirmation_for,
            extra_block_patterns     = extra_block_patterns,
            extra_confirm_patterns   = extra_confirm_patterns,
            custom_tool_risks        = custom_tool_risks,
        )
        self.gate   = ConfirmationGate(timeout_seconds=confirmation_timeout)
        self.logger = ToolAuditLogger(log_path=log_path)

    # ── Primary interface: streaming events ───────────────────────────────────

    async def execute_stream(
        self,
        call: ToolCall,
        session_id: str = "unknown",
        agent_id:   str = "unknown",
        context:    Optional[dict] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Execute a tool call through the full guardrail pipeline,
        yielding SSE-compatible events.

        Event types emitted:
          {"type": "tool_blocked",           "tool": ..., "reason": ...}
          {"type": "confirmation_required",  "call_id": ..., "tool": ..., "preview": ..., "arguments": ...}
          {"type": "confirmation_approved",  "call_id": ..., "tool": ...}
          {"type": "confirmation_denied",    "call_id": ..., "tool": ..., "reason": ...}
          {"type": "confirmation_timeout",   "call_id": ..., "tool": ...}
          {"type": "tool_result",            "tool": ..., "success": ..., "content": ...}
        """
        classification = self.classifier.classify(call.tool_name, call.arguments)
        call_id = self.logger.log_intent(
            session_id, agent_id, call.tool_name, call.arguments, classification
        )

        # ── BLOCK ──────────────────────────────────────────────────────────────
        if classification.level == RiskLevel.BLOCK:
            self.logger.log_blocked(call_id, call.tool_name, classification.reason)
            yield {
                "type":    "tool_blocked",
                "call_id": call_id,
                "tool":    call.tool_name,
                "preview": classification.preview,
                "reason":  classification.reason,
            }
            return

        # ── CONFIRM ────────────────────────────────────────────────────────────
        if classification.level == RiskLevel.CONFIRM:
            yield {
                "type":      "confirmation_required",
                "call_id":   call_id,
                "tool":      call.tool_name,
                "preview":   classification.preview,
                "reason":    classification.reason,
                "arguments": call.arguments,
                "session_id": session_id,
            }

            approved, deny_reason = await self.gate.request_confirmation(
                session_id = session_id,
                tool       = call.tool_name,
                preview    = classification.preview,
                reason     = classification.reason,
                arguments  = call.arguments,
            )

            if not approved:
                if deny_reason == "timeout":
                    self.logger.log_timeout(call_id, call.tool_name)
                    yield {
                        "type":    "confirmation_timeout",
                        "call_id": call_id,
                        "tool":    call.tool_name,
                    }
                else:
                    self.logger.log_denied(call_id, call.tool_name, deny_reason or "denied")
                    yield {
                        "type":    "confirmation_denied",
                        "call_id": call_id,
                        "tool":    call.tool_name,
                        "reason":  deny_reason,
                    }
                return

            yield {
                "type":    "confirmation_approved",
                "call_id": call_id,
                "tool":    call.tool_name,
            }

        # ── EXECUTE (SAFE / WARN / post-CONFIRM approval) ──────────────────────
        start = time.monotonic()
        result: ToolResult = await self.registry.execute(call, context=context)
        duration_ms = (time.monotonic() - start) * 1000

        self.logger.log_result(
            call_id     = call_id,
            tool        = call.tool_name,
            success     = result.success,
            content     = result.content,
            duration_ms = duration_ms,
        )

        yield {
            "type":    "tool_result",
            "call_id": call_id,
            "tool":    call.tool_name,
            "success": result.success,
            "content": result.content,
        }

    # ── Simple non-streaming interface (for non-SSE contexts) ─────────────────

    async def execute(
        self,
        call: ToolCall,
        session_id: str = "unknown",
        agent_id:   str = "unknown",
        context:    Optional[dict] = None,
    ) -> ToolResult:
        """
        Non-streaming execute. Blocks until confirmed or rejected.
        Returns a ToolResult with success=False if blocked/denied/timeout.
        """
        final: Optional[ToolResult] = None
        async for event in self.execute_stream(call, session_id, agent_id, context):
            if event["type"] == "tool_result":
                final = ToolResult(
                    tool_name = call.tool_name,
                    success   = event["success"],
                    content   = event["content"],
                )
            elif event["type"] in ("tool_blocked", "confirmation_denied", "confirmation_timeout"):
                final = ToolResult(
                    tool_name = call.tool_name,
                    success   = False,
                    content   = f"[{event['type']}] {event.get('reason', '')}",
                )
        return final or ToolResult(
            tool_name=call.tool_name, success=False, content="No result produced."
        )

    # ── Convenience passthrough to gate (for API routes) ──────────────────────

    def approve(self, call_id: str) -> bool:
        return self.gate.approve(call_id)

    def deny(self, call_id: str, reason: str = "User denied") -> bool:
        return self.gate.deny(call_id, reason)

    def get_pending(self, session_id: Optional[str] = None) -> list[dict]:
        return self.gate.get_pending(session_id)

    def get_audit_log(self, n: int = 50) -> list[dict]:
        return self.logger.tail(n)

    def get_session_trace(self, session_id: str) -> list[dict]:
        return self.logger.get_session_trace(session_id)
