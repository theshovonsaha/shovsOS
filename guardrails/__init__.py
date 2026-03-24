"""
Guardrails Package
------------------
Drop-in safety layer for any agent system.

Usage (in main.py / wherever you build your stack):

    from guardrails import GuardrailMiddleware
    from guardrails.api_routes import router as guardrails_router

    middleware = GuardrailMiddleware(registry, require_confirmation_for="all")
    app.include_router(guardrails_router(middleware))

Then swap every registry.execute(call) with:

    await middleware.execute(call, session_id="abc", context={})
"""

from guardrails.middleware import GuardrailMiddleware
from guardrails.risk_classifier import RiskClassifier, RiskLevel
from guardrails.confirmation_gate import ConfirmationGate
from guardrails.tool_logger import ToolAuditLogger

__all__ = [
    "GuardrailMiddleware",
    "RiskClassifier",
    "RiskLevel",
    "ConfirmationGate",
    "ToolAuditLogger",
]
