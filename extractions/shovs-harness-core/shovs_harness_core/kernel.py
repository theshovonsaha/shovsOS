from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attention import AttentionItem, select_attention
from .contract import SourceContract, infer_source_contract
from .ledger import Ledger


@dataclass(frozen=True)
class KernelDecision:
    state: str
    allowed: bool
    reason: str
    next_tool: str = ""
    next_args: dict[str, Any] | None = None


class HarnessKernel:
    """A tiny control kernel around a model, not a replacement for the model."""

    def __init__(self, objective: str, allowed_tools: list[str] | None = None):
        self.contract: SourceContract = infer_source_contract(objective)
        tools = allowed_tools or self.contract.required_tools or ["web_search", "web_fetch"]
        self.ledger = Ledger(objective=objective, allowed_tools=list(dict.fromkeys(tools)))

    def decide(self) -> KernelDecision:
        successful_fetches = [result for result in self.ledger.results if result.name == "web_fetch" and result.ok]
        successful_searches = [result for result in self.ledger.results if result.name == "web_search" and result.ok]
        if self.contract.total_urls and len(successful_fetches) >= self.contract.total_urls:
            return KernelDecision("respond", True, "source quota met")
        if self.contract.entity_count and not successful_searches:
            return KernelDecision("act", True, "need entity discovery/search", "web_search", {"query": self.ledger.objective})
        if self.contract.total_urls:
            return KernelDecision("act", True, "need selected URL fetches", "web_fetch", {"url": "<selected_url>"})
        return KernelDecision("respond", True, "no deterministic source quota inferred")

    def add_tool_result(self, tool: str, args: dict[str, Any], ok: bool, data: dict[str, Any], summary: str = ""):
        call = self.ledger.add_call(tool, args)
        return self.ledger.add_result(call.id, ok, data, summary)

    def attention(self, phase: str):
        items = [
            AttentionItem("objective", "objective", self.ledger.objective),
            AttentionItem("contract", "contract", repr(self.contract), "missing" if self.contract.missing else "active"),
        ]
        for pending in self.ledger.pending:
            items.append(AttentionItem(f"pending:{pending}", "pending", pending, "missing"))
        for result in self.ledger.results:
            status = "done" if result.ok else "error"
            items.append(AttentionItem(result.id, "evidence", result.summary or result.name, status))
        return select_attention(items, phase)
