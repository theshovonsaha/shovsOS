from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .action_runner import enforce_proposed_actions
from .contract import SourceContract, infer_source_contract
from .evals import evaluate_trace
from .kernel import HarnessKernel
from .proposers import ProposedAction


DEFAULT_ENTITIES = ("ROKU", "TBN", "SENEA", "MIKU", "COSTCO", "AGENTTRACE")


class HarnessExtension:
    """Host-facing adapter for the extracted harness.

    This class is intentionally small: it exposes the kernel as a plain Python
    extension surface without importing ShovsOS, FastAPI, an LLM SDK, or web
    tools. A host can mount this behind an API route, CLI command, plugin, or
    frontend demo and still get the same contract/eval semantics.
    """

    extension_id = "shovs.harness_core"
    version = "0.1.0"

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.extension_id,
            "version": self.version,
            "name": "Shovs Harness Core",
            "description": "A small source-contract, ledger, and trace-eval kernel for agent reliability tests.",
            "capabilities": [
                "infer_source_contract",
                "enforce_proposed_actions",
                "evaluate_trace",
                "next_kernel_decision",
                "compare_plain_vs_harness_trace",
                "llama_cpp_openai_compatible_client",
            ],
            "input_schema": {
                "objective": "string",
                "trace": "optional list[event]",
                "actions": "optional list[proposed_action]",
                "entities": "optional list[string]",
                "include_traces": "optional bool",
            },
            "output_schema": {
                "contract": "source workflow contract",
                "decision": "next deterministic kernel decision",
                "reports": "state-based eval reports",
            },
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        objective = str(payload.get("objective") or payload.get("query") or "").strip()
        if not objective:
            objective = "Search top 3 items, search each, fetch 3 URLs each."
        contract = infer_source_contract(objective)
        kernel = HarnessKernel(objective)
        entities = _entities(payload.get("entities"), contract)
        include_traces = bool(payload.get("include_traces", True))

        reports: dict[str, Any] = {
            "plain_loop": asdict(evaluate_trace(contract, _plain_trace(contract, entities))),
            "harness_loop": asdict(evaluate_trace(contract, _harness_trace(contract, entities))),
        }
        provided_trace = payload.get("trace")
        if isinstance(provided_trace, list):
            reports["provided_trace"] = asdict(evaluate_trace(contract, _clean_trace(provided_trace)))
        proposed_actions = _clean_actions(payload.get("actions"))
        if proposed_actions:
            reports["action_enforcement"] = enforce_proposed_actions(
                contract,
                proposed_actions,
                candidate_urls=_candidate_urls(payload.get("candidate_urls")),
            ).to_dict()

        result: dict[str, Any] = {
            "extension": self.manifest(),
            "objective": objective,
            "contract": _contract_dict(contract),
            "decision": asdict(kernel.decide()),
            "reports": reports,
            "verdict": _verdict(reports),
        }
        if include_traces:
            result["traces"] = {
                "plain_loop": _plain_trace(contract, entities),
                "harness_loop": _harness_trace(contract, entities),
            }
        return result


def run_extension_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return HarnessExtension().run(payload)


def _contract_dict(contract: SourceContract) -> dict[str, Any]:
    return {
        "objective": contract.objective,
        "entity_count": contract.entity_count,
        "urls_per_entity": contract.urls_per_entity,
        "total_urls": contract.total_urls,
        "required_tools": list(contract.required_tools),
        "missing": list(contract.missing),
        "final_allowed": contract.final_allowed,
    }


def _entities(raw: Any, contract: SourceContract) -> list[str]:
    if isinstance(raw, list):
        values = [str(item).strip().upper() for item in raw if str(item).strip()]
    else:
        values = []
    target = contract.entity_count or 3
    fallback = [item for item in DEFAULT_ENTITIES if item not in values]
    return (values + fallback)[:target]


def _plain_trace(contract: SourceContract, entities: list[str]) -> list[dict[str, Any]]:
    locked = entities[: contract.entity_count or len(entities) or 1]
    drift = "EPAM" if "EPAM" not in locked else "OFF_LIST"
    trace = [{"kind": "entity_locked", "entity": entity, "summary": "Locked from discovery"} for entity in locked]
    if locked:
        trace.append({"tool": "web_search", "entity": drift, "ok": True, "summary": "Planner drifted to an off-list entity"})
        trace.append(
            {
                "tool": "web_fetch",
                "entity": locked[0],
                "url": f"https://source.test/{locked[0]}/0",
                "ok": True,
                "summary": "Fetched one source, below quota",
            }
        )
    return trace


def _harness_trace(contract: SourceContract, entities: list[str]) -> list[dict[str, Any]]:
    locked = entities[: contract.entity_count or len(entities) or 1]
    trace = [{"kind": "entity_locked", "entity": entity, "summary": "Locked before source collection"} for entity in locked]
    for entity in locked:
        trace.append({"tool": "web_search", "entity": entity, "ok": True, "summary": f"Searched {entity}"})
    per_entity = contract.urls_per_entity or 1
    for entity in locked:
        for index in range(per_entity):
            trace.append(
                {
                    "tool": "web_fetch",
                    "entity": entity,
                    "url": f"https://source.test/{entity}/{index}",
                    "ok": True,
                    "summary": f"Fetched source {index + 1} for {entity}",
                }
            )
    return trace


def _clean_trace(raw: list[Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in raw if isinstance(item, dict)]


def _clean_actions(raw: Any) -> list[ProposedAction]:
    if not isinstance(raw, list):
        return []
    actions: list[ProposedAction] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        actions.append(
            ProposedAction(
                action=str(item.get("action") or ""),
                entity=str(item.get("entity") or ""),
                url=str(item.get("url") or ""),
                entities=tuple(str(entity) for entity in (item.get("entities") or []) if str(entity).strip()),
            )
        )
    return actions


def _candidate_urls(raw: Any) -> dict[str, list[str]] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            out[str(key).upper()] = [str(item) for item in value if str(item).strip()]
    return out


def _verdict(reports: dict[str, Any]) -> dict[str, Any]:
    plain = reports.get("plain_loop") or {}
    harness = reports.get("harness_loop") or {}
    return {
        "plain_ok": bool(plain.get("ok")),
        "harness_ok": bool(harness.get("ok")),
        "improvement": round(float(harness.get("score", 0.0)) - float(plain.get("score", 0.0)), 3),
        "summary": "harness passes state checks that the plain loop fails"
        if harness.get("ok") and not plain.get("ok")
        else "compare reports before making a claim",
    }
