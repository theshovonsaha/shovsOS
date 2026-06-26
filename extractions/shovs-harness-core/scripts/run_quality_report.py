from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shovs_harness_core import HarnessKernel, evaluate_trace, infer_source_contract


def main() -> None:
    objective = "Search top 3 stocks, search each, fetch 3 URLs each."
    contract = infer_source_contract(objective)
    plain_drift_trace = [
        {"kind": "entity_locked", "entity": "ROKU"},
        {"kind": "entity_locked", "entity": "TBN"},
        {"kind": "entity_locked", "entity": "SENEA"},
        {"tool": "web_search", "entity": "EPAM"},
        {"tool": "web_fetch", "entity": "ROKU"},
    ]
    harness_trace = [
        {"kind": "entity_locked", "entity": "ROKU"},
        {"kind": "entity_locked", "entity": "TBN"},
        {"kind": "entity_locked", "entity": "SENEA"},
        {"tool": "web_search", "entity": "ROKU"},
        {"tool": "web_search", "entity": "TBN"},
        {"tool": "web_search", "entity": "SENEA"},
    ]
    for entity in ("ROKU", "TBN", "SENEA"):
        for idx in range(3):
            harness_trace.append(
                {
                    "tool": "web_fetch",
                    "entity": entity,
                    "url": f"https://source.test/{entity}/{idx}",
                }
            )

    kernel = HarnessKernel(objective)
    report = {
        "objective": objective,
        "contract": {
            "entity_count": contract.entity_count,
            "urls_per_entity": contract.urls_per_entity,
            "total_urls": contract.total_urls,
            "required_tools": contract.required_tools,
        },
        "plain_loop": evaluate_trace(contract, plain_drift_trace).__dict__,
        "harness_shape": evaluate_trace(contract, harness_trace).__dict__,
        "first_kernel_decision": kernel.decide().__dict__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
