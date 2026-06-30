from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shovs_harness_core import LlamaCppClient, LlamaCppConfig, enforce_proposed_actions, infer_source_contract
from shovs_harness_core.proposers import LLMProposer


OBJECTIVE = "Search top 3 stocks today, search each, fetch 3 URLs each."
WORLD = {
    "entities": ["ROKU", "TBN", "SENEA", "EPAM"],
    "urls": {
        "ROKU": ["https://src.test/ROKU/0", "https://src.test/ROKU/1", "https://src.test/ROKU/2"],
        "TBN": ["https://src.test/TBN/0", "https://src.test/TBN/1", "https://src.test/TBN/2"],
        "SENEA": ["https://src.test/SENEA/0", "https://src.test/SENEA/1", "https://src.test/SENEA/2"],
        "EPAM": ["https://src.test/EPAM/0"],
    },
}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a local llama.cpp server through the harness action gate.")
    parser.add_argument("--base-url", default=os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=os.getenv("LLAMACPP_DEFAULT_MODEL", "local-model"))
    parser.add_argument("--objective", default=OBJECTIVE)
    args = parser.parse_args()

    contract = infer_source_contract(args.objective)
    client = LlamaCppClient(LlamaCppConfig(base_url=args.base_url, retries=1, timeout=90.0))
    proposer = LLMProposer(client, args.model)
    actions = await proposer.propose(contract, WORLD)
    report = enforce_proposed_actions(contract, actions, candidate_urls=WORLD["urls"])

    print(json.dumps(
        {
            "base_url": args.base_url,
            "model": args.model,
            "contract": {
                "entity_count": contract.entity_count,
                "urls_per_entity": contract.urls_per_entity,
                "total_urls": contract.total_urls,
            },
            "proposed_actions": [
                {
                    "action": item.action,
                    "entity": item.entity,
                    "url": item.url,
                    "entities": list(item.entities),
                }
                for item in actions
            ],
            "enforcement": report.to_dict(),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    asyncio.run(main())
