#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from memory.benchmark_harness import run_memory_benchmark
from memory.benchmark_store import save_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Shovs memory benchmark harness.")
    parser.add_argument("--owner-id", required=True, help="Owner scope for benchmark snapshot.")
    parser.add_argument(
        "--save-latest",
        action="store_true",
        help="Persist benchmark result into logs/memory_benchmark_latest.json",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    result = await run_memory_benchmark(args.owner_id)
    if args.save_latest:
        save_latest(args.owner_id, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())

