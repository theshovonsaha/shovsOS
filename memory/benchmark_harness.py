from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterable

from engine.deterministic_facts import extract_user_stated_fact_updates
from engine.direct_fact_policy import should_answer_direct_fact_from_memory
from memory.semantic_graph import SemanticGraph


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fact_signature(item: dict) -> tuple[str, str, str]:
    return (
        str(item.get("subject") or "").strip().lower(),
        str(item.get("predicate") or "").strip().lower(),
        str(item.get("object") or item.get("object_") or "").strip().lower(),
    )


def _f1(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class _SyntheticSemanticGraph(SemanticGraph):
    async def _get_embedding(self, text: str):  # pragma: no cover - deterministic harness helper
        tokens = {tok.strip(".,!?").lower() for tok in str(text or "").split()}
        basis = {
            "postgresql": [1.0, 0.0, 0.0, 0.0],
            "database": [0.9, 0.1, 0.0, 0.0],
            "sql": [0.9, 0.1, 0.0, 0.0],
            "berlin": [0.0, 1.0, 0.0, 0.0],
            "germany": [0.0, 0.9, 0.1, 0.0],
            "cursor": [0.0, 0.0, 1.0, 0.0],
            "editor": [0.0, 0.0, 0.9, 0.1],
            "pnpm": [0.0, 0.0, 0.0, 1.0],
            "package": [0.0, 0.0, 0.1, 0.9],
        }
        vec = [0.0, 0.0, 0.0, 0.0]
        for token in tokens:
            comp = basis.get(token)
            if not comp:
                continue
            vec = [a + b for a, b in zip(vec, comp)]
        if vec == [0.0, 0.0, 0.0, 0.0]:
            vec = [0.25, 0.25, 0.25, 0.25]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _evaluate_deterministic_extraction() -> dict:
    started = perf_counter()
    extraction_cases = [
        {
            "message": "Call me Alex. I use Linux.",
            "expected": {
                ("user", "preferred_name", "alex"),
                ("user", "operating_system", "linux"),
            },
            "current_facts": [],
            "expected_voids": set(),
        },
        {
            "message": "I prefer concise responses. My pronouns are he/him.",
            "expected": {
                ("user", "response_verbosity", "concise"),
                ("user", "pronouns", "he/him"),
            },
            "current_facts": [],
            "expected_voids": set(),
        },
        {
            "message": "Actually, I moved to Berlin.",
            "expected": {
                ("user", "location", "berlin"),
            },
            "current_facts": [("User", "location", "Toronto")],
            "expected_voids": {("user", "location")},
        },
    ]

    expected_total = 0
    extracted_total = 0
    matched_total = 0
    void_expected_total = 0
    void_matched_total = 0
    case_rows: list[dict] = []

    for case in extraction_cases:
        facts, voids = extract_user_stated_fact_updates(
            case["message"],
            current_facts=case["current_facts"],
        )
        expected = set(case["expected"])
        observed = {_fact_signature(item) for item in facts}
        expected_total += len(expected)
        extracted_total += len(observed)
        matched = expected & observed
        matched_total += len(matched)

        observed_voids = {
            (
                str(item.get("subject") or "").strip().lower(),
                str(item.get("predicate") or "").strip().lower(),
            )
            for item in voids
        }
        expected_voids = set(case.get("expected_voids") or set())
        void_expected_total += len(expected_voids)
        void_matched_total += len(expected_voids & observed_voids)
        case_rows.append(
            {
                "message": case["message"],
                "matched_facts": len(matched),
                "expected_facts": len(expected),
                "extracted_facts": len(observed),
                "matched_voids": len(expected_voids & observed_voids),
                "expected_voids": len(expected_voids),
            }
        )

    precision = matched_total / max(1, extracted_total)
    recall = matched_total / max(1, expected_total)
    void_accuracy = void_matched_total / max(1, void_expected_total) if void_expected_total else 1.0
    duration_ms = round((perf_counter() - started) * 1000.0, 2)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "void_accuracy": round(void_accuracy, 4),
        "duration_ms": duration_ms,
        "cases": case_rows,
    }


def _evaluate_direct_fact_guard() -> dict:
    started = perf_counter()
    current_facts = [
        ("User", "preferred_name", "Alex"),
        ("User", "location", "Berlin"),
        ("User", "response_verbosity", "concise"),
        ("User", "operating_system", "Linux"),
    ]
    cases = [
        ("What is my name?", True),
        ("Where do I live?", True),
        ("Keep responses concise, what is my verbosity preference?", True),
        ("What operating system do I use?", True),
        ("What stock did we discuss?", False),
    ]
    matched = 0
    rows = []
    for query, expected in cases:
        observed = should_answer_direct_fact_from_memory(query, current_facts)
        if observed == expected:
            matched += 1
        rows.append({"query": query, "expected": expected, "observed": observed})
    accuracy = matched / max(1, len(cases))
    duration_ms = round((perf_counter() - started) * 1000.0, 2)
    return {
        "accuracy": round(accuracy, 4),
        "duration_ms": duration_ms,
        "cases": rows,
    }


async def _evaluate_semantic_retrieval() -> dict:
    started = perf_counter()
    with TemporaryDirectory() as tmp_dir:
        graph = _SyntheticSemanticGraph(db_path=str(Path(tmp_dir) / "benchmark_memory.db"))
        await graph.add_triplet("User", "likes", "PostgreSQL database", owner_id="benchmark-owner")
        await graph.add_triplet("User", "location", "Berlin Germany", owner_id="benchmark-owner")
        await graph.add_triplet("User", "preferred_editor", "Cursor", owner_id="benchmark-owner")
        await graph.add_triplet("User", "package_manager", "pnpm", owner_id="benchmark-owner")

        queries = [
            ("database systems sql", "postgresql database"),
            ("where am i located in germany", "berlin germany"),
            ("which editor do i use", "cursor"),
            ("package tool preference", "pnpm"),
        ]
        hit_at_3 = 0
        mrr_total = 0.0
        rows: list[dict] = []

        for query, expected in queries:
            results = await graph.traverse(query, top_k=3, threshold=0.01, owner_id="benchmark-owner")
            ranked = [str(item.get("object") or "").strip().lower() for item in results]
            expected_l = expected.lower()
            rank = None
            for idx, object_ in enumerate(ranked, start=1):
                if expected_l in object_:
                    rank = idx
                    break
            if rank is not None:
                hit_at_3 += 1
                mrr_total += 1.0 / rank
            rows.append(
                {
                    "query": query,
                    "expected": expected,
                    "rank": rank,
                    "top_objects": ranked,
                }
            )

    total = max(1, len(queries))
    duration_ms = round((perf_counter() - started) * 1000.0, 2)
    return {
        "hit_rate_at_3": round(hit_at_3 / total, 4),
        "mrr_at_3": round(mrr_total / total, 4),
        "duration_ms": duration_ms,
        "cases": rows,
    }


async def run_memory_benchmark(owner_id: str) -> dict:
    deterministic = _evaluate_deterministic_extraction()
    direct_fact = _evaluate_direct_fact_guard()
    retrieval = await _evaluate_semantic_retrieval()

    overall = (
        deterministic["f1"] * 0.35
        + deterministic["void_accuracy"] * 0.15
        + direct_fact["accuracy"] * 0.2
        + retrieval["hit_rate_at_3"] * 0.2
        + retrieval["mrr_at_3"] * 0.1
    )

    return {
        "owner_id": owner_id,
        "ran_at": _iso_now(),
        "overall_score": round(overall, 4),
        "metrics": {
            "deterministic_extraction": deterministic,
            "direct_fact_guard": direct_fact,
            "semantic_retrieval": retrieval,
        },
    }

