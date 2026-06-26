from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from run_engine.workflow_contracts import classify_workflow_shape


COHERENCE_EVAL_VERSION = "coherence-eval-v1"


@dataclass(frozen=True)
class IntentCase:
    prompt: str
    expected_shape: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PacketLeakCase:
    name: str
    packet_content: str
    forbidden_terms: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoherenceEvalResult:
    version: str
    success: bool
    score: float
    total: int
    passed: int
    failed: int
    issues: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_INTENT_CASES: tuple[IntentCase, ...] = (
    IntentCase("hi", "simple_chat", ("small_talk",)),
    IntentCase("hi again", "simple_chat", ("small_talk",)),
    IntentCase("thanks that helps", "simple_chat", ("small_talk",)),
    IntentCase("ok got it", "simple_chat", ("small_talk",)),
    IntentCase("Search top 3 stocks, search each separately, fetch 3 URLs each.", "source_collection", ("source",)),
    IntentCase("top 3 sushi places in Toronto, then search each, fetch 3 URLs each", "source_collection", ("source", "local")),
    IntentCase("fix the frontend trace monitor tests", "coding_change", ("coding",)),
    IntentCase("remember that I prefer compact answers", "memory_correction", ("memory",)),
)


def evaluate_intent_classifier(
    cases: tuple[IntentCase, ...] | list[IntentCase] = DEFAULT_INTENT_CASES,
    classifier: Callable[[str], str] = classify_workflow_shape,
) -> CoherenceEvalResult:
    issues: list[dict[str, Any]] = []
    total = len(cases)
    for case in cases:
        observed = classifier(case.prompt)
        if observed != case.expected_shape:
            issues.append({
                "type": "intent_mismatch",
                "prompt": case.prompt,
                "expected": case.expected_shape,
                "observed": observed,
                "tags": list(case.tags),
            })
    passed = total - len(issues)
    score = passed / total if total else 1.0
    return CoherenceEvalResult(
        version=COHERENCE_EVAL_VERSION,
        success=not issues,
        score=round(score, 4),
        total=total,
        passed=passed,
        failed=len(issues),
        issues=issues,
        summary={"eval": "intent_classifier", "case_count": total},
    )


def evaluate_packet_leaks(cases: tuple[PacketLeakCase, ...] | list[PacketLeakCase]) -> CoherenceEvalResult:
    issues: list[dict[str, Any]] = []
    total = len(cases)
    for case in cases:
        lowered = case.packet_content.lower()
        leaks = [term for term in case.forbidden_terms if str(term or "").lower() in lowered]
        if leaks:
            issues.append({
                "type": "packet_context_leak",
                "case": case.name,
                "leaks": leaks,
                "tags": list(case.tags),
            })
    passed = total - len(issues)
    score = passed / total if total else 1.0
    return CoherenceEvalResult(
        version=COHERENCE_EVAL_VERSION,
        success=not issues,
        score=round(score, 4),
        total=total,
        passed=passed,
        failed=len(issues),
        issues=issues,
        summary={"eval": "packet_context_leaks", "case_count": total},
    )
