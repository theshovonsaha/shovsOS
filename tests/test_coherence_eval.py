from run_engine.coherence_eval import (
    DEFAULT_INTENT_CASES,
    PacketLeakCase,
    evaluate_intent_classifier,
    evaluate_packet_leaks,
)


def test_default_intent_coherence_eval_passes():
    result = evaluate_intent_classifier(DEFAULT_INTENT_CASES)

    assert result.success is True
    assert result.score == 1.0
    assert result.issues == []


def test_packet_leak_eval_catches_forbidden_context():
    result = evaluate_packet_leaks([
        PacketLeakCase(
            name="simple-chat",
            packet_content="Current Objective\nhi again\nRuntime Attention: stale plan",
            forbidden_terms=("Runtime Attention", "stale plan"),
            tags=("simple_chat",),
        )
    ])

    assert result.success is False
    assert result.failed == 1
    assert result.issues[0]["type"] == "packet_context_leak"
