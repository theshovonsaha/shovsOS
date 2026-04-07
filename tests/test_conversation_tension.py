from engine.candidate_signals import extract_stance_signals
from engine.conversation_tension import analyze_conversation_tension, render_conversation_tension


def test_analyze_conversation_tension_detects_conflicting_fact_updates():
    tension = analyze_conversation_tension(
        user_message="I live in Berlin now.",
        current_facts=[("User", "location", "Toronto")],
        deterministic_keyed_facts=[
            {
                "subject": "User",
                "predicate": "location",
                "object": "Berlin",
            }
        ],
        session_history=[{"role": "user", "content": "I live in Toronto."}],
    )

    assert tension.drift_detected is True
    assert tension.should_challenge is True
    assert tension.challenge_level in {"medium", "high"}
    assert tension.conflicting_facts[0]["previous"] == "Toronto"
    assert tension.conflicting_facts[0]["current"] == "Berlin"


def test_render_conversation_tension_includes_drift_lines():
    tension = analyze_conversation_tension(
        user_message="Actually, call me Alex.",
        current_facts=[("User", "preferred_name", "Shovon")],
        deterministic_keyed_facts=[
            {
                "subject": "User",
                "predicate": "preferred_name",
                "object": "Alex",
            }
        ],
        session_history=[{"role": "user", "content": "My name is Shovon."}],
    )

    rendered = render_conversation_tension(tension)

    assert "Challenge Level:" in rendered
    assert "Should Challenge: yes" in rendered
    assert "Drift:" in rendered


def test_analyze_conversation_tension_detects_unacknowledged_stance_drift():
    new_stances = extract_stance_signals(
        "I prefer direct contradiction over polite smoothing.",
        turn_index=4,
    )

    tension = analyze_conversation_tension(
        user_message="I prefer direct contradiction over polite smoothing.",
        current_facts=[],
        deterministic_keyed_facts=[],
        session_history=[{"role": "user", "content": "I prefer careful consensus and softer framing."}],
        candidate_signals=[
            {
                "signal_type": "stance",
                "topic": new_stances[0]["topic"],
                "position": "careful consensus and softer framing",
                "confidence": "asserted",
                "turn_index": 1,
                "raw_text": "I prefer careful consensus and softer framing.",
                "superseded": False,
                "source": "stance_extractor",
                "reason": "stance_asserted",
                "text": f"Stance [{new_stances[0]['topic']}]: careful consensus and softer framing",
            }
        ],
        current_stance_signals=new_stances,
    )

    assert tension.drift_detected is True
    assert tension.unacknowledged_drift is True
    assert tension.should_challenge is True
    assert tension.stance_drifts[0]["topic"] == new_stances[0]["topic"]


def test_analyze_conversation_tension_treats_explicit_stance_revision_as_acknowledged():
    new_stances = extract_stance_signals(
        "Actually, I prefer direct contradiction over polite smoothing.",
        turn_index=5,
    )

    tension = analyze_conversation_tension(
        user_message="Actually, I prefer direct contradiction over polite smoothing.",
        current_facts=[],
        deterministic_keyed_facts=[],
        session_history=[{"role": "user", "content": "I prefer careful consensus and softer framing."}],
        candidate_signals=[
            {
                "signal_type": "stance",
                "topic": new_stances[0]["topic"],
                "position": "careful consensus and softer framing",
                "confidence": "asserted",
                "turn_index": 1,
                "raw_text": "I prefer careful consensus and softer framing.",
                "superseded": False,
                "source": "stance_extractor",
                "reason": "stance_asserted",
                "text": f"Stance [{new_stances[0]['topic']}]: careful consensus and softer framing",
            }
        ],
        current_stance_signals=new_stances,
    )

    assert tension.drift_detected is True
    assert tension.unacknowledged_drift is False
    assert tension.should_challenge is False
    assert "Stance Revision" in render_conversation_tension(tension)