from engine.candidate_signals import extract_stance_signals, merge_candidate_signals
from orchestration.session_manager import SessionManager


def test_session_manager_persists_structured_candidate_signals(tmp_path):
    manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = manager.create(model="llama3.2", system_prompt="", agent_id="default")

    manager.update_candidate_signals(
        session.id,
        [
            {
                "text": "General working_on Secret roadmap",
                "reason": "not_grounded",
                "source": "compression_filter",
            }
        ],
    )

    loaded = manager.get(session.id)

    assert loaded is not None
    assert loaded.candidate_signals == [
        {
            "text": "General working_on Secret roadmap",
            "reason": "not_grounded",
            "source": "compression_filter",
        }
    ]
    assert loaded.candidate_context == "- Candidate: General working_on Secret roadmap (reason=not_grounded)"


def test_update_candidate_context_backfills_structured_signals(tmp_path):
    manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = manager.create(model="llama3.2", system_prompt="", agent_id="default")

    manager.update_candidate_context(
        session.id,
        "- Candidate: User prefers weekly summaries (reason=low_grounding)",
    )

    loaded = manager.get(session.id)

    assert loaded is not None
    assert loaded.candidate_signals == [
        {
            "text": "User prefers weekly summaries",
            "reason": "low_grounding",
        }
    ]


def test_extract_stance_signals_builds_structured_stance_candidates():
    signals = extract_stance_signals(
        "I think explicit challenge mode should stay on by default.",
        turn_index=7,
    )

    assert signals
    assert signals[0]["signal_type"] == "stance"
    assert signals[0]["confidence"] == "hedged"
    assert signals[0]["turn_index"] == 7
    assert "Stance [" in signals[0]["text"]


def test_candidate_signal_lifecycle_promotes_then_expires_stale_review_items():
    signals = merge_candidate_signals(
        [],
        [{"fact": "General working_on Secret roadmap", "grounding_reason": "not_grounded"}],
        current_turn=1,
    )

    assert signals[0]["promotion_state"] == "candidate"
    assert signals[0]["prompt_eligible"] is True

    promoted = merge_candidate_signals(
        signals,
        [{"fact": "General working_on Secret roadmap", "grounding_reason": "not_grounded"}],
        current_turn=2,
    )
    assert promoted[0]["promotion_state"] == "promoted"
    assert promoted[0]["expired"] is False

    decayed = merge_candidate_signals(promoted, [], current_turn=25)
    assert decayed == []
