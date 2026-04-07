import uuid

from orchestration.session_manager import SessionManager
from shovs_memory import ShovsMemory


def test_shovs_memory_facade_applies_deterministic_updates_and_inspects_state(tmp_path):
    db_path = tmp_path / "memory_graph.db"
    sessions_db = tmp_path / "sessions.db"
    owner_id = f"shovs-memory-owner-{uuid.uuid4().hex[:8]}"
    session_manager = SessionManager(db_path=str(sessions_db))
    session = session_manager.create(
        model="llama3.2",
        system_prompt="",
        agent_id="default",
        owner_id=owner_id,
    )
    session_manager.update_context(session.id, "Recent work: consumer memory plane")
    session_manager.update_candidate_context(
        session.id,
        "- Candidate: User likes weekly summaries (reason=single_mention)",
    )

    memory = ShovsMemory(
        session_id=session.id,
        owner_id=owner_id,
        db_path=str(db_path),
        session_manager=session_manager,
    )

    first = memory.apply_user_message("My name is Shovon and I live in Vancouver.", turn=1)
    second = memory.apply_user_message("Actually, I moved to Toronto.", turn=2)
    payload = memory.inspect()

    assert len(first["facts"]) == 2
    assert len(second["voids"]) == 1
    assert ("User", "preferred_name", "Shovon") in memory.current_facts()
    assert ("User", "location", "Toronto") in memory.current_facts()
    assert payload["summary"]["deterministic_fact_count"] == 2
    assert payload["summary"]["superseded_fact_count"] == 1
    assert payload["summary"]["candidate_signal_count"] == 1
    assert any(item["object"] == "Toronto" for item in payload["deterministic_facts"])
    assert any(item["object"] == "Vancouver" for item in payload["superseded_facts"])
