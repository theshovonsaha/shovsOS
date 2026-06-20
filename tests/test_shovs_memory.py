import uuid
from pathlib import Path

from orchestration.session_manager import SessionManager
from memory.semantic_graph import SemanticGraph
from shovs_memory import ShovsMemory
from shovs_memory.inspector import build_memory_payload


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


def test_temporal_fact_replacement_rolls_back_void_when_add_fails(tmp_path):
    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    session_id = "session-rollback"
    owner_id = "owner-rollback"
    graph.add_temporal_fact(
        session_id,
        "User",
        "location",
        "Vancouver",
        turn=1,
        owner_id=owner_id,
    )

    try:
        graph.replace_temporal_facts(
            session_id,
            facts=[
                {
                    "subject": "User",
                    "predicate": "location",
                    "object": object(),
                }
            ],
            voids=[{"subject": "User", "predicate": "location"}],
            turn=2,
            owner_id=owner_id,
        )
    except AttributeError:
        pass
    else:
        raise AssertionError("replacement should fail after the void statement")

    assert graph.get_current_facts(session_id, owner_id=owner_id) == [
        ("User", "location", "Vancouver")
    ]


def test_shovs_memory_store_fact_rolls_back_replacement_when_add_fails(tmp_path):
    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    memory = ShovsMemory(
        session_id="session-facade-rollback",
        owner_id="owner-facade-rollback",
        graph=graph,
    )
    memory.store_fact(
        subject="User",
        predicate="location",
        object_="Vancouver",
        turn=1,
    )

    try:
        memory.store_fact(
            subject="User",
            predicate="location",
            object_=object(),
            turn=2,
        )
    except AttributeError:
        pass
    else:
        raise AssertionError("replacement should fail before committing the void")

    assert memory.current_facts() == [("User", "location", "Vancouver")]


def test_shovs_memory_apply_user_message_rolls_back_batch_when_add_fails(tmp_path, monkeypatch):
    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    memory = ShovsMemory(
        session_id="session-message-rollback",
        owner_id="owner-message-rollback",
        graph=graph,
    )
    memory.store_fact(
        subject="User",
        predicate="location",
        object_="Vancouver",
        turn=1,
    )

    monkeypatch.setattr(
        "shovs_memory.memory.extract_user_stated_fact_updates",
        lambda *_args, **_kwargs: (
            [{"subject": "User", "predicate": "location", "object": object()}],
            [{"subject": "User", "predicate": "location"}],
        ),
    )

    try:
        memory.apply_user_message("Actually I moved.", turn=2)
    except AttributeError:
        pass
    else:
        raise AssertionError("batch update should fail before committing the void")

    assert memory.current_facts() == [("User", "location", "Vancouver")]


def test_shovs_memory_default_db_path_is_resolved_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = ShovsMemory(session_id="session-path", owner_id="owner-path")

    assert Path(memory.graph.db_path).is_absolute()


def test_shovs_memory_inspector_requires_explicit_graph(tmp_path):
    session_manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = session_manager.create(
        model="llama3.2",
        system_prompt="",
        agent_id="default",
        owner_id="owner-required-graph",
    )

    try:
        build_memory_payload(
            session=session,
            owner_id="owner-required-graph",
            graph=None,
            context_preview=lambda raw: [],
        )
    except ValueError as exc:
        assert "requires an explicit SemanticGraph" in str(exc)
    else:
        raise AssertionError("inspector should not default-construct a graph")


def test_shovs_memory_inspector_counts_all_current_facts_beyond_timeline_limit(tmp_path):
    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    session_manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    owner_id = "owner-long-count"
    session = session_manager.create(
        model="llama3.2",
        system_prompt="",
        agent_id="default",
        owner_id=owner_id,
    )
    for index in range(65):
        graph.add_temporal_fact(
            session.id,
            "User",
            f"preference_{index}",
            f"value_{index}",
            turn=index + 1,
            owner_id=owner_id,
        )

    payload = build_memory_payload(
        session=session,
        owner_id=owner_id,
        graph=graph,
        timeline_limit=40,
        context_preview=lambda raw: [],
    )

    assert payload["summary"]["deterministic_fact_count"] == 65
    assert len(payload["deterministic_facts"]) == 65
    assert payload["summary"]["timeline_truncated"] is True
    assert payload["graph_db_path"] == str(tmp_path / "memory_graph.db")


def test_shovs_memory_inspector_surfaces_disputed_and_conflict_traced_facts(tmp_path):
    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    session_manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    owner_id = "owner-dispute-demo"
    session = session_manager.create(
        model="llama3.2",
        system_prompt="",
        agent_id="default",
        owner_id=owner_id,
    )
    session_manager.update_candidate_signals(
        session.id,
        [
            {
                "text": "User location may not be Toronto",
                "reason": "evidence_disputed",
                "signal_type": "disputed_fact",
                "subject": "User",
                "predicate": "location",
                "object": "Toronto",
            }
        ],
    )
    graph.add_temporal_fact(
        session.id,
        "User",
        "location",
        "Berlin",
        turn=2,
        owner_id=owner_id,
        conflict_trace=True,
        prior_value_disputed=True,
        conflict_reason="fresh evidence disputed prior location",
    )

    payload = build_memory_payload(
        session=session,
        owner_id=owner_id,
        graph=graph,
        context_preview=lambda raw: [],
    )

    assert payload["summary"]["disputed_fact_count"] == 1
    assert payload["summary"]["conflict_traced_count"] == 1
    assert payload["disputed_facts"][0]["reason"] == "evidence_disputed"
    assert payload["conflict_traced_facts"][0]["conflict_trace"] is True


def test_typed_fact_records_include_lifecycle_provenance_and_supersession(tmp_path):
    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    session_id = "session-typed"
    owner_id = "owner-typed"

    graph.add_temporal_fact(
        session_id,
        "Task",
        "budget_limit",
        "under $200",
        turn=1,
        owner_id=owner_id,
        run_id="run-1",
        memory_type="policy",
        confidence=0.95,
    )
    graph.replace_temporal_facts(
        session_id,
        facts=[
            {
                "subject": "Task",
                "predicate": "budget_limit",
                "object": "under $150",
                "run_id": "run-2",
                "memory_type": "policy",
                "confidence": 1.0,
            }
        ],
        voids=[{"subject": "Task", "predicate": "budget_limit"}],
        turn=2,
        owner_id=owner_id,
    )

    current = graph.get_current_fact_records(
        session_id,
        owner_id=owner_id,
        memory_types=["policy"],
    )
    timeline = graph.list_temporal_facts(session_id, owner_id=owner_id)

    assert current[0]["object"] == "under $150"
    assert current[0]["memory_type"] == "policy"
    assert current[0]["memory_status"] == "active"
    assert current[0]["source_turn_id"] == 2
    assert current[0]["content_hash"]
    old = next(item for item in timeline if item["object"] == "under $200")
    new = next(item for item in timeline if item["object"] == "under $150")
    assert old["memory_status"] == "superseded"
    assert old["superseded_by"] == new["id"]
