import uuid

from fastapi.testclient import TestClient

from api.main import app, consumer_session_manager, session_manager
from config.trace_store import get_trace_store
from memory.semantic_graph import SemanticGraph


def test_session_memory_state_exposes_current_superseded_and_candidate_memory():
    owner_id = f"memory-state-owner-{uuid.uuid4().hex[:8]}"
    graph = SemanticGraph()
    graph.clear(owner_id=owner_id)

    with TestClient(app):
        session = session_manager.create(
            model="groq:moonshotai/kimi-k2-instruct",
            system_prompt="",
            agent_id="default",
            owner_id=owner_id,
        )
        session_manager.set_context_mode(session.id, "v2")
        session_manager.update_context(session.id, "Active Goals: keep answers concise\n- Recent work: budget agent memory")
        session_manager.update_candidate_context(
            session.id,
            "- Candidate: User prefers weekly summaries (reason=low_grounding)\n"
            "- Candidate: Team is moving to Berlin (reason=single_mention)",
        )

        graph.add_temporal_fact(session.id, "User", "preferred_name", "Shovon", turn=1, owner_id=owner_id)
        graph.add_temporal_fact(session.id, "User", "location", "Toronto", turn=2, owner_id=owner_id)
        graph.void_temporal_fact(session.id, "User", "location", turn=4, owner_id=owner_id)
        graph.add_temporal_fact(session.id, "User", "location", "Berlin", turn=4, owner_id=owner_id)

        trace_store = get_trace_store()
        trace_store.append_event(
            "default",
            session.id,
            "deterministic_fact_extractor",
            {
                "fact_count": 1,
                "void_count": 1,
                "facts": [{"subject": "User", "predicate": "location", "object": "Berlin"}],
                "voids": [{"subject": "User", "predicate": "location"}],
            },
            owner_id=owner_id,
        )
        trace_store.append_event(
            "default",
            session.id,
            "memory_fact_filter",
            {
                "blocked_count": 1,
                "blocked": [{"fact": "User prefers weekly summaries", "grounding_reason": "low_grounding"}],
            },
            owner_id=owner_id,
        )

        client = TestClient(app)
        response = client.get(f"/sessions/{session.id}/memory-state", params={"owner_id": owner_id})

        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["deterministic_fact_count"] == 2
        assert payload["summary"]["superseded_fact_count"] == 1
        assert payload["summary"]["candidate_signal_count"] == 2
        assert any(item["object"] == "Berlin" and item["status"] == "current" for item in payload["deterministic_facts"])
        assert any(item["object"] == "Toronto" and item["status"] == "superseded" for item in payload["superseded_facts"])
        assert any(item["reason"] == "low_grounding" for item in payload["candidate_signals"])
        assert any(item["label"] == "Deterministic extractor" for item in payload["recent_memory_signals"])
        assert any("Active Goals" in line for line in payload["context_preview"])

        session_manager.delete(session.id, owner_id=owner_id)
        graph.clear(owner_id=owner_id)


def test_consumer_memory_state_endpoint_uses_consumer_sessions():
    owner_id = f"consumer-memory-owner-{uuid.uuid4().hex[:8]}"
    graph = SemanticGraph()
    graph.clear(owner_id=owner_id)

    with TestClient(app):
        session = consumer_session_manager.create(
            model="groq:moonshotai/kimi-k2-instruct",
            system_prompt="",
            agent_id="consumer",
            owner_id=owner_id,
        )
        consumer_session_manager.update_context(session.id, "Consumer summary line")
        graph.add_temporal_fact(session.id, "User", "preferred_name", "Shovon", turn=1, owner_id=owner_id)

        client = TestClient(app)
        response = client.get(f"/consumer/sessions/{session.id}/memory-state", params={"owner_id": owner_id})

        assert response.status_code == 200
        payload = response.json()
        assert payload["agent_id"] == "consumer"
        assert payload["summary"]["deterministic_fact_count"] == 1
        assert payload["context_preview"] == ["Consumer summary line"]

        consumer_session_manager.delete(session.id, owner_id=owner_id)
        graph.clear(owner_id=owner_id)
