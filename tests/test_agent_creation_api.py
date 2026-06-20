from fastapi.testclient import TestClient

import api.main as main_mod
from api.main import app


def test_agent_create_sanitizes_fields_and_invalidates_cache(monkeypatch):
    invalidated: list[tuple[str, str | None]] = []

    def fake_invalidate(agent_id: str, owner_id: str | None = None):
        invalidated.append((agent_id, owner_id))

    monkeypatch.setattr(main_mod.agent_manager, "invalidate_cache", fake_invalidate)
    client = TestClient(app)
    owner_id = "agent-create-owner"
    payload = {
        "id": "api-create-sanitize",
        "owner_id": owner_id,
        "name": " API Create ",
        "model": "ollama:llama3.2",
        "tools": ["web_search", "web_search", "", " query_memory "],
        "bootstrap_files": ["docs/AGENTS.md", "../SOUL.md", "AGENTS.md"],
        "bootstrap_max_chars": 999999,
        "default_context_mode": "bad",
        "ledger_mode": "invalid",
    }

    response = client.post("/agents", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["owner_id"] == owner_id
    assert data["name"] == "API Create"
    assert data["tools"] == ["web_search", "query_memory"]
    assert data["bootstrap_files"] == ["AGENTS.md", "SOUL.md"]
    assert data["bootstrap_max_chars"] == 20000
    assert data["default_context_mode"] == "v2"
    assert data["ledger_mode"] == "shadow"
    assert invalidated == [("api-create-sanitize", owner_id)]


def test_agent_patch_coerces_scalar_fields_before_profile_sanitize():
    client = TestClient(app)
    owner_id = "agent-patch-owner"
    create = client.post(
        "/agents",
        json={
            "id": "api-patch-coerce",
            "owner_id": owner_id,
            "name": "Patch Coerce",
            "tools": ["web_search"],
            "embed_model": "ollama:nomic-embed-text",
        },
    )
    assert create.status_code == 200

    patch = client.patch(
        "/agents/api-patch-coerce",
        json={
            "owner_id": owner_id,
            "tools": "web_search, web_fetch,, query_memory",
            "skills": "alpha\nbeta",
            "bootstrap_files": "docs/AGENTS.md, ../SOUL.md",
            "default_use_planner": "false",
            "unified_model_mode": "false",
            "bootstrap_max_chars": "500",
            "default_context_mode": "bad",
        },
    )

    assert patch.status_code == 200
    data = patch.json()
    assert data["tools"] == ["web_search", "web_fetch", "query_memory"]
    assert data["skills"] == ["alpha", "beta"]
    assert data["bootstrap_files"] == ["AGENTS.md", "SOUL.md"]
    assert data["default_use_planner"] is False
    assert data["unified_model_mode"] is False
    assert data["bootstrap_max_chars"] == 1000
    assert data["default_context_mode"] == "v2"


def test_agent_patch_rejects_embed_model_change():
    client = TestClient(app)
    owner_id = "agent-embed-owner"
    create = client.post(
        "/agents",
        json={
            "id": "api-embed-immutable",
            "owner_id": owner_id,
            "name": "Embed Immutable",
            "embed_model": "ollama:nomic-embed-text",
        },
    )
    assert create.status_code == 200

    patch = client.patch(
        "/agents/api-embed-immutable",
        json={
            "owner_id": owner_id,
            "embed_model": "openai:text-embedding-3-small",
        },
    )

    assert patch.status_code == 400
    assert "embed_model is immutable" in patch.json()["detail"]


def test_session_create_uses_agent_profile_model_when_model_omitted():
    client = TestClient(app)
    owner_id = "agent-session-owner"
    created_agent = client.post(
        "/agents",
        json={
            "id": "api-session-profile-model",
            "owner_id": owner_id,
            "name": "Profile Model",
            "model": "groq:llama-3.1-8b-instant",
            "default_context_mode": "v3",
        },
    )
    assert created_agent.status_code == 200

    session = client.post(
        "/sessions",
        json={
            "owner_id": owner_id,
            "agent_id": "api-session-profile-model",
        },
    )

    assert session.status_code == 200
    data = session.json()
    assert data["agent_id"] == "api-session-profile-model"
    assert data["model"] == "groq:llama-3.1-8b-instant"
    assert data["context_mode"] == "v3"
