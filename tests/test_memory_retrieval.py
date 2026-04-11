import pytest

from memory.retrieval import unified_memory_search


class _FakeGraph:
    async def traverse(self, query, top_k=5, threshold=0.5, owner_id=None, locus_id=None):
        return [
            {
                "id": 1,
                "subject": "User",
                "predicate": "location",
                "object": "Berlin",
                "similarity": 0.88,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]

    def get_current_facts(self, session_id, owner_id=None):
        return [
            ("User", "location", "Berlin"),
            ("User", "preferred_editor", "Cursor"),
        ]

    def list_loci(self, owner_id=None):
        return []

    def get_compiled_drawer(self, locus_id):
        return None


@pytest.mark.asyncio
async def test_unified_memory_search_merges_and_deduplicates_sources(monkeypatch):
    class _FakeVectorEngine:
        def __init__(self, *args, **kwargs):
            pass

        async def query(self, text, limit=5):
            return [
                {
                    "id": "vec-1",
                    "key": "User location",
                    "anchor": "User: I moved to Berlin.\nAssistant: Updated location.",
                    "metadata": {"created_at": "2026-01-02T00:00:00Z"},
                }
            ]

    monkeypatch.setattr("memory.retrieval.VectorEngine", _FakeVectorEngine)

    payload = await unified_memory_search(
        "where do i live",
        owner_id="owner-x",
        session_id="session-x",
        top_k=5,
        graph=_FakeGraph(),
    )

    assert payload["stats"]["session_scoped"] is True
    assert payload["stats"]["source_counts"]["semantic_graph"] == 1
    assert payload["stats"]["source_counts"]["deterministic_fact"] == 2
    assert payload["stats"]["source_counts"]["vector_engine"] == 1
    assert payload["results"]
    first = payload["results"][0]
    assert first["kind"] in {"triplet", "fact"}
    assert "sources" in first
    # Location exists in both deterministic and semantic lanes and should be merged.
    assert any(
        item.get("subject") == "User"
        and item.get("predicate") == "location"
        and "semantic_graph" in (item.get("sources") or [])
        for item in payload["results"]
    )


@pytest.mark.asyncio
async def test_unified_memory_search_without_session_only_uses_semantic():
    payload = await unified_memory_search(
        "location",
        owner_id="owner-x",
        session_id=None,
        top_k=3,
        graph=_FakeGraph(),
    )
    assert payload["stats"]["session_scoped"] is False
    assert payload["stats"]["source_counts"]["semantic_graph"] == 1
    assert payload["stats"]["source_counts"]["deterministic_fact"] == 0
