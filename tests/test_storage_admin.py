from pathlib import Path

from memory.semantic_graph import SemanticGraph
from memory.tool_results_db import ToolResultsDB
from orchestration.agent_profiles import ProfileManager
from orchestration.session_manager import SessionManager
from services.storage_admin import StorageAdminService, StoreSelection


def test_storage_admin_backup_and_reset(tmp_path: Path):
    sessions_db = tmp_path / "sessions.db"
    agents_db = tmp_path / "agents.db"
    graph_db = tmp_path / "memory_graph.db"
    tool_db = tmp_path / "tool_results.db"
    backups_dir = tmp_path / "backups"

    sessions = SessionManager(max_sessions=10, db_path=str(sessions_db))
    profiles = ProfileManager(db_path=str(agents_db))

    session = sessions.create(model="llama3.2", system_prompt="hello")
    sessions.append_message(session.id, "user", "remember me")
    sessions.append_message(session.id, "assistant", "ok")

    graph = SemanticGraph(db_path=str(graph_db))
    graph.add_temporal_fact(session.id, "Shovon", "lives_in", "Toronto", 1)

    tool_results = ToolResultsDB(db_path=str(tool_db))
    tool_results.store(session.id, "web_search", {"query": "test"}, "result text")

    service = StorageAdminService(
        sessions=sessions,
        profiles=profiles,
        backup_root=backups_dir,
        path_overrides={
            "semantic_memory": graph_db,
            "tool_results": tool_db,
            "vector_memory": tmp_path / "chroma_db",
            "session_rag": tmp_path / "session_rag",
        },
    )

    backup = service.backup(StoreSelection(sessions=True, semantic_memory=True, tool_results=True, agents=True, vector_memory=False, session_rag=False), "unit")
    assert backup["items"]
    assert (Path(backup["path"]) / "manifest.json").exists()

    result = service.reset(
        StoreSelection(sessions=True, semantic_memory=True, tool_results=True, agents=False, vector_memory=False, session_rag=False),
        backup_first=False,
    )

    assert result["cleared"]["sessions"]["deleted_rows"] >= 1
    assert result["cleared"]["semantic_memory"]["deleted_rows"] >= 0
    assert result["cleared"]["tool_results"]["deleted_rows"] >= 1
    assert sessions.list_sessions() == []
    assert graph.get_current_facts(session.id) == []
    assert tool_results.count() == 0
