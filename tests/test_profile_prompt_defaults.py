from orchestration.agent_profiles import (
    AgentProfile,
    DEFAULT_AGENT_TOOLS,
    GENERAL_SYSTEM_PROMPT,
    PLATINUM_SYSTEM_PROMPT,
    ProfileManager,
)


def test_default_profile_uses_general_prompt(tmp_path):
    db_path = tmp_path / "agents.db"
    pm = ProfileManager(db_path=str(db_path))

    default = pm.get("default")

    assert default is not None
    assert default.runtime_kind == "managed"
    assert default.system_prompt == GENERAL_SYSTEM_PROMPT
    assert default.system_prompt != PLATINUM_SYSTEM_PROMPT
    assert default.tools == DEFAULT_AGENT_TOOLS
    assert "web_search" in default.tools
    assert "file_create" not in default.tools
    assert "rag_search" not in default.tools


def test_non_visual_profile_is_not_forced_to_platinum_prompt(tmp_path):
    db_path = tmp_path / "agents.db"
    pm = ProfileManager(db_path=str(db_path))
    profile = AgentProfile(
        id="writer_test",
        name="Writer Test",
        description="General writing helper",
        tools=["file_create"],
        system_prompt=PLATINUM_SYSTEM_PROMPT,
    )
    pm.create(profile)

    pm_reloaded = ProfileManager(db_path=str(db_path))
    stored = pm_reloaded.get("writer_test")

    assert stored is not None
    assert stored.system_prompt == GENERAL_SYSTEM_PROMPT


def test_profile_runtime_kind_round_trips(tmp_path):
    db_path = tmp_path / "agents.db"
    pm = ProfileManager(db_path=str(db_path))
    profile = AgentProfile(
        id="runtime_test",
        name="Runtime Test",
        runtime_kind="managed",
        tools=["web_search"],
    )
    pm.create(profile)

    stored = pm.get("runtime_test")

    assert stored is not None
    assert stored.runtime_kind == "managed"


def test_profile_runtime_kind_legacy_aliases_normalize(tmp_path):
    db_path = tmp_path / "agents.db"
    pm = ProfileManager(db_path=str(db_path))
    profile = AgentProfile(
        id="runtime_alias_test",
        name="Runtime Alias Test",
        runtime_kind="legacy",
        tools=["web_search"],
    )
    pm.create(profile)

    stored = pm.get("runtime_alias_test")

    assert stored is not None
    assert stored.runtime_kind == "managed"


def test_profile_bootstrap_fields_round_trip(tmp_path):
    db_path = tmp_path / "agents.db"
    pm = ProfileManager(db_path=str(db_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    profile = AgentProfile(
        id="bootstrap_test",
        name="Bootstrap Test",
        workspace_path=str(workspace),
        bootstrap_files=["AGENTS.md", "SOUL.md"],
        bootstrap_max_chars=4321,
    )
    pm.create(profile)

    stored = pm.get("bootstrap_test")

    assert stored is not None
    assert stored.workspace_path == str(workspace)
    assert stored.bootstrap_files == ["AGENTS.md", "SOUL.md"]
    assert stored.bootstrap_max_chars == 4321


def test_profile_runtime_defaults_and_bootstrap_are_sanitized(tmp_path):
    db_path = tmp_path / "agents.db"
    pm = ProfileManager(db_path=str(db_path))
    profile = AgentProfile(
        id="sanitize_test",
        name=" Sanitize Test ",
        tools=["web_search", "web_search", "", "store_memory"],
        bootstrap_files=["docs/AGENTS.md", "AGENTS.md", "", "../SOUL.md"],
        bootstrap_max_chars=999999,
        default_loop_mode="weird",
        default_context_mode="bad",
        default_use_planner=False,
    )
    pm.create(profile)

    stored = pm.get("sanitize_test")

    assert stored is not None
    assert stored.name == "Sanitize Test"
    assert stored.tools == ["web_search", "store_memory"]
    assert stored.bootstrap_files == ["AGENTS.md", "SOUL.md"]
    assert stored.bootstrap_max_chars == 20000
    assert stored.default_loop_mode == "auto"
    assert stored.default_context_mode == "v2"
    assert stored.default_use_planner is False
