from orchestration.agent_profiles import AgentProfile, ProfileManager
from orchestration.workflow_templates import get_workflow_template, list_workflow_templates


def test_workflow_templates_are_listable_and_defaultable():
    templates = list_workflow_templates()

    assert any(item["id"] == "general_operator_v1" for item in templates)
    assert get_workflow_template("missing").id == "general_operator_v1"
    assert get_workflow_template("research_agent_v1").risk_policy == "evidence_first"


def test_agent_profile_persists_workflow_coherence_fields(tmp_path):
    manager = ProfileManager(db_path=str(tmp_path / "agents.db"))
    created = manager.create(
        AgentProfile(
            id="research",
            name="Research",
            workflow_template="research_agent_v1",
            prompt_version="role_contracts_v1",
            risk_policy="evidence_first",
            ledger_mode="ledger_enforced",
        )
    )
    loaded = manager.get(created.id)

    assert loaded is not None
    assert loaded.workflow_template == "research_agent_v1"
    assert loaded.risk_policy == "evidence_first"
    assert loaded.ledger_mode == "ledger_enforced"


def test_shopping_advisor_profile_fields_round_trip(tmp_path):
    manager = ProfileManager(db_path=str(tmp_path / "agents.db"))
    profile = manager.create(
        AgentProfile(
            id="shopping-advisor",
            name="Shopping Advisor",
            tools=["shopping_advice", "web_search", "web_fetch"],
            workflow_template="shopping_advisor_v1",
            prompt_version="shopping_patch_v1",
            risk_policy="consumer_verified",
        )
    )
    loaded = manager.get(profile.id)

    assert loaded is not None
    assert loaded.workflow_template == "shopping_advisor_v1"
    assert "shopping_advice" in loaded.tools
    assert loaded.prompt_version == "shopping_patch_v1"
