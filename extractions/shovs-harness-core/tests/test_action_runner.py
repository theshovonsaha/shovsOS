from shovs_harness_core import enforce_proposed_actions, infer_source_contract
from shovs_harness_core.proposers import ProposedAction


def test_action_runner_accepts_complete_contract_shape():
    contract = infer_source_contract("Search top 2 stocks, search each, fetch 2 URLs each.")
    actions = [
        ProposedAction("lock", entities=("AAA", "BBB")),
        ProposedAction("fetch", entity="AAA", url="https://src.test/AAA/0"),
        ProposedAction("fetch", entity="AAA", url="https://src.test/AAA/1"),
        ProposedAction("fetch", entity="BBB", url="https://src.test/BBB/0"),
        ProposedAction("fetch", entity="BBB", url="https://src.test/BBB/1"),
        ProposedAction("respond"),
    ]

    report = enforce_proposed_actions(contract, actions)

    assert report.can_respond is True
    assert report.eval.ok is True
    assert report.violations == []


def test_action_runner_blocks_drift_and_premature_response():
    contract = infer_source_contract("Search top 3 stocks, search each, fetch 3 URLs each.")
    actions = [
        ProposedAction("lock", entities=("ROKU", "TBN", "SENEA")),
        ProposedAction("fetch", entity="EPAM", url="https://src.test/EPAM/0"),
        ProposedAction("fetch", entity="ROKU", url="https://src.test/ROKU/0"),
        ProposedAction("respond"),
    ]

    report = enforce_proposed_actions(contract, actions)

    assert report.can_respond is False
    assert [item.code for item in report.violations] == ["entity_drift", "premature_respond"]
    assert "missing_fetch_quota" in report.eval.failures


def test_action_runner_rejects_off_contract_urls():
    contract = infer_source_contract("Find top 1 sushi place, search each, fetch 1 URL each.")
    actions = [
        ProposedAction("lock", entities=("MIKU",)),
        ProposedAction("fetch", entity="MIKU", url="https://bad.test/miku"),
    ]

    report = enforce_proposed_actions(
        contract,
        actions,
        candidate_urls={"MIKU": ["https://good.test/miku"]},
    )

    assert report.can_respond is False
    assert [item.code for item in report.violations] == ["off_contract_url"]
    assert "missing_fetch_quota" in report.eval.failures
