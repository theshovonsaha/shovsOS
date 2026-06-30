from shovs_harness_core import HarnessExtension, run_extension_payload


def test_extension_manifest_is_hostable():
    manifest = HarnessExtension().manifest()
    assert manifest["id"] == "shovs.harness_core"
    assert "evaluate_trace" in manifest["capabilities"]
    assert manifest["input_schema"]["objective"] == "string"


def test_extension_compares_plain_loop_to_harness_loop():
    report = run_extension_payload(
        {
            "objective": "Search top 3 stocks today, search each, fetch 3 URLs each.",
            "entities": ["ROKU", "TBN", "SENEA"],
        }
    )

    assert report["contract"]["total_urls"] == 9
    assert report["reports"]["plain_loop"]["ok"] is False
    assert "entity_drift:EPAM" in report["reports"]["plain_loop"]["failures"]
    assert report["reports"]["harness_loop"]["ok"] is True
    assert report["verdict"]["improvement"] > 0


def test_extension_evaluates_provided_trace_without_fake_success():
    report = run_extension_payload(
        {
            "objective": "Find top 2 sushi places in Toronto, search each, fetch 2 URLs each.",
            "trace": [
                {"kind": "entity_locked", "entity": "MIKU"},
                {"kind": "entity_locked", "entity": "SAITO"},
                {"tool": "web_search", "entity": "MIKU", "ok": True},
                {"tool": "web_fetch", "entity": "SAITO", "ok": False},
            ],
            "include_traces": False,
        }
    )

    provided = report["reports"]["provided_trace"]
    assert provided["ok"] is False
    assert "missing_fetch_quota" in provided["failures"]
    assert "traces" not in report


def test_extension_enforces_model_proposed_actions():
    report = run_extension_payload(
        {
            "objective": "Search top 2 stocks, search each, fetch 2 URLs each.",
            "actions": [
                {"action": "lock", "entities": ["AAA", "BBB"]},
                {"action": "fetch", "entity": "AAA", "url": "https://src.test/AAA/0"},
                {"action": "fetch", "entity": "EPAM", "url": "https://src.test/EPAM/0"},
                {"action": "respond"},
            ],
            "include_traces": False,
        }
    )

    enforcement = report["reports"]["action_enforcement"]
    assert enforcement["can_respond"] is False
    assert [item["code"] for item in enforcement["violations"]] == ["entity_drift", "premature_respond"]
