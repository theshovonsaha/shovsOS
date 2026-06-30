from engine.context_schema import ContextPhase
from run_engine.control_policies import resolve_control_policy
from run_engine.language_kernel import build_kernel_snapshot
from run_engine.ledger import RunLedger


def _ledger_with_blocked_source_contract() -> RunLedger:
    ledger = RunLedger(
        run_id="run-kernel",
        session_id="session-kernel",
        turn_id="turn-1",
        objective="Search top 3 stocks, search each separately, fetch 3 URLs each, write a TLDR table.",
        allowed_tools=["web_search", "web_fetch"],
        ledger_mode="ledger_enforced",
    )
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="plan_execute"))
    ledger.lock_entities(["ROKU", "TBN", "SENEA"], entity_type="ticker")
    ledger.set_source_contract({
        "next_tool": "web_search",
        "next_arguments": {"query": "ROKU stock news June 13 2026"},
        "allowed_fetch_urls_by_entity": {
            "ROKU": ["https://news.example/roku-1", "https://news.example/roku-2", "https://news.example/roku-3"],
            "TBN": ["https://news.example/tbn-1", "https://news.example/tbn-2", "https://news.example/tbn-3"],
            "SENEA": ["https://news.example/senea-1", "https://news.example/senea-2", "https://news.example/senea-3"],
        },
        "urls_per_entity": 3,
        "total_urls": 9,
        "forbid_unlocked_entity_drift": True,
    })
    call = ledger.add_tool_call("web_search", {"query": "ROKU stock news June 13 2026"})
    result = ledger.link_tool_result(
        tool_call_id=call.id,
        tool_name="web_search",
        success=True,
        status="success",
        summary="Found ROKU sources including https://news.example/roku-1",
    )
    ledger.add_evidence_from_result(result)
    return ledger


def test_language_kernel_snapshot_closes_all_seven_research_lanes():
    ledger = _ledger_with_blocked_source_contract()

    snapshot = build_kernel_snapshot(
        ledger,
        ContextPhase.ACTING,
        compact_memory="User prefers exact URLs and deterministic source collection.",
        relevant_blocks=[{"id": "block-1", "content": "Use Morningstar movers before locking ticker entities."}],
        raw_payloads=[{"id": "raw-1", "content": "<html>large payload</html>"}],
    )
    data = snapshot.to_dict()

    assert data["version"] == "language-kernel-v1"

    # 1. Context attention layer.
    assert data["attention"]["version"] == "runtime-attention-v1"
    assert data["context_ladder"]["steps"][0]["level"] == "evidence_reference"
    assert any(step["level"] == "raw_payload_ref" for step in data["context_ladder"]["steps"])

    # 2. Tool contract compiler / gate.
    assert data["tool_gate"]["next_required_action"]["tool"] == "web_search"
    assert data["prompt_contract"]["allowed_next_arguments"]["query"] == "ROKU stock news June 13 2026"
    assert data["prompt_contract"]["final_answer_allowed"] is False

    # 3. Memory immune system.
    assert data["memory_immune_report"]["safe_to_commit"] is True

    # 4. Experience graph.
    assert data["experience_graph"]["summary"]["successful_tools"] == 1
    assert any(node["kind"] == "evidence" for node in data["experience_graph"]["nodes"])

    # 5. Proxy state eval.
    assert data["proxy_state_eval"]["passed"] is False
    assert "fetched_urls:0/9" in data["proxy_state_eval"]["issues"]

    # 6. Small model micro-agent jobs.
    job_ids = {job["id"] for job in data["micro_agent_jobs"]}
    assert {"gap_classifier", "verifier_precheck", "evidence_summarizer"}.issubset(job_ids)

    # 7. Agent-native UI run map.
    assert data["ui_run_map"]["version"] == "agent-native-run-map-v1"
    assert data["ui_run_map"]["next_focus"] in {"plan", "tools", "memory", "verification", "response"}


def test_language_kernel_prompt_contract_blocks_fake_tool_claims():
    ledger = RunLedger(
        run_id="run-fake-claim",
        session_id="session-fake-claim",
        turn_id="turn-1",
        objective="Fetch exact article then summarize.",
        allowed_tools=["web_fetch"],
    )

    snapshot = build_kernel_snapshot(
        ledger,
        ContextPhase.RESPONSE,
        response_text="I fetched URL 1 and verified the article.",
    )
    data = snapshot.to_dict()

    assert data["proxy_state_eval"]["passed"] is False
    assert "response_claims_tool_work_without_successful_tool_result" in data["proxy_state_eval"]["issues"]
    assert "response_claims_fetch_without_successful_web_fetch" in data["proxy_state_eval"]["issues"]
    assert data["prompt_contract"]["successful_tool_result_ids"] == []


def test_language_kernel_memory_immune_surfaces_disputed_writes():
    ledger = RunLedger(
        run_id="run-memory-dispute",
        session_id="session-memory-dispute",
        turn_id="turn-1",
        objective="Store a corrected user preference.",
        allowed_tools=[],
    )
    ledger.add_memory_write(
        status="disputed",
        summary="Old preference demoted by newer user correction.",
        data={
            "source": "user_message",
            "conflict_trace": {"old": "likes seafood", "new": "no seafood"},
        },
    )

    snapshot = build_kernel_snapshot(ledger, ContextPhase.MEMORY_COMMIT)
    report = snapshot.to_dict()["memory_immune_report"]

    assert report["safe_to_commit"] is True
    assert report["disputed_writes"][0]["conflict_trace"]["new"] == "no seafood"
    assert snapshot.to_dict()["micro_agent_jobs"][0]["id"] == "memory_eligibility"
