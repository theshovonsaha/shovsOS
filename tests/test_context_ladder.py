from run_engine.context_ladder import build_context_ladder, render_context_ladder


def test_context_ladder_returns_compact_signal_before_raw_payload():
    ladder = build_context_ladder(
        query="ROKU stock news",
        compact_memory="Old line\nROKU source workflow in progress\nOther note",
        relevant_blocks=[{"id": "block-1", "title": "Prior run", "content": "TBN note\nROKU needs two URLs"}],
        evidence_refs=[{"id": "result-1", "summary": "ROKU source candidate found", "source": "tool:web_search"}],
        raw_payloads=[{"id": "raw-1", "content": "{\"large\": \"payload\"}"}],
        include_raw=False,
    )

    levels = [step["level"] for step in ladder["steps"]]
    rendered = render_context_ladder(ladder)

    assert levels[0] == "evidence_reference"
    assert "evidence_reference" in levels
    assert "relevant_block" in levels
    assert "raw_payload_ref" in levels
    assert "Raw payload available on demand" in rendered
    assert "{\"large\"" not in rendered
