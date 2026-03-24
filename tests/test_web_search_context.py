import json

from plugins.tools import _format_search_results, _normalize_search_results


def test_normalize_search_results_dedupes_and_fills_missing_snippets():
    raw = [
        {"title": "Result 1", "url": "https://example.com/a", "snippet": "", "source": "duckduckgo"},
        {"title": "Result 1 duplicate", "url": "https://example.com/a/", "snippet": "Duplicate", "source": "duckduckgo"},
        {"title": "", "url": "https://example.com/b", "snippet": "  Useful   context   text  ", "source": "duckduckgo"},
    ]

    normalized = _normalize_search_results(raw, max_results=5)

    assert len(normalized) == 2
    assert normalized[0]["snippet"] == "Result 1"
    assert normalized[1]["snippet"] == "Useful context text"


def test_format_search_results_includes_engine_and_context_summary():
    raw = [
        {"title": "R1", "url": "https://example.com/1", "snippet": "Alpha fact", "source": "duckduckgo"},
        {"title": "R2", "url": "https://example.com/2", "snippet": "Beta fact", "source": "duckduckgo"},
    ]

    payload = json.loads(_format_search_results("test query", raw, engine="duckduckgo"))

    assert payload["type"] == "web_search_results"
    assert payload["engine"] == "duckduckgo"
    assert "Alpha fact" in payload["context_summary"]
    assert len(payload["results"]) == 2
