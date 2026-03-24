import json

import pytest

from plugins import tools_web


@pytest.mark.asyncio
async def test_web_search_curates_and_dedupes_results(monkeypatch):
    async def fake_searxng(_query, _num_results):
        return [
            {
                "title": "<b>Alpha Result</b>",
                "url": "https://example.com/post?utm_source=newsletter&id=1",
                "snippet": "  Alpha   snippet with   extra spaces  ",
                "source": "searxng",
            },
            {
                "title": "Alpha Result duplicate",
                "url": "https://example.com/post?id=1&utm_campaign=test",
                "snippet": "Duplicate entry should be removed",
                "source": "searxng",
            },
            {
                "title": "Beta Result",
                "url": "https://another.example.org/news#section",
                "snippet": "<p>Beta snippet</p>",
                "source": "searxng",
            },
            {
                "title": "",
                "url": "",
                "snippet": "",
                "source": "searxng",
            },
        ]

    monkeypatch.setattr(tools_web, "_search_searxng", fake_searxng)

    raw = await tools_web._web_search("test query", num_results=3, backend="searxng")
    data = json.loads(raw)

    assert data["type"] == "web_search_results"
    assert data["backend"] == "searxng"
    assert data["engine"] == "searxng"
    assert len(data["results"]) == 2
    assert data["results"][0]["title"] == "Alpha Result"
    assert data["results"][0]["snippet"] == "Alpha snippet with extra spaces"
    assert data["results"][0]["url"] == "https://example.com/post?id=1"
    assert data["results"][1]["url"] == "https://another.example.org/news"
    assert all(r["title"] or r["url"] or r["snippet"] for r in data["results"])
    assert data["context_summary"]["raw_results_considered"] == 4
    assert data["context_summary"]["curated_results"] == 2
    assert any(domain == "example.com" for domain in data["context_summary"]["unique_domains"])


@pytest.mark.asyncio
async def test_web_search_auto_accumulates_more_results(monkeypatch):
    async def fake_searxng(_query, _num_results):
        return [
            {
                "title": "Only One",
                "url": "https://source-one.example/item",
                "snippet": "first backend",
                "source": "searxng",
            }
        ]

    async def fake_brave(_query, _num_results):
        return [
            {
                "title": "Two",
                "url": "https://source-two.example/a",
                "snippet": "second backend a",
                "source": "brave",
            },
            {
                "title": "Three",
                "url": "https://source-three.example/b",
                "snippet": "second backend b",
                "source": "brave",
            },
        ]

    async def fake_none(_query, _num_results):
        return None

    monkeypatch.setattr(tools_web, "_search_searxng", fake_searxng)
    monkeypatch.setattr(tools_web, "_search_brave", fake_brave)
    monkeypatch.setattr(tools_web, "_search_tavily", fake_none)
    monkeypatch.setattr(tools_web, "_search_exa", fake_none)
    monkeypatch.setattr(tools_web, "_search_groq", fake_none)

    raw = await tools_web._web_search("test query", num_results=3, backend="auto")
    data = json.loads(raw)

    assert len(data["results"]) == 3
    assert data["backend"] == "searxng+brave"
    assert data["context_summary"]["curated_results"] == 3
    assert data["context_summary"]["raw_results_considered"] == 3
