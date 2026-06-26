from memory import reranker


def test_lexical_fallback_promotes_exact_query_matches(monkeypatch):
    monkeypatch.setattr(reranker, "_get", lambda: False)

    results = [
        {
            "content": "Generic market recap with broad sector notes.",
            "source": "generic",
            "score": 0.99,
        },
        {
            "content": "ROKU analyst reaction after the stock jump.",
            "source": "news",
            "score": 0.2,
        },
        {
            "content": "TBN article without the requested ticker.",
            "source": "news",
            "score": 0.4,
        },
    ]

    ranked = reranker.rerank("ROKU stock news", results, top_n=2)

    assert ranked[0]["content"].startswith("ROKU analyst")
    assert len(ranked) == 2
