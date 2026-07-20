"""Security boundaries for the citation-graph research client."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.orchestration import research_subagent as subject


def test_paper_id_is_bounded_encoded_and_response_is_bounded(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return {"paperId": "safe"}

    monkeypatch.setattr(subject, "safe_get_json", fake_get)
    walker = subject.CitationGraphWalker(api_key="runtime-only", rate_limit_seconds=0)

    assert walker.fetch_paper("doi/value?redirect=unexpected") == {"paperId": "safe"}
    assert captured["url"].endswith("doi%2Fvalue%3Fredirect%3Dunexpected")
    assert captured["params"] == {"fields": subject._S2_FIELDS}
    assert captured["max_bytes"] == 4 * 1024 * 1024
    assert captured["max_redirects"] == 0
    assert walker.fetch_paper("x" * (subject._MAX_PAPER_ID_BYTES + 1)) is None


def test_cache_is_lru_bounded(monkeypatch):
    monkeypatch.setattr(subject, "_MAX_PAPER_CACHE_ENTRIES", 2)
    monkeypatch.setattr(
        subject,
        "safe_get_json",
        lambda url, **_kwargs: {"paperId": url.rsplit("/", 1)[-1]},
    )
    walker = subject.CitationGraphWalker(rate_limit_seconds=0)

    for paper_id in ("one", "two", "three"):
        assert walker.fetch_paper(paper_id) is not None

    assert list(walker._cache) == ["two", "three"]


@pytest.mark.parametrize(
    ("depth", "per_level"),
    ((-1, 1), (subject._MAX_CITATION_DEPTH + 1, 1), (1, 0), (1, 101)),
)
def test_traversal_bounds_fail_before_network(depth, per_level):
    walker = subject.CitationGraphWalker(rate_limit_seconds=0)
    with pytest.raises(ValueError, match="outside the supported boundary"):
        walker.get_citations("paper", max_depth=depth, max_per_level=per_level)
