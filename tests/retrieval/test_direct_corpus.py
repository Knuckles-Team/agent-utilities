#!/usr/bin/python
"""Tests for Direct Corpus Interaction (grep/read/search) and its wiring.

CONCEPT:AU-KG.retrieval.memory-first-retrieval
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.retrieval.direct_corpus import (
    DciResult,
    DirectCorpusSearcher,
    searcher_from_nodes,
)

pytestmark = pytest.mark.concept("AU-KG.retrieval.memory-first-retrieval")

_CORPUS = {
    "d1": "The HybridRetriever combines vector and graph search.\nIt also supports reranking.",
    "d2": "A recipe for tomato pasta.\nBoil water, add pasta.",
    "d3": "def retrieve_hybrid(query):\n    return vector_search(query)",
}


# --- grep ------------------------------------------------------------------


def test_grep_literal_match():
    s = DirectCorpusSearcher(_CORPUS)
    hits = s.grep("pasta")
    assert {h.doc_id for h in hits} == {"d2"}
    assert all(h.line_no >= 1 for h in hits)


def test_grep_regex_and_context():
    s = DirectCorpusSearcher(_CORPUS)
    hits = s.grep(r"def \w+\(", regex=True, context_lines=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "d3"
    assert hits[0].after  # context captured


def test_grep_invalid_regex_raises():
    s = DirectCorpusSearcher(_CORPUS)
    with pytest.raises(ValueError):
        s.grep("(unclosed", regex=True)


# --- read ------------------------------------------------------------------


def test_read_full_and_range():
    s = DirectCorpusSearcher(_CORPUS)
    assert "vector and graph" in s.read("d1")
    assert s.read("d1", start=2, end=2).strip() == "It also supports reranking."


def test_read_missing_doc_raises():
    with pytest.raises(KeyError):
        DirectCorpusSearcher(_CORPUS).read("nope")


# --- ranked search ---------------------------------------------------------


def test_search_ranks_relevant_first_with_localization():
    s = DirectCorpusSearcher(_CORPUS)
    res = s.search("vector graph search", top_k=5)
    assert isinstance(res, DciResult)
    assert res.hits[0].doc_id == "d1"
    assert res.hits[0].term_coverage > 0
    assert res.hits[0].match_lines  # localization present
    assert res.docs_searched == 3


def test_search_empty_query():
    res = DirectCorpusSearcher(_CORPUS).search("")
    assert res.hits == []


def test_search_no_match():
    res = DirectCorpusSearcher(_CORPUS).search("xylophone quasar")
    assert res.hits == []


# --- node adapter ----------------------------------------------------------


def test_searcher_from_nodes_uses_text_keys():
    nodes = [
        {"id": "a", "content": "alpha vector index"},
        {"id": "b", "name": "beta", "summary": "graph traversal"},
        {"id": "c"},  # no text → skipped
    ]
    s = searcher_from_nodes(nodes)
    assert set(s.documents) == {"a", "b"}


# --- live path: HybridRetriever.direct_search ------------------------------


def test_live_path_direct_search_over_engine_nodes():
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    engine = MagicMock()

    def _execute(cypher, *args, **kwargs):
        if "Article" in cypher:
            return [
                {
                    "id": "a1",
                    "data": {"id": "a1", "content": "vector graph search index"},
                },
                {
                    "id": "a2",
                    "data": {"id": "a2", "content": "unrelated cooking notes"},
                },
            ]
        return []

    engine.backend.execute.side_effect = _execute
    r = HybridRetriever(engine)
    res = r.direct_search("vector graph", top_k=5)
    assert res.hits and res.hits[0].doc_id == "a1"
