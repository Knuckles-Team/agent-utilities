"""Regression: hybrid search must expose its relevance under the public `score` key.

The hybrid retriever ranks into the internal `_score` (it re-weights it during
fusion); the MCP formatter and ~24 other consumers read `score`. Before the fix no
layer mapped one to the other, so every search result rendered as `Score: 0.00`
even though the ranking was correct — a producer/consumer contract mismatch.
"""

from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin


class _StubRetriever:
    def retrieve_hybrid(self, query, **kwargs):  # noqa: ANN001, ANN003
        return [
            {"id": "a", "type": "Code", "_score": 0.73, "status": "ACTIVE"},
            {"id": "b", "type": "Code", "_score": 0.41, "status": "ACTIVE"},
        ]


class _Engine(QueryMixin):
    def __init__(self) -> None:
        self.hybrid_retriever = _StubRetriever()
        self.active_schema_pack = None


def test_search_hybrid_projects_internal_rank_to_public_score() -> None:
    out = _Engine().search_hybrid("anything", top_k=5)
    assert out, "expected results"
    # The public contract is `score`; the value comes from the retriever's `_score`.
    assert out[0]["score"] == 0.73
    assert out[1]["score"] == 0.41
    # The internal key is preserved (it is still used by downstream re-rankers).
    assert out[0]["_score"] == 0.73


def test_search_hybrid_does_not_overwrite_an_existing_score() -> None:
    class _ScoredRetriever:
        def retrieve_hybrid(self, query, **kwargs):  # noqa: ANN001, ANN003
            return [{"id": "a", "score": 0.9, "_score": 0.1, "status": "ACTIVE"}]

    eng = _Engine()
    eng.hybrid_retriever = _ScoredRetriever()
    out = eng.search_hybrid("anything", top_k=5)
    # An explicit `score` (e.g. the encoder path) wins over the internal `_score`.
    assert out[0]["score"] == 0.9
