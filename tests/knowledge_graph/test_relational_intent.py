from __future__ import annotations

"""Tests for deterministic relational-intent retrieval.

CONCEPT:KG-2.34 — Relational-Intent Retrieval
"""


from unittest.mock import MagicMock, patch

from agent_utilities.knowledge_graph.retrieval.relational_intent import (
    parse_relational_intent,
    traverse,
)
from agent_utilities.models.schema_packs import get_schema_pack

_VERBS = get_schema_pack("research-state").relational_verbs
_EMBED = "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"


class TestParse:
    def test_outgoing(self):
        rq = parse_relational_intent("which papers support transformers", _VERBS)
        assert rq is not None
        assert rq.verb_edge == "supports_belief"
        assert rq.direction == "out"
        assert rq.seed_text == "transformers"

    def test_weakens(self):
        rq = parse_relational_intent("what weakens the scaling hypothesis", _VERBS)
        assert rq is not None and rq.verb_edge == "weakens"

    def test_inverse_direction(self):
        rq = parse_relational_intent("what is cited by Smith2020", _VERBS)
        assert rq is not None
        assert rq.verb_edge == "cites_source"
        assert rq.direction == "in"
        assert rq.seed_text == "Smith2020"

    def test_non_relational_is_noop(self):
        assert parse_relational_intent("summarize the dataset", _VERBS) is None
        assert parse_relational_intent("tell me about transformers", _VERBS) is None

    def test_empty_vocab_is_noop(self):
        assert parse_relational_intent("which papers support X", {}) is None


class TestTraverse:
    def test_resolves_seed_and_walks_edge(self):
        engine = MagicMock()
        engine._search_keyword.return_value = [{"id": "seed:1"}]
        engine.backend.execute.return_value = [
            {"id": "paper:2", "data": {"id": "paper:2", "name": "Paper Two"}}
        ]
        from agent_utilities.knowledge_graph.retrieval.relational_intent import (
            RelationalQuery,
        )

        rq = RelationalQuery(verb_edge="supports_belief", direction="out", seed_text="x")
        out = traverse(engine, rq, top_k=5)
        assert [n["id"] for n in out] == ["paper:2"]
        assert out[0]["_relational_hit"] == "supports_belief"

    def test_unresolved_seed_returns_empty(self):
        engine = MagicMock()
        engine._search_keyword.return_value = []
        from agent_utilities.knowledge_graph.retrieval.relational_intent import (
            RelationalQuery,
        )

        rq = RelationalQuery(verb_edge="supports_belief", direction="out", seed_text="x")
        assert traverse(engine, rq) == []


@patch(_EMBED, side_effect=Exception("no embed in test"))
def test_live_path_retrieve_hybrid_invokes_relational_arm(_m):
    """LIVE-PATH: retrieve_hybrid actually runs the relational arm and merges hits."""
    from agent_utilities.knowledge_graph.retrieval import (
        relational_intent as ri_module,
    )
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    engine = MagicMock()
    engine.backend = None  # skip vector + graph-traversal branches deterministically
    engine._search_keyword.return_value = []
    r = HybridRetriever(engine, schema_pack=get_schema_pack("research-state"))

    sentinel = [{"id": "rel:hit", "_score": 1.0, "_relational_hit": "supports_belief"}]
    with patch.object(
        ri_module, "traverse", return_value=sentinel
    ), patch.object(
        ri_module,
        "parse_relational_intent",
        return_value=ri_module.RelationalQuery("supports_belief", "out", "x"),
    ):
        results = r.retrieve_hybrid("which papers support transformers", context_window=5)

    assert any(n.get("id") == "rel:hit" for n in results)
