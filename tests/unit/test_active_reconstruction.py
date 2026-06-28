"""Active Cue-Tag-Content graph reconstruction (CONCEPT:KG-2.275, MRAgent).

Unit tests for the dependency-injected reconstruction walk plus a LIVE-PATH test
that drives it through the real ``entity_context`` provider / context plane — the
way ``graph_analyze action=explain target="entity:why"`` reaches it.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval import active_reconstruction as ar
from agent_utilities.knowledge_graph.retrieval.context_plane import synthesize_context
from agent_utilities.knowledge_graph.retrieval.entity_context import entity_context


# ── primitives ────────────────────────────────────────────────────────────────
@pytest.mark.concept("KG-2.275")
def test_lexical_relevance_is_query_coverage():
    assert ar.lexical_relevance("home country", "Sweden home country") == 1.0
    assert ar.lexical_relevance("home country", "dance studio") == 0.0
    assert ar.lexical_relevance("", "anything") == 0.0


@pytest.mark.concept("KG-2.275")
def test_humanize_tag_and_node_text():
    assert ar.humanize_tag("MOVED_TO") == "moved to"
    assert ar.humanize_tag("rdf:type") == "rdf type"
    assert ar.node_text({"id": "x", "summary": "s"}) == "s"
    assert ar.node_text({"id": "x"}) == "x"  # falls back to id


# ── the reconstruction walk (no engine, no LLM) ───────────────────────────────
def _toy_neighbors():
    edges = {
        "alice": [("DANCES_AT", "studio"), ("MOVED_TO", "sweden")],
        "studio": [("HOSTED", "recital"), ("DANCES_AT", "alice")],
        "sweden": [("MOVED_TO", "alice")],
        "recital": [("HOSTED", "studio")],
    }
    names = {
        "alice": "Alice",
        "studio": "dance studio",
        "sweden": "Sweden home country",
        "recital": "summer recital",
    }

    def neighbor_fn(node_id):
        return [
            (tag, {"id": tid, "name": names[tid], "label": "Person"})
            for tag, tid in edges.get(node_id, [])
        ]

    return neighbor_fn


@pytest.mark.concept("KG-2.275")
def test_reconstruct_finds_evidence_and_prunes_irrelevant_branch():
    recon = ar.reconstruct(
        "where did alice move to her home country",
        ["alice"],
        neighbor_fn=_toy_neighbors(),
    )
    ids = [e.id for e in recon.evidence]
    # Sweden (the home-country answer) is reconstructed and ranked first.
    assert recon.evidence and recon.evidence[0].id == "sweden"
    assert recon.evidence[0].via_tag == "MOVED_TO"
    # The irrelevant "dance studio" branch is pruned (score below floor).
    assert "studio" not in ids
    assert recon.steps[0].pruned >= 1
    # The walk self-terminates rather than running forever.
    assert recon.stop_reason


@pytest.mark.concept("KG-2.275")
def test_reconstruct_no_seed_is_a_clean_noop():
    recon = ar.reconstruct("q", [], neighbor_fn=_toy_neighbors())
    assert recon.evidence == [] and recon.stop_reason == "no_seed"


@pytest.mark.concept("KG-2.275")
def test_reconstruct_tag_top_k_bounds_expansion():
    # With tag_top_k=1 only the single most-relevant tag is expanded per hop.
    recon = ar.reconstruct(
        "sweden home country",
        ["alice"],
        neighbor_fn=_toy_neighbors(),
        tag_top_k=1,
    )
    assert all(len(s.activated_tags) <= 1 for s in recon.steps)


# ── engine-backed adapters + LIVE entity_context "why" path ───────────────────
class FakeGraphEngine:
    """Minimal engine supporting ``_search_keyword`` + the neighbour Cypher."""

    NODES = {
        "alice": "Alice",
        "studio": "dance studio",
        "sweden": "Sweden home country",
    }
    EDGES = {
        "alice": [("DANCES_AT", "studio"), ("MOVED_TO", "sweden")],
        "sweden": [("MOVED_TO", "alice")],
        "studio": [("DANCES_AT", "alice")],
    }

    def _search_keyword(self, q, top_k=3):
        ql = (q or "").lower()
        return [
            {"id": nid, "name": name}
            for nid, name in self.NODES.items()
            if name.split()[0].lower() in ql
        ][:top_k]

    def query_cypher(self, cypher, params):
        if "-[r]-(t)" in cypher:
            nid = params.get("id")
            return [
                {
                    "tag": tag,
                    "id": tid,
                    "name": self.NODES[tid],
                    "title": None,
                    "text": None,
                    "summary": None,
                    "label": "Person",
                }
                for tag, tid in self.EDGES.get(nid, [])
            ]
        return []  # census + CONTAINS fallback: nothing


@pytest.mark.concept("KG-2.275")
def test_engine_neighbor_fn_and_resolve_seeds():
    eng = FakeGraphEngine()
    seeds = ar.resolve_seeds(eng, "tell me about alice")
    assert [s["id"] for s in seeds] == ["alice"]
    neigh = ar.engine_neighbor_fn(eng)("alice")
    by_id = {n["id"]: (tag, n) for tag, n in neigh}
    assert by_id["sweden"][0] == "MOVED_TO"
    assert by_id["sweden"][1]["name"] == "Sweden home country"


@pytest.mark.concept("KG-2.275")
def test_entity_context_why_runs_active_reconstruction_live():
    """The live provider path: intent='why' reconstructs instead of a census."""
    res = entity_context(
        FakeGraphEngine(),
        query="where did alice move to her home country",
        intent="why",
        domain="entity",
    )
    assert res["status"] == "ok"
    assert res["used_primitives"] == ["active_reconstruction"]
    assert res["capability_id"] == "entity:entity:why"
    assert "Reconstructed" in res["answer"] and "Sweden" in res["answer"]
    cited = {c["id"] for c in res["citations"]}
    assert "sweden" in cited and "studio" not in cited
    assert res["sections"]["stop_reason"]


@pytest.mark.concept("KG-2.275")
def test_entity_context_why_falls_back_to_census_when_no_seed():
    """No resolvable seed → falls through to the census view (no regression)."""
    res = entity_context(
        FakeGraphEngine(), query="why does zzz nothing match", intent="why"
    )
    assert res["status"] == "ok"
    assert res["used_primitives"] != ["active_reconstruction"]


@pytest.mark.concept("KG-2.275")
def test_context_plane_routes_entity_why_to_reconstruction():
    """End-to-end through synthesize_context, as the MCP/REST surface calls it."""
    res = synthesize_context(
        FakeGraphEngine(),
        domain="entity",
        query="where did alice move to her home country",
        intent="why",
    )
    assert res["domain"] == "entity"
    assert res["used_primitives"] == ["active_reconstruction"]
