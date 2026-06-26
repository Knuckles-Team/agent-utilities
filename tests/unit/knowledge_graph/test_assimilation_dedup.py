#!/usr/bin/python
"""Cross-source feature dedup (VU-2).

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import dedup_features

pytestmark = pytest.mark.concept("KG-2.7")


class _Graph:
    def __init__(self, nodes):
        self._n = nodes  # id -> attrs dict

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)


class _Engine:
    """Local fake engine: typed embedded nodes + link_nodes (uppercases like real)."""

    def __init__(self, nodes):
        self.graph = _Graph(nodes)
        self.edges: list[tuple[str, str, str, dict]] = []

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        self.edges.append((src, dst, str(rel_type).upper(), properties or {}))


def _feat(emb, importance=0.5, ntype="capability"):
    return {"type": ntype, "embedding": emb, "importance_score": importance}


def test_links_similar_and_supersedes_duplicates():
    # f1 & f2 near-identical (duplicates); f3 orthogonal (unrelated).
    engine = _Engine(
        {
            "f1": _feat([1.0, 0.0, 0.0], importance=0.9),
            "f2": _feat([0.99, 0.01, 0.0], importance=0.4),
            "f3": _feat([0.0, 0.0, 1.0], importance=0.7),
        }
    )
    report = dedup_features(engine, similar_threshold=0.8, dup_threshold=0.95)
    assert report.candidates == 3
    assert report.similar_pairs == 1  # only f1~f2
    assert report.clusters == 1
    assert report.duplicates_superseded == 1
    assert report.survivors == ["f1"]  # higher importance survives

    sim = [e for e in engine.edges if e[2] == "SIMILAR_TO"]
    sup = [e for e in engine.edges if e[2] == "SUPERSEDES"]
    assert len(sim) == 1 and sim[0][3]["score"] >= 0.95
    assert sup == [("f1", "f2", "SUPERSEDES", sup[0][3])]  # survivor → duplicate


def test_dry_run_writes_nothing():
    engine = _Engine({"a": _feat([1.0, 0.0]), "b": _feat([1.0, 0.0])})
    report = dedup_features(engine, write=False)
    assert report.similar_pairs == 1 and report.duplicates_superseded == 1
    assert engine.edges == []  # analysis-only


def test_restrict_to_incremental():
    engine = _Engine(
        {
            "old1": _feat([1.0, 0.0, 0.0]),
            "old2": _feat([1.0, 0.0, 0.0]),  # old1~old2 but neither is "new"
            "new1": _feat([0.0, 1.0, 0.0]),
            "new2": _feat([0.0, 1.0, 0.0]),  # new1~new2
        }
    )
    report = dedup_features(engine, restrict_to={"new1", "new2"})
    # only pairs touching the new ids are considered
    assert report.similar_pairs == 1
    assert {"new1", "new2"} & set(report.survivors)


def test_ignores_untyped_and_unembedded_nodes():
    engine = _Engine(
        {
            "f1": _feat([1.0, 0.0]),
            "doc": {"type": "document", "embedding": [1.0, 0.0]},  # wrong type
            "f2": {"type": "feature"},  # no embedding
        }
    )
    report = dedup_features(engine)
    assert report.candidates == 1  # only f1 qualifies
    assert report.similar_pairs == 0


def test_name_resolution_merges_despite_disagreeing_embeddings():
    # AHE-3.69: f1 & f2 are the SAME entity by name but their embeddings are
    # orthogonal (cosine 0 < dup_threshold), so the embedding path alone would
    # NOT merge them. The entropy-gated name fast-path must.
    engine = _Engine(
        {
            "f1": {**_feat([1.0, 0.0, 0.0], importance=0.9), "name": "Kubernetes"},
            "f2": {**_feat([0.0, 1.0, 0.0], importance=0.3), "name": "kubernetes"},
            "f3": {**_feat([0.0, 0.0, 1.0]), "name": "PostgreSQL"},
        }
    )
    report = dedup_features(engine, similar_threshold=0.8, dup_threshold=0.95)
    assert report.name_resolved_pairs == 1  # f1~f2 by name
    assert report.duplicates_superseded == 1
    assert report.survivors == ["f1"]  # higher importance survives
    sup = [e for e in engine.edges if e[2] == "SUPERSEDES"]
    assert sup == [("f1", "f2", "SUPERSEDES", sup[0][3])]


def test_name_resolution_skips_generic_low_entropy_names():
    # Two nodes both named "System" with orthogonal embeddings must NOT merge.
    engine = _Engine(
        {
            "a": {**_feat([1.0, 0.0, 0.0]), "name": "System"},
            "b": {**_feat([0.0, 1.0, 0.0]), "name": "System"},
        }
    )
    report = dedup_features(engine, similar_threshold=0.8, dup_threshold=0.95)
    assert report.name_resolved_pairs == 0
    assert report.low_entropy_skipped == 2
    assert report.duplicates_superseded == 0


def test_uses_engine_batch_op_when_present():
    nodes = {"f1": _feat([1.0, 0.0]), "f2": _feat([0.2, 0.9])}

    class _BatchEngine(_Engine):
        def compute_similarity_edges(self, threshold):
            # engine asserts f1~f2 are similar regardless of local cosine
            return [
                ("f1", "f2", 0.97),
                ("f1", "zzz", 0.99),
            ]  # zzz filtered (not a candidate)

    engine = _BatchEngine(nodes)
    report = dedup_features(engine, similar_threshold=0.8, dup_threshold=0.95)
    assert report.similar_pairs == 1  # zzz pair filtered to candidate set
    assert report.duplicates_superseded == 1


def test_version_variants_linked_not_superseded():
    # AHE-3.70: "Llama 2" and "Llama 3" are siblings (version variants) — they must
    # be LINKED as VARIANT_OF, NOT merged/superseded as duplicates.
    engine = _Engine(
        {
            "v2": {**_feat([1.0, 0.0, 0.0], importance=0.9), "name": "Llama 2"},
            "v3": {**_feat([0.0, 1.0, 0.0], importance=0.3), "name": "Llama 3"},
        }
    )
    report = dedup_features(engine, similar_threshold=0.8, dup_threshold=0.95)
    assert report.variants_linked == 1
    assert report.duplicates_superseded == 0
    var = [e for e in engine.edges if e[2] == "VARIANT_OF"]
    assert len(var) == 1
    assert {var[0][0], var[0][1]} == {"v2", "v3"}


def test_engine_resolve_candidates_escalation():
    # KG-2.260: when the engine exposes ResolveCandidates, ambiguous residuals
    # escalate to it. f1/f2 are generic ("System") so name-resolution leaves them
    # residual; the engine returns a same_as (merge) + an extends (variant) proposal.
    class _ResolveEngine(_Engine):
        def resolve_candidates(self, sim_threshold, merge_threshold, node_type):
            return [
                {
                    "canonical": "f1",
                    "members": ["f1", "f2"],
                    "score": 0.99,
                    "kind": "same_as",
                },
                {
                    "canonical": "f1",
                    "members": ["f1", "f3"],
                    "score": 0.85,
                    "kind": "extends",
                },
            ]

    engine = _ResolveEngine(
        {
            "f1": {**_feat([1.0, 0.0, 0.0], importance=0.9), "name": "System"},
            "f2": {**_feat([0.0, 1.0, 0.0], importance=0.3), "name": "System"},
            "f3": {**_feat([0.0, 0.0, 1.0], importance=0.5), "name": "System"},
        }
    )
    report = dedup_features(engine, similar_threshold=0.8, dup_threshold=0.95)
    assert report.engine_proposals >= 2
    # the same_as proposal drove a merge (f1 supersedes f2)
    assert report.duplicates_superseded >= 1
    # the extends proposal linked a VARIANT_OF edge
    assert any(e[2] == "VARIANT_OF" for e in engine.edges)
