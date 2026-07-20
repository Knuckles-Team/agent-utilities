#!/usr/bin/python
"""Comparative feature / innovation matrix (CONCEPT:AU-KG.research.default-so-every-cycle).

Materializes the post-assimilation graph (coverage edges + leverage + synergy)
into one queryable + rendered deliverable.
"""

from typing import Any

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    build_feature_matrix,
    materialize_feature_matrix,
    render_markdown,
)

pytestmark = pytest.mark.concept("AU-KG.research.default-so-every-cycle")


class _Graph:
    def __init__(self, nodes):
        self._n = nodes
        self._out: dict = {}
        self._in: dict = {}

    def nodes(self, data=False):
        # data=True → NX-style (id, data) items; data=False → the dict itself, which
        # supports BOTH iteration (keys) AND [id] lookup (the per-id scoped fetch path,
        # CONCEPT:AU-KG.ingest.fetch-only-requested-ids) so build_feature_matrix(restrict_to=...) stays O(cohort).
        return list(self._n.items()) if data else self._n

    def add_edge(self, src, dst, props):
        self._out.setdefault(src, []).append((src, dst, props))
        self._in.setdefault(dst, []).append((src, dst, props))

    def out_edges(self, nid, data=False):
        e = self._out.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]

    def in_edges(self, nid, data=False):
        e = self._in.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]


class _Engine:
    def __init__(self, nodes):
        self.graph = _Graph(nodes)
        self.added: dict = {}

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        self.graph.add_edge(src, dst, properties or {})

    def add_node(self, node_id, node_type=None, properties=None, **_kw):
        self.added[node_id] = {"node_type": node_type, "properties": properties or {}}


def _f(concept, sources=(), name=""):
    return {
        "type": "capability",
        "name": name or concept,
        "concept_ids": [concept],
        "research_sources": list(sources),
        "status": "open",
    }


def _engine():
    eng = _Engine(
        {
            "f_kg": _f(
                "AU-KG.memory.tiered-memory-caching",
                sources=["paperA"],
                name="Latent rollout memory",
            ),
            "f_orch": _f(
                "AU-ORCH.adapter.hot-cache-invalidation",
                sources=["paperA", "paperB"],
                name="Diverse query",
            ),
            "f_ahe": _f(
                "AU-AHE.harness.harness-evolution",
                sources=["paperB"],
                name="Cache-tier reward",
            ),
        }
    )
    # f_kg → related (novel-but-relevant) with a recorded novelty
    eng.link_nodes(
        "f_kg",
        "AU-KG.ontology.kg-3",
        "RELATES_TO",
        properties={"_rel": "RELATES_TO", "novelty": 0.7},
    )
    # f_ahe → covered (we already built this)
    eng.link_nodes(
        "f_ahe",
        "AU-AHE.harness.harness-evolution",
        "SATISFIED_BY",
        properties={"_rel": "SATISFIED_BY"},
    )
    # cross-pillar synergy: KG ~ ORCH
    eng.link_nodes("f_kg", "f_orch", "SIMILAR_TO", properties={"_rel": "SIMILAR_TO"})
    return eng


def test_matrix_coverage_novelty_and_synergy():
    matrix = build_feature_matrix(_engine())
    rows = {r.feature_id: r for r in matrix.rows}

    assert rows["f_ahe"].coverage == "covered"
    assert rows["f_ahe"].concept_id == "AU-AHE.harness.harness-evolution"
    assert rows["f_kg"].coverage == "related"
    assert rows["f_kg"].concept_id == "AU-KG.ontology.kg-3"
    assert rows["f_kg"].novelty_score == 0.7
    assert rows["f_orch"].coverage == "novel"
    assert rows["f_orch"].novelty_score == 1.0

    # one cross-pillar synergy bundle; both members list each other as partners
    assert matrix.counts["bundles"] == 1
    assert rows["f_kg"].synergy_partners == ["f_orch"]
    assert sorted(rows["f_kg"].synergy_pillars) == ["KG", "ORCH"]

    # leverage: f_orch (2 sources) outranks f_kg (1 source); covered f_ahe is 0
    assert rows["f_orch"].leverage_score > rows["f_kg"].leverage_score
    assert rows["f_ahe"].leverage_score == 0.0

    # novel gaps = the two OPEN features, leverage-ranked
    assert [r.feature_id for r in matrix.novel_gaps()] == ["f_orch", "f_kg"]
    assert matrix.counts == {
        "total": 3,
        "covered": 1,
        "related": 1,
        "novel": 1,
        "bundles": 1,
        "sources": 2,
    }


def test_render_markdown_has_all_sections():
    md = render_markdown(build_feature_matrix(_engine()))
    assert "# Comparative Feature / Innovation Matrix" in md
    assert "Novel gaps to implement" in md
    assert "Cross-source synergies" in md
    assert "Per-source contribution" in md


def test_materialize_writes_feature_matrix_node_idempotently():
    eng = _engine()
    matrix = build_feature_matrix(eng)
    s1 = materialize_feature_matrix(eng, matrix)
    assert s1["persisted"] is True
    assert eng.added["feature_matrix:latest"]["node_type"] == "feature_matrix"
    # second materialize reuses the same node id (upsert, not accumulate)
    s2 = materialize_feature_matrix(eng, matrix)
    assert s2["node_id"] == s1["node_id"] == "feature_matrix:latest"
    assert s2["counts"] == s1["counts"]


# ---------------------------------------------------------------------------
# X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): the matrix node is
# a materialized JOIN over the summarized features/concepts — it registers as
# a live TruthMaintenance materialization and ACTUALLY goes Stale when one of
# those base facts changes.
# ---------------------------------------------------------------------------


class _TmsAwareEngine(_Engine):
    """``_Engine`` + ``add_edge`` + a minimal, faithful in-process
    TruthMaintenance index (same contract as ``test_insight_validation.py``'s
    ``_TmsAwareInsightStubEngine``): every ``add_node`` bumps that id's
    version; ``register_materialization`` snapshots the CURRENT version of
    every ``:DerivedFrom`` target; ``materialization_status`` reports "Stale"
    the instant a snapshotted dependency's version has since moved."""

    def __init__(self, nodes):
        super().__init__(nodes)
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self._versions: dict[str, int] = {nid: 1 for nid in nodes}
        self._materializations: dict[str, dict[str, int]] = {}

    def add_node(self, node_id, node_type=None, properties=None, **_kw):
        super().add_node(node_id, node_type, properties, **_kw)
        self._versions[node_id] = self._versions.get(node_id, 0) + 1

    def add_edge(self, source, target, rel_type="", **properties):
        self.edges.append((source, target, {"rel_type": rel_type, **properties}))

    def register_materialization(self, derived_id):
        deps = {
            target
            for source, target, props in self.edges
            if source == derived_id and props.get("relationship_type") == "DERIVED_FROM"
        }
        self._materializations[derived_id] = {d: self._versions.get(d, 0) for d in deps}
        return {
            "id": derived_id,
            "depends_on": sorted(deps),
            "generating_activity": None,
        }

    def materialization_status(self, derived_id):
        snapshot = self._materializations.get(derived_id)
        if snapshot is None:
            return None
        for dep, ver in snapshot.items():
            if self._versions.get(dep, 0) != ver:
                return "Stale"
        return "Fresh"


def _tms_engine():
    eng = _TmsAwareEngine(
        {
            "f_kg": _f(
                "AU-KG.memory.tiered-memory-caching",
                sources=["paperA"],
                name="Latent rollout memory",
            ),
            "f_orch": _f(
                "AU-ORCH.adapter.hot-cache-invalidation",
                sources=["paperA", "paperB"],
                name="Diverse query",
            ),
            "f_ahe": _f(
                "AU-AHE.harness.harness-evolution",
                sources=["paperB"],
                name="Cache-tier reward",
            ),
        }
    )
    eng.link_nodes(
        "f_kg",
        "AU-KG.ontology.kg-3",
        "RELATES_TO",
        properties={"_rel": "RELATES_TO", "novelty": 0.7},
    )
    eng.link_nodes(
        "f_ahe",
        "AU-AHE.harness.harness-evolution",
        "SATISFIED_BY",
        properties={"_rel": "SATISFIED_BY"},
    )
    eng.link_nodes("f_kg", "f_orch", "SIMILAR_TO", properties={"_rel": "SIMILAR_TO"})
    return eng


def test_materialize_registers_and_goes_stale_when_a_summarized_feature_changes():
    eng = _tms_engine()
    matrix = build_feature_matrix(eng)
    materialize_feature_matrix(eng, matrix)

    assert eng.materialization_status("feature_matrix:latest") == "Fresh"

    # A summarized feature is later revised through the normal write path.
    eng.add_node("f_kg", "capability", properties={"revised": True})

    assert eng.materialization_status("feature_matrix:latest") == "Stale"


def test_materialize_registration_failure_does_not_break_the_persist():
    class _BoomEngine(_TmsAwareEngine):
        def register_materialization(self, derived_id):
            raise RuntimeError("kg unreachable")

    eng = _BoomEngine(
        {"f_kg": _f("AU-KG.memory.tiered-memory-caching", sources=["paperA"])}
    )
    matrix = build_feature_matrix(eng)
    summary = materialize_feature_matrix(eng, matrix)
    assert summary["persisted"] is True  # unaffected by the registration failure


def test_scoped_matrix_restricts_to_cohort_sources():
    """restrict_to scopes the WHOLE build to a feature subset (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) —
    per-id collection + per-id coverage + scoped leverage/synergy."""
    eng = _engine()
    m = build_feature_matrix(eng, restrict_to={"f_kg"})
    ids = {r.feature_id for r in m.rows}
    assert ids == {"f_kg"}  # only the cohort source, not the whole graph
    row = m.rows[0]
    assert row.coverage == "related" and row.concept_id == "AU-KG.ontology.kg-3"
    assert row.novelty_score == 0.7
    assert m.counts["total"] == 1
    # a covered feature outside the scope is excluded entirely
    m2 = build_feature_matrix(eng, restrict_to={"f_orch", "f_ahe"})
    assert {r.feature_id for r in m2.rows} == {"f_orch", "f_ahe"}


def test_scoped_matrix_uses_per_id_fetch_not_full_scan(monkeypatch):
    """Scoped build must NOT iterate the whole graph — it fetches only restrict_to ids."""
    eng = _engine()

    def _boom(*a, **k):  # nodes(data=True) = the whole-graph pull; must not be called
        raise AssertionError("whole-graph nodes(data=True) called in scoped mode")

    # allow no-arg nodes() (per-id view) but fail the data=True full scan
    real = eng.graph.nodes

    def _guard(data=False):
        if data:
            _boom()
        return real(data=False)

    monkeypatch.setattr(eng.graph, "nodes", _guard)
    m = build_feature_matrix(eng, restrict_to={"f_kg"})
    assert {r.feature_id for r in m.rows} == {"f_kg"}
