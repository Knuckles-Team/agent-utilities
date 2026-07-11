"""Tests for the Enterprise Operations Causal Graph join + analysis layer (Codex X-2).

CONCEPT:AU-KG.enrichment.ops-causal-graph

Builds a small SYNTHETIC ops causal graph — failure trace -> agent -> service
-> deployment -> commit -> incident -> capability -> policy -> evidence — and
asserts:

* root-cause ranking surfaces the actual root-cause CHANGE (the commit), not
  just the closest intermediate node, for a given failure trace.
* blast-radius returns the correct downstream set for a change.
* change-risk composes blast-radius + historical incident severity.
* control-evidence gathers + verifies the chain for a governance control.

All of this runs OFFLINE (no live engine/backend), reusing
:class:`~agent_utilities.knowledge_graph.core.formal_reasoning_core.StructuralCausalModel`
— the same convention ``tests/test_causal_reasoning.py`` uses for the base SCM.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.models import ExtractionBatch
from agent_utilities.knowledge_graph.enrichment.ops_causal_graph import (
    CATEGORY,
    OpsCausalLink,
    OpsCausalModel,
    blast_radius_analysis,
    build_causal_model,
    change_risk_score,
    control_evidence_chain,
    load_ops_causal_neighborhood,
    materialize_ops_causal_links,
    root_cause_rank,
)
from tests.kg_recording_backend import RecordingGraphBackend


# ── synthetic ops causal graph fixture ───────────────────────────────────────
#
#   commit:bad123 --affects--> stack:checkout-v3 --deploys_software--> svc:checkout
#   commit:bad123 --affects--> svc:checkout
#   svc:checkout  --part_of(reversed)--> agent:checkout --executed_by(reversed)--> trace:1
#   commit:bad123 --caused_incident--> incident:INC001
#   svc:checkout  --supports--> cap:payments
#   incident:INC001 --applies_to--> cap:payments
#   policy:pci    --governs--> cap:payments
#   policy:pci    --has_evidence--> evidence:ev1
#
#   commit:good999 --affects--> svc:other   (an unrelated, innocent change)
@pytest.fixture
def ops_links() -> list[OpsCausalLink]:
    return [
        OpsCausalLink("commit:bad123", "stack:checkout-v3", "affects", strength=0.9),
        OpsCausalLink("commit:bad123", "svc:checkout", "affects", strength=0.9),
        OpsCausalLink("stack:checkout-v3", "svc:checkout", "deploys_software"),
        OpsCausalLink("svc:checkout", "agent:checkout", "part_of"),
        OpsCausalLink("agent:checkout", "trace:1", "executed_by"),
        OpsCausalLink(
            "commit:bad123", "incident:INC001", "caused_incident", strength=1.0
        ),
        OpsCausalLink("commit:good999", "svc:other", "affects", strength=0.9),
        OpsCausalLink("svc:checkout", "cap:payments", "supports"),
        OpsCausalLink("incident:INC001", "cap:payments", "applies_to"),
        OpsCausalLink("policy:pci", "cap:payments", "governs"),
        OpsCausalLink("policy:pci", "evidence:ev1", "has_evidence"),
    ]


@pytest.fixture
def model(ops_links: list[OpsCausalLink]) -> OpsCausalModel:
    return build_causal_model(ops_links)


# ── build_causal_model ───────────────────────────────────────────────────────
def test_build_causal_model_creates_factors_and_edges(model: OpsCausalModel):
    scm = model.scm
    assert scm.has_node("commit:bad123")
    assert scm.has_node("trace:1")
    assert scm.has_edge("commit:bad123", "svc:checkout")
    assert scm.edge_count == 11
    # 11 distinct nodes across the fixture.
    assert scm.factor_count == 11


def test_build_causal_model_dedupes_repeated_edges():
    links = [
        OpsCausalLink("a", "b", "affects"),
        OpsCausalLink("a", "b", "affects"),  # duplicate — must not double-add
    ]
    model = build_causal_model(links)
    assert model.scm.edge_count == 1


# ── root_cause_rank ───────────────────────────────────────────────────────────
def test_root_cause_rank_surfaces_the_actual_change_first(model: OpsCausalModel):
    """The failure trace's TRUE root cause (the commit — a topological source,
    nothing further upstream caused it) must outrank the merely-closer
    intermediate nodes (agent, service) that themselves have a recorded cause.
    """
    ranked = root_cause_rank(model, "trace:1")
    assert ranked, "expected at least one ranked root-cause candidate"
    assert ranked[0]["node_id"] == "commit:bad123"
    assert ranked[0]["is_root"] is True

    ranked_ids = [r["node_id"] for r in ranked]
    assert "commit:good999" not in ranked_ids  # unrelated change never appears
    assert "agent:checkout" in ranked_ids
    assert "svc:checkout" in ranked_ids


def test_root_cause_rank_unknown_node_returns_empty(model: OpsCausalModel):
    assert root_cause_rank(model, "no-such-node") == []


def test_root_cause_rank_respects_max_results(model: OpsCausalModel):
    ranked = root_cause_rank(model, "trace:1", max_results=1)
    assert len(ranked) == 1
    assert ranked[0]["node_id"] == "commit:bad123"


def test_root_cause_rank_recency_weighting_prefers_recent_cause():
    now = 10_000.0
    links = [
        OpsCausalLink("old_cause", "failure", "affects", observed_at=now - 100_000),
        OpsCausalLink("recent_cause", "failure", "affects", observed_at=now - 60),
    ]
    model = build_causal_model(links)
    ranked = root_cause_rank(model, "failure", now=now)
    # Both are 1-hop, equally-strong, equally "root" (both topological sources)
    # — recency must be the deciding tie-breaker.
    assert ranked[0]["node_id"] == "recent_cause"


# ── blast_radius_analysis ────────────────────────────────────────────────────
def test_blast_radius_returns_downstream_set(model: OpsCausalModel):
    radius = blast_radius_analysis(model, "commit:bad123")
    ids = {r["node_id"] for r in radius}
    assert ids == {
        "stack:checkout-v3",
        "svc:checkout",
        "agent:checkout",
        "trace:1",
        "incident:INC001",
        "cap:payments",
    }
    # unrelated branch never appears
    assert "svc:other" not in ids
    assert "commit:good999" not in ids

    by_id = {r["node_id"]: r for r in radius}
    assert by_id["stack:checkout-v3"]["depth"] == 1
    assert by_id["trace:1"]["depth"] == 3


def test_blast_radius_unknown_node_returns_empty(model: OpsCausalModel):
    assert blast_radius_analysis(model, "no-such-node") == []


def test_blast_radius_via_live_engine_delegates_to_get_blast_radius():
    """Given a live engine (not an OpsCausalModel), delegate straight to its
    already-shipped ``get_blast_radius`` — no second traversal implementation."""

    class _FakeEngine:
        def get_blast_radius(self, node_id, depth):
            assert node_id == "commit:bad123"
            assert depth == 6
            return [
                {"id": "svc:checkout", "type": "System", "depth": 1},
                {"id": "trace:1", "type": "Trace", "depth": 3},
            ]

    radius = blast_radius_analysis(_FakeEngine(), "commit:bad123")
    assert radius == [
        {"node_id": "svc:checkout", "depth": 1, "type": "System"},
        {"node_id": "trace:1", "depth": 3, "type": "Trace"},
    ]


def test_blast_radius_rejects_object_without_get_blast_radius():
    with pytest.raises(TypeError):
        blast_radius_analysis(object(), "commit:bad123")


# ── change_risk_score ─────────────────────────────────────────────────────────
def test_change_risk_score_combines_exposure_and_history(model: OpsCausalModel):
    risky = change_risk_score(
        model,
        "commit:bad123",
        incident_history=[{"node_id": "incident:INC001", "severity": 0.8}],
    )
    assert risky["blast_radius_size"] == 6
    assert risky["structural_exposure"] == pytest.approx(0.6)
    assert risky["historical_severity"] == pytest.approx(0.8)
    assert risky["risk_score"] == pytest.approx(0.7)
    assert len(risky["contributing_incidents"]) == 1


def test_change_risk_score_zero_history_is_purely_structural(model: OpsCausalModel):
    safe = change_risk_score(model, "commit:good999", incident_history=[])
    assert safe["historical_severity"] == 0.0
    assert safe["risk_score"] == pytest.approx(0.5 * safe["structural_exposure"])


def test_change_risk_ignores_incidents_outside_blast_radius(model: OpsCausalModel):
    # incident:INC001 IS in commit:bad123's blast radius but NOT in
    # commit:good999's — history must only count when it overlaps.
    risk = change_risk_score(
        model,
        "commit:good999",
        incident_history=[{"node_id": "incident:INC001", "severity": 1.0}],
    )
    assert risk["contributing_incidents"] == []
    assert risk["historical_severity"] == 0.0


def test_change_risk_more_exposure_and_history_ranks_higher(model: OpsCausalModel):
    risky = change_risk_score(
        model,
        "commit:bad123",
        incident_history=[{"node_id": "incident:INC001", "severity": 0.9}],
    )
    safe = change_risk_score(model, "commit:good999", incident_history=[])
    assert risky["risk_score"] > safe["risk_score"]


# ── control_evidence_chain ────────────────────────────────────────────────────
def test_control_evidence_chain_gathers_governed_and_upstream(model: OpsCausalModel):
    chain = control_evidence_chain(model, "policy:pci")
    assert set(chain["governs"]) == {"cap:payments", "evidence:ev1"}
    upstream = set(chain["upstream_operational_history"])
    assert {"commit:bad123", "incident:INC001", "svc:checkout", "stack:checkout-v3"} <= upstream
    assert "policy:pci" not in upstream
    assert chain["is_consistent"] is True
    assert chain["consistency_score"] == 1.0
    assert chain["violations"] == []


def test_control_evidence_chain_unknown_control_returns_empty(model: OpsCausalModel):
    chain = control_evidence_chain(model, "no-such-control")
    assert chain["governs"] == []
    assert chain["is_consistent"] is True


def test_control_evidence_chain_control_with_no_downstream_is_empty():
    from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
        CausalFactor,
    )

    links = [OpsCausalLink("a", "b", "affects")]
    isolated_model = build_causal_model(links)
    # A totally isolated control node (no outgoing edges at all) governs nothing.
    isolated_model.scm.add_factor(CausalFactor(id="isolated_control"))
    chain = control_evidence_chain(isolated_model, "isolated_control")
    assert chain["governs"] == []


# ── materialize_ops_causal_links (write path) ────────────────────────────────
def test_materialize_ops_causal_links_persists_edges_only(
    ops_links: list[OpsCausalLink],
):
    backend = RecordingGraphBackend()
    nodes_written, edges_written = materialize_ops_causal_links(backend, ops_links)
    assert nodes_written == 0  # join layer creates zero new entities
    assert edges_written == len(ops_links)
    assert ("commit:bad123", "svc:checkout", "affects") in backend.edges


def test_as_enrichment_edge_carries_strength_and_recency():
    link = OpsCausalLink("a", "b", "affects", strength=0.5, observed_at=123.0)
    edge = link.as_enrichment_edge()
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.rel_type == "affects"
    assert edge.props["strength"] == 0.5
    assert edge.props["observed_at"] == 123.0


def test_materialize_batch_category_is_ops_causal(ops_links: list[OpsCausalLink]):
    batch = ExtractionBatch(category=CATEGORY, edges=[ops_links[0].as_enrichment_edge()])
    assert batch.category == "ops_causal"
    assert batch.nodes == []


# ── load_ops_causal_neighborhood (production loader, degrade path) ──────────
def test_load_ops_causal_neighborhood_no_backend_returns_empty():
    class _NoBackendEngine:
        backend = None

    assert load_ops_causal_neighborhood(_NoBackendEngine(), "trace:1") == []


def test_load_ops_causal_neighborhood_degrades_on_query_failure():
    class _BrokenBackend:
        def execute(self, query, params):
            raise RuntimeError("boom")

    class _Engine:
        backend = _BrokenBackend()

    assert load_ops_causal_neighborhood(_Engine(), "trace:1") == []
