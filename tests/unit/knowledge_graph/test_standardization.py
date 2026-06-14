#!/usr/bin/python
"""Enterprise standardization & consolidation engine (CONCEPT:KG-2.49).

Covers: drift scoring from interface gaps, asset routing, per-org/domain
aggregation, candidate cohorts (capability + dedup clusters), the consolidation
decision engine (north-star selection, scoring monotonicity, idempotency), and
the end-to-end propose-only pass.
"""

import pytest

from agent_utilities.knowledge_graph.standardization import (
    applicable_standard,
    drift_score,
    run_standardization_pass,
)
from agent_utilities.knowledge_graph.standardization.cohorts import (
    read_assets,
    read_cohorts,
)
from agent_utilities.knowledge_graph.standardization.consolidation import (
    recommend_consolidations,
)
from agent_utilities.knowledge_graph.standardization.drift import score_conformance

pytestmark = pytest.mark.concept("KG-2.49")


# ── Fake engine double (mirrors the assimilation test doubles) ───────────────
class _Graph:
    def __init__(self, nodes, edges):
        self._n = nodes  # id -> attrs dict
        self._e = edges  # list[(src, dst, props)]

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)

    def edges(self, data=False):
        return list(self._e) if data else [(s, d) for s, d, _ in self._e]


class _Engine:
    """Fake engine: nodes + bulk edges + link_nodes/add_node/delete_edge."""

    def __init__(self, nodes=None):
        self._nodes = dict(nodes or {})
        self._edges: list[tuple[str, str, dict]] = []
        self.graph = _Graph(self._nodes, self._edges)

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        props = dict(properties or {})
        props.setdefault("_rel", str(rel_type).split(".")[-1].upper())
        self._edges.append((src, dst, props))

    def add_node(self, node_id, node_type, properties=None):
        data = dict(properties or {})
        data["type"] = node_type
        self._nodes[node_id] = data

    def delete_edge(self, src, dst, rel):
        rel_u = str(rel).split(".")[-1].upper()
        self._edges[:] = [
            (s, d, p)
            for (s, d, p) in self._edges
            if not (s == src and d == dst and str(p.get("_rel", "")).upper() == rel_u)
        ]

    def get_blast_radius(self, node_id, depth):
        return []


def _app(**over):
    base = {
        "type": "deployed_software_component",
        "capability": "itsm",
        "owner": "team-x",
        "lifecycle_state": "active",
        "data_classification": "internal",
        "cost_center": "cc-100",
        "link_types": {"belongs_to_organization"},
    }
    base.update(over)
    return base


# ── drift_score (interface gaps → 0..1) ──────────────────────────────────────
def test_conformant_application_has_zero_drift():
    drift, gaps = drift_score(_app(), "ManagedApplication")
    assert drift == 0.0
    assert gaps == []


def test_missing_two_of_six_slots_is_one_third_drift():
    # ManagedApplication = 5 required props + 1 link = 6 slots. Drop cost_center
    # and the org link → 2 gaps.
    asset = _app()
    asset.pop("cost_center")
    asset["link_types"] = set()
    drift, gaps = drift_score(asset, "ManagedApplication")
    assert len(gaps) == 2
    assert round(drift, 3) == round(2 / 6, 3)


def test_org_specific_extensions_do_not_count_against_conformance():
    # Extra org property (pci_scope) is ignored by the interface shape.
    drift, _ = drift_score(
        _app(pci_scope="level-1", pii_region="eu"), "ManagedApplication"
    )
    assert drift == 0.0


# ── routing ──────────────────────────────────────────────────────────────────
def test_routes_by_capability_then_type():
    assert applicable_standard({"capability": "vcs"}) == "ManagedApplication"
    assert applicable_standard({"capability": "process"}) == "BusinessProcess"
    assert applicable_standard({"type": "dataset"}) == "DataAsset"
    assert applicable_standard({"capability": "weather", "type": "unknown"}) is None


# ── conformance aggregation + CONFORMS_TO edges ──────────────────────────────
def test_score_conformance_aggregates_and_writes_edges():
    engine = _Engine(
        {
            "app1": _app(organization="org-a"),  # conformant
            "app2": _app(organization="org-a", cost_center=None),  # 1 gap
            "proc1": {
                "type": "business_process",
                "capability": "process",
                "owner": "p",
                "process_tier": "t1",
                "organization": "org-b",
            },  # missing capability? no — has capability=process; all 3 present? owner,capability,process_tier yes → conformant
        }
    )
    # app2 cost_center=None means key present but value None → invalid → still a gap.
    report = score_conformance(engine)
    assert report.assets_scored == 3
    assert report.edges_written == 3
    assert "ManagedApplication" in report.per_domain
    assert "BusinessProcess" in report.per_domain
    # org-a has one conformant + one drifting app → mean drift > 0.
    assert report.per_org["org-a"].mean_drift > 0.0
    # CONFORMS_TO edges materialized.
    conforms = [e for e in engine._edges if e[2].get("_rel") == "CONFORMS_TO"]
    assert len(conforms) == 3


# ── cohorts ──────────────────────────────────────────────────────────────────
def test_capability_cohort_groups_cross_vendor_redundancy():
    engine = _Engine(
        {
            "snow": _app(capability="itsm", vendor="servicenow"),
            "erp": _app(capability="itsm", vendor="erpnext"),
            "solo": _app(capability="crm", vendor="twenty"),
        }
    )
    groups = read_cohorts(engine)
    cohorts = [g for g in groups if g.origin == "cohort"]
    assert len(cohorts) == 1
    g = cohorts[0]
    assert g.capability == "itsm"
    assert set(g.members) == {"snow", "erp"}
    assert g.sources == {"servicenow", "erpnext"}
    assert g.kind == "retire_tool"


def test_supersedes_cluster_becomes_merge_candidate():
    engine = _Engine({"r1": _app(vendor="gitlab"), "r2": _app(vendor="gitlab")})
    # dedup wrote a SUPERSEDES edge survivor->dup.
    engine.link_nodes("r1", "r2", "SUPERSEDES", properties={"_rel": "SUPERSEDES"})
    groups = read_cohorts(engine)
    dedup = [g for g in groups if g.origin == "dedup"]
    assert len(dedup) == 1
    assert set(dedup[0].members) == {"r1", "r2"}
    assert dedup[0].kind == "merge_codebases"


# ── consolidation decision engine ────────────────────────────────────────────
def test_north_star_is_lowest_drift_member():
    engine = _Engine(
        {
            # both itsm from different vendors → a cohort; snow conformant, erp drifts.
            "snow": _app(capability="itsm", vendor="servicenow"),
            "erp": _app(
                capability="itsm", vendor="erpnext", owner=None, cost_center=None
            ),
        }
    )
    report = recommend_consolidations(engine, top_n=10)
    assert report.recommendations
    rec = report.recommendations[0]
    assert rec.north_star == "snow"  # lowest drift survives
    # proposed ABSORBED_INTO edge erp -> snow written.
    absorbed = [e for e in engine._edges if e[2].get("_rel") == "ABSORBED_INTO"]
    assert ("erp", "snow") == (absorbed[0][0], absorbed[0][1])
    assert absorbed[0][2]["status"] == "proposed"


def test_cost_raises_consolidation_priority():
    def _two_vendor(cap, cost):
        return _Engine(
            {
                f"{cap}-a": _app(capability=cap, vendor="va", annual_cost=cost),
                f"{cap}-b": _app(capability=cap, vendor="vb", annual_cost=cost),
            }
        )

    cheap = recommend_consolidations(_two_vendor("itsm", 0), write=False)
    pricey = recommend_consolidations(_two_vendor("itsm", 1_000_000), write=False)
    assert pricey.recommendations[0].priority >= cheap.recommendations[0].priority
    assert pricey.recommendations[0].value_score > cheap.recommendations[0].value_score


def test_consolidation_is_idempotent():
    engine = _Engine(
        {
            "snow": _app(capability="itsm", vendor="servicenow"),
            "erp": _app(capability="itsm", vendor="erpnext"),
        }
    )
    recommend_consolidations(engine, top_n=10)
    n_after_first = len(
        [e for e in engine._edges if e[2].get("_rel") == "ABSORBED_INTO"]
    )
    recommend_consolidations(engine, top_n=10)
    n_after_second = len(
        [e for e in engine._edges if e[2].get("_rel") == "ABSORBED_INTO"]
    )
    assert n_after_first == n_after_second == 1


# ── end-to-end pass ──────────────────────────────────────────────────────────
def test_run_standardization_pass_end_to_end():
    engine = _Engine(
        {
            "snow": _app(capability="itsm", vendor="servicenow", organization="org-a"),
            "erp": _app(
                capability="itsm",
                vendor="erpnext",
                organization="org-b",
                cost_center=None,
            ),
            "data1": {
                "type": "dataset",
                "capability": "data",
                "owner": "d",
                "data_classification": "internal",
                "retention_policy": "30d",
                "link_types": {"was_derived_from"},
            },
        }
    )
    rep = run_standardization_pass(engine, top_n=5)
    assert rep["standards_materialized"] == 3
    assert rep["assets_scored"] == 3
    assert rep["conformance_edges"] == 3
    assert "ManagedApplication" in rep["drift_by_domain"]
    assert rep["groups_considered"] >= 1
    assert rep["recommendations"]
    # EnterpriseStandard nodes materialized + a recommendation node persisted.
    std_nodes = [
        n for n, d in engine._nodes.items() if d.get("type") == "enterprise_standard"
    ]
    rec_nodes = [
        n
        for n, d in engine._nodes.items()
        if d.get("type") == "consolidation_recommendation"
    ]
    assert len(std_nodes) == 3
    assert rec_nodes


def test_read_assets_excludes_ungoverned():
    engine = _Engine(
        {
            "app": _app(),
            "weather": {"type": "sensor", "capability": "weather"},
        }
    )
    assets = read_assets(engine)
    assert "app" in assets
    assert "weather" not in assets


# ── Wire-First live-path: reachable through kg.ontology + the MCP action ──────
def test_ontology_system_binds_enterprise_standards_live_path():
    """The execution plane reaches standards only through kg.ontology (cardinal rule)."""
    from agent_utilities.knowledge_graph.ontology import OntologySystem

    onto = OntologySystem()  # no graph needed for the registry/scoring surface
    assert {i.name for i in onto.standards.list_interfaces()} == {
        "ManagedApplication",
        "BusinessProcess",
        "DataAsset",
    }
    # standard_for / drift_for resolve through the bound system.
    assert onto.standard_for(_app()) == "ManagedApplication"
    drift, gaps = onto.drift_for(_app())
    assert drift == 0.0 and gaps == []


def test_ontology_standardize_runs_pass_against_engine():
    from agent_utilities.knowledge_graph.ontology import OntologySystem

    engine = _Engine(
        {
            "snow": _app(capability="itsm", vendor="servicenow"),
            "erp": _app(capability="itsm", vendor="erpnext"),
        }
    )
    rep = OntologySystem().standardize(engine=engine, top_n=5)
    assert rep["assets_scored"] == 2
    assert rep["recommendations"]


def test_loop_standardize_stage_live_path():
    """run_one_cycle(standardize=True) writes consolidation recommendations."""
    from agent_utilities.knowledge_graph.research.loop_controller import (
        LoopController,
    )

    engine = _Engine(
        {
            "snow": _app(capability="itsm", vendor="servicenow"),
            "erp": _app(capability="itsm", vendor="erpnext"),
        }
    )
    # Only the standardize stage; skip assimilate/intake/synthesize (need richer graph).
    report = LoopController(engine).run_one_cycle(
        standardize=True, assimilate=False, synthesize=False, max_topics=0
    )
    assert report["standardize"] is not None
    assert report["standardize"]["assets_scored"] == 2
    rec_nodes = [
        n
        for n, d in engine._nodes.items()
        if d.get("type") == "consolidation_recommendation"
    ]
    assert rec_nodes


def test_kg_server_exposes_standardize_action():
    """The graph_orchestrate 'standardize' action + enterprise registry are wired."""
    import inspect

    from agent_utilities.mcp import kg_server

    src = inspect.getsource(kg_server)
    assert 'action == "standardize"' in src
    assert "run_standardization_pass" in src
    assert "ENTERPRISE_STANDARD_REGISTRY" in src
