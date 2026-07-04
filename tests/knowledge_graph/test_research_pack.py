from __future__ import annotations

"""Tests for the flagship research-state Schema-Pack 2.0 profile.

CONCEPT:AU-KG.research.research-state-domain-pack — Research-State Domain Pack
"""


from agent_utilities.models.knowledge_graph import RegistryEdgeType
from agent_utilities.models.schema_packs import get_schema_pack, list_schema_packs


def test_dedicated_edge_types_exist():
    assert RegistryEdgeType.WEAKENS.value == "weakens"
    assert RegistryEdgeType.USES_DATASET.value == "uses_dataset"


def test_pack_registered_and_round_trips():
    names = {p["name"] for p in list_schema_packs()}
    assert "research-state" in names
    pack = get_schema_pack("research-state")
    assert pack.name == "research-state"


def test_pack_activates_research_edges():
    pack = get_schema_pack("research-state")
    assert RegistryEdgeType.WEAKENS in pack.edge_types
    assert RegistryEdgeType.USES_DATASET in pack.edge_types
    assert RegistryEdgeType.CITED_BY_PAPER in pack.edge_types


def test_link_inference_covers_supports_and_weakens():
    pack = get_schema_pack("research-state")
    edges = {r.edge_type for r in pack.link_inference}
    assert {"supports_belief", "weakens", "uses_dataset", "cites_source"} <= edges


def test_relational_verbs_present():
    pack = get_schema_pack("research-state")
    assert pack.relational_verbs["support"] == "supports_belief"
    assert pack.relational_verbs["weakens"] == "weakens"


def test_owl_closure_declarations():
    pack = get_schema_pack("research-state")
    transitive, _symmetric, inverse = pack.get_owl_closure_sets()
    assert "supports_belief" in transitive
    assert inverse.get("cites_source") == "cited_by_paper"


def test_retrieval_signals_configured():
    pack = get_schema_pack("research-state")
    assert pack.recency_spec_for("document") is not None
    assert pack.trust_for("arxiv") > 1.0
    assert pack.autocut_enabled is True
