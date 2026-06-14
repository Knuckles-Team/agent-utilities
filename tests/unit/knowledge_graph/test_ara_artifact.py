"""ARA — OWL-native research artifact: projection + ontology registration (KG-2.80)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.owl_bridge import (
    ARA_INVERSE_EDGES,
    ARA_TRANSITIVE_EDGES,
    PROMOTABLE_EDGE_TYPES,
    PROMOTABLE_NODE_TYPES,
    OWLBridge,
)
from agent_utilities.knowledge_graph.ontology.interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
)
from agent_utilities.knowledge_graph.ontology.links import DEFAULT_LINK_REGISTRY
from agent_utilities.knowledge_graph.ontology.value_types import get_value_type
from agent_utilities.knowledge_graph.research.ara import ResearchArtifact


class _Engine:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"type": ntype, **(properties or {})}

    def add_edge(self, src, dst, rel_type="", **props):
        self.edges.append((src, dst, rel_type))


def _artifact() -> ResearchArtifact:
    return ResearchArtifact.from_extracted(
        "2604.24658",
        "Agent-Native Research Artifacts",
        claims=["ARA lifts paper-QA to 93.7%"],
        evidence=["benchmark run #1: 93.7% on paper-QA"],
        code_specs=["compile(paper) -> ARA"],
        summary="recast research as a 4-layer artifact",
    )


# ── projection into nodes/forensic edges ────────────────────────────────────


def test_projection_emits_layered_nodes_and_forensic_edges():
    nodes, edges = _artifact().to_graph_payload()
    by_type: dict[str, int] = {}
    for n in nodes:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
    assert by_type.get("research_artifact") == 1
    assert by_type.get("claim") == 1
    assert by_type.get("evidence") == 1
    assert by_type.get("code_spec") == 1

    rels = {e["type"] for e in edges}
    # envelope→logic, logic→evidence (grounding), logic→src (implementation)
    assert "contains" in rels
    assert "grounded_in" in rels
    assert "implemented_by" in rels
    # provenance edge for the HasProvenance shape
    assert "was_derived_from" in rels


def test_materialize_writes_nodes_and_edges_best_effort():
    eng = _Engine()
    stats = _artifact().materialize(eng)
    assert stats["nodes"] >= 4 and stats["edges"] >= 3
    assert "research_artifact:2604.24658" in eng.nodes
    # the claim is grounded in the evidence (a reasoning-traversable edge)
    assert any(rel == "grounded_in" for _, _, rel in eng.edges)


def test_from_research_artifact_lifts_legacy_extraction():
    class _Legacy:
        article_id = "2606.11198"
        title = "Some Paper"
        summary = "s"
        key_contributions = ["claim A", "claim B"]
        methods = ["method X"]
        suggested_experiments = ["try Y"]
        authors = ["A. Author"]
        source_url = "https://arxiv.org/abs/2606.11198"

    art = ResearchArtifact.from_research_artifact(_Legacy())
    assert len(art.claims) == 2
    assert len(art.code_specs) == 1
    assert len(art.exploration) == 1 and art.exploration[0].kind == "experiment"


# ── ontology registration (always-on, no facade) ────────────────────────────


def test_ara_node_and_edge_types_are_promotable():
    for nt in ("research_artifact", "claim", "code_spec", "exploration_node"):
        assert nt in PROMOTABLE_NODE_TYPES
    for et in ("grounded_in", "implemented_by", "contains", "has_evidence"):
        assert et in PROMOTABLE_EDGE_TYPES


def test_ara_interfaces_registered_with_implementers():
    reg = DEFAULT_INTERFACE_REGISTRY
    assert "VerifiableClaim" in reg._interfaces
    assert "ResearchArtifactShape" in reg._interfaces
    # claim implements VerifiableClaim; research_artifact implements ResearchArtifactShape
    assert "claim" in reg.find_implementers("VerifiableClaim")
    assert "research_artifact" in reg.find_implementers("ResearchArtifactShape")
    owl = reg.to_owl()
    assert "VerifiableClaim" in owl and "ResearchArtifactShape" in owl


def test_ara_typed_links_carry_grounding_inverse():
    grounds = DEFAULT_LINK_REGISTRY.get("grounds")
    assert grounds is not None
    assert grounds.edge_type.value == "grounded_in"
    assert grounds.inverse_edge_type is not None
    assert grounds.inverse_edge_type.value == "supports"
    assert DEFAULT_LINK_REGISTRY.get("artifact_contains_claim") is not None
    assert DEFAULT_LINK_REGISTRY.get("implements_claim") is not None


def test_ara_claim_confidence_value_type_constrained():
    vt = get_value_type("ClaimConfidence")
    assert vt is not None
    assert vt.validate(0.7) and not vt.validate(1.5)


def test_owl_bridge_seeds_ara_object_property_characteristics():
    # a bridge with no schema pack still treats grounding as transitive + inverse
    bridge = OWLBridge(graph=None, owl_backend=None)
    assert "grounded_in" in bridge._pack_transitive
    assert bridge._pack_inverse.get("grounded_in") == "supports"
    # the module constants are the always-on source
    assert "grounded_in" in ARA_TRANSITIVE_EDGES
    assert ARA_INVERSE_EDGES.get("grounded_in") == "supports"
