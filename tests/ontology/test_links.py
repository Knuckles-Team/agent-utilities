"""Tests for the first-class ontology link layer (CONCEPT:KG-2.26).

Covers each cardinality, direct-link edge materialization, round-trip junction
(reified many-to-many) materialization, reverse traversal, and the
import-populated default registry.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.links import (
    DEFAULT_LINK_REGISTRY,
    JunctionLinkType,
    LinkCardinality,
    LinkType,
    LinkTypeRegistry,
    endpoints_of,
    is_junction_node,
    junctions_for,
    neighbors_via,
    register_builtin_links,
)
from agent_utilities.models.knowledge_graph import (
    RegistryEdge,
    RegistryEdgeType,
    RegistryNode,
    RegistryNodeType,
)


# ── cardinality ──────────────────────────────────────────────────────────────


def test_one_to_one_is_functional_and_inverse_functional() -> None:
    link = LinkType(
        name="primary_contact",
        source_type=RegistryNodeType.AGENT,
        target_type=RegistryNodeType.USER,
        edge_type=RegistryEdgeType.RELATED_TO,
        cardinality=LinkCardinality.ONE_TO_ONE,
    )
    assert link.is_functional()
    assert link.is_inverse_functional()


def test_one_to_many_is_inverse_functional_only() -> None:
    link = LinkType(
        name="authored",
        source_type=RegistryNodeType.PERSON,
        target_type=RegistryNodeType.DOCUMENT,
        edge_type=RegistryEdgeType.AUTHORED,
        cardinality=LinkCardinality.ONE_TO_MANY,
    )
    assert not link.is_functional()
    assert link.is_inverse_functional()


def test_many_to_many_is_neither() -> None:
    link = LinkType(
        name="tagged",
        source_type=RegistryNodeType.DOCUMENT,
        target_type=RegistryNodeType.CONCEPT,
        edge_type=RegistryEdgeType.ABOUT,
        cardinality=LinkCardinality.MANY_TO_MANY,
    )
    assert not link.is_functional()
    assert not link.is_inverse_functional()


def test_direct_link_materializes_single_edge() -> None:
    link = LinkType(
        name="authored",
        source_type=RegistryNodeType.PERSON,
        target_type=RegistryNodeType.DOCUMENT,
        edge_type=RegistryEdgeType.AUTHORED,
        inverse_edge_type=RegistryEdgeType.WAS_ATTRIBUTED_TO,
    )
    edge = link.materialize("person:ada", "doc:notes", metadata={"k": "v"})
    assert isinstance(edge, RegistryEdge)
    assert edge.source == "person:ada"
    assert edge.target == "doc:notes"
    assert edge.type == RegistryEdgeType.AUTHORED
    assert edge.metadata["link_type"] == "authored"
    assert edge.metadata["k"] == "v"

    owl = link.owl_object_property()
    assert owl.edge_type == RegistryEdgeType.AUTHORED.value
    assert owl.inverse_of == RegistryEdgeType.WAS_ATTRIBUTED_TO.value


# ── junction reification ──────────────────────────────────────────────────────


def _agent_skill_link() -> JunctionLinkType:
    return JunctionLinkType(
        name="agent_skill",
        source_type=RegistryNodeType.AGENT,
        target_type=RegistryNodeType.SKILL,
        junction_type=RegistryNodeType.RELATIONSHIP,
        source_role="agent",
        target_role="skill",
        edge_type=RegistryEdgeType.HAS_SKILL,
        target_edge_type=RegistryEdgeType.PROVIDES_CAPABILITY,
        junction_properties=["proficiency"],
    )


def test_junction_forces_many_to_many() -> None:
    # Even if the caller passes a different cardinality, a reified link is M:N.
    link = JunctionLinkType(
        name="x",
        source_type=RegistryNodeType.AGENT,
        target_type=RegistryNodeType.SKILL,
        source_role="agent",
        target_role="skill",
        edge_type=RegistryEdgeType.HAS_SKILL,
        target_edge_type=RegistryEdgeType.PROVIDES_CAPABILITY,
        cardinality=LinkCardinality.ONE_TO_ONE,
    )
    assert link.cardinality == LinkCardinality.MANY_TO_MANY


def test_junction_requires_distinct_roles() -> None:
    with pytest.raises(ValueError):
        JunctionLinkType(
            name="bad",
            source_type=RegistryNodeType.AGENT,
            target_type=RegistryNodeType.SKILL,
            source_role="same",
            target_role="same",
            edge_type=RegistryEdgeType.HAS_SKILL,
            target_edge_type=RegistryEdgeType.PROVIDES_CAPABILITY,
        )


def test_materialize_junction_triple_shape() -> None:
    link = _agent_skill_link()
    node, edge_a, edge_b = link.materialize_junction(
        "agent:planner", "skill:research", {"proficiency": 0.9}
    )

    assert isinstance(node, RegistryNode)
    assert node.type == RegistryNodeType.RELATIONSHIP
    assert is_junction_node(node)
    assert node.metadata["properties"] == {"proficiency": 0.9}
    assert node.metadata["agent"] == "agent:planner"
    assert node.metadata["skill"] == "skill:research"

    # edge_a: source endpoint -> junction
    assert edge_a.source == "agent:planner"
    assert edge_a.target == node.id
    assert edge_a.type == RegistryEdgeType.HAS_SKILL
    assert edge_a.metadata["role"] == "agent"

    # edge_b: target endpoint -> junction
    assert edge_b.source == "skill:research"
    assert edge_b.target == node.id
    assert edge_b.type == RegistryEdgeType.PROVIDES_CAPABILITY
    assert edge_b.metadata["role"] == "skill"


def test_junction_id_is_deterministic_idempotent() -> None:
    link = _agent_skill_link()
    n1, _, _ = link.materialize_junction("agent:a", "skill:s", {"proficiency": 0.5})
    n2, _, _ = link.materialize_junction("agent:a", "skill:s", {"proficiency": 0.7})
    # Same endpoint pair => same junction id (upsert, not duplicate).
    assert n1.id == n2.id
    # Different pair => different id.
    n3, _, _ = link.materialize_junction("agent:a", "skill:other", {})
    assert n3.id != n1.id


def test_round_trip_endpoints_recovered() -> None:
    link = _agent_skill_link()
    node, _, _ = link.materialize_junction("agent:planner", "skill:research", {})
    src, tgt = endpoints_of(node)
    assert src == "agent:planner"
    assert tgt == "skill:research"


def test_endpoints_of_non_junction_returns_none() -> None:
    plain = RegistryNode(id="x", type=RegistryNodeType.AGENT, name="x")
    assert endpoints_of(plain) == (None, None)


# ── reverse traversal over reified M:N ────────────────────────────────────────


def test_reverse_traversal_bidirectional() -> None:
    link = _agent_skill_link()
    # One agent linked to two skills + a second agent on one of them.
    j1, a1, b1 = link.materialize_junction("agent:planner", "skill:research", {})
    j2, a2, b2 = link.materialize_junction("agent:planner", "skill:coding", {})
    j3, a3, b3 = link.materialize_junction("agent:builder", "skill:coding", {})

    junctions = [j1, j2, j3]
    edges = [a1, b1, a2, b2, a3, b3]

    # From the agent: which junctions does it participate in?
    assert set(junctions_for("agent:planner", edges)) == {j1.id, j2.id}

    # Forward neighbors of the agent => the two skills.
    planner_skills = {other for other, _ in neighbors_via("agent:planner", junctions, edges)}
    assert planner_skills == {"skill:research", "skill:coding"}

    # Reverse neighbors of a skill => the agents that have it (M:N reverse walk).
    coding_agents = {other for other, _ in neighbors_via("skill:coding", junctions, edges)}
    assert coding_agents == {"agent:planner", "agent:builder"}


def test_neighbors_via_filters_by_link_name() -> None:
    link = _agent_skill_link()
    j1, a1, b1 = link.materialize_junction("agent:planner", "skill:research", {})

    other_link = JunctionLinkType(
        name="agent_team",
        source_type=RegistryNodeType.AGENT,
        target_type=RegistryNodeType.TEAM,
        source_role="agent",
        target_role="team",
        edge_type=RegistryEdgeType.BELONGS_TO,
        target_edge_type=RegistryEdgeType.CONTAINS,
    )
    j2, a2, b2 = other_link.materialize_junction("agent:planner", "team:alpha", {})

    junctions = [j1, j2]
    edges = [a1, b1, a2, b2]

    only_skill = neighbors_via(
        "agent:planner", junctions, edges, link_name="agent_skill"
    )
    assert [o for o, _ in only_skill] == ["skill:research"]


# ── registry (import-populated, live) ─────────────────────────────────────────


def test_default_registry_is_populated_with_builtins() -> None:
    assert len(DEFAULT_LINK_REGISTRY) >= 2
    assert "authored" in DEFAULT_LINK_REGISTRY
    assert "agent_skill" in DEFAULT_LINK_REGISTRY
    # The reified built-in is discoverable as a junction.
    names = {j.name for j in DEFAULT_LINK_REGISTRY.junctions()}
    assert "agent_skill" in names


def test_registry_rejects_duplicates_and_lookups() -> None:
    reg = LinkTypeRegistry()
    register_builtin_links(reg)
    with pytest.raises(ValueError):
        register_builtin_links(reg)  # second pass duplicates "authored"

    assert reg.get("authored") is not None
    person_links = reg.links_for_type(RegistryNodeType.PERSON)
    assert any(link.name == "authored" for link in person_links)
    agent_links = reg.links_for_type(RegistryNodeType.AGENT)
    assert any(link.name == "agent_skill" for link in agent_links)
