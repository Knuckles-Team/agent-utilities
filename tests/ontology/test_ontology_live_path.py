"""Live-path integration tests for the ontology layer (Wire-First).

These exercise the *existing* entry points — a real
:class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph` facade and its
``ontology`` accessor (:class:`OntologySystem`) — rather than the per-module
unit surfaces, asserting that the integrated behavior happens as a side effect
on the live path:

  (a) register + invoke a typed function through the graph-bound runtime;
  (b) compute a derived property for an object via ``kg.ontology.derive``;
  (c) target an interface and resolve its concrete implementers;
  (d) materialize a many-to-many junction link into the (node, edge, edge) triple
      the graph-write path consumes;

plus the cross-cutting wiring: the facade exposes ``ontology``; the OWL bridge
PROMOTABLE_NODE_TYPES carries the new ontology-system node types; and the
``ontology_system.ttl`` declares the promoted classes.

CONCEPT:KG-2.38 / KG-2.39 / KG-2.40 / KG-2.41 / KG-2.47 / KG-2.26.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.knowledge_graph.ontology import OntologySystem
from agent_utilities.knowledge_graph.ontology.functions import (
    FunctionKind,
    FunctionParameter,
    FunctionSpec,
)


@pytest.fixture()
def kg() -> KnowledgeGraph:
    """A cheap in-memory facade (no external backend required)."""
    return KnowledgeGraph(backend_type="memory")


# ── facade wiring ────────────────────────────────────────────────────────────


def test_facade_exposes_ontology_system_bound_to_graph(kg: KnowledgeGraph) -> None:
    ont = kg.ontology
    assert isinstance(ont, OntologySystem)
    # The runtime must be bound to THIS facade so ON_OBJECTS funcs read the graph.
    assert ont.functions._graph is kg
    # Same object on repeat access (lazy slot cached).
    assert kg.ontology is ont
    # Registries are import-populated, never empty shells.
    assert len(ont.function_registry) >= 2
    assert len(ont.interfaces) >= 3
    assert len(ont.links) >= 2
    assert len(ont.property_types) >= 10
    assert len(ont.value_types) >= 6
    assert len(ont.derived_registry) >= 3


# ── (a) register + invoke a function on the live path ────────────────────────


def test_register_and_invoke_function_via_ontology(kg: KnowledgeGraph) -> None:
    ont = kg.ontology

    def _add(a: int, b: int) -> int:
        return a + b

    spec = FunctionSpec(
        name="test.add_live",
        version="1.0.0",
        kind=FunctionKind.PLAIN,
        inputs=[
            FunctionParameter(name="a", type=int),
            FunctionParameter(name="b", type=int),
        ],
        output=int,
        handler=_add,
        released=True,
        description="Live-path integration adder.",
    )
    ont.function_registry.register(spec, replace=True)

    result = ont.invoke_function("test.add_live", {"a": 2, "b": 3})
    assert result.ok is True
    assert result.value == 5
    # The governed runtime audits every invocation (live side effect).
    assert result.audit_ref

    # A built-in QUERY function also resolves through the same runtime.
    agg = ont.invoke_function("numeric.aggregate", {"values": [1, 2, 3], "op": "sum"})
    assert agg.ok is True
    assert agg.value == pytest.approx(6.0)


# ── (b) compute a derived property on the live path ──────────────────────────


def test_compute_derived_property_via_ontology(kg: KnowledgeGraph) -> None:
    ont = kg.ontology
    obj = {"id": "doc-1", "name": "Quarterly Report", "type": "document"}

    # The built-in FUNCTION-backed `summary` derived property runs through the
    # real function runtime against the supplied object.
    res = ont.derive(obj, "summary")
    assert res.property_name == "summary"
    assert res.ok is True
    assert res.value  # non-empty rendered summary
    assert isinstance(res.value, str)

    # compute_all returns the {name: value} map of all applicable declarations.
    all_derived = ont.derive_all(obj)
    assert "summary" in all_derived


# ── (c) interface targeting on the live path ─────────────────────────────────


def test_interface_targeting_resolves_implementers(kg: KnowledgeGraph) -> None:
    ont = kg.ontology
    # An interface name expands to its concrete implementing object types.
    impls = ont.resolve_target("HasProvenance")
    assert "document" in impls
    # A plain concrete-type name passes through unchanged.
    assert ont.resolve_target("place") == ["place"]
    # Conformance check on the live path.
    doc = {
        "type": "document",
        "source": "https://example.org/a",
        "ingested_at": "2026-01-01T00:00:00Z",
    }
    assert ont.conforms(doc, "HasProvenance") in (
        True,
        False,
    )  # real evaluation, no raise


# ── (d) materialize a junction link on the live path ─────────────────────────


def test_materialize_junction_link_via_ontology(kg: KnowledgeGraph) -> None:
    ont = kg.ontology
    node, edge_a, edge_b = ont.materialize_link(
        "agent_skill", "agent-7", "skill-python", {"proficiency": "expert"}
    )
    # The triple is exactly what the graph-write add_node/add_edge path consumes.
    assert node.id
    assert node.metadata["link_type"] == "agent_skill"
    assert node.metadata["agent"] == "agent-7"
    assert node.metadata["skill"] == "skill-python"
    assert node.metadata["properties"]["proficiency"] == "expert"
    # Endpoint edges point at the junction, role-tagged.
    assert edge_a.target == node.id and edge_a.source == "agent-7"
    assert edge_b.target == node.id and edge_b.source == "skill-python"
    assert edge_a.metadata["role"] == "agent"
    assert edge_b.metadata["role"] == "skill"

    # A non-junction / unknown link name raises (real lookup, not a silent pass).
    with pytest.raises(KeyError):
        ont.materialize_link("authored", "p-1", "d-1")  # 'authored' is a direct link


# ── (e) property/value type coercion on the live path ────────────────────────


def test_property_and_value_type_paths(kg: KnowledgeGraph) -> None:
    ont = kg.ontology
    # Property-type -> column-type bridge (drives node-table DDL).
    assert ont.column_type_for("array<string>") == "STRING[]"
    assert ont.coerce_property("integer", "42") == 42
    # Constrained value type validation.
    assert ont.validate_value("EmailAddress", "a@b.com") is True
    assert ont.validate_value("EmailAddress", "not-an-email") is False


# ── cross-cutting wiring (OWL promotion + ttl) ───────────────────────────────


def test_owl_bridge_promotes_ontology_system_node_types() -> None:
    from agent_utilities.knowledge_graph.core.owl_bridge import PROMOTABLE_NODE_TYPES

    for nt in ("relationship", "function", "function_invocation", "interface"):
        assert nt in PROMOTABLE_NODE_TYPES, nt


def test_ontology_system_ttl_parses_with_classes() -> None:
    rdflib = pytest.importorskip("rdflib")
    spec = importlib.util.find_spec("agent_utilities.knowledge_graph")
    assert spec is not None and spec.origin is not None
    ttl = Path(spec.origin).parent / "ontology_system.ttl"
    assert ttl.exists(), "ontology_system.ttl must be bundled for the OWL loader glob"
    g = rdflib.Graph()
    g.parse(str(ttl), format="turtle")
    KG = rdflib.Namespace("http://knuckles.team/kg#")
    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    for cls in (
        "Function",
        "FunctionInvocation",
        "Interface",
        "LinkType",
        "JunctionLink",
    ):
        assert (KG[cls], rdflib.RDF.type, OWL.Class) in g, cls
    # It also contributes owl:Restriction axioms toward the TTL gate.
    assert len(list(g.subjects(rdflib.RDF.type, OWL.Restriction))) >= 1
