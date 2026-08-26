#!/usr/bin/python
"""The bundled OWL library, rendered as reasoner input.

CONCEPT:AU-KG.ontology.ontology-driven-reasoning

The platform ships 29 ``ontology*.ttl`` modules declaring ~365 object
properties, 20 ``owl:TransitiveProperty``, 8 ``owl:SymmetricProperty``, 30
``owl:inverseOf``, 4 ``owl:propertyChainAxiom`` and 376 ``rdfs:subClassOf``
axioms. None of them ever reached a reasoner: the one wired reasoning path
(:meth:`...core.owl_bridge.OWLBridge.run_cycle` ← ``enrichment/materialize.py``)
synthesises its ontology from two Python string sets, so in production it
reasoned over a single axiom (``au:grounded_in a owl:TransitiveProperty``) and
the graph contained zero edges carrying ``inferred = true``.

This module is the missing join. It parses the SAME bundled TBox
:func:`...core.ontology_publisher.collect_bundled_ontology_graph` already
collects — no second copy of the ontology-discovery logic — and projects it into
:func:`closure_sets`, the ``(transitive, symmetric, inverse)`` property sets the
existing reasoning cycle already knows how to close over. Wiring is therefore a
union into :class:`...core.owl_bridge.OWLBridge`'s own declaration sets, not a
new reasoning path.

Naming. OWL local names are camelCase (``dependsOn``) and the labelled property
graph stores relationship types in ``UPPER_SNAKE`` (``DEPENDS_ON``), so every
property name is translated once here. Class local names are used verbatim —
they are already the schema labels (``Concept``, ``WorkItem``).

Subclass, subproperty, domain and range axioms are read but deliberately NOT
fed to any writer. The engine's own materialising reasoner
(``RunDatalogReasoning``) consumes exactly those, and it is unsafe on this
storage: the native graph holds at most ONE edge per ordered node pair, so
committing a derived relation over an already-connected pair REPLACES the
asserted one (measured: an asserted ``a -PART_OF-> b`` became
``a -DEPENDS_ON-> b`` under the ontology's own
``PART_OF rdfs:subPropertyOf DEPENDS_ON``). They are exposed here for
inspection so a future guarded materialiser has them ready, not because
anything writes them today.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["OntologyAxioms", "closure_sets", "ontology_axioms"]

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
#: Guard against a pathological ontology turning into an unbounded RPC payload.
_MAX_PER_KIND = 2000


def _relationship(iri: Any) -> str:
    """``…#dependsOn`` → ``DEPENDS_ON`` (the labelled-property-graph edge type)."""
    local = str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1].strip()
    if not local or not local[0].isalpha():
        return ""
    return _CAMEL_BOUNDARY.sub("_", local).upper()


def _class_name(iri: Any) -> str:
    """``…#Concept`` → ``Concept`` (already the schema's ``node_type`` label)."""
    local = str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1].strip()
    return local if local and local[0].isalpha() else ""


@dataclass(frozen=True)
class OntologyAxioms:
    """The bundled TBox's edge-producing axioms, in labelled-property-graph terms."""

    transitive_properties: tuple[str, ...] = ()
    symmetric_properties: tuple[str, ...] = ()
    inverse_properties: tuple[tuple[str, str], ...] = ()
    subproperty_relations: tuple[tuple[str, str], ...] = ()
    subclass_relations: tuple[tuple[str, str], ...] = ()
    property_chains: tuple[tuple[str, str, str], ...] = ()
    #: Diagnostics — how many axioms of each kind were read, for logging/tests.
    counts: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(
            self.transitive_properties
            or self.symmetric_properties
            or self.inverse_properties
            or self.subproperty_relations
            or self.subclass_relations
            or self.property_chains
        )


def _chain_members(graph: Any, head: Any, rdf: Any) -> list[Any]:
    """Walk an RDF collection (``owl:propertyChainAxiom`` is an rdf:List)."""
    members: list[Any] = []
    node = head
    while node is not None and node != rdf.nil and len(members) <= 8:
        first = graph.value(node, rdf.first)
        if first is None:
            break
        members.append(first)
        node = graph.value(node, rdf.rest)
    return members


@lru_cache(maxsize=1)
def ontology_axioms() -> OntologyAxioms:
    """Parse the bundled TBox once and cache the projected axiom sets.

    Best-effort by construction: an unavailable ``rdflib`` or an unparseable
    module yields an empty (falsey) :class:`OntologyAxioms` rather than raising,
    because reasoning must never break a write path. The cache means the 29
    modules are parsed at most once per process.
    """
    try:
        import rdflib
        from rdflib.namespace import OWL, RDF, RDFS

        from ..core.ontology_publisher import collect_bundled_ontology_graph

        graph = collect_bundled_ontology_graph()
    except Exception as exc:  # noqa: BLE001 — no ontology is a degraded mode, never a failure
        logger.debug("bundled ontology unavailable (%s)", type(exc).__name__)
        return OntologyAxioms()

    if graph is None or len(graph) == 0:
        return OntologyAxioms()

    def named(term: Any) -> bool:
        return isinstance(term, rdflib.URIRef)

    transitive = {
        rel
        for prop in graph.subjects(RDF.type, OWL.TransitiveProperty)
        if named(prop) and (rel := _relationship(prop))
    }
    symmetric = {
        rel
        for prop in graph.subjects(RDF.type, OWL.SymmetricProperty)
        if named(prop) and (rel := _relationship(prop))
    }
    inverse = {
        (left, right)
        for subject, obj in graph.subject_objects(OWL.inverseOf)
        if named(subject)
        and named(obj)
        and (left := _relationship(subject))
        and (right := _relationship(obj))
        and left != right
    }
    subproperty = {
        (child, parent)
        for subject, obj in graph.subject_objects(RDFS.subPropertyOf)
        if named(subject)
        and named(obj)
        and (child := _relationship(subject))
        and (parent := _relationship(obj))
        and child != parent
    }
    subclass = {
        (child, parent)
        for subject, obj in graph.subject_objects(RDFS.subClassOf)
        # A blank-node object is an owl:Restriction, not a named superclass.
        if named(subject)
        and named(obj)
        and (child := _class_name(subject))
        and (parent := _class_name(obj))
        and child != parent
    }
    chains: set[tuple[str, str, str]] = set()
    for subject, head in graph.subject_objects(OWL.propertyChainAxiom):
        inferred = _relationship(subject) if named(subject) else ""
        members = [
            _relationship(member)
            for member in _chain_members(graph, head, RDF)
            if named(member)
        ]
        # The engine's datalog chain rule is binary: prop1 ∘ prop2 → inferred.
        if inferred and len(members) == 2 and all(members):
            chains.add((members[0], members[1], inferred))

    axioms = OntologyAxioms(
        transitive_properties=tuple(sorted(transitive)[:_MAX_PER_KIND]),
        symmetric_properties=tuple(sorted(symmetric)[:_MAX_PER_KIND]),
        inverse_properties=tuple(sorted(inverse)[:_MAX_PER_KIND]),
        subproperty_relations=tuple(sorted(subproperty)[:_MAX_PER_KIND]),
        subclass_relations=tuple(sorted(subclass)[:_MAX_PER_KIND]),
        property_chains=tuple(sorted(chains)[:_MAX_PER_KIND]),
        counts={
            "triples": len(graph),
            "transitive": len(transitive),
            "symmetric": len(symmetric),
            "inverse": len(inverse),
            "subproperty": len(subproperty),
            "subclass": len(subclass),
            "property_chains": len(chains),
        },
    )
    logger.info("bundled ontology axioms projected for reasoning: %s", axioms.counts)
    return axioms


def closure_sets() -> tuple[
    frozenset[str], frozenset[str], tuple[tuple[str, str], ...]
]:
    """``(transitive, symmetric, inverse_pairs)`` for the lightweight closure.

    Property names are casefolded here because the closure compares them against
    ``edge["relationship"]`` casefolded.
    """
    axioms = ontology_axioms()
    return (
        frozenset(name.casefold() for name in axioms.transitive_properties),
        frozenset(name.casefold() for name in axioms.symmetric_properties),
        tuple(
            (left.casefold(), right.casefold())
            for left, right in axioms.inverse_properties
        ),
    )
