#!/usr/bin/python
from __future__ import annotations

"""Research ontology objects (CONCEPT:AU-KG.ontology.kg-2).

Makes the unified research-intelligence pipeline *ontologically driven*: research
papers (``Article``) and ecosystem capabilities (``Concept``) become first-class
ontology **Interfaces**, and the ConceptMatcher's verdicts are first-class typed
**Links**:

* ``ADDRESSES`` — a paper addresses a research topic/concept (inverse ADDRESSED_BY),
* ``SATISFIED_BY`` — the paper's contribution is a capability we already built,
* ``RELATES_TO`` — novel-but-relevant (the gap stays open).

Registered into the import-populated default registries so ``kg.ontology``
discovers them with no configuration (KG-2.38 interfaces, KG-2.26 links). The
matcher keeps writing raw edges for robustness; this layer gives the same edges a
governed ontology schema (conformance, link cardinality, discovery). (CONCEPT:AU-KG.ontology.kg-2)
"""

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
    Interface,
    InterfaceLinkConstraint,
    InterfaceProperty,
    InterfaceRegistry,
)
from .links import (
    DEFAULT_LINK_REGISTRY,
    LinkCardinality,
    LinkType,
    LinkTypeRegistry,
)

_RESEARCH_INTERFACES = "ResearchPaper", "ResearchConcept"


def register_research_ontology(
    interfaces: InterfaceRegistry, links: LinkTypeRegistry
) -> None:
    """Register the research-intelligence interfaces + typed links (CONCEPT:AU-KG.ontology.kg-2).

    Idempotent: skips an interface/link already registered (re-import safe).
    """
    if interfaces.get("ResearchPaper") is None:
        interfaces.register(
            Interface(
                name="ResearchPaper",
                description=(
                    "Abstract shape for an ingested research item (Article): a title "
                    "and summary, and — once the ConceptMatcher has run — at least "
                    "one ADDRESSES/SATISFIED_BY/RELATES_TO link to the ecosystem "
                    "Concept registry."
                ),
                properties=[
                    InterfaceProperty(
                        name="name",
                        type_ref="string",
                        description="The paper title.",
                    ),
                    InterfaceProperty(
                        name="summary",
                        type_ref="string",
                        description="Abstract / extracted summary.",
                    ),
                ],
                link_constraints=[
                    InterfaceLinkConstraint(
                        name="addresses",
                        edge_type=RegistryEdgeType.ADDRESSES,
                        min_count=0,
                        description="Links to the ecosystem concept(s) it addresses.",
                    ),
                ],
            )
        )
        interfaces.register(
            Interface(
                name="ResearchConcept",
                description=(
                    "Abstract shape for an ecosystem capability concept the matcher "
                    "compares research against: a canonical concept_id and a "
                    "description."
                ),
                properties=[
                    InterfaceProperty(
                        name="concept_id",
                        type_ref="string",
                        description="Canonical concept id (e.g. KG-2.7).",
                    ),
                    InterfaceProperty(
                        name="description",
                        type_ref="string",
                        description="What the capability does.",
                    ),
                ],
            )
        )

    for link in (
        LinkType(
            name="paper_addresses",
            source_type=RegistryNodeType.ARTICLE,
            target_type=RegistryNodeType.CONCEPT,
            edge_type=RegistryEdgeType.ADDRESSES,
            cardinality=LinkCardinality.MANY_TO_MANY,
            inverse_edge_type=RegistryEdgeType.ADDRESSED_BY,
            description="A research paper addresses an ecosystem concept/topic.",
        ),
        LinkType(
            name="paper_satisfied_by",
            source_type=RegistryNodeType.ARTICLE,
            target_type=RegistryNodeType.CONCEPT,
            edge_type=RegistryEdgeType.SATISFIED_BY,
            cardinality=LinkCardinality.MANY_TO_MANY,
            description="The paper's capability is already built (covered).",
        ),
        LinkType(
            name="paper_relates_to",
            source_type=RegistryNodeType.ARTICLE,
            target_type=RegistryNodeType.CONCEPT,
            edge_type=RegistryEdgeType.RELATES_TO,
            cardinality=LinkCardinality.MANY_TO_MANY,
            description="Novel-but-relevant to an ecosystem concept (stays a gap).",
        ),
    ):
        if links.get(link.name) is None:
            links.register(link)


# Import-populated, never an empty shell — so kg.ontology discovers them by default.
register_research_ontology(DEFAULT_INTERFACE_REGISTRY, DEFAULT_LINK_REGISTRY)

__all__ = ["register_research_ontology"]
