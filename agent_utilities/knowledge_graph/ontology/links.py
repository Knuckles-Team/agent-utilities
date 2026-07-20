#!/usr/bin/python
from __future__ import annotations

"""Ontology Link Types — typed link cardinalities + junction reification.

CONCEPT:AU-KG.domains.trade-journal-bias-auditor — First-Class Link Types

Palantir Foundry doc matched: *object-link-types / type-reference link
cardinalities*, specifically the **many-to-many link backed by an object type**
(the "intermediary"/"junction" object) pattern. In Foundry's ontology a link
between two object types declares a cardinality (one-to-one, one-to-many,
many-to-many); a many-to-many link may be *reified* as its own intermediary
object type that carries the link's own properties (e.g. an
``Employee``↔``Project`` link reified as an ``Assignment`` object holding
``role`` and ``allocation``). In an LPG/RDF substrate this is exactly a reified
edge: a junction NODE plus two directed edges to the endpoints.

This module gives the agent-utilities ontology that missing first-class link
layer:

  - :class:`LinkCardinality` — ONE_TO_ONE / ONE_TO_MANY / MANY_TO_MANY.
  - :class:`LinkType` — a *definition* of a typed link between two ontology
    object (node) types, mirroring a Foundry link-type. Edge direction is named
    (``edge_type`` is a real :class:`RegistryEdgeType`), so the link can be
    written and traversed deterministically.
  - :class:`JunctionLinkType` — a many-to-many link reified through an
    intermediary junction object type. It declares the junction node type, the
    two endpoint roles, the two edge types, and the property schema the junction
    instance carries.
  - :func:`materialize_junction` — turns a junction link + a concrete pair of
    endpoint ids + the junction's own properties into a
    ``(junction_node, edge_a, edge_b)`` triple (``RegistryNode`` +
    two ``RegistryEdge``) ready for the graph write path. **Real**
    materialization — deterministic id, stamped provenance, no stub.
  - Reverse-traversal helpers (:func:`endpoints_of`, :func:`neighbors_via`,
    :func:`junctions_for`) that walk a set of materialized edges back to the
    endpoints / junctions, so a reified many-to-many is navigable both ways.

Reuses the existing fabric — nothing reinvented:
  - :class:`RegistryEdgeType` / :class:`RegistryNode` / :class:`RegistryEdge`
    from ``models.knowledge_graph`` are the write-path contracts. The junction
    node defaults to ``RegistryNodeType.RELATIONSHIP`` — the repo's existing
    *first-class reified relationship* node type (see ``RelationshipNode``).
  - The cardinality → OWL-characteristic mapping reuses the
    :class:`OwlObjectProperty` notion from ``models.schema_pack`` (an inverse +
    functional/inverse-functional declaration), and link inference reuses the
    :class:`LinkInferenceRule` notion so a pack can deterministically infer a
    typed link from document text.

A module-level :data:`DEFAULT_LINK_REGISTRY` is populated at import with real
built-in link types (an authored ``person→document`` one-to-many link and an
``agent⇄skill`` many-to-many junction reified as a ``relationship`` node) — a
live registry, never an empty shell.
"""

import hashlib
import time
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ...models.knowledge_graph import (
    RegistryEdge,
    RegistryEdgeType,
    RegistryNode,
    RegistryNodeType,
)
from ...models.schema_pack import LinkInferenceRule, OwlObjectProperty


class LinkCardinality(StrEnum):
    """Cardinality of an ontology link, matching Foundry link-type cardinalities.

    CONCEPT:AU-KG.domains.trade-journal-bias-auditor

    - ``ONE_TO_ONE``: each source relates to at most one target and vice-versa
      (an ``owl:FunctionalProperty`` *and* ``owl:InverseFunctionalProperty``).
    - ``ONE_TO_MANY``: a source may relate to many targets, but each target to
      at most one source (the forward edge is inverse-functional).
    - ``MANY_TO_MANY``: unconstrained on both ends — the only cardinality a
      junction object can reify (a many-to-many link *backed by an object*).
    """

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"


class LinkType(BaseModel):
    """A first-class, typed link between two ontology object types.

    CONCEPT:AU-KG.domains.trade-journal-bias-auditor — mirrors a Palantir Foundry *link type*: a named, directed,
    cardinality-bearing relationship between a source object type and a target
    object type. Direct (non-reified) links materialise as a single
    :class:`RegistryEdge`; for a ``MANY_TO_MANY`` link that needs its own
    properties, use :class:`JunctionLinkType` instead.

    Attributes:
        name: Unique link-type name (the registry key), e.g. ``"authored"``.
        source_type: The ontology node type at the link's tail.
        target_type: The ontology node type at the link's head.
        edge_type: The concrete :class:`RegistryEdgeType` written for the
            forward direction.
        cardinality: One of :class:`LinkCardinality`.
        inverse_edge_type: Optional named inverse edge (``owl:inverseOf``); when
            set, reverse traversal/closure can materialise the back-edge.
        description: Human/LLM-facing description.
        link_inference: Optional zero-LLM regex rules that infer this link from
            document text on write (reuses :class:`LinkInferenceRule`).
    """

    name: str
    source_type: RegistryNodeType
    target_type: RegistryNodeType
    edge_type: RegistryEdgeType
    cardinality: LinkCardinality = LinkCardinality.ONE_TO_MANY
    inverse_edge_type: RegistryEdgeType | None = None
    description: str = ""
    link_inference: list[LinkInferenceRule] = Field(default_factory=list)

    def materialize(
        self,
        source_id: str,
        target_id: str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> RegistryEdge:
        """Return the single forward :class:`RegistryEdge` for this direct link.

        For a many-to-many link that needs to carry its own properties prefer a
        :class:`JunctionLinkType`; this method writes a plain typed edge.
        """
        md: dict[str, Any] = {"link_type": self.name, "cardinality": self.cardinality}
        if metadata:
            md.update(metadata)
        return RegistryEdge(
            source=source_id,
            target=target_id,
            type=self.edge_type,
            weight=weight,
            metadata=md,
        )

    def owl_object_property(self) -> OwlObjectProperty:
        """Return the OWL object-property characteristics implied by cardinality.

        CONCEPT:AU-KG.ontology.pack-owl-closure — reuses the pack-driven OWL closure notion. The
        ``inverse_of`` is carried so the existing reasoning closure can
        materialise the back-edge. (Functional / inverse-functional flags are
        cardinality-derived and surfaced via :meth:`is_functional` /
        :meth:`is_inverse_functional` for the SHACL/uniqueness gate, since
        ``OwlObjectProperty`` itself models only transitive/symmetric/inverse.)
        """
        return OwlObjectProperty(
            edge_type=self.edge_type.value,
            transitive=False,
            symmetric=False,
            inverse_of=(
                self.inverse_edge_type.value if self.inverse_edge_type else None
            ),
        )

    def is_functional(self) -> bool:
        """True when each source maps to at most one target (one-to-one)."""
        return self.cardinality == LinkCardinality.ONE_TO_ONE

    def is_inverse_functional(self) -> bool:
        """True when each target maps to at most one source (1:1 or 1:many)."""
        return self.cardinality in (
            LinkCardinality.ONE_TO_ONE,
            LinkCardinality.ONE_TO_MANY,
        )


class JunctionLinkType(LinkType):
    """A many-to-many link reified through an intermediary junction object.

    CONCEPT:AU-KG.domains.trade-journal-bias-auditor — the Palantir *many-to-many link backed by an object type*.
    The link is not a single edge but a junction NODE carrying the link's own
    properties, joined to its two endpoints by two directed edges::

        (source) --edge_a--> (junction:junction_type {props}) <--edge_b-- (target)

    The junction node type defaults to ``RegistryNodeType.RELATIONSHIP`` (the
    repo's existing first-class reified-relationship node). The two role names
    label which endpoint each edge attaches to, so reverse traversal can recover
    ``(source, target)`` unambiguously from the materialized edges.

    Attributes:
        junction_type: Node type of the reified junction (default
            ``RELATIONSHIP``).
        source_role: Role label for the source endpoint (e.g. ``"agent"``).
        target_role: Role label for the target endpoint (e.g. ``"skill"``).
        edge_type: Edge from the *source* endpoint into the junction
            (``edge_a``). Inherited from :class:`LinkType`.
        target_edge_type: Edge from the *target* endpoint into the junction
            (``edge_b``).
        junction_properties: Declared property names the junction may carry
            (informational schema; extra props are still accepted on write).
    """

    junction_type: RegistryNodeType = RegistryNodeType.RELATIONSHIP
    source_role: str = "source"
    target_role: str = "target"
    target_edge_type: RegistryEdgeType
    junction_properties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_many_to_many(self) -> JunctionLinkType:
        """A reified link is many-to-many by construction (Foundry rule)."""
        # A junction object only ever reifies an unconstrained M:N link — that is
        # the whole reason to back the link with an object. Normalize rather than
        # reject so callers needn't restate the obvious.
        object.__setattr__(self, "cardinality", LinkCardinality.MANY_TO_MANY)
        if self.source_role == self.target_role:
            raise ValueError(
                "JunctionLinkType requires distinct source_role/target_role "
                f"(both were {self.source_role!r})"
            )
        return self

    def junction_id(self, source_id: str, target_id: str) -> str:
        """Return a stable, deterministic id for the junction instance.

        Deterministic on ``(name, source_id, target_id)`` so re-materializing
        the same pair is idempotent (upserts the same junction node) — the
        write path can MERGE rather than duplicate.
        """
        digest = hashlib.sha256(
            f"{self.name}\x1f{source_id}\x1f{target_id}".encode()
        ).hexdigest()[:16]
        return f"junction:{self.name}:{digest}"

    def materialize_junction(
        self,
        source_id: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        weight: float = 1.0,
    ) -> tuple[RegistryNode, RegistryEdge, RegistryEdge]:
        """Reify the link as ``(junction_node, edge_a, edge_b)`` for the writer.

        CONCEPT:AU-KG.domains.trade-journal-bias-auditor — the core materialization. Produces:

          - a :class:`RegistryNode` of ``junction_type`` whose ``metadata``
            carries the endpoint ids, roles, link-type name and the supplied
            junction properties (the link's own attributes);
          - ``edge_a``: ``source_id --edge_type--> junction`` tagged with the
            ``source_role``;
          - ``edge_b``: ``target_id --target_edge_type--> junction`` tagged with
            the ``target_role``.

        The triple is exactly what the graph write path consumes (add_node +
        add_edge × 2). Deterministic ids make re-writes idempotent.

        Args:
            source_id: Concrete id of the source endpoint node.
            target_id: Concrete id of the target endpoint node.
            properties: The junction's own link properties (e.g. ``role``,
                ``allocation``). Stored on the junction node's ``metadata``.
            name: Optional human name for the junction node (defaults to
                ``"<link_name>(<source>↔<target>)"``).
            weight: Edge weight stamped on both endpoint edges.
        """
        props = dict(properties or {})
        jid = self.junction_id(source_id, target_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = RegistryNode(
            id=jid,
            type=self.junction_type,
            name=name or f"{self.name}({source_id}↔{target_id})",
            description=self.description,
            metadata={
                "link_type": self.name,
                "is_junction": True,
                self.source_role: source_id,
                self.target_role: target_id,
                "source_role": self.source_role,
                "target_role": self.target_role,
                "source_edge_type": self.edge_type.value,
                "target_edge_type": self.target_edge_type.value,
                "properties": props,
                "materialized_at": now,
            },
        )
        edge_a = RegistryEdge(
            source=source_id,
            target=jid,
            type=self.edge_type,
            weight=weight,
            metadata={
                "link_type": self.name,
                "role": self.source_role,
                "junction": jid,
            },
        )
        edge_b = RegistryEdge(
            source=target_id,
            target=jid,
            type=self.target_edge_type,
            weight=weight,
            metadata={
                "link_type": self.name,
                "role": self.target_role,
                "junction": jid,
            },
        )
        return node, edge_a, edge_b


# ── Reverse-traversal helpers ────────────────────────────────────────────────


def is_junction_node(node: RegistryNode) -> bool:
    """True when ``node`` was materialized as a reified link junction."""
    return bool(node.metadata.get("is_junction"))


def endpoints_of(junction: RegistryNode) -> tuple[str | None, str | None]:
    """Recover ``(source_id, target_id)`` from a materialized junction node.

    CONCEPT:AU-KG.domains.trade-journal-bias-auditor — reverse traversal entry point. Reads the role-keyed
    endpoint ids stamped onto the junction's metadata by
    :meth:`JunctionLinkType.materialize_junction`. Returns ``(None, None)`` for a
    node that is not a junction.
    """
    if not is_junction_node(junction):
        return None, None
    md = junction.metadata
    src_role = md.get("source_role", "source")
    tgt_role = md.get("target_role", "target")
    return md.get(src_role), md.get(tgt_role)


def junctions_for(
    endpoint_id: str,
    edges: Iterable[RegistryEdge],
) -> list[str]:
    """Return the junction ids reachable from ``endpoint_id`` via reified edges.

    Walks the materialized endpoint→junction edges (those carrying a
    ``"junction"`` metadata tag) and returns the distinct junction node ids the
    endpoint participates in — one hop of reverse traversal over an M:N link.
    """
    seen: list[str] = []
    for e in edges:
        if e.source != endpoint_id:
            continue
        jid = e.metadata.get("junction")
        if isinstance(jid, str) and jid == e.target and jid not in seen:
            seen.append(jid)
    return seen


def neighbors_via(
    endpoint_id: str,
    junctions: Sequence[RegistryNode],
    edges: Iterable[RegistryEdge],
    *,
    link_name: str | None = None,
) -> list[tuple[str, RegistryNode]]:
    """Return ``(other_endpoint_id, junction_node)`` neighbors of ``endpoint_id``.

    CONCEPT:AU-KG.domains.trade-journal-bias-auditor — full reverse traversal of a reified many-to-many link.
    Given the endpoint, the junction nodes, and the materialized edges, this
    resolves every *other* endpoint the node is linked to (and the junction that
    carries the link's properties), optionally filtered to one ``link_name``.

    Bidirectional: works whether ``endpoint_id`` is the source or target role of
    the junction — the missing-direction back-edge is not required because the
    junction metadata records both endpoint ids.
    """
    reachable = set(junctions_for(endpoint_id, edges))
    by_id = {j.id: j for j in junctions}
    out: list[tuple[str, RegistryNode]] = []
    for jid in reachable:
        jnode = by_id.get(jid)
        if jnode is None:
            continue
        if link_name is not None and jnode.metadata.get("link_type") != link_name:
            continue
        src, tgt = endpoints_of(jnode)
        other = tgt if src == endpoint_id else src
        if isinstance(other, str):
            out.append((other, jnode))
    return out


# ── Link-type registry (import-populated; live, not an empty shell) ───────────


class LinkTypeRegistry:
    """Registry of first-class link types, keyed by name. CONCEPT:AU-KG.domains.trade-journal-bias-auditor.

    Mirrors :class:`ActionRegistry`: rejects duplicate names, supports lookup by
    name and by the ontology object types a link connects.
    """

    def __init__(self) -> None:
        self._links: dict[str, LinkType] = {}

    def register(self, link: LinkType) -> None:
        """Register a link type; raises ``ValueError`` on a duplicate name."""
        if link.name in self._links:
            raise ValueError(f"Link type already registered: {link.name!r}")
        self._links[link.name] = link

    def get(self, name: str) -> LinkType | None:
        """Return the link type named ``name``, or ``None``."""
        return self._links.get(name)

    def list_links(self) -> list[LinkType]:
        """Return all registered link types."""
        return list(self._links.values())

    def links_for_type(self, node_type: RegistryNodeType) -> list[LinkType]:
        """Return links whose source or target is ``node_type``."""
        return [
            link
            for link in self._links.values()
            if node_type in (link.source_type, link.target_type)
        ]

    def junctions(self) -> list[JunctionLinkType]:
        """Return only the reified (junction-backed) link types."""
        return [
            link for link in self._links.values() if isinstance(link, JunctionLinkType)
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._links

    def __len__(self) -> int:
        return len(self._links)


def register_builtin_links(registry: LinkTypeRegistry) -> None:
    """Register the built-in link types into ``registry``.

    Two real links, exercising both a direct typed link and a reified junction:

      - ``authored`` — ``person --authored--> document`` (ONE_TO_MANY), with an
        ``was_attributed_to`` inverse and a zero-LLM ``[[wikilink]]`` inference
        rule so the link can be inferred from prose on write.
      - ``agent_skill`` — ``agent ⇄ skill`` reified as a ``relationship``
        junction carrying ``proficiency`` (MANY_TO_MANY) via ``has_skill`` /
        ``provides_capability`` endpoint edges.
    """
    registry.register(
        LinkType(
            name="authored",
            source_type=RegistryNodeType.PERSON,
            target_type=RegistryNodeType.DOCUMENT,
            edge_type=RegistryEdgeType.AUTHORED,
            cardinality=LinkCardinality.ONE_TO_MANY,
            inverse_edge_type=RegistryEdgeType.WAS_ATTRIBUTED_TO,
            description="A person authored a document.",
            link_inference=[
                LinkInferenceRule(
                    pattern=r"authored by \[\[([^\]]+)\]\]",
                    edge_type=RegistryEdgeType.AUTHORED.value,
                    source="group:1",
                    target="doc",
                )
            ],
        )
    )
    registry.register(
        JunctionLinkType(
            name="agent_skill",
            source_type=RegistryNodeType.AGENT,
            target_type=RegistryNodeType.SKILL,
            junction_type=RegistryNodeType.RELATIONSHIP,
            source_role="agent",
            target_role="skill",
            edge_type=RegistryEdgeType.HAS_SKILL,
            target_edge_type=RegistryEdgeType.PROVIDES_CAPABILITY,
            junction_properties=["proficiency", "since"],
            description=(
                "An agent possesses a skill at some proficiency — a many-to-many "
                "link reified as a relationship object carrying its own props."
            ),
        )
    )
    # ── Agent-Native Research Artifact forensic bindings (CONCEPT:AU-KG.ontology.verified-by-implemented-by) ──
    # The ARA 4-layer artifact as first-class typed links so the reasoner traverses
    # /logic↔/evidence↔/src. ``grounds`` carries the grounded_in↔supports inverse
    # (and grounded_in is transitive in owl_bridge), so reasoning chains a claim to
    # the ecosystem code/services that substantiate it.
    registry.register(
        LinkType(
            name="artifact_contains_claim",
            source_type=RegistryNodeType.RESEARCH_ARTIFACT,
            target_type=RegistryNodeType.CLAIM,
            edge_type=RegistryEdgeType.CONTAINS,
            cardinality=LinkCardinality.ONE_TO_MANY,
            description="A research artifact contains its /logic-layer claims.",
        )
    )
    registry.register(
        LinkType(
            name="grounds",
            source_type=RegistryNodeType.CLAIM,
            target_type=RegistryNodeType.EVIDENCE,
            edge_type=RegistryEdgeType.GROUNDED_IN,
            cardinality=LinkCardinality.ONE_TO_MANY,
            inverse_edge_type=RegistryEdgeType.SUPPORTS,
            description="A claim is grounded in evidence (inverse: evidence supports).",
        )
    )
    registry.register(
        LinkType(
            name="implements_claim",
            source_type=RegistryNodeType.CLAIM,
            target_type=RegistryNodeType.CODE_SPEC,
            edge_type=RegistryEdgeType.IMPLEMENTED_BY,
            cardinality=LinkCardinality.ONE_TO_MANY,
            description="A claim's /src layer: the code spec that implements it.",
        )
    )
    # CONCEPT:AU-KG.ontology.typed-ontology-links-binding — Typed ontology links binding Model/InferenceProfile to TaskClass/Role/Agent (HAS_PROFILE/PROFILE_OF/TUNED_FOR/BOUND_TO_ROLE/USES_PROFILE) for profile extrapolation.
    # First-class typed links so OWL reasoning extrapolates which sampling profile
    # fits a task class / role / model from how related ones are tuned.
    registry.register(
        LinkType(
            name="model_has_profile",
            source_type=RegistryNodeType.MODEL,
            target_type=RegistryNodeType.INFERENCE_PROFILE,
            edge_type=RegistryEdgeType.HAS_PROFILE,
            cardinality=LinkCardinality.ONE_TO_MANY,
            inverse_edge_type=RegistryEdgeType.PROFILE_OF,
            description="A model carries a tuned inference profile (inverse: profile_of).",
        )
    )
    registry.register(
        LinkType(
            name="profile_tuned_for",
            source_type=RegistryNodeType.INFERENCE_PROFILE,
            target_type=RegistryNodeType.TASK_CLASS,
            edge_type=RegistryEdgeType.TUNED_FOR,
            cardinality=LinkCardinality.MANY_TO_MANY,
            description="An inference profile is tuned for a task class.",
        )
    )
    registry.register(
        LinkType(
            name="profile_bound_to_role",
            source_type=RegistryNodeType.INFERENCE_PROFILE,
            target_type=RegistryNodeType.ROLE,
            edge_type=RegistryEdgeType.BOUND_TO_ROLE,
            cardinality=LinkCardinality.MANY_TO_MANY,
            description="An inference profile is bound to a functional role.",
        )
    )
    registry.register(
        LinkType(
            name="agent_uses_profile",
            source_type=RegistryNodeType.AGENT,
            target_type=RegistryNodeType.INFERENCE_PROFILE,
            edge_type=RegistryEdgeType.USES_PROFILE,
            cardinality=LinkCardinality.MANY_TO_MANY,
            description="An agent uses an inference profile for a task class.",
        )
    )


# CONCEPT:AU-KG.domains.trade-journal-bias-auditor — populated at import with real built-ins, never empty.
DEFAULT_LINK_REGISTRY = LinkTypeRegistry()
register_builtin_links(DEFAULT_LINK_REGISTRY)


__all__ = [
    "LinkCardinality",
    "LinkType",
    "JunctionLinkType",
    "LinkTypeRegistry",
    "DEFAULT_LINK_REGISTRY",
    "register_builtin_links",
    "is_junction_node",
    "endpoints_of",
    "junctions_for",
    "neighbors_via",
]
