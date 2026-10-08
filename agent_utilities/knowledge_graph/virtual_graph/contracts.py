"""Virtual-graph contracts: a source connection, its metadata, its mappings.

The rule: materialize metadata and hot subsets, virtualize the rest. A
:class:`SourceConnection` names a live source by reference only. A
:class:`MetadataContract` records what discovery found there: entities,
fields, relationships and the operation that lists each entity. A
:class:`VirtualMapping` binds one ontology class to one discovered entity.
Only metadata becomes graph facts (:func:`metadata_triples`); entity rows stay
at the source and are read live through the mapping.

The vocabulary terms are the EG core schema source :data:`SOURCE_ID`
(EG-UNIFIED-DATA-PLANE-R037). AU ships no ``.ttl`` for it and refers to its
terms through :data:`VG_NS` only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

#: The EG core schema source that owns this vocabulary.
SOURCE_ID = "core:virtual-graph@1"
#: The vocabulary namespace.
VG_NS = "http://knuckles.team/kg/virtual#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

#: Source kinds a connection may name; one live binding family each.
SOURCE_KINDS = frozenset(
    {"api", "mcp", "a2a", "graphql", "sql", "iceberg", "teradata", "sparql"}
)
_REF = re.compile(r"^[a-z][a-z0-9+.-]*://[A-Za-z0-9_./#:-]+$")
_SECRET_HINTS = ("@", "password=", "token=", "secret=")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SourceConnection:
    """A live source named by reference; never a DSN with credentials."""

    source_id: str
    kind: str
    endpoint_ref: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.source_id), "a source connection needs a source_id")
        _require(self.kind in SOURCE_KINDS, f"unknown source kind {self.kind!r}")
        _require(bool(_REF.match(self.endpoint_ref)), "endpoint_ref is not a ref")
        lowered = self.endpoint_ref.lower()
        _require(
            not any(hint in lowered for hint in _SECRET_HINTS),
            "endpoint_ref must not carry credentials",
        )

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True, slots=True)
class DiscoveredEntity:
    """One entity a source exposes, with the operation that lists it."""

    name: str
    key: str
    fields: tuple[str, ...]
    operation: str


@dataclass(frozen=True, slots=True)
class DiscoveredRelationship:
    """``from_entity.field`` holds keys of ``to_entity``."""

    from_entity: str
    field: str
    to_entity: str


@dataclass(frozen=True, slots=True)
class MetadataContract:
    """What metadata discovery found at one source at one schema version."""

    source_id: str
    schema_version: str
    entities: tuple[DiscoveredEntity, ...]
    relationships: tuple[DiscoveredRelationship, ...] = ()

    def entity(self, name: str) -> DiscoveredEntity | None:
        return next((e for e in self.entities if e.name == name), None)

    @property
    def digest(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VirtualMapping:
    """One ontology class bound to one discovered entity of one source.

    ``predicates`` maps an ontology property IRI to the source field that
    holds its value. For an object property, the field holds the key of the
    related entity. A mapping serves reads only after approval.
    """

    mapping_id: str
    source_id: str
    entity: str
    class_iri: str
    key_field: str
    predicates: tuple[tuple[str, str], ...] = ()
    approved: bool = False
    version: str = "1"

    def field_for(self, predicate: str) -> str | None:
        return dict(self.predicates).get(predicate)


def check_mapping(mapping: VirtualMapping, contract: MetadataContract) -> None:
    """Refuse a mapping that names an entity or field discovery never saw."""
    _require(mapping.source_id == contract.source_id, "mapping/source mismatch")
    entity = contract.entity(mapping.entity)
    _require(entity is not None, f"undiscovered entity {mapping.entity!r}")
    known = {entity.key, *entity.fields} if entity else set()
    used = {mapping.key_field, *(f for _, f in mapping.predicates)}
    unknown = sorted(used - known)
    _require(not unknown, f"undiscovered fields {unknown}")


def _iri(kind: str, *parts: str) -> str:
    return f"{VG_NS}{kind}/" + "/".join(parts)


def metadata_triples(
    connection: SourceConnection,
    contract: MetadataContract,
    mappings: Sequence[VirtualMapping] = (),
) -> list[tuple[str, str, str]]:
    """The facts that ARE materialized: connection, contract and mappings."""
    src = _iri("source", connection.source_id)
    meta = _iri("contract", contract.source_id, contract.schema_version)
    triples = [
        (src, RDF_TYPE, VG_NS + "SourceConnection"),
        (src, VG_NS + "sourceKind", connection.kind),
        (src, VG_NS + "endpointRef", connection.endpoint_ref),
        (meta, RDF_TYPE, VG_NS + "MetadataContract"),
        (meta, VG_NS + "describes", src),
        (meta, VG_NS + "contractDigest", contract.digest),
    ]
    triples += [(src, VG_NS + "capability", c) for c in connection.capabilities]
    for entity in contract.entities:
        triples.append((meta, VG_NS + "exposesEntity", entity.name))
    for mapping in mappings:
        node = _iri("mapping", mapping.mapping_id)
        triples += [
            (node, RDF_TYPE, VG_NS + "VirtualMapping"),
            (node, VG_NS + "mapsClass", mapping.class_iri),
            (node, VG_NS + "fromSource", src),
            (node, VG_NS + "approved", str(mapping.approved).lower()),
        ]
    return triples


@dataclass(frozen=True, slots=True)
class MaterializationPolicy:
    """Materialize metadata always; copy entity rows only for hot subsets.

    A mapping is ``materialize`` when its observed reads reach
    ``hot_reads`` and its source declares the ``copy`` capability. Every
    other mapping stays ``virtual`` and is read live at query time.
    """

    hot_reads: int = 50
    pinned: frozenset[str] = field(default_factory=frozenset)

    def decide(
        self, mapping: VirtualMapping, connection: SourceConnection, reads: int
    ) -> str:
        if not connection.supports("copy"):
            return "virtual"
        hot = reads >= self.hot_reads or mapping.mapping_id in self.pinned
        return "materialize" if hot else "virtual"

    def plan(
        self,
        mappings: Sequence[VirtualMapping],
        connections: Mapping[str, SourceConnection],
        reads: Mapping[str, int],
    ) -> dict[str, str]:
        return {
            m.mapping_id: self.decide(
                m, connections[m.source_id], int(reads.get(m.mapping_id, 0))
            )
            for m in mappings
        }


__all__ = [
    "DiscoveredEntity",
    "DiscoveredRelationship",
    "MaterializationPolicy",
    "MetadataContract",
    "SOURCE_ID",
    "SOURCE_KINDS",
    "SourceConnection",
    "VG_NS",
    "VirtualMapping",
    "check_mapping",
    "metadata_triples",
]
