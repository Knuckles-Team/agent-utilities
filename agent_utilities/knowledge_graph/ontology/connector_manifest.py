"""Connector Ontology Manifest — source-agnostic schema (CONCEPT:AU-KG.ontology.connector-manifest-schema).

Generalizes ``ontology.leanix_metamodel``'s ``ClassSpec``/``ObjectPropertySpec``/
``DatatypePropertySpec`` (LeanIX-only) into source-agnostic OWL primitives, plus the
declarative **Connector Ontology Manifest** every fleet connector compiles to: what
resources/actions/events it exposes, how records map onto the canonical ontology
(``schema_mappings``), how it syncs (``sync``, from the ``mcp_tool`` preset shape,
CONCEPT:AU-KG.ingest.mcp-tool-connector), what it protects (``identity``/``permissions``/``policy``),
and — the X6 supply-chain-integrity leg — a signed, hash-pinned ``provenance`` block.

This module is pure schema (Pydantic + dataclasses); the compiler that turns a
manifest into OWL/SHACL lives in :mod:`manifest_compiler`, and the hash/signature
primitives reused by ``provenance.integrity`` live in :mod:`ontology_integrity`.
No network, no LLM — every field here is either read verbatim from an existing
connector artifact (ontology ``.ttl``, ``mcp_source_presets.json``, ``a2a.json``) or a
documented heuristic default flagged in ``review_todos`` for human/LLM follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .leanix_metamodel import _XSD as XSD_TYPES

# Reused, not duplicated (CONCEPT:AU-KG.ontology.connector-manifest-schema): the LeanIX
# compiler already owns the field-value-type -> XSD map and the ArchiMate crosswalk
# heuristic; every source-agnostic manifest reuses the same tables.
from .leanix_metamodel import DEFAULT_ARCHIMATE_CROSSWALK

__all__ = [
    "XSD_TYPES",
    "DEFAULT_ARCHIMATE_CROSSWALK",
    "HUB_NAME_HEURISTIC_CROSSWALK",
    "nearest_hub_class",
    "PII_HEURISTIC_FIELD_NAMES",
    "OntologyClassSpec",
    "OntologyObjectPropertySpec",
    "OntologyDatatypePropertySpec",
    "OntologySpec",
    "ResourceRelation",
    "SchemaMapping",
    "ResourceSpec",
    "ActionSpec",
    "EventSpec",
    "IdentitySpec",
    "PermissionsSpec",
    "SyncSpec",
    "IntegrityInfo",
    "ProvenanceSpec",
    "PolicySpec",
    "ConnectorManifest",
]

# ── D16 residue fallback: nearest hub-canonical-class-by-name (DRAFT, never authoritative) ──
# When a resource has neither an explicit ``rdfs:subClassOf`` (its OWN ontology's stated
# parent) nor a hit in :data:`DEFAULT_ARCHIMATE_CROSSWALK` (LeanIX/ArchiMate fact-sheet
# types only), this conservative keyword table gives a best-effort DRAFT crosswalk to the
# canonical hub ``agent_utilities/knowledge_graph/ontology.ttl`` class of the same/nearest
# *name* — e.g. a connector's ``Incident``/``Issue``/``Bug`` resource -> hub ``:Incident``.
# Deliberately small and conservative: only common, low-ambiguity domain nouns get an
# entry; anything else is left unresolved (``None``) rather than guessed. Every hit is
# flagged in ``review_todos`` as a DRAFT requiring human sign-off — it is NEVER
# auto-enforced (CONCEPT:AU-KG.ontology.connector-manifest-schema, task D16).
HUB_NAME_HEURISTIC_CROSSWALK: dict[str, str] = {
    "action": "Action",
    "agent": "Agent",
    "agreement": "Agreement",
    "alert": "Incident",
    "article": "Document",
    "bug": "Incident",
    "code": "Code",
    "company": "Organization",
    "concept": "Concept",
    "contact": "Person",
    "dataset": "Dataset",
    "decision": "Decision",
    "department": "Organization",
    "doc": "Document",
    "document": "Document",
    "employee": "Person",
    "event": "Event",
    "evidence": "Evidence",
    "file": "File",
    "framework": "Framework",
    "goal": "Goal",
    "group": "Organization",
    "host": "Server",
    "incident": "Incident",
    "insight": "Insight",
    "issue": "Incident",
    "job": "Action",
    "location": "Place",
    "log": "Event",
    "member": "Person",
    "memory": "Memory",
    "module": "Module",
    "note": "Document",
    "objective": "Goal",
    "org": "Organization",
    "organization": "Organization",
    "page": "Document",
    "person": "Person",
    "place": "Place",
    "playbook": "Playbook",
    "policy": "Policy",
    "problem": "Incident",
    "procedure": "Procedure",
    "process": "Action",
    "project": "SoftwareProject",
    "repo": "SoftwareProject",
    "repository": "SoftwareProject",
    "role": "Role",
    "rule": "GovernanceRule",
    "server": "Server",
    "service": "System",
    "site": "Place",
    "skill": "Skill",
    "system": "System",
    "task": "Action",
    "team": "Team",
    "ticket": "Incident",
    "tool": "Tool",
    "user": "Person",
    "workflow": "Action",
}


def nearest_hub_class(name: str) -> str | None:
    """Best-effort DRAFT crosswalk (D16 residue): try ``name`` whole, then each of its
    camelCase words left-to-right, against :data:`HUB_NAME_HEURISTIC_CROSSWALK`.

    Deterministic, offline, no LLM. Returns ``None`` (never a guess) when nothing in
    the conservative table matches — callers must leave the field unresolved and flag
    it in ``review_todos`` rather than invent a value.
    """
    import re

    whole = HUB_NAME_HEURISTIC_CROSSWALK.get(name.lower())
    if whole:
        return whole
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).split()
    for w in words:
        hit = HUB_NAME_HEURISTIC_CROSSWALK.get(w.lower())
        if hit:
            return hit
    return None


# Field-name heuristic for PII detection (task-specified starter set). Purely a
# default the generator applies to flag candidate PII fields — always left under
# ``policy.pii_fields`` for human/LLM review, never auto-enforced.
PII_HEURISTIC_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "email",
        "email_address",
        "ssn",
        "social_security_number",
        "phone",
        "phone_number",
        "mobile",
        "dob",
        "date_of_birth",
        "dateofbirth",
        "birthdate",
        "address",
        "home_address",
        "ip",
        "ip_address",
        "credit_card",
        "card_number",
        "passport",
        "national_id",
        "first_name",
        "last_name",
        "full_name",
    }
)


# ── Generalized OWL primitives (source-agnostic; was leanix_metamodel.ClassSpec &c.) ──


@dataclass
class OntologyClassSpec:
    """One connector resource as an OWL class (generalized ``ClassSpec``)."""

    local: str  # OWL class local name
    label: str
    parent: str | None  # crosswalk (ArchiMate/base) class local name, or None
    id_prefix: str


@dataclass
class OntologyObjectPropertySpec:
    """One cross-resource relation as an OWL object property (generalized ``ObjectPropertySpec``)."""

    local: str
    label: str
    domain: (
        str | None
    )  # resource local name, or None (fan-in relation, no single domain)
    range: str
    lpg_rel_type: str  # UPPER_SNAKE edge label the extractor emits


@dataclass
class OntologyDatatypePropertySpec:
    """One record field as an OWL datatype property (generalized ``DatatypePropertySpec``).

    Field-level (no ``domain``) — matches the existing fleet ``ontology_*.ttl`` /
    federated ``<pkg>/ontology/*.ttl`` convention where datatype properties are a
    shared, ontology-wide vocabulary rather than per-class.
    """

    local: str
    label: str
    range: str  # an xsd:* curie


@dataclass
class OntologySpec:
    """The compiled manifest: OWL terms plus the maps a connector/extractor consumes."""

    classes: list[OntologyClassSpec] = field(default_factory=list)
    object_properties: list[OntologyObjectPropertySpec] = field(default_factory=list)
    datatype_properties: list[OntologyDatatypePropertySpec] = field(
        default_factory=list
    )
    # resource local name -> (owl class local, node-id prefix)
    type_map: dict[str, tuple[str, str]] = field(default_factory=dict)
    # relation local name -> (lpg_rel_type, target resource)
    relation_map: dict[str, tuple[str, str]] = field(default_factory=dict)


# ── Connector Ontology Manifest (Pydantic; the on-disk connector_manifest.yml) ──


class ResourceRelation(BaseModel):
    """One outgoing relation from a resource (an OWL object property)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str = ""
    target: str  # target resource local name (or "owl:Thing" if unresolved)
    lpg_rel_type: str = ""  # UPPER_SNAKE edge label; derived if left blank


class SchemaMapping(BaseModel):
    """How one resource's fields map onto the canonical ontology (KG-2.9-parity)."""

    model_config = ConfigDict(extra="forbid")

    ontology_class: str | None = None  # ArchiMate/base crosswalk parent (consumed by manifest_compiler + ops_causal_crosswalk)
    fields: dict[str, str] = Field(default_factory=dict)  # field name -> xsd:* curie


class ResourceSpec(BaseModel):
    """One connector resource (a fact-sheet type / table / record kind)."""

    model_config = ConfigDict(extra="forbid")

    name: str  # OWL class local name
    label: str = ""
    id_prefix: str = ""
    relations: list[ResourceRelation] = Field(default_factory=list)


class ActionSpec(BaseModel):
    """One connector action/capability (from the connector's ``a2a.json``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    description: str = ""


class EventSpec(BaseModel):
    """One connector-emitted event (typically an ``updated``/``created`` watermark)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    resource: str = ""
    description: str = ""


class IdentitySpec(BaseModel):
    """Per-resource record identity fields (id/title/text/updated), keyed by resource."""

    model_config = ConfigDict(extra="forbid")

    id_field: dict[str, str] = Field(default_factory=dict)
    title_field: dict[str, str] = Field(default_factory=dict)
    text_field: dict[str, str] = Field(default_factory=dict)
    updated_field: dict[str, str] = Field(default_factory=dict)


class PermissionsSpec(BaseModel):
    """ACL/tenant-scoping seam (feeds :class:`ExternalAccess` sync, AU-KG.ingest.mcp-tool-connector)."""

    model_config = ConfigDict(extra="forbid")

    acl_fields: list[str] = Field(default_factory=list)
    tenant_field: str | None = None
    read_roles: list[str] = Field(default_factory=list)


class SyncSpec(BaseModel):
    """One ``mcp_tool`` source preset (:data:`MCP_TOOL_PRESETS` shape), verbatim + typed."""

    model_config = ConfigDict(extra="forbid")

    preset: str  # the mcp_source_presets.json key
    server: str
    tool: str
    action: str | None = None
    records_path: str | None = None
    id_field: str | None = None
    title_field: str | None = None
    text_field: str | None = None
    updated_field: str | None = None
    pagination: str | None = None
    doc_type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)  # full preset, passthrough


class IntegrityInfo(BaseModel):
    """The URDNA2015-equivalent canonical-graph hash (:mod:`ontology_integrity`)."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "urdna2015-sha256"
    hash: str
    triple_count: int = 0


class ProvenanceSpec(BaseModel):
    """Who/what generated this manifest, and the signed integrity pin (X6)."""

    model_config = ConfigDict(extra="forbid")

    generated_by: str = "scripts/generate_connector_manifests.py"
    generated_at: str = ""  # ISO-8601 UTC; empty only for hand-authored manifests
    source_artifacts: list[str] = Field(default_factory=list)
    integrity: IntegrityInfo
    signer: str | None = None
    signature: str | None = None


class PolicySpec(BaseModel):
    """RLS/ABAC + PII residue — always reviewable, never auto-enforced."""

    model_config = ConfigDict(extra="forbid")

    pii_fields: dict[str, list[str]] = Field(default_factory=dict)  # resource -> fields
    tenant_boundary: str | None = None
    rls: list[str] = Field(default_factory=list)


class ConnectorManifest(BaseModel):
    """The full Connector Ontology Manifest — ``agents/<pkg>/connector_manifest.yml``."""

    model_config = ConfigDict(extra="forbid")

    connector: str
    ontology_source: str = (
        ""  # IRI/ttl-naming slug (e.g. "servicenow" for servicenow-api);
    )
    # empty means "same as connector". Distinct because a connector's package name
    # (agents/<pkg>) and its ontology domain slug (the existing ttl's own declared
    # owl:Ontology IRI local name) commonly differ.
    schema_version: str = "1"
    resources: list[ResourceSpec] = Field(default_factory=list)
    actions: list[ActionSpec] = Field(default_factory=list)
    events: list[EventSpec] = Field(default_factory=list)
    identity: IdentitySpec = Field(default_factory=IdentitySpec)
    permissions: PermissionsSpec = Field(default_factory=PermissionsSpec)
    schema_mappings: dict[str, SchemaMapping] = Field(
        default_factory=dict
    )  # keyed by resource
    sync: list[SyncSpec] = Field(default_factory=list)
    provenance: ProvenanceSpec
    policy: PolicySpec = Field(default_factory=PolicySpec)
    review_todos: list[str] = Field(default_factory=list)

    @property
    def resolved_ontology_source(self) -> str:
        """The ontology IRI/ttl-naming slug: ``ontology_source`` if set, else ``connector``."""
        return self.ontology_source or self.connector
