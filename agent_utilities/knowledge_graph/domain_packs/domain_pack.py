"""Domain Pack manifest schema (CONCEPT:AU-KG.ingest.domain-pack-framework).

A **domain pack** is a versioned, installable/removable package that turns a
predictable-structure corpus (a static markdown knowledge graph, an exported
API dump, ...) into graph facts **without any custom code**: it bundles an
ontology extension, SHACL shapes, an extraction/fragment schema, declarative
mapping rules (the DSL, :mod:`.dsl`), and evaluation cases in ONE on-disk
``domain_pack.yml``.

**Surveyed first: is ``SchemaPack`` (``models/schema_pack.py``) this
primitive already?** No — it solves a different problem, and this module is
deliberately NOT a rival:

* :class:`~agent_utilities.models.schema_pack.SchemaPack` activates a SUBSET
  of the already-fixed, hardcoded ``RegistryNodeType``/``RegistryEdgeType``
  ``StrEnum`` for retrieval tuning (boosts, recency decay, source trust,
  autocut, zero-LLM ``link_inference`` regex-over-raw-text, relational-verb NL
  parsing, OWL closure declarations) — a **lens over what the engine already
  knows**, resolved to exactly ONE process-wide active pack at a time
  (``schema_pack_loader.get_active_pack``/``set_active_pack`` — a singleton
  switch, not coexistence). It has no version field, no on-disk signed
  artifact, no install/remove lifecycle, and cannot declare a genuinely new
  domain class — you cannot add "Runbook" to it without editing the Python
  enum in code.
* This module is the opposite axis: **how a genuinely new, corpus-specific
  class vocabulary gets INTO the graph in the first place**, as a real
  on-disk, hash-verified, semver-versioned artifact that several packs mount
  **simultaneously** (:mod:`.pack_loader`'s ``DomainPackRegistry`` — see that
  module for why this mirrors the fleet's existing "many domain
  ``ontology_*.ttl`` modules, one engine authority" pattern) — exactly the
  operator's ask ("several static KGs mounted at once, epistemic-graph
  authoritative across all of them"), which ``SchemaPack``'s singleton model
  structurally cannot provide.
* They compose, not compete: once a domain pack's facts are ingested, an
  operator who wants retrieval tuning over that new vocabulary still reaches
  for ``SchemaPack`` — declaring one is an orthogonal, later step, not
  something this module needs to reimplement.
* **Why not just add ``version``/install-remove/multi-active to
  ``SchemaPack`` instead of a new format?** Because its foundational
  constraint isn't a missing field, it's the type system underneath it:
  ``SchemaPack.node_types``/``edge_types`` are ``set[RegistryNodeType]``/
  ``set[RegistryEdgeType]`` — Python ``StrEnum`` members fixed at import
  time. A corpus-specific class ("Runbook") cannot become a member of that
  enum without editing and shipping a new ``agent-utilities`` release; a
  domain pack, by construction, must be addable and removable by an
  operator at RUNTIME, with no code change and no redeploy. That is an
  incompatible foundation, not a feature gap ``SchemaPack`` could grow into
  — extending it would mean replacing its enum-typed fields with
  arbitrary-string ones, which breaks every existing ``SchemaPack`` caller
  that pattern-matches on those enums (the engine's ``HybridRetriever``,
  the entity extractor, ``owl_bridge``). A new, additive format was the
  smaller, non-breaking change.
* **Entity-identity is deliberately NOT re-declared here.** A sibling
  universal-ingestion lane (Track 5, entity resolution) added
  :class:`~agent_utilities.models.schema_pack.IdentityRule` to
  ``SchemaPack`` — resolution-time "are these two already-extracted records
  the SAME real-world entity" rules, wired into
  ``knowledge_graph.assimilation.identity_candidates.resolve_identity_candidates``.
  That is a different question from this module's per-rule ``id_template``
  (extraction-time: "what deterministic id string does THIS fragment's fact
  get") — but the name is the same, so a domain pack wanting
  resolution-grade identity rules declares them on its own ``SchemaPack``,
  never on ``DomainPackManifest``. (There is deliberately no
  ``DomainPackManifest.identity`` field — each mapping rule's own
  ``id_template`` already IS this pack's id-derivation strategy.)

**Reused, not reinvented** (the operator's own instruction: *study the
connector-manifest format before designing a new one*):

* :class:`ConnectorManifest`'s ``resources``/``schema_mappings``/
  ``provenance`` shape IS this pack's ``ontology`` field, verbatim — a domain
  pack's ontology extension is compiled, hashed, and anti-sprawl-checked by
  the exact same :func:`manifest_compiler.compile_manifest` /
  :func:`manifest_compiler.export_manifest_ttl` / :func:`manifest_compiler.is_wired`
  / :class:`manifest_compiler.AntiSprawlError` pipeline every connector already
  goes through — a pack cannot graft a new OWL class into the canonical
  ontology any more than a connector can; it must map onto an existing
  canonical class (``schema_mappings[resource].ontology_class``) or go through
  the same one-line ``owl:imports`` review any new ontology module requires.
  The connector-only fields (``actions``/``events``/``identity``/
  ``permissions``/``sync``) are simply left at their empty defaults — a domain
  pack is not an API connector and has no sync presets.
* The pack's integrity hash reuses the exact canonical-JSON hashing recipe
  :func:`ontology_integrity.canonical_manifest_hash` established for connector
  manifests (sorted-key, separator-tight JSON -> sha256) via
  :func:`.pack_loader.pack_integrity_hash` — a thin variant, not a new scheme,
  that additionally excludes the pack's own ``integrity.hash`` field (a
  connector's ``integrity.hash`` pins an *external* compiled-ontology artifact,
  never itself, so it doesn't need this; a domain pack's ``integrity.hash``
  pins the whole document, so it must exclude itself to avoid self-reference).
* The pack's ``mappings`` are the generalization of ``MCP_TOOL_PRESETS``
  (``protocols/source_connectors/connectors/mcp_tool.py``): declarative data
  (server+tool+field-map there; fragment-matcher+target here), never a new
  connector module.

**What a pack cannot do** (design constraint, not an implementation detail):
a pack is a pure, inert data document. Nothing in this module calls the graph
engine, writes a node, or grants an ACL. Loading a pack only *validates* it;
running its mappings (:mod:`.dsl`) only *proposes* facts; only an explicit
call to the existing governed ``ingest_graph_slice``/``ingest_envelope``
(:mod:`..ingestion.envelope_ingest`) can ever write them.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..ontology.connector_manifest import ConnectorManifest, IntegrityInfo
from .fragment_contract import FRAGMENT_TYPES

__all__ = [
    "SEMVER_RE",
    "ColumnMapping",
    "FrontmatterMapping",
    "TableMapping",
    "HeadingMapping",
    "LinkMapping",
    "JSONFieldMapping",
    "MappingRule",
    "EvaluationCase",
    "DomainPackProvenance",
    "DomainPackManifest",
]

# Loose but real semver (major.minor.patch, optional pre-release/build) — a
# domain pack MUST be versioned (the operator's own requirement) so a fact can
# be traced to the exact pack revision that produced it via
# ``ChangeEnvelope.ontology_mapping_version``.
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def _check_fragment_type(value: str) -> str:
    if value not in FRAGMENT_TYPES:
        raise ValueError(
            f"fragment_type must be one of {sorted(FRAGMENT_TYPES)}, got {value!r}"
        )
    return value


class ColumnMapping(BaseModel):
    """How one table column projects onto a property or an edge on the row node."""

    model_config = ConfigDict(extra="forbid")

    produce: Literal["property", "edge"]
    property: str | None = None  # datatype property local name (produce=="property")
    relation: str | None = None  # object property local name (produce=="edge")
    edge_target_type: str | None = None  # ontology class of the edge's target
    edge_target_id_template: str | None = (
        None  # e.g. "person:{value}"; "{value}" is the cell's own text
    )

    @model_validator(mode="after")
    def _require_field_matching_produce(self) -> ColumnMapping:
        if self.produce == "property" and not self.property:
            raise ValueError("produce=='property' requires 'property'")
        if self.produce == "edge" and not self.relation:
            raise ValueError("produce=='edge' requires 'relation'")
        return self


class FrontmatterMapping(BaseModel):
    """Matches one exact frontmatter key on ``frontmatter_key`` fragments."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["frontmatter"] = "frontmatter"
    key: str
    node_type: str
    id_template: str = "{artifact_id}"
    produce: Literal["property", "edge"]
    property: str | None = None
    relation: str | None = None
    edge_target_type: str | None = None
    edge_target_id_template: str | None = None


class TableMapping(BaseModel):
    """Matches ``table_row`` fragments, optionally scoped to a heading section."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    heading_path: str | None = None  # restrict to rows under this heading; None = any
    row_node_type: str
    row_id_template: str  # e.g. "{artifact_id}#row:{row_index}"
    columns: dict[str, ColumnMapping] = Field(default_factory=dict)


class HeadingMapping(BaseModel):
    """Matches ``heading_section`` fragments whose heading text matches a regex."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["heading"] = "heading"
    heading_pattern: str
    node_type: str
    id_template: str  # e.g. "{artifact_id}#section:{heading_path}"
    parent_relation: str | None = None
    parent_id_template: str = "{artifact_id}"

    @field_validator("heading_pattern")
    @classmethod
    def _valid_regex(cls, value: str) -> str:
        re.compile(value)
        return value


class LinkMapping(BaseModel):
    """Matches ``link`` fragments whose href matches an (optional) regex."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["link"] = "link"
    href_pattern: str | None = None
    relation: str
    source_id_template: str = "{artifact_id}"
    target_id_template: str = "{href}"
    target_node_type: str | None = None

    @field_validator("href_pattern")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class JSONFieldMapping(BaseModel):
    """Matches ``json_field`` fragments by exact dotted path.

    Generalizes to both bundled JSON documents and API records: both are, at
    the fragment level, one ``{dotted.path: value}`` pair — the same rule kind
    covers "frontmatter JSON block" and "REST API record field" alike.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["json_field"] = "json_field"
    path: str
    node_type: str
    id_template: str = "{artifact_id}"
    produce: Literal["property", "edge"]
    property: str | None = None
    relation: str | None = None
    edge_target_type: str | None = None
    edge_target_id_template: str | None = None


MappingRule = Annotated[
    FrontmatterMapping | TableMapping | HeadingMapping | LinkMapping | JSONFieldMapping,
    Field(discriminator="kind"),
]


class EvaluationCase(BaseModel):
    """A fixture: given these fragments, the mappings MUST produce exactly these facts.

    This is the pack's own self-check (CONCEPT:AU-KG.ingest.mapping-dsl) — the
    load-time proof that the DSL is inspectable ("an operator must be able to
    read a mapping and predict what facts it will produce"). A pack that
    doesn't reproduce its own declared cases is refused at load time (fail
    closed), never silently accepted with a "best effort" mapping.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    artifact: dict[str, Any]  # ArtifactLike-shaped dict
    fragments: list[dict[str, Any]]  # FragmentLike-shaped dicts
    expect_entities: list[dict[str, Any]] = Field(default_factory=list)
    expect_relationships: list[dict[str, Any]] = Field(default_factory=list)


class DomainPackProvenance(BaseModel):
    """Who/what authored this pack, and its integrity pin.

    Reuses :class:`IntegrityInfo` verbatim from the connector-manifest schema.
    Signature verification is OPTIONAL (unlike a connector's supply-chain
    manifest): a domain pack is typically operator-authored for one corpus,
    not a third-party vendor artifact crossing a supply-chain trust boundary —
    but when ``signer``/``signature`` ARE present they are verified with the
    exact same Ed25519 machinery (:mod:`..ontology.ontology_integrity`), so a
    team that wants connector-grade certification for its packs gets it for
    free.
    """

    model_config = ConfigDict(extra="forbid")

    generated_by: str = "hand-authored"
    generated_at: str = ""
    integrity: IntegrityInfo
    signer: str | None = None
    signature_algorithm: str | None = None
    signing_public_key: str | None = None
    signature: str | None = None


class DomainPackManifest(BaseModel):
    """The full on-disk domain pack — ``<pack>/domain_pack.yml``."""

    model_config = ConfigDict(extra="forbid")

    pack: str
    version: str
    description: str = ""
    ontology: ConnectorManifest
    shacl_shapes: list[str] = Field(default_factory=list)  # paths relative to pack dir
    mappings: list[MappingRule] = Field(default_factory=list)
    evaluation_cases: list[EvaluationCase] = Field(default_factory=list)
    default_classification: str = "internal"
    provenance: DomainPackProvenance

    @field_validator("version")
    @classmethod
    def _valid_semver(cls, value: str) -> str:
        if not SEMVER_RE.match(value):
            raise ValueError(f"version must be semver (X.Y.Z), got {value!r}")
        return value

    @field_validator("default_classification")
    @classmethod
    def _valid_classification(cls, value: str) -> str:
        # Fail closed at LOAD time on a typo'd classification, rather than at
        # write time inside ingest_graph_slice's DataClassification(...) call.
        from ...models.company_brain import DataClassification

        try:
            DataClassification(value)
        except ValueError as exc:
            valid = sorted(c.value for c in DataClassification)
            raise ValueError(
                f"default_classification must be one of {valid}, got {value!r}"
            ) from exc
        return value

    @property
    def mapping_version(self) -> str:
        """The exact string stamped on every produced fact's
        ``ChangeEnvelope.ontology_mapping_version`` — traces a fact to the pack
        revision that produced it."""
        return f"{self.pack}@{self.version}"

    @property
    def connector_id(self) -> str:
        """The ``ChangeEnvelope.connector`` identity for this pack's facts."""
        return f"domain-pack:{self.pack}"
