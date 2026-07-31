"""Shared fixture builder for domain-pack tests (not itself a test module).

Builds a small but real "runbooks" domain pack: a ``Runbook`` table-row
resource crosswalked onto the canonical ``Document``/``Person`` classes, with
frontmatter (status/owner), a table (step/assignee), and a link (references)
mapping — one worked example exercising all four wired rule kinds.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_utilities.knowledge_graph.domain_packs.domain_pack import (
    ColumnMapping,
    DomainPackManifest,
    DomainPackProvenance,
    EvaluationCase,
    FrontmatterMapping,
    LinkMapping,
    TableMapping,
)
from agent_utilities.knowledge_graph.domain_packs.pack_loader import pack_integrity_hash
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
    SchemaMapping,
)

RUNBOOK_MAPPINGS = [
    FrontmatterMapping(
        key="status",
        node_type="Document",
        id_template="{artifact_id}",
        produce="property",
        property="status",
    ),
    FrontmatterMapping(
        key="owner",
        node_type="Document",
        id_template="{artifact_id}",
        produce="edge",
        relation="ownedBy",
        edge_target_type="Person",
        edge_target_id_template="person:{value}",
    ),
    TableMapping(
        heading_path="Steps",
        row_node_type="Runbook",
        row_id_template="{artifact_id}#row:{row_index}",
        columns={
            "step": ColumnMapping(produce="property", property="name"),
            "assignee": ColumnMapping(
                produce="edge",
                relation="assignedTo",
                edge_target_type="Person",
                edge_target_id_template="person:{value}",
            ),
        },
    ),
    LinkMapping(relation="references", target_node_type="Document"),
]


def _synthetic_evaluation_case() -> EvaluationCase:
    artifact_id = "md:test123"
    artifact = {
        "artifact_id": artifact_id,
        "source_path": "runbook.md",
        "content_hash": "0" * 64,
    }
    fragments = [
        {
            "fragment_id": f"{artifact_id}#frontmatter:status",
            "artifact_id": artifact_id,
            "fragment_type": "frontmatter_key",
            "locator": "frontmatter.status",
            "content_hash": "1" * 64,
            "value": "active",
            "metadata": {"key": "status"},
        },
        {
            "fragment_id": f"{artifact_id}#frontmatter:owner",
            "artifact_id": artifact_id,
            "fragment_type": "frontmatter_key",
            "locator": "frontmatter.owner",
            "content_hash": "2" * 64,
            "value": "alice",
            "metadata": {"key": "owner"},
        },
        {
            "fragment_id": f"{artifact_id}#row:0",
            "artifact_id": artifact_id,
            "fragment_type": "table_row",
            "locator": "table[Steps].row[0]",
            "content_hash": "3" * 64,
            "value": {"step": "Provision VM", "assignee": "bob"},
            "metadata": {"heading_path": "Steps", "row_index": 0},
        },
        {
            "fragment_id": f"{artifact_id}#link:0",
            "artifact_id": artifact_id,
            "fragment_type": "link",
            "locator": "link[0]",
            "content_hash": "4" * 64,
            "value": {"text": "reference", "href": "other.md"},
            "metadata": {"text": "reference", "href": "other.md"},
        },
    ]
    row_id = f"{artifact_id}#row:0"
    expect_entities = [
        # The status-property rule and the owner-edge rule both touch the
        # SAME artifact-level Document node; the DSL merges same-id entities
        # into one (ingest_graph_slice requires unique auxiliary node ids).
        {"id": artifact_id, "node_type": "Document", "status": "active"},
        {"id": "person:alice", "node_type": "Person"},
        {"id": "person:bob", "node_type": "Person"},
        {"id": row_id, "node_type": "Runbook", "name": "Provision VM"},
        {"id": "other.md", "node_type": "Document"},
    ]
    expect_relationships = [
        {"source": artifact_id, "target": "person:alice", "relationship": "ownedBy"},
        {"source": row_id, "target": "person:bob", "relationship": "assignedTo"},
        {"source": artifact_id, "target": "other.md", "relationship": "references"},
    ]
    return EvaluationCase(
        name="single-runbook",
        artifact=artifact,
        fragments=fragments,
        expect_entities=expect_entities,
        expect_relationships=expect_relationships,
    )


def _ontology_extension(pack_name: str) -> ConnectorManifest:
    return ConnectorManifest(
        connector=pack_name,
        resources=[
            ResourceSpec(name="Runbook", label="Runbook Step", id_prefix="runbook")
        ],
        schema_mappings={
            "Runbook": SchemaMapping(
                ontology_class="Document",
                fields={"name": "xsd:string"},
            )
        },
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )


def build_manifest(
    pack_name: str = "runbooks",
    *,
    version: str = "1.0.0",
    mappings: list | None = None,
    evaluation_cases: list[EvaluationCase] | None = None,
    default_classification: str = "internal",
) -> DomainPackManifest:
    """A real, hash-consistent :class:`DomainPackManifest` fixture."""
    ontology = _ontology_extension(pack_name)
    manifest = DomainPackManifest(
        pack=pack_name,
        version=version,
        description="Runbook static markdown KG domain pack (test fixture).",
        ontology=ontology,
        mappings=RUNBOOK_MAPPINGS if mappings is None else mappings,
        evaluation_cases=(
            [_synthetic_evaluation_case()]
            if evaluation_cases is None
            else evaluation_cases
        ),
        default_classification=default_classification,
        provenance=DomainPackProvenance(integrity=IntegrityInfo(hash="0" * 64)),
    )
    digest = pack_integrity_hash(manifest)
    return manifest.model_copy(
        update={
            "provenance": manifest.provenance.model_copy(
                update={"integrity": IntegrityInfo(hash=digest)}
            )
        }
    )


def write_pack(root: Path, manifest: DomainPackManifest) -> Path:
    """Write ``manifest`` to ``<root>/<pack>/domain_pack.yml`` and return the pack dir."""
    pack_dir = root / manifest.pack
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "domain_pack.yml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return pack_dir
