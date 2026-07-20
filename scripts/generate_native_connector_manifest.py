#!/usr/bin/env python3
"""Generate the signed capability manifest for Agent Utilities native sources.

The native source connectors are Python implementations shipped inside Agent
Utilities rather than separately versioned provider repositories.  This
generator binds their activation contract to a deterministic closure of critical
local implementation modules and writes both the fingerprint sidecar and the
Ed25519-signed manifest. It is deterministic, offline, and requires the same
explicit release signer as the external connector generators.

Usage:
  python3 scripts/generate_native_connector_manifest.py [--output-dir PATH]
      [--now 2026-07-14T00:00:00Z]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    ConnectorManifest,
    IdentitySpec,
    IntegrityInfo,
    PermissionsSpec,
    PolicySpec,
    ProvenanceSpec,
    ResourceSpec,
    SchemaMapping,
    SyncSpec,
)
from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (  # noqa: E402
    NATIVE_FINGERPRINT_FORMAT,
    SOURCE_TO_CONNECTOR_PACKAGE,
    native_activation_contract,
    native_activation_fingerprints,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    compile_manifest,
    export_manifest_ttl,
)

CONNECTOR = "native-source-connectors"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "agent_utilities"
    / "knowledge_graph"
    / "ontology"
    / "connector_manifests"
    / CONNECTOR
)


def native_source_types() -> tuple[str, ...]:
    """Return the governed in-package source inventory from the runtime aliases."""

    return tuple(
        sorted(
            source
            for source, package in SOURCE_TO_CONNECTOR_PACKAGE.items()
            if package == CONNECTOR
        )
    )


def source_fingerprints() -> dict[str, str]:
    """Hash every native source's critical local-module dependency closure."""

    return native_activation_fingerprints(native_source_types())


def _placeholder(fingerprints: dict[str, str]) -> ConnectorManifest:
    resource = "SourceDocument"
    return ConnectorManifest(
        connector=CONNECTOR,
        ontology_source="native_source_connector",
        resources=[
            ResourceSpec(
                name=resource,
                label="Governed source document",
                id_prefix="source-document",
            )
        ],
        identity=IdentitySpec(
            id_field={resource: "id"},
            title_field={resource: "title"},
            text_field={resource: "text"},
            updated_field={resource: "updated_at"},
        ),
        permissions=PermissionsSpec(
            acl_fields=["external_access"], tenant_field="tenant_id"
        ),
        schema_mappings={
            resource: SchemaMapping(
                ontology_class="Document",
                fields={
                    "title": "xsd:string",
                    "text": "xsd:string",
                    "updated_at": "xsd:dateTime",
                },
            )
        },
        sync=[
            _sync_spec(source_type, fingerprints[source_type])
            for source_type in sorted(fingerprints)
        ],
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
        policy=PolicySpec(
            tenant_boundary="tenant_id",
            rls=["external_access must be enforced before materialization"],
        ),
    )


def _sync_spec(source_type: str, fingerprint: str) -> SyncSpec:
    contract = native_activation_contract(source_type)
    if contract is None:
        raise RuntimeError(f"native source {source_type!r} is unavailable")
    interface, _module_name = contract
    document_source = interface == "source_connector_v1"
    return SyncSpec(
        preset=f"native-{source_type.replace('_', '-')}",
        server="agent-utilities",
        tool=source_type,
        action="load" if document_source else "ingest",
        id_field="id" if document_source else None,
        title_field="title" if document_source else None,
        text_field="text" if document_source else None,
        updated_field="updated_at" if document_source else None,
        doc_type="document" if document_source else None,
        tool_schema_sha256=fingerprint,
        raw={"source_type": source_type, "interface": interface},
    )


def build_manifest(
    *,
    now: datetime | None = None,
    release_signer: ontology_integrity.ReleaseSigner | None = None,
) -> tuple[ConnectorManifest, dict[str, str]]:
    """Build the signed native manifest and its code-fingerprint inventory."""

    fingerprints = source_fingerprints()
    placeholder = _placeholder(fingerprints)
    spec = compile_manifest(placeholder)
    ttl = export_manifest_ttl(spec, source=placeholder.resolved_ontology_source)

    import rdflib

    graph = rdflib.Graph()
    graph.parse(data=ttl, format="turtle")
    digest, triple_count = ontology_integrity.canonical_hash(graph)
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    signer = release_signer or ontology_integrity.ReleaseSigner.from_runtime()
    provenance = ProvenanceSpec(
        generated_by="scripts/generate_native_connector_manifest.py",
        generated_at=stamp,
        source_artifacts=[
            "agent_utilities/protocols/source_connectors/connectors",
            "agent_utilities/knowledge_graph/ingestion/external_graph.py",
            "tool_schema_fingerprints.json",
        ],
        integrity=IntegrityInfo(hash=digest, triple_count=triple_count),
        signer=signer.signer_id,
        signature_algorithm=signer.algorithm,
        signing_public_key=signer.public_key,
        signature=None,
    )
    unsigned = placeholder.model_copy(update={"provenance": provenance})
    manifest_hash = ontology_integrity.canonical_manifest_hash(unsigned)
    signed = unsigned.model_copy(
        update={
            "provenance": provenance.model_copy(
                update={"signature": signer.sign(manifest_hash)}
            )
        }
    )
    return signed, fingerprints


def _manifest_yaml(manifest: ConnectorManifest) -> str:
    import yaml

    return yaml.safe_dump(
        manifest.model_dump(mode="json", exclude_none=False),
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )


def _fingerprints_json(fingerprints: dict[str, str]) -> str:
    return (
        json.dumps(
            {
                "format": NATIVE_FINGERPRINT_FORMAT,
                "sources": dict(sorted(fingerprints.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_bundle(
    output_dir: Path,
    *,
    now: datetime | None = None,
    release_signer: ontology_integrity.ReleaseSigner | None = None,
) -> ConnectorManifest:
    manifest, fingerprints = build_manifest(now=now, release_signer=release_signer)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tool_schema_fingerprints.json").write_text(
        _fingerprints_json(fingerprints), encoding="utf-8"
    )
    (output_dir / "connector_manifest.yml").write_text(
        _manifest_yaml(manifest), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--now", help="ISO-8601 UTC timestamp override")
    args = parser.parse_args()
    now = (
        datetime.strptime(args.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if args.now
        else None
    )
    manifest = write_bundle(args.output_dir, now=now)
    print(
        f"generated {CONNECTOR}: {len(manifest.sync)} native sources, "
        f"{args.output_dir.name}/connector_manifest.yml"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
