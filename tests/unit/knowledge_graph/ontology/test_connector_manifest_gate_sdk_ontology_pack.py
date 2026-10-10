"""AU-BOUNDARY-R030: the compile-before-sync gate compiles through the SDK.

Drives the real gate entry point to show the live-path recompile-and-hash
through ``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology``
still succeeds end to end.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
)


def _sample_manifest() -> ConnectorManifest:
    return ConnectorManifest(
        connector="widget",
        resources=[
            ResourceSpec(
                name="WidgetOrder",
                id_prefix="order",
                relations=[ResourceRelation(name="placedBy", target="Customer")],
            ),
            ResourceSpec(name="Customer"),
        ],
        schema_mappings={
            "WidgetOrder": SchemaMapping(fields={"status": "xsd:string"}),
        },
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )


def test_gate_compiled_manifest_graph_uses_the_sdk_compiler():
    manifest = _sample_manifest()

    g, violations = gate._compiled_manifest_graph(manifest, "widget")

    assert violations == []
    assert g is not None
    assert len(g) > 0
