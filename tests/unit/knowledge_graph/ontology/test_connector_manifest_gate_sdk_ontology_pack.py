"""AU-BOUNDARY-R030: the compile-before-sync gate compiles through the SDK.

Proves equivalence between AU's hand-written Turtle emitter
(``agent_utilities.knowledge_graph.ontology.manifest_compiler``, still the
copy other AU callers use) and the SDK's typed replacement
(``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology``,
now what :func:`connector_manifest_gate._compiled_manifest_graph` calls)
before the AU copy is deleted, then drives the real gate entry point to show
the live-path recompile-and-hash still succeeds end to end.
"""

from __future__ import annotations

import pytest
from agent_connector_sdk.manifest.model import ConnectorManifest as SDKManifest
from agent_connector_sdk.manifest.ontology_pack import compile_manifest_ontology

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
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


@pytest.mark.spec("AU-SEC-R005")
def test_sdk_ontology_pack_reproduces_aus_turtle_byte_for_byte():
    manifest = _sample_manifest()

    au_spec = compile_manifest(manifest)
    au_ttl = export_manifest_ttl(au_spec, source=manifest.resolved_ontology_source)

    sdk_manifest = SDKManifest.model_validate(manifest.model_dump(mode="python"))
    sdk_ttl = compile_manifest_ontology(sdk_manifest)

    assert sdk_ttl == au_ttl


def test_gate_compiled_manifest_graph_uses_the_sdk_compiler():
    manifest = _sample_manifest()

    g, violations = gate._compiled_manifest_graph(manifest, "widget")

    assert violations == []
    assert g is not None
    assert len(g) > 0
