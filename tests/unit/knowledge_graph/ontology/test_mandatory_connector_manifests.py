"""Fleet CI contract: every ecosystem provider ships a signed capability manifest.

Each provider-owned ``connector_manifest.yml`` must compile, hash-match, and be
wired. Source-capable packages must also resolve through the same signed
precheck used by runtime ingestion. The complete fleet is bundled into Agent
Utilities so standalone GraphOS deployments retain the same fail-closed floor.
"""

from __future__ import annotations

import pytest
import yaml

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
)

NAMED_CONNECTOR_PACKAGES: tuple[str, ...] = tuple(
    sorted(gate.mandatory_connector_packages())
)


def test_entire_workspace_ecosystem_is_tracked():
    assert len(NAMED_CONNECTOR_PACKAGES) == 62
    assert len(set(NAMED_CONNECTOR_PACKAGES)) == len(NAMED_CONNECTOR_PACKAGES)


@pytest.mark.parametrize("package", NAMED_CONNECTOR_PACKAGES)
def test_named_connector_has_bundled_manifest(package: str):
    path = gate.bundled_manifests_root() / package / "connector_manifest.yml"
    assert path.exists(), (
        f"{package} is an ecosystem provider but has no bundled "
        "connector_manifest.yml; regenerate the capability bundle."
    )


@pytest.mark.parametrize("package", NAMED_CONNECTOR_PACKAGES)
def test_named_connector_manifest_passes_gate(package: str):
    path = gate.bundled_manifests_root() / package / "connector_manifest.yml"
    violations = gate.check_manifest_bytes(path, require_signature=True)
    assert violations == [], f"{package}: {violations}"


def test_all_source_connectors_resolve_through_precheck_source(monkeypatch):
    """Every ``MANDATORY_NAMED_CONNECTOR_SOURCES`` identifier finds ITS bundled

    manifest and passes the gate via the exact same ``precheck_source`` path
    ``sync_source`` uses in production — not just via a direct file check.
    """
    provider_catalog: dict[str, dict[str, dict]] = {}
    schema_catalog: dict[str, dict[str, str]] = {}
    source_packages: list[str] = []
    for package in NAMED_CONNECTOR_PACKAGES:
        path = gate.bundled_manifests_root() / package / "connector_manifest.yml"
        manifest = ConnectorManifest.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        provider_catalog[manifest.connector] = {
            sync.preset: dict(sync.raw) for sync in manifest.sync
        }
        schema_catalog[manifest.connector] = {
            sync.tool: str(sync.tool_schema_sha256) for sync in manifest.sync
        }
        if manifest.sync:
            source_packages.append(package)
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_presets",
        provider_catalog.get,
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_schema_fingerprints",
        schema_catalog.get,
    )

    assert len(source_packages) == 60
    for source in source_packages:
        result = gate.precheck_source(source)
        assert result["checked"] is True, source
        assert result["ok"] is True, (source, result["violations"])


def test_mandatory_precheck_rejects_missing_provider(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_presets",
        lambda _provider: None,
    )

    result = gate.precheck_source("container-manager-mcp")

    assert result["ok"] is False
    assert any("[provider]" in violation for violation in result["violations"])


def test_direct_external_graph_passes_the_signed_native_activation_gate():
    result = gate.precheck_source("external_graph")

    assert result["checked"] is True
    assert result["ok"] is True, result["violations"]
    assert result["connector"] == "native-source-connectors"
    assert result["manifest_path"] == "native-source-connectors/connector_manifest.yml"


def test_mandatory_set_is_a_superset_of_named_connector_packages():
    """Every package in :data:`NAMED_CONNECTOR_PACKAGES` is reachable from at

    least one identifier in :data:`gate.MANDATORY_NAMED_CONNECTOR_SOURCES`.
    """
    root = gate.bundled_manifests_root()
    resolved_packages = {
        gate.resolve_connector_package(source, agents_root=root)
        for source in gate.MANDATORY_NAMED_CONNECTOR_SOURCES
    }
    assert resolved_packages <= set(NAMED_CONNECTOR_PACKAGES)
    assert len(resolved_packages) == 12


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
