"""Tests for the ``source_sync`` compile-before-sync gate (D17, C5 fleet rollout).

Builds a throwaway ``agents/<pkg>/connector_manifest.yml`` on disk and proves the
gate: (1) finds it via the alias/suffix connector-package resolver, (2) passes a
clean certified manifest through, (3) fails closed on a tampered/hand-edited one,
and (4) rejects a source with no owned connector package.
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
    SchemaMapping,
    SyncSpec,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
)
from agent_utilities.knowledge_graph.ontology.ontology_integrity import (
    DEFAULT_SIGNER_ID,
    ReleaseSigner,
    canonical_hash,
    canonical_manifest_hash,
)

_WIDGET_SCHEMA_SHA256 = "1" * 64


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch):
    private_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", private_key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    signer = ReleaseSigner.from_runtime()
    monkeypatch.setenv("ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS", signer.public_key)


def _write_clean_manifest(root: Path, pkg: str) -> Path:
    """A manifest whose ``provenance.integrity.hash`` genuinely matches its compile."""
    (root / pkg).mkdir(parents=True)
    manifest = ConnectorManifest(
        connector=pkg,
        resources=[ResourceSpec(name="Widget", id_prefix="widget")],
        schema_mappings={"Widget": SchemaMapping(fields={"name": "xsd:string"})},
        sync=[
            SyncSpec(
                preset="widget-records",
                server="widget-mcp",
                tool="widget_records",
                action="list",
                records_path="items",
                id_field="id",
                text_field="summary",
                tool_schema_sha256=_WIDGET_SCHEMA_SHA256,
                raw={
                    "server": "widget-mcp",
                    "tool": "widget_records",
                    "action": "list",
                    "records_path": "items",
                    "id_field": "id",
                    "text_field": "summary",
                },
            )
        ],
        provenance=ProvenanceSpec(
            integrity=IntegrityInfo(hash="0" * 64), signer=DEFAULT_SIGNER_ID
        ),
    )
    spec = compile_manifest(manifest)
    ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
    import rdflib

    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, n = canonical_hash(g)
    signer = ReleaseSigner.from_runtime()
    unsigned = manifest.model_copy(
        update={
            "provenance": ProvenanceSpec(
                integrity=IntegrityInfo(hash=digest, triple_count=n),
                signer=signer.signer_id,
                signature_algorithm=signer.algorithm,
                signing_public_key=signer.public_key,
            )
        }
    )
    manifest = unsigned.model_copy(
        update={
            "provenance": unsigned.provenance.model_copy(
                update={"signature": signer.sign(canonical_manifest_hash(unsigned))}
            )
        }
    )

    import yaml

    path = root / pkg / "connector_manifest.yml"
    path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _install_widget_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_presets",
        lambda provider: (
            {
                "widget-records": {
                    "server": "widget-mcp",
                    "tool": "widget_records",
                    "action": "list",
                    "records_path": "items",
                    "id_field": "id",
                    "text_field": "summary",
                }
            }
            if provider == "widget-mcp"
            else None
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_schema_fingerprints",
        lambda provider: (
            {"widget_records": _WIDGET_SCHEMA_SHA256}
            if provider == "widget-mcp"
            else None
        ),
    )


def test_resolve_connector_package_via_alias(tmp_path: Path):
    (tmp_path / "atlassian-agent").mkdir()
    assert (
        gate.resolve_connector_package("jira", agents_root=tmp_path)
        == "atlassian-agent"
    )
    assert (
        gate.resolve_connector_package("confluence", agents_root=tmp_path)
        == "atlassian-agent"
    )


def test_direct_external_graph_uses_one_current_native_activation_contract() -> None:
    assert (
        gate.resolve_connector_package(
            "external_graph", agents_root=gate.bundled_manifests_root()
        )
        == "native-source-connectors"
    )
    assert gate.native_activation_contract("external_graph") == (
        "external_graph_ingestion_v1",
        "agent_utilities.knowledge_graph.ingestion.external_graph",
    )


def test_native_fingerprint_binds_external_graph_governance_closure() -> None:
    modules = set(gate.native_activation_fingerprint_modules("external_graph"))

    assert {
        "agent_utilities.core.config",
        "agent_utilities.knowledge_graph.core.connection_registry",
        "agent_utilities.knowledge_graph.ingestion.change_envelope",
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest",
        "agent_utilities.knowledge_graph.ingestion.external_graph",
        "agent_utilities.knowledge_graph.ingestion.external_graph_schema",
        "agent_utilities.security.persistence_privacy",
        "agent_utilities.security.secrets_client",
    } <= modules
    assert all("/" not in module and "\\" not in module for module in modules)


def test_native_fingerprint_changes_for_transitive_dependency_tampering(
    tmp_path: Path,
) -> None:
    package = tmp_path / "fixturepkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    activation = package / "activation.py"
    activation.write_text(
        "from fixturepkg import dependency\n"
        "try:\n"
        "    import fixturepkg.optional\n"
        "except ImportError:\n"
        "    pass\n",
        encoding="utf-8",
    )
    dependency = package / "dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    activation_bytes = activation.read_bytes()

    first, first_modules = gate._local_module_closure_fingerprint(
        ("fixturepkg.activation",),
        package_name="fixturepkg",
        package_root=package,
    )
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    second, second_modules = gate._local_module_closure_fingerprint(
        ("fixturepkg.activation",),
        package_name="fixturepkg",
        package_root=package,
    )

    assert activation.read_bytes() == activation_bytes
    assert first != second
    assert first_modules == second_modules
    assert "fixturepkg.dependency" in second_modules

    (package / "optional.py").write_text("ENABLED = True\n", encoding="utf-8")
    third, third_modules = gate._local_module_closure_fingerprint(
        ("fixturepkg.activation",),
        package_name="fixturepkg",
        package_root=package,
    )
    assert third != second
    assert "fixturepkg.optional" in third_modules


def test_resolve_connector_package_via_suffix_guess(tmp_path: Path):
    (tmp_path / "widget-mcp").mkdir()
    assert (
        gate.resolve_connector_package("widget", agents_root=tmp_path) == "widget-mcp"
    )


def test_resolve_connector_package_none_when_no_match(tmp_path: Path):
    assert (
        gate.resolve_connector_package("totally-unknown-source", agents_root=tmp_path)
        is None
    )


def test_precheck_source_fails_closed_when_no_manifest(tmp_path: Path):
    result = gate.precheck_source("no-such-source", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is False
    assert any("[missing]" in violation for violation in result["violations"])


def test_precheck_source_ok_on_clean_manifest(tmp_path: Path, monkeypatch):
    _write_clean_manifest(tmp_path, "widget-mcp")
    _install_widget_provider(monkeypatch)
    result = gate.precheck_source("widget", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is True
    assert result["connector"] == "widget-mcp"
    assert result["violations"] == []


def test_precheck_source_fails_closed_on_tampered_manifest(tmp_path: Path):
    path = _write_clean_manifest(tmp_path, "widget-mcp")
    # Hand-edit the on-disk manifest post-signing (add a resource without recompiling).
    text = path.read_text(encoding="utf-8")
    tampered = text.replace("name: Widget", "name: TamperedWidget", 1)
    assert tampered != text
    path.write_text(tampered, encoding="utf-8")

    result = gate.precheck_source("widget", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is False
    assert any("[integrity]" in v for v in result["violations"])


def test_complete_manifest_signature_blocks_sync_preset_tampering(tmp_path: Path):
    path = _write_clean_manifest(tmp_path, "widget-mcp")
    text = path.read_text(encoding="utf-8")
    tampered = text.replace("tool: widget_records", "tool: renamed_records", 1)
    assert tampered != text
    path.write_text(tampered, encoding="utf-8")

    violations = gate.check_manifest_bytes(path, require_signature=True)

    assert not any("[integrity]" in violation for violation in violations)
    assert any("[signature]" in violation for violation in violations)


def test_required_signature_rejects_unsigned_manifest(tmp_path: Path):
    import yaml

    path = _write_clean_manifest(tmp_path, "widget-mcp")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["provenance"]["signature"] = None
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    violations = gate.check_manifest_bytes(path, require_signature=True)

    assert any("manifest is unsigned" in violation for violation in violations)


def test_sync_source_refuses_on_failed_gate(tmp_path: Path, monkeypatch):
    """The ``sync_source`` entrypoint itself refuses to dispatch on a failed gate."""
    from agent_utilities.knowledge_graph.core import source_sync

    path = _write_clean_manifest(tmp_path, "widget-mcp")
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("name: Widget", "name: TamperedWidget", 1), encoding="utf-8"
    )

    real_precheck = gate.precheck_source

    def fake_precheck(source, *, agents_root=None):
        return real_precheck(source, agents_root=tmp_path)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        fake_precheck,
    )

    called = {"dispatched": False}

    def fake_dispatch(engine, source, *, mode="delta", ids=None, client=None):
        called["dispatched"] = True
        return {"status": "ok"}

    monkeypatch.setattr(source_sync, "_dispatch_sync_source", fake_dispatch)

    out = source_sync.sync_source(object(), "widget", mode="delta")
    assert out["status"] == "error"
    assert called["dispatched"] is False


def test_manifest_required_empty_by_default():
    """Every bundled ecosystem provider is mandatory without an opt-in."""
    expected = set(gate.MANDATORY_NAMED_CONNECTOR_SOURCES) | set(
        gate.mandatory_connector_packages()
    )
    assert gate.required_connector_sources() == expected
    assert gate.manifest_required("servicenow") is True
    assert gate.manifest_required("jira") is True
    assert gate.manifest_required("archivebox") is True
    assert gate.manifest_required("plane-agent") is True
    # Unknown sources are governed too; they must be onboarded before use.
    assert gate.manifest_required("unbundled-development-source") is True
    assert gate.manifest_required("") is False


# ── rdflib packaging gap (kg-exhaustive-smoke.md): the compile-before-sync
# gate needs rdflib (deliberately excluded from the lean `serving` plane,
# CONCEPT:AU-KG.ontology.connector-manifest-gate — it lives only in the `[owl]`
# extra) to parse/hash the
# compiled manifest. When it's absent this must degrade to ONE clear,
# actionable violation line, not a bare ModuleNotFoundError bubbling up as a
# generic "manifest does not compile cleanly" message. ──────────────────────


def test_check_manifest_bytes_degrades_clearly_when_rdflib_missing(
    tmp_path: Path, monkeypatch
):
    path = _write_clean_manifest(tmp_path, "widget-mcp")

    # Force `import rdflib` to raise, without needing to actually uninstall it
    # from the dev/test environment (which has it via the `[owl]`/`[test]` extras).
    monkeypatch.setitem(__import__("sys").modules, "rdflib", None)

    violations = gate.check_manifest_bytes(path)

    assert len(violations) == 1
    assert violations[0].startswith("[dependency]")
    assert "owl" in violations[0]
    assert "pip install" in violations[0]
    # must NOT be conflated with a generic compile failure
    assert "does not compile cleanly" not in violations[0]


def test_manifest_required_inventory_is_bundled_and_universal():
    assert gate.required_connector_sources() == set(
        gate.MANDATORY_NAMED_CONNECTOR_SOURCES
    ) | set(gate.mandatory_connector_packages())
    assert gate.manifest_required("archivebox") is True
    assert gate.manifest_required("ARCHIVEBOX") is True
    assert gate.manifest_required("twenty") is True
    # the AU-P1-6 mandatory 12 are required regardless of the env var.
    assert gate.manifest_required("servicenow") is True
    assert gate.manifest_required("jira") is True
    assert gate.manifest_required("plane") is True
    assert gate.manifest_required("unbundled-development-source") is True


def test_mandatory_named_connector_sources_is_the_au_p1_6_twelve():
    """The set covers 12 packages through 13 source/package identifiers (by their

    resolvable ``sync_source``/package identifiers) — atlassian contributes two
    (jira, confluence).
    """
    assert gate.MANDATORY_NAMED_CONNECTOR_SOURCES == frozenset(
        {
            "jira",
            "confluence",
            "gitlab",
            "servicenow",
            "leanix",
            "langfuse",
            "tunnel_manager",
            "microsoft-agent",
            "container-manager-mcp",
            "documentdb-mcp",
            "repository-manager",
            "systems-manager",
            "vector-mcp",
        }
    )


def test_precheck_source_fails_closed_when_manifest_missing(tmp_path: Path):
    """A source with NO manifest fails closed (CONCEPT:AU-P0-4).

    This returns a checked+not-ok result so ``sync_source`` refuses the sync
    exactly like a tampered manifest — "unknown" never means "allowed".
    """
    result = gate.precheck_source("widget", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is False
    assert any("[missing]" in v for v in result["violations"])


def test_precheck_source_rejects_non_catalog_source_missing_manifest(
    tmp_path: Path, monkeypatch
):
    """An unknown source is not less governed than a catalog source."""
    result = gate.precheck_source("some-other-unrelated-source", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is False
    assert any("[missing]" in violation for violation in result["violations"])


def test_precheck_source_ok_when_manifest_present(tmp_path: Path, monkeypatch):
    """Once a manifest is provided, it compiles and verifies normally."""
    _write_clean_manifest(tmp_path, "widget-mcp")
    _install_widget_provider(monkeypatch)
    result = gate.precheck_source("widget", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is True


def test_manifest_gate_rejects_provider_preset_drift(tmp_path: Path, monkeypatch):
    _write_clean_manifest(tmp_path, "widget-mcp")
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_presets",
        lambda _provider: {
            "widget-records": {
                "server": "widget-mcp",
                "tool": "renamed_widget_records",
            }
        },
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_schema_fingerprints",
        lambda _provider: {"widget_records": _WIDGET_SCHEMA_SHA256},
    )

    result = gate.precheck_source("widget", agents_root=tmp_path)

    assert result["ok"] is False
    assert any("[tool-schema]" in violation for violation in result["violations"])


def test_manifest_gate_rejects_missing_live_schema_certification(
    tmp_path: Path, monkeypatch
):
    _write_clean_manifest(tmp_path, "widget-mcp")
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_presets",
        lambda _provider: {
            "widget-records": {
                "server": "widget-mcp",
                "tool": "widget_records",
                "action": "list",
                "records_path": "items",
                "id_field": "id",
                "text_field": "summary",
            }
        },
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool.provider_tool_schema_fingerprints",
        lambda _provider: None,
    )

    result = gate.precheck_source("widget", agents_root=tmp_path)

    assert result["ok"] is False
    assert any("tool_schema_fingerprints.json" in item for item in result["violations"])


def test_sync_source_fails_closed_when_precheck_raises(monkeypatch):
    from agent_utilities.knowledge_graph.core import source_sync

    def broken_precheck(_source):
        raise RuntimeError("gate unavailable")

    monkeypatch.setattr(gate, "precheck_source", broken_precheck)
    called = False

    def fake_dispatch(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"status": "ok"}

    monkeypatch.setattr(source_sync, "_dispatch_sync_source", fake_dispatch)
    result = source_sync.sync_source(object(), "widget")

    assert result["status"] == "error"
    assert "RuntimeError" in result["reason"]
    assert called is False


def test_sync_source_refuses_source_with_no_manifest(monkeypatch):
    """The ``sync_source`` entrypoint refuses every source lacking a manifest."""
    from agent_utilities.knowledge_graph.core import source_sync

    called = {"dispatched": False}

    def fake_dispatch(engine, source, *, mode="delta", ids=None, client=None):
        called["dispatched"] = True
        return {"status": "ok"}

    monkeypatch.setattr(source_sync, "_dispatch_sync_source", fake_dispatch)

    out = source_sync.sync_source(object(), "totally-ungated-source", mode="delta")
    assert out["status"] == "error"
    assert called["dispatched"] is False


def test_sync_source_refuses_unchecked_gate_result(monkeypatch):
    """A malformed/legacy unchecked result cannot bypass the gate."""
    from agent_utilities.knowledge_graph.core import source_sync

    def fake_precheck(source, *, agents_root=None):
        return {"checked": False}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        fake_precheck,
    )

    def fake_dispatch(engine, source, *, mode="delta", ids=None, client=None):
        return {"status": "ok", "source": source}

    monkeypatch.setattr(source_sync, "_dispatch_sync_source", fake_dispatch)

    out = source_sync.sync_source(object(), "leanix", mode="delta")
    assert out["status"] == "error"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
