"""Tests for the ``source_sync`` compile-before-sync gate (D17, C5 fleet rollout).

Builds a throwaway ``agents/<pkg>/connector_manifest.yml`` on disk and proves the
gate: (1) finds it via the alias/suffix connector-package resolver, (2) passes a
clean manifest through, (3) fails closed on a tampered/hand-edited one, and (4) is
a silent no-op for a source with no connector package at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ontology import connector_manifest_gate as gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceSpec,
    SchemaMapping,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
)
from agent_utilities.knowledge_graph.ontology.ontology_integrity import canonical_hash


def _write_clean_manifest(root: Path, pkg: str) -> Path:
    """A manifest whose ``provenance.integrity.hash`` genuinely matches its compile."""
    (root / pkg).mkdir(parents=True)
    manifest = ConnectorManifest(
        connector=pkg,
        resources=[ResourceSpec(name="Widget", id_prefix="widget")],
        schema_mappings={"Widget": SchemaMapping(fields={"name": "xsd:string"})},
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    spec = compile_manifest(manifest)
    ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
    import rdflib

    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, n = canonical_hash(g)
    manifest = manifest.model_copy(
        update={"provenance": ProvenanceSpec(integrity=IntegrityInfo(hash=digest, triple_count=n))}
    )

    import yaml

    path = root / pkg / "connector_manifest.yml"
    path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_resolve_connector_package_via_alias(tmp_path: Path):
    (tmp_path / "atlassian-agent").mkdir()
    assert gate.resolve_connector_package("jira", agents_root=tmp_path) == "atlassian-agent"
    assert gate.resolve_connector_package("confluence", agents_root=tmp_path) == "atlassian-agent"


def test_resolve_connector_package_via_suffix_guess(tmp_path: Path):
    (tmp_path / "widget-mcp").mkdir()
    assert gate.resolve_connector_package("widget", agents_root=tmp_path) == "widget-mcp"


def test_resolve_connector_package_none_when_no_match(tmp_path: Path):
    assert gate.resolve_connector_package("totally-unknown-source", agents_root=tmp_path) is None


def test_precheck_source_passthrough_when_no_manifest(tmp_path: Path):
    result = gate.precheck_source("no-such-source", agents_root=tmp_path)
    assert result == {"checked": False, "reason": "no connector_manifest.yml for this source"}


def test_precheck_source_ok_on_clean_manifest(tmp_path: Path):
    _write_clean_manifest(tmp_path, "widget-mcp")
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


def test_sync_source_refuses_on_failed_gate(tmp_path: Path, monkeypatch):
    """The ``sync_source`` entrypoint itself refuses to dispatch on a failed gate."""
    from agent_utilities.knowledge_graph.core import source_sync

    path = _write_clean_manifest(tmp_path, "widget-mcp")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("name: Widget", "name: TamperedWidget", 1), encoding="utf-8")

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
    """No ``CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE`` set -> nothing is opted in."""
    assert gate.enterprise_required_sources() == set()
    assert gate.manifest_required("servicenow") is False


def test_manifest_required_reads_allowlist(monkeypatch):
    monkeypatch.setenv("CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE", "ServiceNow, Twenty ,")
    assert gate.enterprise_required_sources() == {"servicenow", "twenty"}
    assert gate.manifest_required("servicenow") is True
    assert gate.manifest_required("SERVICENOW") is True
    assert gate.manifest_required("twenty") is True
    assert gate.manifest_required("jira") is False


def test_precheck_source_fails_closed_when_enterprise_gated_manifest_missing(
    tmp_path: Path, monkeypatch
):
    """A source opted into the enterprise policy with NO manifest -> fail-closed (CONCEPT:AU-P0-4).

    Unlike the default silent pass-through, this returns a checked+not-ok
    result so ``sync_source`` refuses the sync exactly like a tampered
    manifest — "unknown" never silently means "allowed" for a source an
    operator explicitly designated as enterprise-gated.
    """
    monkeypatch.setenv("CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE", "widget")
    result = gate.precheck_source("widget", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is False
    assert any("[missing]" in v for v in result["violations"])


def test_precheck_source_still_passthrough_for_non_enterprise_sources_missing_manifest(
    tmp_path: Path, monkeypatch
):
    """Only the OPTED-IN source is gated — every other dev/local source stays permissive."""
    monkeypatch.setenv("CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE", "widget")
    result = gate.precheck_source("some-other-unrelated-source", agents_root=tmp_path)
    assert result == {"checked": False, "reason": "no connector_manifest.yml for this source"}


def test_precheck_source_enterprise_gated_source_ok_when_manifest_present(
    tmp_path: Path, monkeypatch
):
    """Once a manifest IS provided for the gated source, it compiles/verifies normally."""
    monkeypatch.setenv("CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE", "widget")
    _write_clean_manifest(tmp_path, "widget-mcp")
    result = gate.precheck_source("widget", agents_root=tmp_path)
    assert result["checked"] is True
    assert result["ok"] is True


def test_sync_source_refuses_enterprise_gated_source_with_no_manifest(monkeypatch):
    """The ``sync_source`` entrypoint refuses an enterprise-gated source lacking a manifest."""
    from agent_utilities.knowledge_graph.core import source_sync

    monkeypatch.setenv("CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE", "totally-ungated-source")

    called = {"dispatched": False}

    def fake_dispatch(engine, source, *, mode="delta", ids=None, client=None):
        called["dispatched"] = True
        return {"status": "ok"}

    monkeypatch.setattr(source_sync, "_dispatch_sync_source", fake_dispatch)

    out = source_sync.sync_source(object(), "totally-ungated-source", mode="delta")
    assert out["status"] == "error"
    assert called["dispatched"] is False


def test_sync_source_dispatches_normally_when_no_manifest(monkeypatch):
    """No connector_manifest.yml for a source -> the gate is a silent pass-through."""
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
    assert out["status"] == "ok"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
