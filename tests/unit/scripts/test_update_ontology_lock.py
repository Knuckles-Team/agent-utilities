"""Tests for the fleet ``ontology.lock`` pinning script (D15, C5 fleet rollout)."""

from __future__ import annotations

import base64
import importlib.util
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

_GEN_SPEC = importlib.util.spec_from_file_location(
    "generate_connector_manifests",
    Path(__file__).resolve().parents[3] / "scripts" / "generate_connector_manifests.py",
)
gen = importlib.util.module_from_spec(_GEN_SPEC)
assert _GEN_SPEC.loader is not None
_GEN_SPEC.loader.exec_module(gen)

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "update_ontology_lock.py"

_ONTOLOGY = """\
@prefix : <http://knuckles.team/kg#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://knuckles.team/kg/acme> a owl:Ontology ;
    rdfs:label "Acme" ;
    owl:imports <http://knuckles.team/kg> .

:Order a owl:Class ; rdfs:label "Order" .
:total a owl:DatatypeProperty ; rdfs:label "total" ; rdfs:range xsd:decimal .
"""

_NOW = datetime(2026, 7, 9, tzinfo=UTC)


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch):
    private_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", private_key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )


@pytest.fixture
def agents_root(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    connector = root / "acme-api"
    mod = connector / "acme_api" / "ontology"
    mod.mkdir(parents=True)
    (mod / "acme.ttl").write_text(_ONTOLOGY)
    manifest = gen.build_manifest(connector, now=_NOW)
    (connector / "connector_manifest.yml").write_text(gen._to_yaml(manifest))
    return root


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.yml"
    path.write_text(
        "subdirectories:\n"
        "  agent-packages:\n"
        "    subdirectories:\n"
        "      agents:\n"
        "        repositories:\n"
        '          - url: "https://example.invalid/providers/acme-api.git"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def native_manifest(tmp_path: Path) -> Path:
    connector = tmp_path / "native-source-connectors"
    ontology = connector / "native_source_connectors" / "ontology"
    ontology.mkdir(parents=True)
    (ontology / "native.ttl").write_text(_ONTOLOGY, encoding="utf-8")
    manifest = gen.build_manifest(connector, now=_NOW)
    path = connector / "connector_manifest.yml"
    path.write_text(gen._to_yaml(manifest), encoding="utf-8")
    return path


def _command(
    *,
    agents_root: Path,
    workspace: Path,
    native_manifest: Path,
    lock_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(_SCRIPT),
        "--agents-root",
        str(agents_root),
        "--workspace",
        str(workspace),
        "--native-manifest",
        str(native_manifest),
        "--lock-path",
        str(lock_path),
    ]


def test_update_ontology_lock_pins_hash_and_removes_stale_provider(
    tmp_path: Path,
    agents_root: Path,
    workspace: Path,
    native_manifest: Path,
):
    lock_path = tmp_path / "ontology.lock"
    lock_path.write_text(
        "agents/stale-api/connector_manifest.yml:\n"
        "  hash: stale\n"
        "ontology/core.ttl:\n"
        "  hash: retained\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        _command(
            agents_root=agents_root,
            workspace=workspace,
            native_manifest=native_manifest,
            lock_path=lock_path,
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "pinned 2 artifact" in result.stdout

    entries = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert "agents/acme-api/connector_manifest.yml" in entries
    assert "agents/native-source-connectors/connector_manifest.yml" in entries
    assert "agents/stale-api/connector_manifest.yml" not in entries
    assert entries["ontology/core.ttl"] == {"hash": "retained"}
    entry = entries["agents/acme-api/connector_manifest.yml"]
    assert entry["algorithm"] == "urdna2015-sha256"
    assert entry["signature_algorithm"] == "ed25519"
    assert entry["signing_public_key"]
    assert len(entry["hash"]) == 64


def test_update_ontology_lock_reports_compile_failure(
    tmp_path: Path,
    agents_root: Path,
    workspace: Path,
    native_manifest: Path,
):
    bad = agents_root / "acme-api" / "connector_manifest.yml"
    bad.write_text("not: [valid, connector, manifest")  # malformed YAML
    lock_path = tmp_path / "ontology.lock"
    original = b"existing:\n  hash: unchanged\n"
    lock_path.write_bytes(original)
    result = subprocess.run(
        _command(
            agents_root=agents_root,
            workspace=workspace,
            native_manifest=native_manifest,
            lock_path=lock_path,
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAILED to compile" in result.stdout
    assert lock_path.read_bytes() == original


def test_update_ontology_lock_rejects_missing_workspace_provider_without_write(
    tmp_path: Path,
    agents_root: Path,
    workspace: Path,
    native_manifest: Path,
):
    workspace.write_text(
        workspace.read_text(encoding="utf-8")
        + '          - url: "https://example.invalid/providers/missing-api.git"\n',
        encoding="utf-8",
    )
    lock_path = tmp_path / "ontology.lock"
    original = b"existing:\n  hash: unchanged\n"
    lock_path.write_bytes(original)

    result = subprocess.run(
        _command(
            agents_root=agents_root,
            workspace=workspace,
            native_manifest=native_manifest,
            lock_path=lock_path,
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "membership differs" in result.stdout
    assert lock_path.read_bytes() == original


def test_update_ontology_lock_refuses_untrusted_self_signed_manifest(
    tmp_path: Path,
    agents_root: Path,
    workspace: Path,
    native_manifest: Path,
    monkeypatch,
):
    monkeypatch.delenv("ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF", raising=False)
    monkeypatch.delenv("ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS", raising=False)
    lock_path = tmp_path / "ontology.lock"

    result = subprocess.run(
        _command(
            agents_root=agents_root,
            workspace=workspace,
            native_manifest=native_manifest,
            lock_path=lock_path,
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "no valid trusted signing key" in result.stdout
    assert not lock_path.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
