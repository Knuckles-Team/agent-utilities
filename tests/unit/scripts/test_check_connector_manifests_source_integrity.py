"""Drive the source CLI without presenting source integrity as admission."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import rdflib
import yaml

from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
)
from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (
    check_manifest_bytes,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
)
from agent_utilities.knowledge_graph.ontology.ontology_integrity import canonical_hash

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_connector_manifests.py"


def _manifest(tmp_path: Path) -> Path:
    manifest = ConnectorManifest(
        connector="widget-mcp",
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    graph = rdflib.Graph().parse(
        data=export_manifest_ttl(
            compile_manifest(manifest), source=manifest.resolved_ontology_source
        ),
        format="turtle",
    )
    manifest.provenance.integrity.hash = canonical_hash(graph)[0]
    path = tmp_path / "connector_manifest.yml"
    path.write_text(yaml.safe_dump(manifest.model_dump(mode="json")), encoding="utf-8")
    return path


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--manifest", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_source_integrity_does_not_claim_release_or_runtime_admission(tmp_path: Path):
    path = _manifest(tmp_path)
    result = _run(path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SOURCE INTEGRITY OK" in result.stdout
    assert (
        "attestation, semantic admission, and attachment are not checked"
        in result.stdout
    )
    assert any(
        "[signature]" in violation
        for violation in check_manifest_bytes(path, require_signature=True)
    )
    assert check_manifest_bytes(path, require_release_pin=True)


def test_source_cli_rejects_hash_tampering(tmp_path: Path):
    path = _manifest(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["provenance"]["integrity"]["hash"] = "0" * 64
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = _run(path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[integrity]" in result.stdout
    assert "SOURCE INTEGRITY OK" not in result.stdout


def test_source_cli_rejects_invalid_schema(tmp_path: Path):
    path = tmp_path / "connector_manifest.yml"
    path.write_text("resources: invalid\n", encoding="utf-8")
    result = _run(path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[schema]" in result.stdout
