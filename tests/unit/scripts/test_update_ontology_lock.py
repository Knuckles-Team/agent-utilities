"""Tests for the fleet ``ontology.lock`` pinning script (D15, C5 fleet rollout)."""

from __future__ import annotations

import importlib.util
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


def test_update_ontology_lock_pins_hash(tmp_path: Path, agents_root: Path):
    lock_path = tmp_path / "ontology.lock"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--agents-root",
            str(agents_root),
            "--lock-path",
            str(lock_path),
            "--artifact-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "pinned 1 artifact" in result.stdout

    entries = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert "agents/acme-api/connector_manifest.yml" in entries
    entry = entries["agents/acme-api/connector_manifest.yml"]
    assert entry["algorithm"] == "urdna2015-sha256"
    assert len(entry["hash"]) == 64


def test_update_ontology_lock_reports_compile_failure(tmp_path: Path, agents_root: Path):
    bad = agents_root / "acme-api" / "connector_manifest.yml"
    bad.write_text("not: [valid, connector, manifest")  # malformed YAML
    lock_path = tmp_path / "ontology.lock"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--agents-root",
            str(agents_root),
            "--lock-path",
            str(lock_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAILED to compile" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
