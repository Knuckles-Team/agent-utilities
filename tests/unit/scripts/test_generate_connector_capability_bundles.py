"""Contracts for privacy-safe connector source attestations."""

from __future__ import annotations

import base64
import importlib.util
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    SyncSpec,
)

_SPEC = importlib.util.spec_from_file_location(
    "generate_connector_capability_bundles",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_connector_capability_bundles.py",
)
generator = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(generator)

_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_connector_capability_bundles_source_attestation",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "check_connector_capability_bundles.py",
)
gate = importlib.util.module_from_spec(_GATE_SPEC)
assert _GATE_SPEC.loader is not None
_GATE_SPEC.loader.exec_module(gate)


@pytest.fixture
def signer(monkeypatch: pytest.MonkeyPatch) -> ontology_integrity.ReleaseSigner:
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    return ontology_integrity.ReleaseSigner.from_runtime()


def _manifest() -> ConnectorManifest:
    return ConnectorManifest(
        connector="fixture-connector",
        sync=[
            SyncSpec(
                preset="widgets",
                server="widget-mcp",
                tool="list_widgets",
                tool_schema_sha256="4" * 64,
            )
        ],
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="5" * 64)),
    )


_ONTOLOGY = """\
@prefix : <http://knuckles.team/kg#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://knuckles.team/kg/fixture> a owl:Ontology .
:Widget a owl:Class ; rdfs:label "Widget" .
"""


def _provider(agents_root: Path, name: str) -> Path:
    repo = agents_root / name
    ontology = repo / name.replace("-", "_") / "ontology"
    ontology.mkdir(parents=True)
    (ontology / "fixture.ttl").write_text(_ONTOLOGY, encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    return repo


def _workspace(path: Path, providers: tuple[str, ...]) -> Path:
    entries = "\n".join(
        f'          - url: "https://example.invalid/providers/{name}.git"'
        for name in providers
    )
    path.write_text(
        "subdirectories:\n"
        "  agent-packages:\n"
        "    subdirectories:\n"
        "      agents:\n"
        "        repositories:\n"
        f"{entries}\n",
        encoding="utf-8",
    )
    return path


def test_source_attestation_never_claims_live_execution(
    signer: ontology_integrity.ReleaseSigner,
) -> None:
    attestation = generator.build_source_attestation(
        _manifest(),
        artifacts={"connector_manifest.yml": "6" * 64},
        now=datetime(2026, 7, 18, tzinfo=UTC),
        release_signer=signer,
    )

    assert attestation["kind"] == "ConnectorSourceAttestation"
    assert attestation["schema_version"] == "2"
    assert attestation["mode"] == "offline-source"
    assert attestation["status"] == "source-validated"
    assert attestation["live_certified"] is False
    assert "live_tool_schema" not in attestation["checks"]
    assert attestation["checks"] == {
        "declared_tool_schema": "passed",
        "synthetic_fixture_contract": "passed",
        "shacl_parse": "passed",
        "privacy_contract": "passed",
        "manifest_integrity": "passed",
    }
    assert attestation["compatibility"] == {
        "agent_utilities": ">=1.27.1,<2",
        "epistemic_graph": ">=2.23.1,<3",
        "bundle_schema": "2",
    }
    assert ontology_integrity.verify_release_signature(
        ontology_integrity.canonical_signed_document_hash(attestation),
        attestation["signature"],
        signer_id=attestation["signer"],
        algorithm=attestation["signature_algorithm"],
        public_key=attestation["signing_public_key"],
        trusted_public_keys=(signer.public_key,),
    )


def test_capability_gate_rejects_old_live_pass_shape() -> None:
    old_shape = {
        "schema_version": "1",
        "connector": "fixture-connector",
        "certified_at": "2026-07-18T00:00:00Z",
        "status": "certified",
        "checks": {
            "live_tool_schema": "passed",
            "synthetic_fixture": "passed",
        },
    }

    violations = gate.source_attestation_violations(old_shape, _manifest())

    assert "source attestation shape differs" in violations
    assert "source attestation identity differs" in violations
    assert "source attestation checks differ" in violations
    assert "source attestation compatibility differs" in violations
    assert "source attestation timestamp is invalid" in violations
    assert "source attestation artifact ledger is invalid" in violations


def test_generation_uses_workspace_membership_and_one_signer(
    tmp_path: Path,
    signer: ontology_integrity.ReleaseSigner,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents_root = tmp_path / "agents"
    configured = _provider(agents_root, "configured-agent")
    extra = _provider(agents_root, "not-a-provider")
    workspace = _workspace(tmp_path / "workspace.yml", (configured.name,))
    calls = 0

    def _runtime_signer(
        cls: type[ontology_integrity.ReleaseSigner],
    ) -> ontology_integrity.ReleaseSigner:
        nonlocal calls
        calls += 1
        return signer

    monkeypatch.setattr(
        ontology_integrity.ReleaseSigner,
        "from_runtime",
        classmethod(_runtime_signer),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_connector_capability_bundles.py",
            "--agents-root",
            str(agents_root),
            "--bundled-output",
            str(tmp_path / "bundled"),
            "--workspace",
            str(workspace),
            "--now",
            "2026-07-18T00:00:00Z",
        ],
    )

    assert generator.main() == 0
    assert calls == 1
    assert "would generate 1 capability bundle(s)" in capsys.readouterr().out
    assert not (configured / "connector_manifest.yml").exists()
    assert not (extra / "connector_manifest.yml").exists()


def test_staging_validates_rotated_manifest_against_current_signer(
    tmp_path: Path,
    signer: ontology_integrity.ReleaseSigner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _provider(tmp_path / "agents", "rotated-agent")
    staging_root = tmp_path / "staging"
    checks: list[bool] = []
    real_check = generator.check_manifest_bytes

    def _check(path: Path, *, require_signature: bool = False) -> list[str]:
        checks.append(require_signature)
        return real_check(path, require_signature=require_signature)

    monkeypatch.setattr(generator, "check_manifest_bytes", _check)
    generator._stage_one(
        repo,
        bundled_output=tmp_path / "bundled",
        staging_root=staging_root,
        now=datetime(2026, 7, 18, tzinfo=UTC),
        release_signer=signer,
    )

    stage_repo = staging_root / "agents" / repo.name
    certification_path = next(stage_repo.glob("*/ontology/certification.json"))
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert checks == [False]
    assert certification["signing_public_key"] == signer.public_key


def test_staging_failure_never_publishes_earlier_provider(
    tmp_path: Path,
    signer: ontology_integrity.ReleaseSigner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agents_root = tmp_path / "agents"
    first = _provider(agents_root, "first-agent")
    second = _provider(agents_root, "second-agent")
    workspace = _workspace(tmp_path / "workspace.yml", (first.name, second.name))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    staged = generator._stage_one
    published = False

    def _stage(repo: Path, **kwargs):
        if repo.name == second.name:
            raise RuntimeError("synthetic staging failure")
        return staged(repo, **kwargs)

    def _publish(_publications) -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(generator, "_stage_one", _stage)
    monkeypatch.setattr(generator, "_publish_transaction", _publish)
    monkeypatch.setattr(
        ontology_integrity.ReleaseSigner,
        "from_runtime",
        classmethod(lambda cls: signer),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_connector_capability_bundles.py",
            "--agents-root",
            str(agents_root),
            "--bundled-output",
            str(bundled),
            "--workspace",
            str(workspace),
            "--now",
            "2026-07-18T00:00:00Z",
            "--apply",
        ],
    )

    assert generator.main() == 1
    assert published is False
    assert not (first / "connector_manifest.yml").exists()
    assert not (bundled / first.name / "connector_manifest.yml").exists()


def test_publication_failure_rolls_back_every_committed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_root = tmp_path / "staged"
    target_root = tmp_path / "target"
    staged_root.mkdir()
    target_root.mkdir()
    publications: list[generator._Publication] = []
    originals: dict[Path, bytes] = {}
    for index in range(2):
        staged = staged_root / f"artifact-{index}.json"
        target = target_root / f"artifact-{index}.json"
        staged.write_bytes(f"new-{index}".encode())
        originals[target] = f"old-{index}".encode()
        target.write_bytes(originals[target])
        publications.append(generator._Publication(staged, target, target_root))

    real_replace = generator.os.replace
    calls = 0

    def _replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        real_replace(source, target)

    monkeypatch.setattr(generator.os, "replace", _replace)
    with pytest.raises(RuntimeError, match="rolled back"):
        generator._publish_transaction(publications)

    assert all(path.read_bytes() == payload for path, payload in originals.items())
    assert not tuple(target_root.glob(".*.tmp"))
