from __future__ import annotations

import base64
import importlib.util
import json
import secrets
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology import (
    connector_manifest_gate as runtime_manifest_gate,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

_SPEC = importlib.util.spec_from_file_location(
    "check_connector_capability_bundles",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "check_connector_capability_bundles.py",
)
gate = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gate)

_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_connector_capability_bundles",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_connector_capability_bundles.py",
)
generator = importlib.util.module_from_spec(_GENERATOR_SPEC)
assert _GENERATOR_SPEC.loader is not None
_GENERATOR_SPEC.loader.exec_module(generator)


@dataclass
class _CopiedBundle:
    agents_root: Path
    repo: Path
    module: Path
    bundled_root: Path
    lock_path: Path
    workspace: Path
    certification_path: Path
    certification: dict[str, Any]
    signer: ontology_integrity.ReleaseSigner


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


def _registry_for_copied_bundle(repo: Path, tmp_path: Path) -> Path:
    """Register ``repo`` under the alias its OWN copied preset already declares.

    CONCEPT:AU-KG.ontology.registry-derived-server-alias — the generator now
    derives every sync preset's ``server`` from the fleet registry and fails
    closed on disagreement. This test exercises the real leanix-agent bundle
    end to end through the gate machinery; it is not about what the live,
    shipped ``deploy/mcp-fleet.registry.yml`` currently maps that provider to
    (that is D-OB-7 drift the provider repo itself must fix). Register
    whichever alias the copied preset already declares so this stays a test of
    staging/gate behavior, not an assertion about live registry content.
    """

    declared: set[str] = set()
    for presets_path in repo.glob("*/connectors/mcp_source_presets.json"):
        data = json.loads(presets_path.read_text(encoding="utf-8"))
        for value in data.values():
            if isinstance(value, dict) and value.get("server"):
                declared.add(str(value["server"]))
    alias = declared.pop() if len(declared) == 1 else repo.name
    path = tmp_path / f"{repo.name}-mcp-fleet.registry.yml"
    path.write_text(
        "version: 1\n"
        "services:\n"
        f"  - name: {alias}\n"
        f"    package: {repo.name}\n"
        f"    console_script: {alias}\n"
        "    host_port: 8200\n",
        encoding="utf-8",
    )
    return path


def _copy_current_bundle(tmp_path: Path, monkeypatch) -> _CopiedBundle:
    source_repo = gate._default_agents_root() / "leanix-agent"
    source_module = gate._module(source_repo)
    source_certification_path = source_module / "ontology" / "certification.json"
    certification = gate._load_json(source_certification_path)
    artifacts = certification.get("artifacts")
    assert isinstance(artifacts, dict)

    agents_root = tmp_path / "agents"
    repo = agents_root / source_repo.name
    for relative in artifacts:
        assert isinstance(relative, str)
        source = source_repo / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    module = repo / source_module.relative_to(source_repo)
    certification_path = module / "ontology" / "certification.json"
    certification_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_certification_path, certification_path)
    shutil.copyfile(source_repo / "pyproject.toml", repo / "pyproject.toml")

    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()

    private_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", private_key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    signer = ontology_integrity.ReleaseSigner.from_runtime()
    monkeypatch.setenv("ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS", signer.public_key)
    registry_path = _registry_for_copied_bundle(repo, tmp_path)
    publications = generator._stage_one(
        repo,
        bundled_output=bundled_root,
        staging_root=tmp_path / "staging",
        now=datetime(2026, 7, 18, tzinfo=UTC),
        release_signer=signer,
        registry_path=registry_path,
    )
    generator._publish_transaction(list(publications))
    certification = gate._load_json(certification_path)
    lock_path = tmp_path / "ontology.lock"
    monkeypatch.setattr(
        runtime_manifest_gate,
        "_manifest_lock_entry",
        lambda manifest: gate._lock_entry(lock_path, manifest.connector),
    )
    return _CopiedBundle(
        agents_root=agents_root,
        repo=repo,
        module=module,
        bundled_root=bundled_root,
        lock_path=lock_path,
        workspace=_workspace(tmp_path / "workspace.yml", (repo.name,)),
        certification_path=certification_path,
        certification=certification,
        signer=signer,
    )


def _resign_and_pin(bundle: _CopiedBundle) -> None:
    certification = bundle.certification
    certification.update(
        {
            "signer": bundle.signer.signer_id,
            "signature_algorithm": bundle.signer.algorithm,
            "signing_public_key": bundle.signer.public_key,
            "signature": None,
        }
    )
    digest = ontology_integrity.canonical_signed_document_hash(certification)
    certification["signature"] = bundle.signer.sign(digest)
    bundle.certification_path.write_text(
        json.dumps(certification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = gate._manifest(bundle.repo / "connector_manifest.yml")
    manifest_hash = ontology_integrity.canonical_manifest_hash(manifest)
    provenance = manifest.provenance
    ontology_integrity.save_lock(
        bundle.lock_path,
        {
            f"agents/{bundle.repo.name}/connector_manifest.yml": {
                "manifest_hash": manifest_hash,
                "signer": provenance.signer,
                "signature_algorithm": provenance.signature_algorithm,
                "signing_public_key": provenance.signing_public_key,
                "signature": provenance.signature,
                "certification_file_sha256": gate._sha256(bundle.certification_path),
                "certification_hash": digest,
                "certification_signer": bundle.signer.signer_id,
                "certification_signature_algorithm": bundle.signer.algorithm,
                "certification_signing_public_key": bundle.signer.public_key,
                "certification_signature": certification["signature"],
            }
        },
    )


def _run_copied_bundle(bundle: _CopiedBundle, monkeypatch, capsys) -> tuple[int, str]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_connector_capability_bundles.py",
            "--agents-root",
            str(bundle.agents_root),
            "--bundled-root",
            str(bundle.bundled_root),
            "--lock-path",
            str(bundle.lock_path),
            "--workspace",
            str(bundle.workspace),
        ],
    )
    result = gate.main()
    return result, capsys.readouterr().out


def test_current_provider_membership_has_one_owned_manifest_per_provider():
    agents_root = gate._default_agents_root()
    workspace = gate._default_workspace()

    configured = set(gate._configured_provider_names(workspace))
    provider_owned = set(gate._provider_owned_names(agents_root))

    assert "leanix-agent" in configured
    assert configured <= provider_owned


def test_current_leanix_provider_bundle_content_passes(tmp_path, monkeypatch):
    bundle = _copy_current_bundle(tmp_path, monkeypatch)
    _resign_and_pin(bundle)
    violations = gate.check_one(
        bundle.repo,
        bundled_root=bundle.bundled_root,
        lock_path=bundle.lock_path,
        privacy=gate._release_privacy_guard(),
    )

    assert violations == []


def test_provider_owned_manifest_outside_workspace_fails_closed(
    tmp_path: Path, monkeypatch, capsys
):
    bundle = _copy_current_bundle(tmp_path, monkeypatch)
    _resign_and_pin(bundle)
    extra = bundle.agents_root / "retired-agent"
    extra.mkdir()
    (extra / "connector_manifest.yml").write_text("invalid: true\n", encoding="utf-8")

    result, output = _run_copied_bundle(bundle, monkeypatch, capsys)

    assert result == 1
    assert "failed for 1 provider(s)" in output
    assert "retired-agent" in output
    assert str(tmp_path) not in output


def test_configured_provider_without_manifest_fails_closed(
    tmp_path: Path, monkeypatch, capsys
):
    agents_root = tmp_path / "agents"
    (agents_root / "sample-agent").mkdir(parents=True)
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    lock_path = tmp_path / "ontology.lock"
    lock_path.write_text("{}\n", encoding="utf-8")
    workspace = _workspace(tmp_path / "workspace.yml", ("sample-agent",))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_connector_capability_bundles.py",
            "--agents-root",
            str(agents_root),
            "--bundled-root",
            str(bundled_root),
            "--lock-path",
            str(lock_path),
            "--workspace",
            str(workspace),
        ],
    )

    assert gate.main() == 1
    output = capsys.readouterr().out
    assert "failed for 1 provider(s)" in output
    assert "sample-agent" in output
    assert str(tmp_path) not in output


def test_empty_provider_membership_fails_closed(tmp_path: Path):
    workspace = tmp_path / "workspace.yml"
    workspace.write_text(
        "subdirectories:\n"
        "  agent-packages:\n"
        "    subdirectories:\n"
        "      agents:\n"
        "        repositories: []\n",
        encoding="utf-8",
    )

    try:
        gate._configured_provider_names(workspace)
    except ValueError as exc:
        assert str(exc) == "provider repository membership must not be empty"
    else:
        raise AssertionError("empty provider membership must fail closed")


def test_crypto_evidence_does_not_trigger_random_pii_format_match(tmp_path: Path):
    path = tmp_path / "certification.json"
    path.write_text(
        json.dumps(
            {
                "signature": "GB82WEST12345698765432",
                "signing_public_key": "FR1420041010050500013M02606",
            }
        ),
        encoding="utf-8",
    )

    scan_text = gate._privacy_scan_text(path)
    _, report = PersistencePrivacyGuard(deny_terms=()).sanitize_text(scan_text)

    assert report.changed is False
    assert "GB82WEST12345698765432" not in scan_text


def test_semantic_content_remains_subject_to_privacy_detection(tmp_path: Path):
    path = tmp_path / "fixture.json"
    path.write_text(
        json.dumps({"description": "GB82WEST12345698765432"}),
        encoding="utf-8",
    )

    scan_text = gate._privacy_scan_text(path)
    _, report = PersistencePrivacyGuard(deny_terms=()).sanitize_text(scan_text)

    assert report.detected_types == ("iban",)


def test_release_privacy_is_independent_of_local_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER", "genius")
    monkeypatch.setenv("LOGNAME", "genius")
    guard = gate._release_privacy_guard()

    clean, report = guard.sanitize_text("connector: genius-agent")

    assert clean == "connector: genius-agent"
    assert report.changed is False


def test_gate_privacy_scans_non_core_signed_artifact_end_to_end(
    tmp_path: Path, monkeypatch, capsys
):
    bundle = _copy_current_bundle(tmp_path, monkeypatch)
    relative = (
        bundle.module.relative_to(bundle.repo)
        / "ontology"
        / "evidence"
        / "supplemental.json"
    ).as_posix()
    artifact = bundle.repo / relative
    artifact.parent.mkdir(parents=True)
    private_value = "synthetic.person" + "@" + "example.test"
    artifact.write_text(
        json.dumps({"description": private_value}),
        encoding="utf-8",
    )
    bundle.certification["artifacts"][relative] = gate._sha256(artifact)
    _resign_and_pin(bundle)

    result, output = _run_copied_bundle(bundle, monkeypatch, capsys)

    assert result == 1
    assert "failed for 1 provider(s)" in output
    assert private_value not in output
    assert str(tmp_path) not in output


def test_gate_rejects_signed_symlink_escape_end_to_end(
    tmp_path: Path, monkeypatch, capsys
):
    bundle = _copy_current_bundle(tmp_path, monkeypatch)
    outside = tmp_path / "outside-artifact.json"
    outside.write_text('{"description":"synthetic"}\n', encoding="utf-8")
    relative = (
        bundle.module.relative_to(bundle.repo)
        / "ontology"
        / "evidence"
        / "supplemental.json"
    ).as_posix()
    artifact = bundle.repo / relative
    artifact.parent.mkdir(parents=True)
    artifact.symlink_to(outside)
    bundle.certification["artifacts"][relative] = gate._sha256(outside)
    _resign_and_pin(bundle)

    result, output = _run_copied_bundle(bundle, monkeypatch, capsys)

    assert result == 1
    assert "failed for 1 provider(s)" in output
    assert outside.name not in output
    assert str(tmp_path) not in output
