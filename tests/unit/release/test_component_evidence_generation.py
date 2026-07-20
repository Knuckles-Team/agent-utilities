"""Contracts for deterministic component evidence generation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scripts.release import check_compatibility, generate_component_evidence

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "script",
    ("generate_component_evidence.py", "generate_release_assembly.py"),
)
def test_release_generators_are_directly_invocable_from_an_unrelated_cwd(
    tmp_path: Path,
    script: str,
) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/release" / script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert b"usage:" in result.stdout


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(path: Path, artifact_digest: str, subject_digest: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "graphos-external-signature/2",
                "scheme": "fixture-signature",
                "subjectDigest": subject_digest,
                "artifactDigest": artifact_digest,
                "signature": "fixture-signature-value",
                "verificationMaterialDigest": "sha256:" + "a" * 64,
                "signerIdentityDigest": "sha256:" + "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def _source_freeze(path: Path) -> Path:
    manifest_payload = (
        ROOT / "deploy" / "release" / "source-freeze-gates.json"
    ).read_bytes()
    manifest = json.loads(manifest_payload)
    repository_ids = [item["id"] for item in manifest["repositories"]]
    repository_digests = {
        identifier: format(index + 5, "064x")
        for index, identifier in enumerate(repository_ids)
    }

    def aggregate(values: dict[str, str]) -> str:
        payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    token = re.compile(r"^\{repo:([a-z][a-z0-9-]{2,63})\}(.*)$")
    commands = []
    for command in manifest["commands"]:
        identifiers = {command["repository"]}
        identifiers.update(
            match.group(1)
            for value in command["argv"]
            if (match := token.fullmatch(value)) is not None
        )
        digest = aggregate(
            {
                identifier: repository_digests[identifier]
                for identifier in repository_ids
                if identifier in identifiers
            }
        )
        commands.append(
            {
                "id": command["id"],
                "status": "passed",
                "exit_code": 0,
                "termination": "exited",
                "source_digest_before": digest,
                "source_digest_after": digest,
            }
        )
    source_digest = aggregate(repository_digests)
    path.write_text(
        json.dumps(
            {
                "schema": "source-freeze-evidence/1",
                "status": "passed",
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "source_digest_before": source_digest,
                "source_digest_after": source_digest,
                "tools": [
                    {"id": "git", "sha256": "3" * 64},
                    {"id": "rg", "sha256": "4" * 64},
                ],
                "repositories": [
                    {
                        "id": identifier,
                        "sha256_before": repository_digests[identifier],
                        "sha256_after": repository_digests[identifier],
                    }
                    for identifier in repository_ids
                ],
                "commands": commands,
                "gates": [
                    {
                        "id": gate["id"],
                        "required_evidence": gate["evidence_classes"],
                        "source_status": (
                            "passed"
                            if "local-source" in gate["evidence_classes"]
                            else "not-applicable"
                        ),
                        "remaining_evidence": [
                            value
                            for value in gate["evidence_classes"]
                            if value != "local-source"
                        ],
                    }
                    for gate in manifest["gates"]
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"entryCount":10}\n', encoding="utf-8")
    source = _source_freeze(tmp_path / "source-freeze.json")
    return artifact, source


def _generate(tmp_path: Path, release_name: str = "release") -> tuple[dict, Path]:
    artifact, source = _inputs(tmp_path)
    release_root = tmp_path / release_name
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            generate_component_evidence,
            "_external_json",
            lambda _env, payload: {
                "schema": "graphos-external-signature/2",
                "scheme": "fixture-signature",
                "subjectDigest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "artifactDigest": _digest(artifact),
                "signature": "fixture-signature-value",
                "verificationMaterialDigest": "sha256:" + "a" * 64,
                "signerIdentityDigest": "sha256:" + "b" * 64,
            },
        )
        declaration = generate_component_evidence.generate(
            name="prebundled-skills",
            version="1",
            kind="catalog",
            artifact_path=artifact,
            source_manifest=source,
            output_dir=release_root / "evidence/prebundled-skills",
            release_root=release_root,
            verifier_env="COMPONENT_SIGNATURE_VERIFIER",
            capabilities=("epistemic-engine-validated", "graph-os-delegation-validated"),
            entry_count=10,
            signer_env="COMPONENT_SIGNATURE_SIGNER",
        )
    return declaration, release_root


def test_component_evidence_is_deterministic_schema_valid_and_path_free(
    tmp_path: Path,
) -> None:
    first, first_root = _generate(tmp_path / "first")
    second, second_root = _generate(tmp_path / "second")

    assert first == second
    for relative in first["evidence"].values():
        assert (first_root / relative).read_bytes() == (second_root / relative).read_bytes()

    source_schema = json.loads(
        (ROOT / "deploy/release/component-source-evidence.schema.json").read_text()
    )
    provenance_schema = json.loads(
        (ROOT / "deploy/release/component-provenance.schema.json").read_text()
    )
    bundle_schema = json.loads(
        (ROOT / "deploy/release/component-signature-bundle.schema.json").read_text()
    )
    for schema in (source_schema, provenance_schema, bundle_schema):
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(source_schema).validate(
        json.loads((first_root / first["evidence"]["source"]).read_text())
    )
    Draft202012Validator(provenance_schema).validate(
        json.loads((first_root / first["evidence"]["provenance"]).read_text())
    )
    Draft202012Validator(bundle_schema).validate(
        json.loads((first_root / first["evidence"]["signatureBundle"]).read_text())
    )
    retained = b"".join(
        (first_root / reference).read_bytes() for reference in first["evidence"].values()
    ).decode("ascii")
    assert str(tmp_path) not in retained
    assert "file://" not in retained
    assert "@example" not in retained


def test_component_evidence_rejects_signature_for_another_artifact(
    tmp_path: Path,
) -> None:
    artifact, source = _inputs(tmp_path)
    wrong = _bundle(
        tmp_path / "wrong.json",
        "sha256:" + "c" * 64,
        "sha256:" + "d" * 64,
    )
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="not bound",
    ):
        generate_component_evidence.generate(
            name="prebundled-skills",
            version="1",
            kind="catalog",
            artifact_path=artifact,
            source_manifest=source,
            output_dir=tmp_path / "release/evidence/prebundled-skills",
            release_root=tmp_path / "release",
            verifier_env="COMPONENT_SIGNATURE_VERIFIER",
            signature_bundle_path=wrong,
        )


def test_component_evidence_rejects_symlink_artifact(tmp_path: Path) -> None:
    artifact, source = _inputs(tmp_path)
    bundle = _bundle(
        tmp_path / "signature.json",
        _digest(artifact),
        "sha256:" + "d" * 64,
    )
    symlink = tmp_path / "artifact-link"
    symlink.symlink_to(artifact)
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="symlink",
    ):
        generate_component_evidence.generate(
            name="prebundled-skills",
            version="1",
            kind="catalog",
            artifact_path=symlink,
            source_manifest=source,
            output_dir=tmp_path / "release/evidence/prebundled-skills",
            release_root=tmp_path / "release",
            verifier_env="COMPONENT_SIGNATURE_VERIFIER",
            signature_bundle_path=bundle,
        )


def _wheel(path: Path, name: str = "fixture-package", version: str = "1.0.0") -> None:
    dist = name.replace("-", "_") + f"-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist}/METADATA",
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n",
        )
        archive.writestr(
            f"{dist}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _oci_archive(
    path: Path,
    *,
    distribution: str = "agent-utilities",
    version: str = "1.0.0",
) -> str:
    metadata = (
        f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\n\n"
    ).encode()
    layer = _tar_bytes(
        {
            (
                "usr/local/lib/python3.12/site-packages/"
                f"{distribution.replace('-', '_')}-{version}.dist-info/METADATA"
            ): metadata
        }
    )
    layer_digest = "sha256:" + hashlib.sha256(layer).hexdigest()
    config = json.dumps(
        {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": [layer_digest]},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                    "digest": layer_digest,
                    "size": len(layer),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    index = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": manifest_digest,
                    "size": len(manifest),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    entries = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index,
        f"blobs/sha256/{config_digest.removeprefix('sha256:')}": config,
        f"blobs/sha256/{layer_digest.removeprefix('sha256:')}": layer,
        f"blobs/sha256/{manifest_digest.removeprefix('sha256:')}": manifest,
    }
    with tarfile.open(path, mode="w") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return manifest_digest


def test_oci_evidence_requires_layout_identity_and_matching_wheelhouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "agent-utilities.oci.tar"
    expected_digest = _oci_archive(artifact)
    source = _source_freeze(tmp_path / "source-freeze.json")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _wheel(
        wheelhouse / "agent_utilities-1.0.0-py3-none-any.whl",
        name="agent-utilities",
    )

    def signer(_env: str, payload: bytes) -> dict[str, str]:
        return {
            "schema": "graphos-external-signature/2",
            "scheme": "fixture-signature",
            "subjectDigest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "artifactDigest": expected_digest,
            "signature": "fixture-signature-value",
            "verificationMaterialDigest": "sha256:" + "a" * 64,
            "signerIdentityDigest": "sha256:" + "b" * 64,
        }

    monkeypatch.setattr(generate_component_evidence, "_external_json", signer)
    declaration = generate_component_evidence.generate(
        name="agent-utilities",
        version="1",
        kind="oci",
        artifact_path=artifact,
        source_manifest=source,
        output_dir=tmp_path / "release/evidence/agent-utilities",
        release_root=tmp_path / "release",
        verifier_env="COMPONENT_SIGNATURE_VERIFIER",
        wheelhouse=wheelhouse,
        signer_env="COMPONENT_SIGNATURE_SIGNER",
    )
    assert declaration["digest"] == expected_digest
    assert declaration["artifact"] == f"oci:agent-utilities@{expected_digest}"

    wrong_wheelhouse = tmp_path / "wrong-wheelhouse"
    wrong_wheelhouse.mkdir()
    _wheel(wrong_wheelhouse / "other_package-1.0.0-py3-none-any.whl", name="other-package")
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="installed distributions differ",
    ):
        generate_component_evidence.generate(
            name="agent-utilities",
            version="1",
            kind="oci",
            artifact_path=artifact,
            source_manifest=source,
            output_dir=tmp_path / "wrong/evidence/agent-utilities",
            release_root=tmp_path / "wrong",
            verifier_env="COMPONENT_SIGNATURE_VERIFIER",
            wheelhouse=wrong_wheelhouse,
            signer_env="COMPONENT_SIGNATURE_SIGNER",
        )


def test_oci_evidence_rejects_arbitrary_file_and_missing_wheelhouse(
    tmp_path: Path,
) -> None:
    artifact, source = _inputs(tmp_path)
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="closed wheelhouse",
    ):
        generate_component_evidence.generate(
            name="agent-utilities",
            version="1",
            kind="oci",
            artifact_path=artifact,
            source_manifest=source,
            output_dir=tmp_path / "release/evidence/agent-utilities",
            release_root=tmp_path / "release",
            verifier_env="COMPONENT_SIGNATURE_VERIFIER",
            signer_env="COMPONENT_SIGNATURE_SIGNER",
        )


def test_component_evidence_rejects_hardlinked_input(tmp_path: Path) -> None:
    artifact, _source = _inputs(tmp_path)
    os.link(artifact, tmp_path / "artifact-alias.json")
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="size boundary",
    ):
        generate_component_evidence._file_digest(
            artifact,
            maximum=generate_component_evidence._MAX_ARTIFACT_BYTES,
        )


def test_component_evidence_rejects_failed_source_freeze(tmp_path: Path) -> None:
    artifact, source = _inputs(tmp_path)
    evidence = json.loads(source.read_text(encoding="utf-8"))
    evidence["status"] = "failed"
    source.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="did not pass",
    ):
        generate_component_evidence.generate(
            name="prebundled-skills",
            version="1",
            kind="catalog",
            artifact_path=artifact,
            source_manifest=source,
            output_dir=tmp_path / "release/evidence/prebundled-skills",
            release_root=tmp_path / "release",
            verifier_env="COMPONENT_SIGNATURE_VERIFIER",
            signer_env="COMPONENT_SIGNATURE_SIGNER",
        )


def test_atomic_output_rejects_hardlink_alias(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    target = root / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = root / "alias.json"
    os.link(target, alias)
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="unaliased",
    ):
        generate_component_evidence._write(alias, b"{}\n", release_root=root)


def test_external_adapter_output_is_bounded() -> None:
    with pytest.raises(
        check_compatibility.CompatibilityError,
        match="output exceeds",
    ):
        check_compatibility._bounded_adapter(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            b"",
            maximum=64,
        )


def test_closed_wheelhouse_rejects_duplicate_distribution(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _wheel(wheelhouse / "fixture_package-1.0.0-py3-none-any.whl")
    _wheel(wheelhouse / "fixture_package-1.0.0-py2-none-any.whl")
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match="duplicate distributions",
    ):
        generate_component_evidence._wheel_components(wheelhouse)


def test_external_signer_and_verifier_are_adapter_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, source = _inputs(tmp_path)
    called: dict[str, object] = {}

    def signer(env_name: str, payload: bytes) -> dict[str, str]:
        called["signer"] = (env_name, payload)
        return {
            "schema": "graphos-external-signature/2",
            "scheme": "fixture-signature",
            "subjectDigest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "artifactDigest": _digest(artifact),
            "signature": "fixture-signature-value",
            "verificationMaterialDigest": "sha256:" + "a" * 64,
            "signerIdentityDigest": "sha256:" + "b" * 64,
        }

    def verifier(name: str, component: dict, bundle: dict) -> None:
        called["verifier"] = (
            name,
            component["digest"],
            bundle["subject"]["encoding"],
            bundle["subject"]["value"],
        )

    monkeypatch.setattr(generate_component_evidence, "_external_json", signer)
    monkeypatch.setattr(check_compatibility, "_verify_signature", verifier)
    generate_component_evidence.generate(
        name="prebundled-skills",
        version="1",
        kind="catalog",
        artifact_path=artifact,
        source_manifest=source,
        output_dir=tmp_path / "release/evidence/prebundled-skills",
        release_root=tmp_path / "release",
        verifier_env="COMPONENT_SIGNATURE_VERIFIER",
        signer_env="COMPONENT_SIGNATURE_SIGNER",
        verify_signature=True,
    )

    signer_payload = called["signer"][1]  # type: ignore[index]
    assert called["signer"][0] == "COMPONENT_SIGNATURE_SIGNER"  # type: ignore[index]
    assert called["verifier"][:3] == (
        "prebundled-skills",
        _digest(artifact),
        "base64",
    )
    assert called["verifier"][3] == base64.b64encode(signer_payload).decode(  # type: ignore[index]
        "ascii"
    )


def test_external_signer_failure_is_fail_closed_and_output_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPONENT_SIGNATURE_SIGNER", '["external-signer"]')
    monkeypatch.setattr(
        check_compatibility,
        "_bounded_adapter",
        lambda *args, **kwargs: (7, b"sensitive-output", b"sensitive-error"),
    )
    with pytest.raises(
        generate_component_evidence.ComponentEvidenceError,
        match=r"output_digest=[a-f0-9]{64}$",
    ) as exc:
        generate_component_evidence._external_json(
            "COMPONENT_SIGNATURE_SIGNER", b"{}\n"
        )
    assert "sensitive" not in str(exc.value)
