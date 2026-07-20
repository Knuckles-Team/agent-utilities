"""Focused security and contract tests for exact OCI-layout export."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXPORTER_PATH = ROOT / "scripts" / "release" / "export_oci_layout.py"
CHECKER_PATH = ROOT / "scripts" / "release" / "check_oci_layout_export.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load(EXPORTER_PATH, "test_export_oci_layout_module")


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _layer() -> bytes:
    payload = b"Metadata-Version: 2.4\nName: fixture\nVersion: 1.0.0\n\n"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(
            "usr/local/lib/python3.12/site-packages/fixture-1.0.0.dist-info/METADATA"
        )
        member.size = len(payload)
        member.mode = 0o644
        member.mtime = 0
        archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def _oci_archive(
    *,
    config_environment: list[str] | None = None,
    foreign_member: bool = False,
    link_member: bool = False,
    corrupt_layer: bool = False,
) -> tuple[bytes, str]:
    layer = _layer()
    layer_digest = "sha256:" + hashlib.sha256(layer).hexdigest()
    config = _canonical(
        {
            "architecture": "amd64",
            "config": {"Env": config_environment or ["LANG=C.UTF-8"]},
            "os": "linux",
            "rootfs": {"diff_ids": [layer_digest], "type": "layers"},
        }
    )
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = _canonical(
        {
            "config": {
                "digest": config_digest,
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": len(config),
            },
            "layers": [
                {
                    "digest": layer_digest,
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                    "size": len(layer),
                }
            ],
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        }
    )
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    index = _canonical(
        {
            "manifests": [
                {
                    "digest": manifest_digest,
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "size": len(manifest),
                }
            ],
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    entries = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index,
        f"blobs/sha256/{config_digest.removeprefix('sha256:')}": config,
        f"blobs/sha256/{manifest_digest.removeprefix('sha256:')}": manifest,
        f"blobs/sha256/{layer_digest.removeprefix('sha256:')}": (
            bytes((layer[0] ^ 1,)) + layer[1:] if corrupt_layer else layer
        ),
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o600
            member.mtime = 0
            archive.addfile(member, io.BytesIO(payload))
        if foreign_member:
            payload = b"foreign"
            member = tarfile.TarInfo("foreign.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if link_member:
            member = tarfile.TarInfo("unsafe-link")
            member.type = tarfile.SYMTYPE
            member.linkname = "index.json"
            archive.addfile(member)
    return stream.getvalue(), manifest_digest


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _container_cli(path: Path) -> Path:
    path.write_bytes(b"exact-container-cli-fixture")
    path.chmod(0o700)
    return path


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    *,
    return_code: int = 0,
) -> list[tuple[list[str], dict[str, Any]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Process:
        pid = 987_654

        def wait(self, timeout: int) -> int:
            assert timeout == exporter._EXPORT_TIMEOUT_SECONDS
            return return_code

    def popen(argv: list[str], **kwargs: Any) -> Process:
        calls.append((argv, kwargs))
        remaining = memoryview(payload)
        while remaining:
            remaining = remaining[os.write(kwargs["stdout"], remaining) :]
        return Process()

    monkeypatch.setattr(exporter.subprocess, "Popen", popen)
    return calls


def test_export_is_no_replace_private_and_shell_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, root_digest = _oci_archive()
    private = _private_directory(tmp_path / "private")
    cli = _container_cli(tmp_path / "podman")
    output = private / "image.oci.tar"
    calls = _install_fake_process(monkeypatch, archive)

    identity = exporter.export_oci_layout(
        image_reference=f"example.invalid/agent@{root_digest}",
        output=output,
        container_cli=str(cli),
    )

    assert output.read_bytes() == archive
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert identity.root_digest == root_digest
    assert identity.archive_sha256 == "sha256:" + hashlib.sha256(archive).hexdigest()
    assert identity.image_manifest_count == 1
    assert [item.name for item in private.iterdir()] == [output.name]
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv == [
        str(cli.resolve()),
        "save",
        "--format",
        "oci-archive",
        f"example.invalid/agent@{root_digest}",
    ]
    assert options["shell"] is False
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stderr"] is subprocess.DEVNULL
    assert options["start_new_session"] is True
    assert options["executable"].startswith("/proc/self/fd/")
    assert options["pass_fds"]

    with pytest.raises(exporter.OciLayoutExportError, match="output_exists"):
        exporter.export_oci_layout(
            image_reference="sha256:" + "a" * 64,
            output=output,
            container_cli=str(cli),
        )
    assert len(calls) == 1


def test_mutable_image_reference_is_rejected_before_invocation(tmp_path: Path) -> None:
    output = _private_directory(tmp_path / "private") / "image.oci.tar"
    with pytest.raises(
        exporter.OciLayoutExportError,
        match="image_reference_not_exact",
    ):
        exporter.export_oci_layout(
            image_reference="agent-utilities:release",
            output=output,
            container_cli="container-cli-must-not-be-resolved",
        )
    assert not output.exists()


def test_output_symlink_and_symlink_parent_are_rejected(tmp_path: Path) -> None:
    private = _private_directory(tmp_path / "private")
    target = private / "target"
    target.write_bytes(b"existing")
    output = private / "image.oci.tar"
    output.symlink_to(target)
    cli = _container_cli(tmp_path / "podman")
    with pytest.raises(exporter.OciLayoutExportError, match="output_exists"):
        exporter.export_oci_layout(
            image_reference="sha256:" + "a" * 64,
            output=output,
            container_cli=str(cli),
        )

    linked_parent = tmp_path / "private-link"
    linked_parent.symlink_to(private, target_is_directory=True)
    with pytest.raises(exporter.OciLayoutExportError, match="output_parent_invalid"):
        exporter.export_oci_layout(
            image_reference="sha256:" + "a" * 64,
            output=linked_parent / "other.oci.tar",
            container_cli=str(cli),
        )


@pytest.mark.parametrize(
    "options, expected",
    (
        ({"foreign_member": True}, "archive_foreign_path"),
        ({"link_member": True}, "archive_member_type_invalid"),
        ({"corrupt_layer": True}, "archive_blob_digest_mismatch"),
    ),
)
def test_invalid_oci_archives_are_rejected_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, bool],
    expected: str,
) -> None:
    archive, _root_digest = _oci_archive(**options)
    private = _private_directory(tmp_path / expected)
    cli = _container_cli(tmp_path / f"podman-{expected}")
    output = private / "image.oci.tar"
    _install_fake_process(monkeypatch, archive)
    with pytest.raises(exporter.OciLayoutExportError, match=expected):
        exporter.export_oci_layout(
            image_reference="sha256:" + "a" * 64,
            output=output,
            container_cli=str(cli),
        )
    assert not output.exists()
    assert list(private.iterdir()) == []


def test_metadata_privacy_violation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _root_digest = _oci_archive(config_environment=["API_TOKEN=private-value"])
    private = _private_directory(tmp_path / "private")
    cli = _container_cli(tmp_path / "podman")
    output = private / "image.oci.tar"
    _install_fake_process(monkeypatch, archive)
    with pytest.raises(
        exporter.OciLayoutExportError,
        match="archive_metadata_privacy_violation",
    ):
        exporter.export_oci_layout(
            image_reference="sha256:" + "a" * 64,
            output=output,
            container_cli=str(cli),
        )
    assert not output.exists()


def test_digest_pinned_name_must_match_exported_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _root_digest = _oci_archive()
    private = _private_directory(tmp_path / "private")
    cli = _container_cli(tmp_path / "podman")
    output = private / "image.oci.tar"
    _install_fake_process(monkeypatch, archive)
    with pytest.raises(
        exporter.OciLayoutExportError,
        match="exported_root_digest_mismatch",
    ):
        exporter.export_oci_layout(
            image_reference="example.invalid/agent@sha256:" + "a" * 64,
            output=output,
            container_cli=str(cli),
        )
    assert not output.exists()


def test_failure_status_is_bounded_and_path_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "operator-identity" / "image.oci.tar"
    result = exporter.main(
        [
            "--image",
            "private.example/agent:mutable",
            "--output",
            str(output),
            "--container-cli",
            str(tmp_path / "private-container-cli"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert len(captured.err) < 200
    assert str(tmp_path) not in captured.err
    assert "private.example" not in captured.err
    assert json.loads(captured.err) == {
        "errorCode": "image_reference_not_exact",
        "schema": "oci-layout-export-status/1",
        "status": "rejected",
    }


def test_exporter_and_source_contract_are_directly_invocable(
    tmp_path: Path,
) -> None:
    help_result = subprocess.run(
        [sys.executable, str(EXPORTER_PATH), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        timeout=10,
    )
    contract_result = subprocess.run(
        [sys.executable, str(CHECKER_PATH)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert help_result.returncode == 0
    assert b"usage:" in help_result.stdout
    assert contract_result.returncode == 0, (
        contract_result.stdout + contract_result.stderr
    )
    assert json.loads(contract_result.stdout) == {
        "digest": json.loads(contract_result.stdout)["digest"],
        "ok": True,
    }
