"""Behavioral contracts for exact local release input assembly."""

from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.release import assemble_exact_local_release as assembler
from scripts.release import promote_local_release as promoter

_DIGEST = "sha256:" + "a" * 64


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _wheel(
    root: Path,
    name: str,
    *,
    extras: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    native: bool = False,
) -> Path:
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-1.0.0.dist-info"
    filename = f"{distribution}-1.0.0-py3-none-any.whl"
    metadata_headers = [f"Name: {name}", "Version: 1.0.0"]
    metadata_headers.extend(f"Provides-Extra: {extra}" for extra in extras)
    metadata_headers.extend(f"Requires-Dist: {value}" for value in requires)
    members = {
        f"{dist_info}/METADATA": ("\n".join(metadata_headers) + "\n\n").encode(),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: exact-release-test\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n\n"
        ),
    }
    if native:
        members[f"{distribution}-1.0.0.data/scripts/epistemic-graph-server"] = (
            b"engine-binary"
        )
        members["epistemic_graph/numeric.abi3.so"] = b"numeric-extension"
    record_name = f"{dist_info}/RECORD"
    rows = [
        f"{member},{_record_digest(payload)},{len(payload)}"
        for member, payload in sorted(members.items())
    ]
    rows.append(f"{record_name},,")
    members[record_name] = ("\n".join(rows) + "\n").encode()
    path = root / filename
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in sorted(members.items()):
            archive.writestr(member, payload)
    return path


def _source(root: Path, *, unrelated: bool = False) -> Path:
    source = root / "source"
    source.mkdir(mode=0o700)
    _wheel(source, "epistemic-graph", extras=("full",), native=True)
    _wheel(source, "agent-utilities", extras=("mcp", "serving"))
    _wheel(source, "langfuse-agent", extras=("mcp",))
    if unrelated:
        _wheel(source, "unrelated-package")
    return source


def _toolchain() -> assembler.ToolchainIdentity:
    return assembler.ToolchainIdentity(
        python_version="3.12.10",
        python_digest=_DIGEST,
        uv_version="0.11.7",
        uv_digest="sha256:" + "b" * 64,
    )


def test_assembly_is_deterministic_private_minimal_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    monkeypatch.setattr(assembler, "_toolchain_identity", lambda _uv: _toolchain())
    first = private / "release-inputs-a"
    second = private / "release-inputs-b"

    first_result = assembler.assemble_exact_local_release(
        release_id="release-test",
        source=source,
        uv="verified-uv",
        output=first.absolute(),
    )
    second_result = assembler.assemble_exact_local_release(
        release_id="release-test",
        source=source,
        uv="verified-uv",
        output=second.absolute(),
    )

    assert set(path.name for path in first.iterdir()) == {
        "exact-local-release.json",
        "wheelhouse",
    }
    assert first.stat().st_mode & 0o077 == 0
    assert (first / "wheelhouse").stat().st_mode & 0o077 == 0
    assert (first / "exact-local-release.json").read_bytes() == (
        second / "exact-local-release.json"
    ).read_bytes()
    assert (first / "wheelhouse" / "release-requirements.txt").read_bytes() == (
        second / "wheelhouse" / "release-requirements.txt"
    ).read_bytes()
    assert first_result == second_result
    assert first_result["packageCount"] == 3
    assert first_result["pythonVersion"] == "3.12.10"
    assert first_result["uvVersion"] == "0.11.7"
    assert all(str(tmp_path) not in str(value) for value in first_result.values())

    spec = promoter.load_spec(
        first / "exact-local-release.json",
        release_id="release-test",
    )
    locked, wheels, _payload = promoter.validate_wheelhouse(
        first / "wheelhouse",
        spec,
    )
    assert set(locked) == {
        "agent-utilities",
        "epistemic-graph",
        "langfuse-agent",
    }
    assert set(wheels) == set(locked)

    with pytest.raises(promoter.ReleaseError, match="assembly-output-exists"):
        assembler.assemble_exact_local_release(
            release_id="release-test",
            source=source,
            uv="verified-uv",
            output=first.absolute(),
        )


def test_unreachable_wheel_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, unrelated=True)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    output = private / "release-inputs"
    monkeypatch.setattr(assembler, "_toolchain_identity", lambda _uv: _toolchain())

    with pytest.raises(
        promoter.ReleaseError,
        match="assembly-wheelhouse-not-minimal",
    ):
        assembler.assemble_exact_local_release(
            release_id="release-test",
            source=source,
            uv="verified-uv",
            output=output.absolute(),
        )
    assert not output.exists()


def test_missing_native_member_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    engine = _wheel(source, "epistemic-graph", extras=("full",))
    artifact = assembler._scan_source_wheels(source)["epistemic-graph"]

    with pytest.raises(
        promoter.ReleaseError,
        match="assembly-native-artifact-inventory-invalid",
    ):
        assembler._native_artifacts(artifact)
    assert engine.is_file()


def test_source_has_no_suppression_or_temporary_assembler_dependency() -> None:
    source = Path(assembler.__file__).read_text(encoding="utf-8")

    assert "noqa" not in source
    assert "type: ignore" not in source
    assert "pragma: no cover" not in source
    assert "/tmp/" not in source
