"""Focused contracts for the epistemic-graph image client preflight."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import py_compile
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts.release import check_epistemic_graph_client_preflight as preflight


def _wheel(
    root: Path,
    *,
    filename: str = "epistemic_graph-2.26.2-py3-none-any.whl",
    metadata_name: str = "epistemic-graph",
    metadata_version: str = "2.26.2",
) -> Path:
    path = root / filename
    dist_info = f"epistemic_graph-{metadata_version}.dist-info"
    metadata = (
        f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {metadata_version}\n\n"
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
    return path


def _record_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _recorded_wheel(root: Path, *, include_script: bool = False) -> Path:
    path = root / "epistemic_graph-2.26.2-py3-none-any.whl"
    dist_info = "epistemic_graph-2.26.2.dist-info"
    members = {
        "epistemic_graph/__init__.py": b"\n",
        "epistemic_graph/client_capabilities.py": (
            b'WORK_ITEM_METADATA_CAS_CAPABILITY = "work_items.cas_metadata"\n'
            b"\n"
            b"def require_client_capabilities(required):\n"
            b"    if tuple(required) != (WORK_ITEM_METADATA_CAS_CAPABILITY,):\n"
            b'        raise RuntimeError("unexpected capability request")\n'
            b"    return {\n"
            b'        "package": "epistemic-graph",\n'
            b'        "package_version": "2.26.2",\n'
            b'        "capabilities": {WORK_ITEM_METADATA_CAS_CAPABILITY: True},\n'
            b"    }\n"
        ),
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\nName: epistemic-graph\nVersion: 2.26.2\n\n"
        ),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\n\n",
    }
    if include_script:
        members["epistemic_graph-2.26.2.data/scripts/epistemic-graph-server"] = (
            b"#!/bin/sh\n"
        )
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        for name, payload in sorted(members.items()):
            writer.writerow((name, _record_hash(payload), str(len(payload))))
        writer.writerow((f"{dist_info}/RECORD", "", ""))
        archive.writestr(f"{dist_info}/RECORD", output.getvalue())
    return path


def _install_recorded_wheel(
    wheel: Path,
    root: Path,
    *,
    direct_url: str,
    script_root: Path | None = None,
) -> None:
    root.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        record_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        for name in archive.namelist():
            if name == record_name or name.endswith("/"):
                continue
            parts = Path(name).parts
            if len(parts) >= 3 and parts[0].endswith(".data"):
                assert script_root is not None and parts[1] == "scripts"
                destination = script_root.joinpath(*parts[2:])
            else:
                destination = root.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))

    dist_info = record_name.rsplit("/", 1)[0]
    adaptations = {
        f"{dist_info}/direct_url.json": direct_url.encode(),
        f"{dist_info}/INSTALLER": b"uv\n",
        f"{dist_info}/REQUESTED": b"",
    }
    for name, payload in adaptations.items():
        destination = root.joinpath(*Path(name).parts)
        destination.write_bytes(payload)

    rows: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(wheel) as archive:
        for row in csv.reader(
            io.StringIO(archive.read(record_name).decode(), newline="")
        ):
            if row[0] != record_name:
                parts = Path(row[0]).parts
                if len(parts) >= 3 and parts[0].endswith(".data"):
                    assert script_root is not None and parts[1] == "scripts"
                    installed = Path(
                        os.path.relpath(script_root.joinpath(*parts[2:]), root)
                    ).as_posix()
                    rows.append((installed, row[1], row[2]))
                else:
                    rows.append((row[0], row[1], row[2]))
    for name, payload in adaptations.items():
        rows.append((name, _record_hash(payload), str(len(payload))))
    rows.append((record_name, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    (root / record_name).write_text(output.getvalue(), encoding="utf-8")


def _compile_cache(
    root: Path,
    relative_source: str = preflight.CLIENT_MODULE_PATH,
    *,
    optimization: int = 0,
    source_override: bytes | None = None,
) -> Path:
    source = root / relative_source
    if source_override is not None:
        source.write_bytes(source_override)
    cache = Path(
        importlib.util.cache_from_source(str(source), optimization=optimization)
    )
    py_compile.compile(
        str(source),
        cfile=str(cache),
        dfile=str(source),
        optimize=optimization,
    )
    return cache


def _append_record_row(root: Path, row: tuple[str, str, str]) -> None:
    record = next(root.glob("*.dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    rows.append(list(row))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    record.write_text(output.getvalue(), encoding="utf-8")


def _isolated_installer_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    return environment


@pytest.mark.parametrize("installer", ("uv", "pip"))
def test_real_installer_preflight_binds_local_wheel(
    tmp_path: Path, installer: str
) -> None:
    if installer == "uv":
        executable = shutil.which("uv")
        if executable is None:
            pytest.skip("uv installer executable is unavailable")
    else:
        executable = None

    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = _recorded_wheel(wheel_dir)
    evidence = preflight.select_wheel(wheel_dir)
    venv = tmp_path / f"{installer}-venv"
    environment = _isolated_installer_environment()
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if installer == "uv":
        # Mirror the production image layer: uv may leave valid bytecode
        # caches that are absent from RECORD when this is enabled.
        environment["UV_COMPILE_BYTECODE"] = "1"
    if installer == "pip":
        pip_probe = subprocess.run(
            [str(python), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        if pip_probe.returncode != 0:
            ensurepip_probe = subprocess.run(
                [str(python), "-m", "ensurepip", "--version"],
                capture_output=True,
                text=True,
                env=environment,
                timeout=120,
            )
            if ensurepip_probe.returncode != 0:
                pytest.skip("pip installer executable is unavailable")
            subprocess.run(
                [str(python), "-m", "ensurepip", "--upgrade"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=120,
            )
        install_command = [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ]
    else:
        install_command = [
            executable,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-index",
            "--no-deps",
            str(wheel),
        ]
    install = subprocess.run(
        install_command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    direct_url_payload = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m; "
            "print(m.distribution('epistemic-graph').read_text('direct_url.json'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
    ).stdout
    direct_url = json.loads(direct_url_payload)
    assert direct_url["url"] == wheel.as_uri()
    archive_info = direct_url.get("archive_info", {})
    assert isinstance(archive_info, dict)
    if archive_info:
        assert archive_info.get("hash") == f"sha256={evidence.sha256}"
        assert archive_info.get("hashes", {}).get("sha256") == evidence.sha256

    script = Path(__file__).resolve().parents[3] / (
        "scripts/release/check_epistemic_graph_client_preflight.py"
    )
    preflight_run = subprocess.run(
        [
            str(python),
            str(script),
            "--wheel-dir",
            str(wheel_dir),
            "--require-installed",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
    )
    assert preflight_run.returncode == 0, preflight_run.stdout + preflight_run.stderr


def _installed_surface(root: Path) -> tuple[_Distribution, ModuleType]:
    distribution = _Distribution(root)
    module = ModuleType(preflight.CAPABILITIES_MODULE)
    module.__file__ = str(root / preflight.CLIENT_MODULE_PATH)
    module.require_client_capabilities = lambda _required: _manifest()
    return distribution, module


def test_select_wheel_proves_exact_filename_and_metadata(tmp_path: Path) -> None:
    path = _wheel(tmp_path)

    evidence = preflight.select_wheel(tmp_path)

    assert evidence.path == path
    assert evidence.name == preflight.PACKAGE_NAME
    assert evidence.version == preflight.EXPECTED_VERSION


@pytest.mark.parametrize(
    ("filename", "metadata_version", "error"),
    (
        (
            "epistemic_graph-2.26.1-py3-none-any.whl",
            "2.26.2",
            "wheel-filename-version-mismatch",
        ),
        (
            "epistemic_graph-2.26.2-py3-none-any.whl",
            "2.26.1",
            "wheel-metadata-version-mismatch",
        ),
    ),
)
def test_stale_or_mismatched_wheel_versions_fail_closed(
    tmp_path: Path,
    filename: str,
    metadata_version: str,
    error: str,
) -> None:
    _wheel(tmp_path, filename=filename, metadata_version=metadata_version)

    with pytest.raises(preflight.PreflightError, match=error):
        preflight.select_wheel(tmp_path)


def test_multiple_staged_wheels_are_ambiguous(tmp_path: Path) -> None:
    _wheel(tmp_path)
    _wheel(
        tmp_path,
        filename="epistemic_graph-2.26.1-py3-none-any.whl",
        metadata_version="2.26.1",
    )

    with pytest.raises(preflight.PreflightError, match="wheel-count-invalid"):
        preflight.select_wheel(tmp_path)


def test_missing_wheel_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(preflight.PreflightError, match="wheel-count-invalid"):
        preflight.select_wheel(tmp_path)


def test_symlinked_wheel_is_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _wheel(source_dir)
    staged = tmp_path / "staged"
    staged.mkdir()
    os.symlink(source, staged / source.name)

    with pytest.raises(preflight.PreflightError, match="wheel-containment-invalid"):
        preflight.select_wheel(staged)


def test_unified_image_wires_both_preflight_phases() -> None:
    root = Path(__file__).resolve().parents[3]
    dockerfile = (root / "docker" / "graphos-unified.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "2.26.1" not in dockerfile
    assert "COPY build-artifacts/eg-wheel/*.whl /tmp/wheels/" in dockerfile
    assert "--print-wheel-basename" in dockerfile
    assert "--require-installed" in dockerfile


class _Distribution:
    def __init__(self, root: Path, *, version: str = "2.26.2") -> None:
        self.root = root
        self.version = version
        self.direct_url: str | None = None

    def locate_file(self, relative: str) -> Path:
        return self.root / relative

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self.direct_url


def _set_direct_url(distribution: _Distribution, payload: str) -> None:
    """Keep the synthetic direct_url file and its RECORD row coherent."""

    distribution.direct_url = payload
    direct_url = next(distribution.root.glob("*.dist-info/direct_url.json"))
    direct_url.write_text(payload, encoding="utf-8")
    record = next(distribution.root.glob("*.dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    relative = direct_url.relative_to(distribution.root).as_posix()
    for row in rows:
        if row[0] == relative:
            encoded = payload.encode("utf-8")
            row[1] = _record_hash(encoded)
            row[2] = str(len(encoded))
            break
    else:
        raise AssertionError("direct_url.json is absent from RECORD")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    record.write_text(output.getvalue(), encoding="utf-8")


def _client_surface(tmp_path: Path) -> tuple[_Distribution, ModuleType]:
    package_root = tmp_path / "site-packages"
    module_file = package_root / preflight.CLIENT_MODULE_PATH
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# synthetic installed module\n", encoding="utf-8")
    distribution = _Distribution(package_root)
    module = ModuleType(preflight.CAPABILITIES_MODULE)
    module.__file__ = str(module_file)
    return distribution, module


def _manifest(capability_value: Any = True) -> dict[str, Any]:
    return {
        "package": preflight.PACKAGE_NAME,
        "package_version": preflight.EXPECTED_VERSION,
        "client_build_identity": "synthetic-client/2.26.2",
        "capabilities": {preflight.REQUIRED_CAPABILITY: capability_value},
    }


def test_installed_client_invokes_producer_capability_gate(tmp_path: Path) -> None:
    distribution, module = _client_surface(tmp_path)
    calls: list[tuple[str, ...]] = []
    bytecode_policy: list[bool] = []
    previous_bytecode_policy = sys.dont_write_bytecode
    module.WORK_ITEM_METADATA_CAS_CAPABILITY = preflight.REQUIRED_CAPABILITY

    def require(required: tuple[str, ...]) -> dict[str, Any]:
        calls.append(required)
        return _manifest()

    module.require_client_capabilities = require

    manifest = preflight.validate_installed_client(
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: (
            bytecode_policy.append(sys.dont_write_bytecode) or module
        ),
    )

    assert manifest["capabilities"] == {preflight.REQUIRED_CAPABILITY: True}
    assert calls == [(preflight.REQUIRED_CAPABILITY,)]
    assert bytecode_policy == [True]
    assert sys.dont_write_bytecode is previous_bytecode_policy


def test_installed_client_binds_to_selected_wheel_provenance(tmp_path: Path) -> None:
    wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    _set_direct_url(
        distribution,
        json.dumps(
            {
                "url": wheel_path.as_uri(),
                "archive_info": {"hash": f"sha256={evidence.sha256}"},
            }
        ),
    )
    module.require_client_capabilities = lambda _required: _manifest()

    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )

    other_dir = wheel_path.parent / "other"
    other_dir.mkdir()
    distribution.direct_url = json.dumps(
        {
            "url": (other_dir / wheel_path.name).as_uri(),
            "archive_info": {"hash": f"sha256={evidence.sha256}"},
        }
    )
    with pytest.raises(
        preflight.PreflightError,
        match="client-artifact-provenance-mismatch",
    ):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def _fallback_binding(
    tmp_path: Path,
) -> tuple[Path, preflight.WheelEvidence, _Distribution, ModuleType]:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel_path = _recorded_wheel(wheel_dir)
    evidence = preflight.select_wheel(wheel_dir)
    installed_root = tmp_path / "installed"
    direct_url = json.dumps(
        {"url": wheel_path.as_uri(), "archive_info": {}}, separators=(",", ":")
    )
    _install_recorded_wheel(wheel_path, installed_root, direct_url=direct_url)
    distribution, module = _installed_surface(installed_root)
    distribution.direct_url = direct_url
    return wheel_path, evidence, distribution, module


def test_empty_archive_info_binds_installed_wheel_record_to_selected_archive(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)

    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )


def test_hash_bearing_pep610_still_proves_installed_tree(tmp_path: Path) -> None:
    wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    _set_direct_url(
        distribution,
        json.dumps(
            {
                "url": wheel_path.as_uri(),
                "archive_info": {"hash": f"sha256={evidence.sha256}"},
            }
        ),
    )

    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )

    (distribution.root / preflight.CLIENT_MODULE_PATH).write_text(
        "# changed after archive authentication\n", encoding="utf-8"
    )
    with pytest.raises(
        preflight.PreflightError, match="installed-wheel-content-mismatch"
    ):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_unrecorded_bytecode_cache_must_derive_from_wheel_source(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    cache = _compile_cache(distribution.root)
    assert cache.exists()

    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )


def test_structural_code_equality_does_not_require_marshal_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    cache = _compile_cache(distribution.root)
    payload = cache.read_bytes()
    loaded = preflight.marshal.loads(payload[16:])
    source = distribution.root / preflight.CLIENT_MODULE_PATH
    expected = compile(
        source.read_bytes(),
        loaded.co_filename,
        "exec",
        optimize=0,
        dont_inherit=True,
    )
    assert expected == loaded

    def reject_serialization(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("marshal serialization is not code identity")

    monkeypatch.setattr(preflight.marshal, "dumps", reject_serialization)
    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )


def test_recorded_empty_identity_bytecode_cache_is_verified(tmp_path: Path) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    cache = _compile_cache(distribution.root)
    _append_record_row(
        distribution.root,
        (cache.relative_to(distribution.root).as_posix(), "", ""),
    )

    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )


def test_foreign_bytecode_at_exact_cache_path_is_rejected(tmp_path: Path) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    installed_source = distribution.root / preflight.CLIENT_MODULE_PATH
    source = tmp_path / "foreign.py"
    foreign = b"foreign = True\n"
    source.write_bytes(
        foreign + b"#" * (len(installed_source.read_bytes()) - len(foreign))
    )
    cache = Path(
        importlib.util.cache_from_source(str(installed_source), optimization=0)
    )
    py_compile.compile(str(source), cfile=str(cache), dfile=str(installed_source))

    with pytest.raises(
        preflight.PreflightError, match="installed-bytecode-code-mismatch"
    ):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_bytecode_header_and_cache_tag_are_bound(tmp_path: Path) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    cache = _compile_cache(distribution.root)
    invalid_header = bytearray(cache.read_bytes())
    invalid_header[4:8] = (0x02).to_bytes(4, "little")
    cache.write_bytes(invalid_header)
    with pytest.raises(preflight.PreflightError, match="installed-bytecode-invalid"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )

    cache.write_bytes(_compile_cache(distribution.root).read_bytes())
    foreign_cache = cache.with_name(
        cache.name.replace(sys.implementation.cache_tag, "foreign")
    )
    cache.rename(foreign_cache)
    with pytest.raises(preflight.PreflightError, match="installed-file-unexpected"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_mismatched_pep610_archive_hash_is_not_replaced_by_record_fallback(
    tmp_path: Path,
) -> None:
    wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    distribution.direct_url = json.dumps(
        {
            "url": wheel_path.as_uri(),
            "archive_info": {"hash": "sha256=" + ("0" * 64)},
        }
    )

    with pytest.raises(
        preflight.PreflightError, match="client-artifact-provenance-mismatch"
    ):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_data_scripts_member_binds_after_controlled_install_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel_path = _recorded_wheel(wheel_dir, include_script=True)
    evidence = preflight.select_wheel(wheel_dir)
    installed_root = tmp_path / "installed"
    script_root = tmp_path / "bin"
    script_root.mkdir()
    monkeypatch.setattr(
        preflight.sysconfig,
        "get_path",
        lambda key: str(script_root) if key == "scripts" else None,
    )
    direct_url = json.dumps(
        {"url": wheel_path.as_uri(), "archive_info": {}}, separators=(",", ":")
    )
    _install_recorded_wheel(
        wheel_path,
        installed_root,
        direct_url=direct_url,
        script_root=script_root,
    )
    distribution, module = _installed_surface(installed_root)
    distribution.direct_url = direct_url

    preflight.validate_installed_client(
        artifact=evidence,
        distribution_reader=lambda _name: distribution,
        module_importer=lambda _name: module,
    )


def test_installed_byte_mutation_is_rejected_without_archive_hash(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    module_path = distribution.root / preflight.CLIENT_MODULE_PATH
    module_path.write_text("# mutated installed module\n", encoding="utf-8")

    with pytest.raises(
        preflight.PreflightError, match="installed-wheel-content-mismatch"
    ):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_missing_installed_member_is_rejected_without_archive_hash(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    (distribution.root / preflight.CLIENT_MODULE_PATH).unlink()

    with pytest.raises(preflight.PreflightError, match="installed-file-missing"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_preexisting_bytecode_cache_is_rejected_without_archive_hash(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    cache = (
        distribution.root
        / "epistemic_graph"
        / "__pycache__"
        / ("client.cpython-314.pyc")
    )
    cache.parent.mkdir()
    cache.write_bytes(b"forged bytecode")

    with pytest.raises(preflight.PreflightError, match="installed-file-unexpected"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_unexpected_installed_member_is_rejected_without_archive_hash(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    extra = distribution.root / "epistemic_graph" / "unexpected.py"
    extra.write_text("# not in wheel\n", encoding="utf-8")

    with pytest.raises(preflight.PreflightError, match="installed-file-unexpected"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_hardlinked_installed_member_is_rejected_without_archive_hash(
    tmp_path: Path,
) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    module_path = distribution.root / preflight.CLIENT_MODULE_PATH
    hardlink = distribution.root / "epistemic_graph" / "hardlink.py"
    os.link(module_path, hardlink)

    with pytest.raises(preflight.PreflightError, match="installed-file-not-regular"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_installed_symlink_is_rejected_without_archive_hash(tmp_path: Path) -> None:
    _wheel_path, evidence, distribution, module = _fallback_binding(tmp_path)
    module_path = distribution.root / preflight.CLIENT_MODULE_PATH
    payload = module_path.with_name("payload.py")
    payload.write_bytes(module_path.read_bytes())
    module_path.unlink()
    os.symlink(payload, module_path)

    with pytest.raises(preflight.PreflightError, match="installed-path-symlink"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


@pytest.mark.parametrize(
    "malformed_member",
    ("../outside.py", "epistemic_graph/client.py"),
)
def test_wheel_traversal_or_duplicate_members_fail_closed(
    tmp_path: Path,
    malformed_member: str,
) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    path = wheel_dir / "epistemic_graph-2.26.2-py3-none-any.whl"
    metadata = b"Metadata-Version: 2.1\nName: epistemic-graph\nVersion: 2.26.2\n\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("epistemic_graph-2.26.2.dist-info/METADATA", metadata)
        archive.writestr(malformed_member, b"one")
        if malformed_member == "epistemic_graph/client.py":
            archive.writestr(malformed_member, b"two")
    evidence = preflight.select_wheel(wheel_dir)
    distribution, module = _client_surface(tmp_path / "installed")
    distribution.direct_url = json.dumps({"url": path.as_uri(), "archive_info": {}})

    with pytest.raises(
        preflight.PreflightError,
        match="wheel-member-(path-invalid|duplicate)",
    ):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_wheel_symlink_member_fails_closed(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    path = wheel_dir / "epistemic_graph-2.26.2-py3-none-any.whl"
    metadata = b"Metadata-Version: 2.1\nName: epistemic-graph\nVersion: 2.26.2\n\n"
    symlink = zipfile.ZipInfo("epistemic_graph/client.py")
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("epistemic_graph-2.26.2.dist-info/METADATA", metadata)
        archive.writestr(symlink, b"target")
    evidence = preflight.select_wheel(wheel_dir)
    distribution, module = _client_surface(tmp_path / "installed")
    distribution.direct_url = json.dumps({"url": path.as_uri(), "archive_info": {}})

    with pytest.raises(preflight.PreflightError, match="wheel-member-symlink"):
        preflight.validate_installed_client(
            artifact=evidence,
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_missing_capability_module_or_gate_fails_closed(tmp_path: Path) -> None:
    distribution, module = _client_surface(tmp_path)

    with pytest.raises(
        preflight.PreflightError,
        match="client-capabilities-module-unavailable",
    ):
        preflight.validate_installed_client(
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: (_ for _ in ()).throw(ModuleNotFoundError()),
        )

    with pytest.raises(
        preflight.PreflightError,
        match="client-capability-gate-unavailable",
    ):
        preflight.validate_installed_client(
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


@pytest.mark.parametrize("surface", ("false", "unknown"))
def test_false_or_unknown_capability_fails_closed(tmp_path: Path, surface: str) -> None:
    distribution, module = _client_surface(tmp_path)
    if surface == "unknown":
        module.WORK_ITEM_METADATA_CAS_CAPABILITY = "unknown.capability"
    else:

        def require(_required: tuple[str, ...]) -> dict[str, Any]:
            return _manifest(False)

        module.require_client_capabilities = require

    expected_error = (
        "client-capability-unknown"
        if surface == "unknown"
        else "client-capability-unavailable"
    )
    with pytest.raises(preflight.PreflightError, match=expected_error):
        preflight.validate_installed_client(
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_source_shadowed_module_is_rejected(tmp_path: Path) -> None:
    distribution, module = _client_surface(tmp_path)
    source_file = tmp_path / "source-checkout" / preflight.CLIENT_MODULE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# source shadow\n", encoding="utf-8")
    module.__file__ = str(source_file)
    module.require_client_capabilities = lambda _required: _manifest()

    with pytest.raises(preflight.PreflightError, match="client-source-shadowed"):
        preflight.validate_installed_client(
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )


def test_editable_distribution_is_rejected(tmp_path: Path) -> None:
    distribution, module = _client_surface(tmp_path)
    distribution.direct_url = json.dumps({"dir_info": {"editable": True}})
    module.require_client_capabilities = lambda _required: _manifest()

    with pytest.raises(preflight.PreflightError, match="client-editable-install"):
        preflight.validate_installed_client(
            distribution_reader=lambda _name: distribution,
            module_importer=lambda _name: module,
        )
