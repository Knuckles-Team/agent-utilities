from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _module():
    source = Path(__file__).parents[3] / "scripts" / "audit_fleet_dependencies.py"
    spec = importlib.util.spec_from_file_location("audit_fleet_dependencies", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(source.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_cargo_lock_requires_registry_checksum(tmp_path: Path) -> None:
    module = _module()
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        """\
version = 4
[[package]]
name = "example"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
""",
        encoding="utf-8",
    )

    with pytest.raises(module.AuditError, match="unverified registry package"):
        module.parse_cargo_lock(lock)


def test_pnpm_lock_requires_integrity(tmp_path: Path) -> None:
    module = _module()
    lock = tmp_path / "pnpm-lock.yaml"
    lock.write_text(
        """\
lockfileVersion: '9.0'
packages:
  example@1.0.0:
    resolution: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(module.AuditError, match="unverified package"):
        module.parse_pnpm_lock(lock)


def test_single_source_snapshot_inventories_no_git_root(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "uv.lock").write_text(
        """\
version = 1
revision = 3
requires-python = ">=3.11"

[[package]]
name = "example"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
wheels = [
  { url = "https://files.pythonhosted.org/example.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
]
""",
        encoding="utf-8",
    )

    repositories = module._single_source_snapshot(tmp_path)
    result = module.inventory(tmp_path, repositories=repositories)

    assert len(result) == 1
    assert result[0].coordinates == {
        module.Coordinate("PyPI", "example", "1.0.0")
    }


def test_single_source_snapshot_rejects_symlink(tmp_path: Path) -> None:
    module = _module()
    repository = tmp_path / "repository"
    repository.mkdir()
    link = tmp_path / "repository-link"
    link.symlink_to(repository, target_is_directory=True)

    with pytest.raises(module.AuditError, match="must be a directory"):
        module._single_source_snapshot(link)
