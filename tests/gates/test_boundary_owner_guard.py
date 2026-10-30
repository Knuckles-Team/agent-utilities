"""Exercise the owner guard with isolated, service-free checkout fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_boundary_owners import check


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    (tmp_path / "architecture").mkdir()
    (tmp_path / "meristem").mkdir()
    (tmp_path / "meristem/__init__.py").write_text("")
    (tmp_path / "meristem/main.py").write_text("")
    registry = {
        "schema": "au-boundary-owners/v1",
        "owners": {
            "meristem": {"modules": ["meristem/__init__.py", "meristem/main.py"]},
            "arbora": {"prefixes": ["meristem/deployment"]},
            "scion-sdk": {"prefixes": ["meristem/protocols/source_connectors"]},
            "medulla": {"prefixes": ["meristem/knowledge_graph"]},
        },
    }
    (tmp_path / "architecture/component-registry.yml").write_text(json.dumps(registry))
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\nagent = "meristem.main:main"\n'
    )
    return tmp_path


def test_owner_guard_accepts_registered_harness(checkout: Path) -> None:
    assert check(checkout) == []


@pytest.mark.spec("MERISTEM-BOUNDARY-R038")
@pytest.mark.parametrize(
    ("path", "owner"),
    [
        ("meristem/deployment/server.py", "arbora"),
        ("meristem/protocols/source_connectors/source.py", "scion-sdk"),
        ("meristem/knowledge_graph/store.py", "medulla"),
    ],
)
def test_owner_guard_rejects_foreign_module_even_if_registered(
    checkout: Path, path: str, owner: str
) -> None:
    source = checkout / path
    source.parent.mkdir(parents=True)
    source.write_text("")
    registry_path = checkout / "architecture/component-registry.yml"
    registry = json.loads(registry_path.read_text())
    registry["owners"]["meristem"]["modules"].append(path)
    registry_path.write_text(json.dumps(registry))

    assert f"{owner}-owned module in meristem: {path}" in check(checkout)


@pytest.mark.spec("MERISTEM-BOUNDARY-R038")
def test_owner_guard_rejects_wrong_owner_script_and_unregistered_module(
    checkout: Path,
) -> None:
    (checkout / "meristem/unregistered.py").write_text("")
    (checkout / "pyproject.toml").write_text(
        '[project.scripts]\nserver = "meristem.deployment.server:main"\n'
    )

    assert check(checkout) == [
        "unregistered module: meristem/unregistered.py",
        "unowned console script target: server",
    ]


def test_owner_guard_rejects_stale_manifest(checkout: Path) -> None:
    (checkout / "meristem/main.py").unlink()
    assert "missing registered module: meristem/main.py" in check(checkout)
