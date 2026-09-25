"""AUD-31 destination gate contracts."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_component_destinations.py"
SPEC = importlib.util.spec_from_file_location("check_component_destinations", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _owner(
    path: Path, repository: str, component: str, legacy: list[str], scripts: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "architecture-destination-owner/v1",
                "owner_repository": repository,
                "components": [
                    {
                        "component_id": component,
                        "parent_layer": "adapters",
                        "capability_ids": [component],
                        "owned_source_roots": [repository.replace("-", "_")],
                        "legacy_au_source_roots": legacy,
                        "console_scripts": scripts,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def owned_repos(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    au = tmp_path / "au"
    au.mkdir()
    graph = tmp_path / "graph.yml"
    sdk = tmp_path / "sdk.yml"
    au_manifest = au / "architecture/component-registry.destinations.yml"
    _owner(au_manifest, "agent-utilities", "au.agent", [], [])
    _owner(
        graph, "graph-os", "graph-os.gateway", ["agent_utilities/gateway"], ["graph-os"]
    )
    _owner(sdk, "agent-connector-sdk", "sdk.runner", ["agent_utilities/connector"], [])
    (au / "pyproject.toml").write_text(
        '[project.scripts]\ngraph-os = "agent_utilities.gateway.old:main"\n',
        encoding="utf-8",
    )
    _git(au, "init", "-q")
    _git(
        au,
        "add",
        "--",
        "pyproject.toml",
        "architecture/component-registry.destinations.yml",
    )
    _git(
        au,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "base",
    )
    return au, au_manifest, graph, sdk


def _validate(repos: tuple[Path, Path, Path, Path], paths: list[str]) -> list[str]:
    au, au_manifest, graph, sdk = repos
    return gate.validate_changes(
        au_repo=au,
        au_manifest=au_manifest,
        graph_manifest=graph,
        sdk_manifest=sdk,
        base="HEAD",
        changed_paths=paths,
    )


def test_unchanged_legacy_script_is_allowed_during_cutover(
    owned_repos: tuple[Path, Path, Path, Path],
) -> None:
    assert _validate(owned_repos, []) == []


def test_new_au_code_in_graph_os_destination_fails(
    owned_repos: tuple[Path, Path, Path, Path],
) -> None:
    assert _validate(owned_repos, ["agent_utilities/gateway/new.py"]) == [
        "agent_utilities/gateway/new.py: destination is graph-os.gateway"
    ]


def test_changed_foreign_console_script_fails(
    owned_repos: tuple[Path, Path, Path, Path],
) -> None:
    au = owned_repos[0]
    (au / "pyproject.toml").write_text(
        '[project.scripts]\ngraph-os = "agent_utilities.gateway.new:main"\n',
        encoding="utf-8",
    )
    assert _validate(owned_repos, ["pyproject.toml"]) == [
        "project.scripts.graph-os: destination is graph-os.gateway"
    ]


def test_duplicate_owner_component_rejected(
    owned_repos: tuple[Path, Path, Path, Path],
) -> None:
    sdk = owned_repos[3]
    data = yaml.safe_load(sdk.read_text(encoding="utf-8"))
    data["components"][0]["component_id"] = "graph-os.gateway"
    sdk.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate component authority"):
        _validate(owned_repos, [])
