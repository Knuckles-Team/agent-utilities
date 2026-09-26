"""AU cannot load the WebUI package or serve its application."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_au_has_no_reverse_webui_package_dependency() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    assert "ag-ui" not in extras
    assert not any(
        "agent-webui" in dependency
        for dependencies in extras.values()
        for dependency in dependencies
    )

    for source in (REPO_ROOT / "agent_utilities").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not item.name.startswith("agent_webui") for item in node.names
                ), source
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("agent_webui"), source
