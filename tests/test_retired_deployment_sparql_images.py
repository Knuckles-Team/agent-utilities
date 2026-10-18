"""Deletion censuses for retired pre-3.0.0 AU paths (boundary deconstruction).

After the 3.0.0 cut agent-utilities is the in-run harness only. These tests
prove, from any checkout of main without network or git subprocesses, that
the retired paths are absent from the tree and that no remaining source
imports them.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _python_files() -> list[Path]:
    """Every ``.py`` file in the checkout, skipping hidden directories."""
    return sorted(
        path
        for path in _ROOT.rglob("*.py")
        if not any(part.startswith(".") for part in path.parts)
    )


def _absolute_imports(py_file: Path) -> list[tuple[int, str]]:
    """All imports in ``py_file`` as ``(line, absolute module)`` pairs.

    Relative imports are resolved against the file's containing package.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    base = py_file.relative_to(_ROOT).with_suffix("").parts[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            anchor = base[: len(base) - (level - 1)] if level else []
            module = ".".join((*anchor, *node.module.split(".")) if node.module else anchor)
            if module:
                found.append((node.lineno, module))
    return found


def _importers_of(module: str) -> list[str]:
    """Report lines for every source importing ``module`` or a sub-module."""
    hits: list[str] = []
    for py_file in _python_files():
        try:
            imported = _absolute_imports(py_file)
        except SyntaxError as exc:
            pytest.fail(f"unparseable source {py_file.relative_to(_ROOT)}: {exc}")
        hits.extend(
            f"{py_file.relative_to(_ROOT)}:{line} imports {name}"
            for line, name in imported
            if name == module or name.startswith(module + ".")
        )
    return hits


@pytest.mark.spec("AU-BOUNDARY-R002.2")
def test_deployment_package_and_backends_module_deleted() -> None:
    """AU-BOUNDARY-R002.2: ``deployment/backends.py`` and ``deployment/`` are gone.

    Acceptance: the file and its production imports are gone. Proves the
    path is absent from the tree and that no remaining source imports
    ``agent_utilities.deployment``.
    """
    deployment = _ROOT / "agent_utilities" / "deployment"
    assert not deployment.exists(), "agent_utilities/deployment/ is still present on main"
    hits = _importers_of("agent_utilities.deployment")
    assert not hits, "importers of agent_utilities.deployment remain:\n" + "\n".join(hits)
