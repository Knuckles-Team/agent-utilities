"""Proof that the retired loopback certification entry point stays removed."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.spec("AU-BOUNDARY-R002.3.3")
def test_retired_loopback_oidc_has_no_entry_point_or_imports():
    assert not (ROOT / "scripts/certification/loopback_oidc.py").exists()
    assert not (ROOT / "tests/unit/scripts/test_loopback_oidc.py").exists()
    for directory in (ROOT / "scripts", ROOT / "tests"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    names.extend(alias.name for alias in node.names)
                elif (
                    isinstance(node, ast.Call)
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "__import__"
                        or isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module"
                    )
                ):
                    names = [node.args[0].value]
                else:
                    continue
                assert not any("loopback_oidc" in name.split(".") for name in names), (
                    path
                )
