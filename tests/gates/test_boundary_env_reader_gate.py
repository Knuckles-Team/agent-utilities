"""No environment variable name is read from more than one call site."""

import ast
from collections import defaultdict
from pathlib import Path

import pytest
from _census import PACKAGE


def duplicate_env_readers(root: Path) -> dict[str, list[str]]:
    """Map each literal environment name read at several sites to its sites."""
    sites: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*.py")):
        if "skills" in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = _env_name(node)
            if name is not None:
                sites[name].append(f"{path.relative_to(root)}:{node.lineno}")
    return {name: where for name, where in sites.items() if len(where) > 1}


def _env_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and node.args:
        func = ast.unparse(node.func)
        arg = node.args[0]
        if func in {"os.environ.get", "os.getenv", "environ.get", "getenv"}:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            return index.value
    return None


@pytest.mark.spec("MERISTEM-BOUNDARY-R016.1")
def test_retained_package_has_no_duplicate_environment_reader() -> None:
    assert duplicate_env_readers(PACKAGE) == {}


@pytest.mark.spec("MERISTEM-BOUNDARY-R016.1")
def test_gate_fails_on_an_injected_duplicate_reader(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import os\nX = os.environ.get('ARBORA_X')\n")
    (tmp_path / "b.py").write_text("import os\nY = os.getenv('ARBORA_X')\n")
    assert set(duplicate_env_readers(tmp_path)) == {"ARBORA_X"}
