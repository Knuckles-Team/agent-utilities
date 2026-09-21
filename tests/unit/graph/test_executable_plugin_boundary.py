import ast
from pathlib import Path

import agent_utilities
from agent_utilities.core.decorators import require_auth
from agent_utilities.graph import initialize_graph_from_workspace

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "agent_utilities"
FORBIDDEN_MODULES = frozenset(
    {
        "agent_utilities.agent_utilities",
        "agent_utilities.decorators",
        "agent_utilities.graph_orchestration",
    }
)


def _absolute_import_from(path: Path, node: ast.ImportFrom) -> str:
    package_parts = ["agent_utilities", *path.relative_to(PACKAGE_ROOT).parent.parts]
    prefix = package_parts[: len(package_parts) - node.level + 1]
    return ".".join([*prefix, *(node.module or "").split(".")]).rstrip(".")


def _import_from_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level:
        return _absolute_import_from(path, node)
    return node.module or ""


def _from_import_modules(path: Path, node: ast.ImportFrom) -> set[str]:
    base = _import_from_module(path, node)
    return {
        base,
        *(
            f"{base}.{alias.name}".lstrip(".")
            for alias in node.names
            if alias.name != "*"
        ),
    }


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = list(ast.walk(tree))
    direct = {
        alias.name
        for node in nodes
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = {
        module
        for node in nodes
        if isinstance(node, ast.ImportFrom)
        for module in _from_import_modules(path, node)
    }
    return direct | from_imports


def test_deleted_facade_paths_do_not_exist() -> None:
    """None of the retired facade paths remains in the package tree."""
    for relative_path in (
        "agent_utilities.py",
        "decorators.py",
        "graph_orchestration.py",
    ):
        assert not (PACKAGE_ROOT / relative_path).exists()
    assert not (PACKAGE_ROOT / "graph" / "adapters").exists()


def test_deleted_facade_modules_have_no_production_imports() -> None:
    """Production code imports owning modules rather than deleted facades."""
    imported_modules = set().union(
        *(_imported_modules(path) for path in PACKAGE_ROOT.rglob("*.py"))
    )

    assert FORBIDDEN_MODULES.isdisjoint(imported_modules)
    assert not any(
        module == "agent_utilities.graph.adapters"
        or module.startswith("agent_utilities.graph.adapters.")
        for module in imported_modules
    )


def test_import_scanner_qualifies_from_import_aliases() -> None:
    """A facade imported as a parent-package attribute cannot evade the scan."""
    tree = ast.parse("from agent_utilities import decorators")
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)
    assert "agent_utilities.decorators" in _from_import_modules(PACKAGE_ROOT, node)


def test_canonical_replacement_exports_remain_supported() -> None:
    """Current public imports cover the live capabilities the facades exposed."""
    assert callable(agent_utilities.to_boolean)
    assert callable(agent_utilities.create_model)
    assert callable(require_auth)
    assert callable(initialize_graph_from_workspace)
