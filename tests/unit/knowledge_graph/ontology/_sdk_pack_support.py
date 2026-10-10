"""Shared AST helpers for the SDK-pack wiring tests (AU-BOUNDARY-R030.x)."""

from __future__ import annotations

import ast
from pathlib import Path

LOCAL_COMPILER = "agent_utilities.knowledge_graph.ontology.manifest_compiler"
SDK_PACK = "agent_connector_sdk.manifest.ontology_pack"
SDK_MODEL = "agent_connector_sdk.manifest.model"


def parse_script(script: Path) -> ast.Module:
    assert script.is_file()
    return ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def imported_names_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def assert_local_compiler_dropped(script: Path) -> None:
    tree = parse_script(script)
    assert LOCAL_COMPILER not in imported_modules(tree)
    local = imported_names_from(tree, LOCAL_COMPILER)
    assert "compile_manifest" not in local
    assert "export_manifest_ttl" not in local


def assert_sdk_pack_wired(script: Path) -> None:
    tree = parse_script(script)
    assert SDK_PACK in imported_modules(tree)
    assert "compile_manifest_ontology" in imported_names_from(tree, SDK_PACK)
    assert SDK_MODEL in imported_modules(tree)
