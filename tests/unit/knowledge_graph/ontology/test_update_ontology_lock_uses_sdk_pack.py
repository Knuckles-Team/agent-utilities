"""AU-BOUNDARY-R030.3: ``scripts/update_ontology_lock.py`` is wired off AU's
hand-written ``agent_utilities.knowledge_graph.ontology.manifest_compiler``
and onto the SDK's typed, source-agnostic pack compiler
(``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology``) —
mirroring the re-validate-then-compile pattern
``connector_manifest_gate._compiled_manifest_graph`` already uses for the
compile-before-sync gate (AU-BOUNDARY-R030.1).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "update_ontology_lock.py"
)


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _imported_names_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.spec("AU-BOUNDARY-R030.3")
def test_update_ontology_lock_no_longer_imports_the_local_compiler():
    assert _SCRIPT.is_file()
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"), filename=str(_SCRIPT))

    modules = _imported_modules(tree)

    assert "agent_utilities.knowledge_graph.ontology.manifest_compiler" not in modules
    assert "compile_manifest" not in _imported_names_from(
        tree, "agent_utilities.knowledge_graph.ontology.manifest_compiler"
    )
    assert "export_manifest_ttl" not in _imported_names_from(
        tree, "agent_utilities.knowledge_graph.ontology.manifest_compiler"
    )


@pytest.mark.spec("AU-BOUNDARY-R030.3")
def test_update_ontology_lock_calls_the_sdk_pack_compiler():
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"), filename=str(_SCRIPT))

    assert "agent_connector_sdk.manifest.ontology_pack" in _imported_modules(tree)
    assert "compile_manifest_ontology" in _imported_names_from(
        tree, "agent_connector_sdk.manifest.ontology_pack"
    )
    assert "agent_connector_sdk.manifest.model" in _imported_modules(tree)
