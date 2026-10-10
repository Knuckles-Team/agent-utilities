"""AU-SEMANTIC-R006.5.5: the bundle generator submits shapes text, not an rdflib graph."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.integrations import connector_certification

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_connector_capability_bundles.py"
)


@pytest.mark.spec("AU-SEMANTIC-R006.5.5")
def test_generator_imports_no_rdflib() -> None:
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "rdflib" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "rdflib"


@pytest.mark.spec("AU-SEMANTIC-R006.5.5")
def test_shapes_text_goes_to_engine_with_empty_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_bundles_r00655", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen: list[tuple[str, str]] = []

    def fake(data: str, shapes: str) -> str:
        seen.append((data, shapes))
        return "epistemic-graph"

    monkeypatch.setattr(connector_certification, "_validate_native_shacl", fake)
    module._validate_shapes_text("@prefix sh: <http://www.w3.org/ns/shacl#> .")
    assert seen == [("", "@prefix sh: <http://www.w3.org/ns/shacl#> .")]
