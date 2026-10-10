"""AU-BOUNDARY-R030.9.2: the leg scaffold compiles via the SDK pack compiler."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import rdflib

from scripts import scaffold_ontology_leg as script


@pytest.mark.spec("AU-BOUNDARY-R030.9.2")
def test_scaffolds_leg_as_parseable_turtle() -> None:
    ttl = script.scaffold_leg("fixture-mcp", ["Widget"])
    graph = rdflib.Graph()
    graph.parse(data=ttl, format="turtle")
    assert len(graph) > 0
    assert "Widget" in ttl


@pytest.mark.spec("AU-BOUNDARY-R030.9.2")
def test_refuses_empty_input() -> None:
    with pytest.raises(ValueError):
        script.build_declaration("fixture-mcp", [])
    with pytest.raises(ValueError):
        script.build_declaration(" ", ["Widget"])
    assert script.main(["fixture-mcp"]) == 2


@pytest.mark.spec("AU-BOUNDARY-R030.9.2")
def test_imports_no_local_emitter() -> None:
    tree = ast.parse(Path(script.__file__).read_text(encoding="utf-8"))
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(
        "manifest_compiler" in m or "ontology.value_types" in m for m in modules
    )
    assert any(m.endswith("core.ontology_publisher") for m in modules)
