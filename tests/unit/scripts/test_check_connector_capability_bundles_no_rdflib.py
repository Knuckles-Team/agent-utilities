"""AU-SEMANTIC-R006.5.4: the capability-bundle gate reads shapes without rdflib."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "check_connector_capability_bundles.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_ccb_no_rdflib", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.spec("AU-SEMANTIC-R006.5.4")
def test_gate_script_imports_no_rdflib():
    tree = ast.parse(_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "rdflib" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "rdflib"


@pytest.mark.spec("AU-SEMANTIC-R006.5.4")
def test_shape_target_classes_reads_prefixed_and_full_iri_forms():
    module = _load()
    text = (
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        "shape:A sh:targetClass :Ticket ;\n sh:property [] .\n"
        "shape:B sh:targetClass <http://knuckles.team/kg#Document> .\n"
        "shape:C sh:targetClass <http://other.example/#Nope> .\n"
    )
    assert module._shape_target_classes(text) == {"Ticket", "Document"}
    assert module._shape_target_classes("not turtle") == set()
