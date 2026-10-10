"""AU-SEMANTIC-R006.4: connector certification submits Turtle, not an rdflib graph."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.integrations import connector_certification as cc

_MODULE_PATH = Path(cc.__file__)
_SUBJECT = "<urn:graphos:connector-certification:{}>"


@pytest.mark.spec("AU-SEMANTIC-R006.4")
def test_connector_certification_imports_no_rdflib():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "rdflib" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "rdflib"


@pytest.mark.spec("AU-SEMANTIC-R006.4")
def test_certification_turtle_payload_and_engine_routing(monkeypatch):
    envelopes = [
        SimpleNamespace(typed_payload={"type": "Ticket"}),
        SimpleNamespace(typed_payload=None),
    ]
    turtle = cc._build_certification_turtle(envelopes)
    assert f"{_SUBJECT.format(0)} a <http://knuckles.team/kg#Ticket>" in turtle
    assert f"{_SUBJECT.format(1)} a <http://knuckles.team/kg#Document>" in turtle
    assert '<http://knuckles.team/kg#tenantReference> "bound"' in turtle

    seen = {}

    def fake(data, shapes):
        seen["data"] = data
        return "epistemic-graph"

    monkeypatch.setattr(cc, "_validate_native_shacl", fake)
    bundle = SimpleNamespace(shapes_text="shapes")
    assert (
        cc._semantic_validation(bundle, envelopes, require_engine_shacl=True)
        == "epistemic-graph"
    )
    assert seen["data"] == turtle
