"""AU-SEMANTIC-R006.3: shacl_gate.py builds Turtle without rdflib.

Covers the import census (no ``rdflib`` import anywhere in the module) and
the behavior through the engine path: ``build_data_graph`` renders a plain
Turtle document (reusing R006.2's ``workflow_gate`` serializer helpers) that
is handed, unparsed, straight to the engine's
``shacl_validate_committed``/``shacl_validate_committed_async`` methods.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.pipeline.phases import shacl_gate

_MODULE_PATH = Path(shacl_gate.__file__)


@pytest.mark.spec("AU-SEMANTIC-R006.3")
def test_shacl_gate_module_imports_no_rdflib():
    """Import census: no ``import rdflib`` / ``from rdflib`` anywhere."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] != "rdflib" for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "rdflib"


class _FakeEngineGraph:
    """Minimal stand-in for the LPG the fallback path iterates."""

    def __init__(self, nodes: dict[str, dict]) -> None:
        self._nodes = nodes

    def get_rdf(self):
        return None

    def nodes(self, data: bool = False):
        if data:
            return list(self._nodes.items())
        return list(self._nodes.keys())

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes


@pytest.mark.spec("AU-SEMANTIC-R006.3")
def test_build_data_graph_renders_turtle_via_reused_serializer():
    """No-engine fallback renders Turtle text (not an rdflib Graph object)."""
    graph = _FakeEngineGraph(
        {"tool-1": {"node_type": "tool", "name": "Example"}}
    )
    turtle = shacl_gate.build_data_graph(graph)
    assert isinstance(turtle, str)
    assert "<http://knuckles.team/kg#tool-1>" in turtle
    assert "a <http://knuckles.team/kg#Tool>" in turtle
    assert '"Example"' in turtle


class _FakeReport:
    conforms = True
    results: list = []


class _FakeCommittedGraph(_FakeEngineGraph):
    """Proves the rendered Turtle is handed straight to the engine's
    committed-SHACL method with no further rdflib round-trip."""

    def shacl_validate_committed(self, turtle: str):
        self.received_turtle = turtle
        return _FakeReport()


@pytest.mark.spec("AU-SEMANTIC-R006.3")
def test_validate_graph_routes_rendered_turtle_through_engine():
    graph = _FakeCommittedGraph({"tool-1": {"node_type": "tool"}})
    conforms, violations, _report_text = shacl_gate.validate_graph(graph)
    assert conforms is True
    assert violations == {}
    assert "<http://knuckles.team/kg#tool-1>" in graph.received_turtle
