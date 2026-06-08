"""Discover + evolve the pydantic-graph orchestration graph (CONCEPT:KG-2.10)."""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.enrichment.models import CodeEntity
from agent_utilities.knowledge_graph.enrichment.orchestration import (
    workflow_to_batch,
)
from agent_utilities.knowledge_graph.enrichment.pydantic_graph import (
    discover_pydantic_graph,
    persist,
    propose_workflow_evolution,
    pydantic_graph_to_workflow,
)
from agent_utilities.knowledge_graph.enrichment.registry import write_batch


class FakeBackend:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props.get("rel_type")))


def _cls(name, bases=None, methods=None, file_path="flow.py"):
    return CodeEntity(
        id=f"code:{file_path}::{name}",
        name=name,
        qualname=name,
        kind="class",
        file_path=file_path,
        line=1,
        ast_hash="h",
        bases=bases or [],
        methods=methods or [],
    )


def _fn(name, file_path="flow.py"):
    return CodeEntity(
        id=f"code:{file_path}::{name}",
        name=name,
        qualname=name,
        kind="function",
        file_path=file_path,
        line=1,
        ast_hash="h",
    )


def _sample_code():
    return [
        _cls("StartNode", bases=["BaseNode"], methods=["run"]),
        _cls("ProcessNode", bases=["BaseNode[State]"], methods=["run"]),
        _cls("FinishNode", bases=["pydantic_graph.BaseNode"], methods=["run"]),
        _cls("PlainConfig", bases=["BaseModel"]),  # unrelated
        _fn("helper"),  # unrelated function
    ]


def test_discover_finds_exactly_the_node_classes():
    d = discover_pydantic_graph(_sample_code())
    assert d["nodes"] == ["StartNode", "ProcessNode", "FinishNode"]
    assert d["node_ids"] == [
        "code:flow.py::StartNode",
        "code:flow.py::ProcessNode",
        "code:flow.py::FinishNode",
    ]
    assert d["file_paths"]["StartNode"] == "flow.py"
    # PlainConfig / helper must not be picked up.
    assert "PlainConfig" not in d["nodes"]
    assert "helper" not in d["nodes"]
    # Entrypoint heuristic finds the Start-like node.
    assert d["entrypoint"] == "StartNode"


def test_entrypoint_prefers_graph_assembly_class():
    code = _sample_code() + [_cls("OrderGraph", bases=["object"])]
    d = discover_pydantic_graph(code)
    assert d["entrypoint"] == "OrderGraph"


def test_pydantic_graph_to_workflow():
    d = discover_pydantic_graph(_sample_code())
    w = pydantic_graph_to_workflow(d, name="order-graph")
    assert w.name == "order-graph"
    assert w.steps == ["StartNode", "ProcessNode", "FinishNode"]
    assert w.orchestrates == [
        "code:flow.py::StartNode",
        "code:flow.py::ProcessNode",
        "code:flow.py::FinishNode",
    ]


def test_propose_workflow_evolution_parses_proposals():
    proposals = [
        {"change": "parallelize ProcessNode", "rationale": "no data dependency"},
        {"change": "add validation step", "rationale": "guard bad input"},
    ]
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Sure, here you go:\n" + json.dumps(proposals) + "\nThanks!"

    d = discover_pydantic_graph(_sample_code())
    out = propose_workflow_evolution(d, fake_llm, context="optimize latency")
    assert out == proposals
    # Prompt carried the discovered nodes + the context.
    assert "StartNode" in captured["prompt"]
    assert "optimize latency" in captured["prompt"]


def test_propose_workflow_evolution_empty_on_garbage():
    d = discover_pydantic_graph(_sample_code())
    assert propose_workflow_evolution(d, lambda _p: "no json here") == []
    assert propose_workflow_evolution(d, lambda _p: "{not a list}") == []


def test_persist_writes_workflow_and_orchestrates_edges():
    d = discover_pydantic_graph(_sample_code())
    backend = FakeBackend()
    n, e = persist(backend, d, name="order-graph")

    assert backend.nodes["workflow:order-graph"]["type"] == "Workflow"
    assert n == 1
    rels = {r for _, _, r in backend.edges}
    assert rels == {"ORCHESTRATES"}
    targets = {t for _, t, _ in backend.edges}
    assert targets == {
        "code:flow.py::StartNode",
        "code:flow.py::ProcessNode",
        "code:flow.py::FinishNode",
    }
    assert e == 3


def test_persist_matches_direct_workflow_batch():
    d = discover_pydantic_graph(_sample_code())
    spec = pydantic_graph_to_workflow(d, name="order-graph")
    expected = workflow_to_batch(spec)

    backend = FakeBackend()
    write_batch(backend, expected)
    direct_edges = set(backend.edges)

    backend2 = FakeBackend()
    persist(backend2, d, name="order-graph")
    assert set(backend2.edges) == direct_edges
