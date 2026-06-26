"""KG-native trace backend (CONCEPT:OS-5.68).

Every trace persists as a TraceNode → SpanNode/GenerationNode subgraph so traces are
graph-queryable (the moat over an opaque trace store), with per-generation cost from
the shared pricing catalog.
"""

from __future__ import annotations

import asyncio

from agent_utilities.harness.trace_backend import (
    KGTraceBackend,
    create_trace_backend,
)
from agent_utilities.models.knowledge_graph import (
    GenerationNode,
    RegistryEdgeType,
    SpanNode,
    TraceNode,
)


class _FakeKG:
    """Duck-typed KG facade: records add_node / link_nodes calls."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id: str, **props) -> None:
        self.nodes[node_id] = props

    def link_nodes(self, src: str, dst: str, rel, **_kw) -> None:
        self.edges.append((src, dst, str(rel)))


def test_emit_persists_trace_subgraph_and_rolls_up_cost():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)

    trace = TraceNode(id="trace:1", name="agent.run", agent="planner")
    span = SpanNode(id="span:1", name="retrieve", trace_id="trace:1", span_kind="retrieval")
    gen = GenerationNode(
        id="gen:1",
        name="llm",
        trace_id="trace:1",
        parent_span_id="span:1",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
    )

    be.emit_trace(trace, spans=[span], generations=[gen])

    # All three nodes persisted with their types.
    assert kg.nodes["trace:1"]["type"] == "trace"
    assert kg.nodes["span:1"]["type"] == "span"
    assert kg.nodes["gen:1"]["type"] == "generation"
    # Subgraph edges: trace HAS_SPAN span; span HAS_GENERATION gen.
    assert ("trace:1", "span:1", str(RegistryEdgeType.HAS_SPAN)) in kg.edges
    assert ("span:1", "gen:1", str(RegistryEdgeType.HAS_GENERATION)) in kg.edges
    # Trace-level token rollup from its generations.
    assert trace.input_tokens == 1000
    assert trace.output_tokens == 500


def test_readback_summary_and_scores():
    be = KGTraceBackend()
    trace = TraceNode(id="trace:2", name="run", metadata={"score": 0.8})
    gen = GenerationNode(
        id="gen:2", name="llm", trace_id="trace:2", model="gpt-4o", error="boom"
    )
    be.emit_trace(trace, generations=[gen])

    summary = asyncio.run(be.get_trace_summary("trace:2"))
    assert summary["id"] == "trace:2"
    assert summary["error"] == "boom"

    scores = asyncio.run(be.get_trace_scores(["trace:2"]))
    assert scores["trace:2"] == 0.8

    missing = asyncio.run(be.get_trace_summary("nope"))
    assert missing["error"] == "not_found"


def test_kg_is_the_default_backend_when_no_vendor_configured(monkeypatch):
    # No Langfuse/OTel/trace_dir → KG-native is the default sink (always-on, OS-5.68).
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    be = create_trace_backend()
    assert isinstance(be, KGTraceBackend)
    assert isinstance(create_trace_backend(backend_type="kg"), KGTraceBackend)
