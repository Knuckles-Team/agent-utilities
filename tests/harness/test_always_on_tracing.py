"""Always-on KG-native tracing via the decorator path (CONCEPT:AU-OS.config.model-factory-passthrough).

When a KG trace sink is injected (as the daemon does at startup), the @trace /
@generation decorators capture EVERY call as a Trace/Span/Generation subgraph —
independent of any Langfuse key. With no sink and no Langfuse, tracing short-circuits
(zero overhead).
"""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.harness import tracing
from agent_utilities.harness.tracing import (
    generation,
    get_kg_trace_sink,
    set_kg_trace_sink,
    trace,
)
from agent_utilities.harness.trace_backend import KGTraceBackend


class _FakeKG:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id: str, **props) -> None:
        self.nodes[node_id] = props

    def link_nodes(self, src: str, dst: str, rel, **_kw) -> None:
        self.edges.append((src, dst, str(rel)))


@pytest.fixture
def kg_sink():
    prev = get_kg_trace_sink()
    kg = _FakeKG()
    set_kg_trace_sink(KGTraceBackend(backend=kg))
    try:
        yield kg
    finally:
        set_kg_trace_sink(prev)


def test_tracing_inactive_without_sink_or_langfuse(monkeypatch):
    # No sink, no Langfuse → decorator is a pass-through (no capture, zero overhead).
    set_kg_trace_sink(None)
    monkeypatch.setattr(tracing.config, "langfuse_secret_key", "", raising=False)
    assert tracing._tracing_active() is False

    @trace(name="noop")
    async def f() -> str:
        return "ok"

    assert asyncio.run(f()) == "ok"


def test_decorators_capture_trace_and_generation_subgraph(kg_sink):
    @generation(name="llm_call", model="gpt-4o")
    async def call_llm(prompt: str) -> str:
        return f"answer to {prompt}"

    @trace(name="agent_run", tags=["live"])
    async def agent_run() -> str:
        return await call_llm("hi")

    out = asyncio.run(agent_run())
    assert out == "answer to hi"

    # The root trace + the generation child were persisted to the KG backend.
    types = {p.get("type") for p in kg_sink.nodes.values()}
    assert "trace" in types
    assert "generation" in types
    # The generation captured its model and is linked under the trace.
    gen = next(p for p in kg_sink.nodes.values() if p.get("type") == "generation")
    assert gen.get("model") == "gpt-4o"
    assert any(str(rel).endswith("has_generation") for _s, _d, rel in kg_sink.edges)


def test_error_marks_trace_status(kg_sink):
    @trace(name="boom")
    async def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        asyncio.run(boom())
    trace_node = next(p for p in kg_sink.nodes.values() if p.get("type") == "trace")
    assert trace_node.get("status") == "error"
