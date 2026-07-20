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
from agent_utilities.harness.trace_backend import KGTraceBackend
from agent_utilities.harness.tracing import (
    generation,
    get_kg_trace_sink,
    set_kg_trace_sink,
    trace,
)


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
    monkeypatch.setattr(tracing.config, "langfuse_secret_key_ref", "", raising=False)
    monkeypatch.setattr(tracing.config, "trace_export_enabled", False)
    assert tracing._tracing_active() is False

    @trace(name="noop")
    async def f() -> str:
        return "ok"

    assert asyncio.run(f()) == "ok"


def test_langfuse_credentials_do_not_authorize_trace_export(monkeypatch):
    from agent_utilities.harness import trace_backend

    calls = 0

    def create_backend(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Langfuse backend must not be created")

    set_kg_trace_sink(None)
    monkeypatch.setattr(trace_backend, "create_trace_backend", create_backend)
    monkeypatch.setattr(
        tracing.config,
        "langfuse_secret_key_ref",
        "env://SYNTHETIC_LANGFUSE_SECRET",
    )
    monkeypatch.setattr(tracing.config, "trace_export_enabled", False)
    assert tracing._tracing_active() is False

    @trace(name="export_disabled")
    async def export_disabled() -> str:
        return "ok"

    assert asyncio.run(export_disabled()) == "ok"
    assert calls == 0


def test_metadata_only_mode_drops_kg_input_and_output(monkeypatch, kg_sink):
    monkeypatch.setattr(tracing.config, "trace_export_enabled", False)
    monkeypatch.setattr(tracing.config, "langfuse_capture_content", False)

    @trace(name="metadata_only")
    async def metadata_only(value: str) -> str:
        return f"result:{value}"

    assert asyncio.run(metadata_only("private-content")) == "result:private-content"
    trace_node = next(
        p for p in kg_sink.nodes.values() if p.get("node_type") == "trace"
    )
    assert trace_node.get("input") == ""
    assert trace_node.get("output") == ""


def test_metadata_only_mode_omits_langfuse_payload_content(monkeypatch):
    from agent_utilities.harness import trace_backend

    class _Api:
        def __init__(self) -> None:
            self.batches: list[list[dict]] = []

        def ingestion_batch(self, *, batch):
            self.batches.append(batch)

    api = _Api()
    backend = trace_backend.LangfuseTraceBackend()
    backend._api = api
    monkeypatch.setattr(
        trace_backend, "create_trace_backend", lambda **_kwargs: backend
    )
    monkeypatch.setattr(
        tracing.config,
        "langfuse_secret_key_ref",
        "env://SYNTHETIC_LANGFUSE_SECRET",
    )
    monkeypatch.setattr(tracing.config, "trace_export_enabled", True)
    monkeypatch.setattr(tracing.config, "langfuse_capture_content", False)
    set_kg_trace_sink(None)

    @generation(name="metadata_generation", model="synthetic-model")
    async def generate(prompt: str) -> str:
        return f"generated:{prompt}"

    @trace(
        name="metadata_trace",
        metadata={
            "private_note": "private-content",
            "provider": "synthetic-provider",
        },
    )
    async def run(value: str) -> str:
        return await generate(value)

    assert asyncio.run(run("private-content")) == "generated:private-content"
    bodies = [event["body"] for batch in api.batches for event in batch]
    assert bodies
    assert all("input" not in body and "output" not in body for body in bodies)
    assert all("private_note" not in body["metadata"] for body in bodies)
    assert all("private-content" not in repr(body) for body in bodies)


def test_metadata_only_mode_drops_exception_content(monkeypatch):
    from agent_utilities.harness import trace_backend

    class _Api:
        def __init__(self) -> None:
            self.batches: list[list[dict]] = []

        def ingestion_batch(self, *, batch):
            self.batches.append(batch)

    api = _Api()
    backend = trace_backend.LangfuseTraceBackend()
    backend._api = api
    monkeypatch.setattr(
        trace_backend, "create_trace_backend", lambda **_kwargs: backend
    )
    monkeypatch.setattr(
        tracing.config,
        "langfuse_secret_key_ref",
        "env://SYNTHETIC_LANGFUSE_SECRET",
    )
    monkeypatch.setattr(tracing.config, "trace_export_enabled", True)
    monkeypatch.setattr(tracing.config, "langfuse_capture_content", False)
    set_kg_trace_sink(None)

    @trace(name="metadata_error")
    async def fail() -> None:
        raise ValueError("private-content")

    with pytest.raises(ValueError, match="private-content"):
        asyncio.run(fail())
    bodies = [event["body"] for batch in api.batches for event in batch]
    assert bodies
    assert all("private-content" not in repr(body) for body in bodies)


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
    types = {p.get("node_type") for p in kg_sink.nodes.values()}
    assert "trace" in types
    assert "generation" in types
    # The generation captured its model and is linked under the trace.
    gen = next(p for p in kg_sink.nodes.values() if p.get("node_type") == "generation")
    assert gen.get("model") == "gpt-4o"
    assert any(str(rel).endswith("has_generation") for _s, _d, rel in kg_sink.edges)


def test_error_marks_trace_status(kg_sink):
    @trace(name="boom")
    async def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        asyncio.run(boom())
    trace_node = next(
        p for p in kg_sink.nodes.values() if p.get("node_type") == "trace"
    )
    assert trace_node.get("status") == "error"
