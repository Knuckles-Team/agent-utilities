"""Engine-RPC OTel span coverage (CONCEPT:AU-OS.observability.telemetry-observability, X2).

``_SessionRoutedAsyncClient._send`` is the sole choke point every engine RPC
funnels through (the explicit service-level methods AND every dynamically
dispatched namespace method — Cypher, ``add_node``/``add_edge``, ...), so it
is wrapped once with ``@_traced_rpc`` rather than instrumenting dozens of
individual ``GraphComputeEngine`` methods. These tests exercise the decorator
directly against a minimal stub function (matching this file's sibling tests'
established pattern of bypassing the real session/routing machinery via
``__new__`` + targeted stubbing) — no real engine connection, session, or
authentication is needed to prove the span contract.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import graph_compute
from agent_utilities.knowledge_graph.core.graph_compute import (
    _SessionRoutedAsyncClient,
    _traced_rpc,
)

pytestmark = pytest.mark.concept("AU-OS.observability.telemetry-observability")


def _client_stub(*, fixed_graph: str | None = None, graph_name: str = "") -> object:
    """A bare ``_SessionRoutedAsyncClient``-shaped stub carrying only the two
    attributes ``_traced_rpc`` reads to resolve ``engine.graph`` — no ``base``/
    namespace/session wiring, matching this directory's ``__new__``-bypass
    convention for testing one seam in isolation."""
    stub = _SessionRoutedAsyncClient.__new__(_SessionRoutedAsyncClient)
    stub._fixed_graph = fixed_graph
    stub._graph_name = graph_name
    return stub


def _telemetry_with_in_memory_exporter(monkeypatch: pytest.MonkeyPatch):
    """A real TracerProvider + InMemorySpanExporter, substituted for
    ``graph_compute``'s module-level ``_otel_trace`` reference — NOT via the
    process-global ``opentelemetry.trace.set_tracer_provider()`` (which only
    succeeds ONCE per process and would make this test order-dependent
    against every other test file that also registers a global provider,
    e.g. ``test_telemetry_engine.py``). ``_traced_rpc`` only ever calls
    ``_otel_trace.get_tracer(...)`` — substituting the module attribute
    exercises the exact same call while staying fully test-isolated; it is
    still the SAME "route through the shared provider" contract, since in
    production ``_otel_trace`` IS the real ``opentelemetry.trace`` module."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    class _FakeOtelTrace:
        @staticmethod
        def get_tracer(name: str) -> object:
            return provider.get_tracer(name)

    monkeypatch.setattr(graph_compute, "_otel_trace", _FakeOtelTrace())
    return exporter


# --------------------------------------------------------------------------- #
# The decorator IS attached to the real _send method
# --------------------------------------------------------------------------- #


def test_send_is_wrapped_by_traced_rpc():
    """Wire-First: the instrumentation is actually attached to the real
    engine-RPC choke point, not just defined and unused."""
    assert _SessionRoutedAsyncClient._send.__wrapped__ is not None
    # functools.wraps preserves the original name/doc on the wrapper.
    assert _SessionRoutedAsyncClient._send.__name__ == "_send"


# --------------------------------------------------------------------------- #
# One span per RPC, attributed with method + graph ONLY
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_traced_rpc_produces_one_span_named_for_the_method(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = _telemetry_with_in_memory_exporter(monkeypatch)

    async def _fake_send(
        self, method, params=None, graph=None, *, idempotency_key=None
    ):
        return {"ok": True}

    wrapped = _traced_rpc(_fake_send)
    stub = _client_stub(graph_name="__commons__")

    result = await wrapped(stub, "Ping")

    assert result == {"ok": True}
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "engine.Ping"
    assert spans[0].attributes["engine.method"] == "Ping"


@pytest.mark.asyncio
async def test_traced_rpc_span_carries_method_and_graph(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = _telemetry_with_in_memory_exporter(monkeypatch)

    async def _fake_send(
        self, method, params=None, graph=None, *, idempotency_key=None
    ):
        return None

    wrapped = _traced_rpc(_fake_send)
    stub = _client_stub(graph_name="tenant-42")

    await wrapped(stub, "ApplyChangeEnvelope", {"envelope": {"mutation": {}}})

    span = exporter.get_finished_spans()[0]
    assert span.attributes["engine.method"] == "ApplyChangeEnvelope"
    assert span.attributes["engine.graph"] == "tenant-42"


@pytest.mark.asyncio
async def test_traced_rpc_prefers_explicit_graph_over_the_client_default(
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = _telemetry_with_in_memory_exporter(monkeypatch)

    async def _fake_send(
        self, method, params=None, graph=None, *, idempotency_key=None
    ):
        return None

    wrapped = _traced_rpc(_fake_send)
    stub = _client_stub(fixed_graph="fixed-graph", graph_name="default-graph")

    await wrapped(stub, "Health", graph="explicit-graph")

    span = exporter.get_finished_spans()[0]
    assert span.attributes["engine.graph"] == "explicit-graph"


@pytest.mark.asyncio
async def test_traced_rpc_never_stamps_params_as_a_span_attribute(
    monkeypatch: pytest.MonkeyPatch,
):
    """Redaction (X2 hard requirement): ``params`` may carry row content or
    secrets — it must never reach a span attribute, only ``method``/``graph``."""
    exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    secret_payload = {
        "envelope": {"mutation": {"secret": "sk-live-AKIAFAKESECRETVALUE0123"}}  # nosec B105 - test fixture
    }

    async def _fake_send(
        self, method, params=None, graph=None, *, idempotency_key=None
    ):
        return None

    wrapped = _traced_rpc(_fake_send)
    stub = _client_stub(graph_name="g1")

    await wrapped(stub, "ApplyChangeEnvelope", secret_payload)

    span = exporter.get_finished_spans()[0]
    assert set(span.attributes.keys()) == {"engine.method", "engine.graph"}
    for value in span.attributes.values():
        assert "sk-live-AKIAFAKESECRETVALUE0123" not in str(value)


@pytest.mark.asyncio
async def test_traced_rpc_propagates_the_wrapped_functions_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tracing must never swallow a real engine error."""
    _telemetry_with_in_memory_exporter(monkeypatch)

    async def _boom(self, method, params=None, graph=None, *, idempotency_key=None):
        raise RuntimeError("engine unreachable")

    wrapped = _traced_rpc(_boom)
    stub = _client_stub(graph_name="g1")

    with pytest.raises(RuntimeError, match="engine unreachable"):
        await wrapped(stub, "Health")


# --------------------------------------------------------------------------- #
# Import-guard — zero overhead when opentelemetry itself is unavailable
# --------------------------------------------------------------------------- #


def test_traced_rpc_returns_the_bare_function_when_otel_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the ``opentelemetry`` API import failed at module load (simulated
    here), ``_send`` must be returned COMPLETELY UNCHANGED — no wrapper
    indirection, zero overhead, matching the current (OTel-absent) behavior."""
    monkeypatch.setattr(graph_compute, "_otel_trace", None)

    async def _fake_send(
        self, method, params=None, graph=None, *, idempotency_key=None
    ):
        return "unchanged"

    wrapped = graph_compute._traced_rpc(_fake_send)
    assert wrapped is _fake_send
