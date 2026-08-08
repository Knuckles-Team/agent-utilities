"""D-CDX-21: a run whose ``on_graph_start``/``on_graph_end`` pair crosses an
OTel ``Context`` boundary must not attempt the corrupting cross-context
``opentelemetry.context.detach()``.

``TelemetryEngine.on_graph_start`` opens the run's ``agent.run`` span via
``tracer.start_as_current_span(...)``, but calls its context manager's
``__enter__()``/``__exit__()`` manually and split across two separate methods
(``on_graph_start``/``on_graph_end``) so the span stays "current" (ambient)
for the run's whole lifetime, letting unrelated code elsewhere nest child
spans under it. ``opentelemetry.trace.use_span`` — what ``start_as_current_
span`` is built on — does ``token = context.attach(...)`` then, in its
``finally``, ``context.detach(token)``. Per PEP 567, a ``contextvars.Token``
is only valid to ``.reset()`` (what ``context.detach`` does under the hood)
in the EXACT ``Context`` it was created in. A multi-stage delegation run
routinely crosses an asyncio Task boundary between ``on_graph_start``'s
attach and ``on_graph_end``'s detach (an MCP child call runs in its own
shielded task — ``mcp/child_resilience.py::_call_once`` — and a ServiceNow
session teardown was the reproduction case), so this is not hypothetical.

OTel's own ``context.detach()`` already swallows the resulting ``ValueError``
internally (logs "Failed to detach context" via ``logger.exception``), so
nothing crashes — the damage is silent: the run's own span still exports
fine, but the ORIGINATING context's ambient "current span" state is left
corrupted (never reset), which can misattribute whatever runs next in that
context's lineage. The fix snapshots the attach ``Context`` in
``on_graph_start`` and, in ``on_graph_end``, only calls the real
``span_cm.__exit__()`` (which invokes ``context.detach()``) when the current
``Context`` still IS that same object; otherwise it ends the span directly
(``span.end()``) without touching ambient context state at all.

The test below FAILS against the pre-fix code (unconditional
``span_cm.__exit__()``): reverting the ``on_graph_end`` guard makes
``opentelemetry.context.detach`` get invoked even when the contexts differ,
which this test asserts must NOT happen.
"""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.observability import TelemetryEngine


def _telemetry_with_in_memory_exporter(monkeypatch: pytest.MonkeyPatch):
    """Same pattern as ``tests/unit/test_telemetry_engine.py``."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = TelemetryEngine(enable_audit=False, enable_tokens=False)
    telemetry._initialized = True
    telemetry._tracer = provider.get_tracer("test-cross-context")
    return telemetry, exporter


def test_on_graph_start_and_end_in_the_same_context_still_detaches_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: the common case (both calls in the same Context/Task) must
    be unaffected by the fix -- the real ``context.detach`` still runs."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)

    import opentelemetry.context as otel_context_module

    detach_calls: list[object] = []
    real_detach = otel_context_module.detach

    def _spy_detach(token):
        detach_calls.append(token)
        return real_detach(token)

    monkeypatch.setattr(otel_context_module, "detach", _spy_detach)

    ctx = contextvars.Context()
    ctx.run(telemetry.on_graph_start, run_id="run-same", agent_id="a", query="q")
    ctx.run(telemetry.on_graph_end, run_id="run-same", status="success")

    assert len(detach_calls) == 1, "same-context close must still call context.detach"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "agent.run"


def test_on_graph_end_in_a_different_context_skips_the_corrupting_detach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The D-CDX-21 reproduction: on_graph_start attaches in one Context
    (simulating one asyncio Task -- e.g. the task running ``run_agent``
    before an MCP child call), on_graph_end runs in a DIFFERENT, fresh
    Context (simulating the finalizer running after a task-boundary-crossing
    ServiceNow MCP teardown). ``opentelemetry.context.detach`` must NOT be
    invoked at all in this case -- calling it with a token from a foreign
    Context is exactly what raised ``ValueError: Token ... was created in a
    different Context`` (swallowed internally by OTel, but still corrupting).
    """
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)

    import opentelemetry.context as otel_context_module

    detach_calls: list[object] = []
    real_detach = otel_context_module.detach

    def _spy_detach(token):
        detach_calls.append(token)
        return real_detach(token)

    monkeypatch.setattr(otel_context_module, "detach", _spy_detach)

    start_ctx = contextvars.Context()
    start_ctx.run(
        telemetry.on_graph_start, run_id="run-cross", agent_id="a", query="q"
    )

    # A genuinely different, unrelated Context -- exactly what a Task created
    # via asyncio.ensure_future/create_task after on_graph_start gets.
    end_ctx = contextvars.Context()
    end_ctx.run(telemetry.on_graph_end, run_id="run-cross", status="success")

    assert detach_calls == [], (
        "context.detach() must be SKIPPED when on_graph_end's Context differs "
        "from on_graph_start's -- calling it anyway is the D-CDX-21 defect "
        "(a cross-context Token.reset(), silently swallowed by OTel but still "
        "corrupting the originating context's ambient span state)"
    )

    # The span itself must still be finalized/exported even though the
    # ambient-context restore was skipped.
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "agent.run"
    assert spans[0].end_time is not None


def test_cross_context_close_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """No exception escapes on_graph_end even across a Context boundary --
    this was already true pre-fix (OTel swallows the ValueError internally),
    so this alone does not prove the fix, but it is a real invariant the fix
    must preserve."""
    telemetry, _exporter = _telemetry_with_in_memory_exporter(monkeypatch)

    start_ctx = contextvars.Context()
    start_ctx.run(
        telemetry.on_graph_start, run_id="run-noraise", agent_id="a", query="q"
    )

    end_ctx = contextvars.Context()
    end_ctx.run(telemetry.on_graph_end, run_id="run-noraise", status="success")
