"""``run_agent`` OTel span coverage (CONCEPT:AU-OS.observability.telemetry-observability, X2).

``run_agent`` (the primary entry point for ``graph_orchestrate``) opens one
``graph.run`` span per execution via ``TelemetryEngine.on_graph_start`` and
closes it — on EVERY exit path (success/degraded/failed/enterprise) — through
the shared ``_record_execution_trace`` helper's new ``on_graph_end`` call,
carrying gen_ai attrs (model, tool-call count) when available.

Mirrors this package's established convention for a function this large and
KG/LLM-entangled (see ``test_delegation_degraded_outcome.py``,
``test_agent_stack_seams.py``): exercise the extracted, directly-callable
helper (``_record_execution_trace``) with the real ``TelemetryEngine``, and
prove ``run_agent`` itself calls ``on_graph_start`` by making the very next
internal call raise — confirming the span opens before any of the heavier
KG/execution machinery runs, without needing to stand up a real engine, LLM,
or KG-resolution path.
"""

from __future__ import annotations

import pytest

from agent_utilities.observability import TelemetryEngine
from agent_utilities.orchestration import agent_runner
from agent_utilities.orchestration.agent_runner import _record_execution_trace

pytestmark = pytest.mark.concept("AU-OS.observability.telemetry-observability")

_DEAD_COLLECTOR = "http://127.0.0.1:1"


def _telemetry_with_in_memory_exporter(monkeypatch: pytest.MonkeyPatch):
    """Same pattern as ``tests/unit/test_telemetry_engine.py``: a manually
    injected tracer needs no live OTLP endpoint and yields real, inspectable
    exported spans."""
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
    telemetry._tracer = provider.get_tracer("test-run-agent-span")
    return telemetry, exporter


# --------------------------------------------------------------------------- #
# _record_execution_trace closes the run's span with gen_ai attrs, on every
# exit path — including the ``engine=None`` case (no KG write happens at all).
# --------------------------------------------------------------------------- #


def test_record_execution_trace_closes_the_run_span_with_gen_ai_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    monkeypatch.setattr(
        "agent_utilities.observability.get_telemetry_engine", lambda: telemetry
    )

    telemetry.on_graph_start(run_id="run-record-1", agent_id="my-agent", query="hi")

    _record_execution_trace(
        None,  # no engine -> the KG-write portion is skipped entirely
        "run-record-1",
        "my-agent",
        "do the thing",
        status="completed",
        duration_ms=42.0,
        model_name="qwen2.5-72b-instruct",
        tool_call_count=3,
    )

    assert "run-record-1" not in telemetry._active_spans
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "graph.run"
    assert span.attributes["gen_ai.system"] == "pydantic_ai"
    assert span.attributes["gen_ai.request.model"] == "qwen2.5-72b-instruct"
    assert span.attributes["gen_ai.response.tool_call_count"] == 3
    assert span.attributes["status"] == "completed"
    assert span.attributes["duration_ms"] == 42.0


def test_record_execution_trace_closes_the_span_even_on_the_failed_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failed-dispatch exit path passes no model/tool_call_count (the
    exception happened before either was known) — the span must still close
    cleanly with just status/duration, no fabricated gen_ai attrs."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    monkeypatch.setattr(
        "agent_utilities.observability.get_telemetry_engine", lambda: telemetry
    )

    telemetry.on_graph_start(run_id="run-record-2", agent_id="my-agent", query="hi")
    _record_execution_trace(
        None,
        "run-record-2",
        "my-agent",
        "do the thing",
        status="failed",
        error="boom",
    )

    span = exporter.get_finished_spans()[0]
    assert span.attributes["status"] == "failed"
    assert "gen_ai.request.model" not in span.attributes
    assert "gen_ai.response.tool_call_count" not in span.attributes


def test_record_execution_trace_is_a_noop_when_otel_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero endpoint configured (the default posture) -> no span was ever
    opened -> closing one must be a silent, exception-free no-op, and the
    (unrelated) engine-guard early return still applies."""
    monkeypatch.delenv("EPISTEMIC_GRAPH_OBS_ADDR", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    # No engine -> function returns right after the OTel no-op; must not raise.
    _record_execution_trace(
        None, "run-never-started", "my-agent", "task", status="completed"
    )


# --------------------------------------------------------------------------- #
# run_agent itself calls on_graph_start before any heavier machinery runs
# --------------------------------------------------------------------------- #


async def test_run_agent_opens_a_span_before_dispatching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent``'s ``on_graph_start`` call fires unconditionally, right
    after minting the run id — proven by making the very next call
    (``_get_or_create_engine``) raise and confirming ``on_graph_start`` had
    already run with the right args before that failure propagated."""
    calls: list[dict[str, object]] = []

    class _FakeTelemetry:
        def on_graph_start(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "agent_utilities.observability.get_telemetry_engine",
        lambda: _FakeTelemetry(),
    )

    class _StopHere(Exception):
        pass

    def _raise(*_a: object, **_k: object) -> None:
        raise _StopHere("stop right after on_graph_start ran")

    monkeypatch.setattr(agent_runner, "_get_or_create_engine", _raise)

    with pytest.raises(_StopHere):
        await agent_runner.run_agent("some-agent", "do the thing", engine=None)

    assert len(calls) == 1
    assert calls[0]["agent_id"] == "some-agent"
    assert calls[0]["query"] == "do the thing"


async def test_run_agent_span_start_failure_never_blocks_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracing must never break a run: a broken telemetry engine at
    ``on_graph_start`` is swallowed, and dispatch proceeds to the NEXT step
    (proven the same way — the following call still raises on schedule)."""

    class _BrokenTelemetry:
        def on_graph_start(self, **_kwargs: object) -> None:
            raise RuntimeError("telemetry backend exploded")

    monkeypatch.setattr(
        "agent_utilities.observability.get_telemetry_engine",
        lambda: _BrokenTelemetry(),
    )

    class _StopHere(Exception):
        pass

    def _raise(*_a: object, **_k: object) -> None:
        raise _StopHere("reached past the broken telemetry call")

    monkeypatch.setattr(agent_runner, "_get_or_create_engine", _raise)

    with pytest.raises(_StopHere):
        await agent_runner.run_agent("some-agent", "do the thing", engine=None)
