"""Tests for ``TelemetryEngine``'s real OpenTelemetry wiring (L24 / OBS-P1-2).

CONCEPT:AU-OS.observability.telemetry-observability — Telemetry Engine

``observability/__init__.py``'s ``TelemetryEngine`` used to carry an "OS-5.8
placeholder" comment for OTel trace/metric export: nothing actually
configured a ``TracerProvider``/``MeterProvider``. These tests prove the
replacement is REAL, not another no-op facade:

* opt-in — no collector endpoint configured (or ``enable_otel=False``) means
  a clean no-op, never a crash;
* once BOTH the constructor opt-in and a collector endpoint are present,
  ``TelemetryEngine`` configures an actual ``opentelemetry.sdk.trace.
  TracerProvider`` and ``opentelemetry.sdk.metrics.MeterProvider``, each
  wired with a real OTLP/HTTP exporter (never a stub object);
* ``on_graph_start``/``on_graph_end`` drive a real span through that
  provider, and ``on_response`` records real counter instruments.

No network access happens in these tests: constructing the OTel SDK
providers/exporters never performs I/O (only export — triggered by
``shutdown()``/an elapsed batch interval — does, and any failure there is
caught internally by the SDK, never raised). Pointing at a closed local port
keeps any such attempt instant and side-effect-free.
"""

from __future__ import annotations

import pytest

from agent_utilities.observability import TelemetryEngine

pytestmark = pytest.mark.concept("AU-OS.observability.telemetry-observability")

# A well-formed URL with nothing listening — safe to construct exporters
# against (construction never dials out) and instant-refused if a flush is
# ever attempted (e.g. during ``shutdown()``).
_DEAD_COLLECTOR = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def _clean_otel_env(monkeypatch: pytest.MonkeyPatch):
    """Every test starts with a clean slate for the OTel endpoint settings."""
    monkeypatch.delenv("EPISTEMIC_GRAPH_OBS_ADDR", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    yield


def _shutdown(telemetry: TelemetryEngine) -> None:
    """Best-effort provider teardown so tests never leak exporter threads."""
    try:
        telemetry.shutdown()
    except Exception:  # noqa: BLE001 — teardown must never fail the test
        pass


# ---------------------------------------------------------------------------
# Opt-in / no-op posture
# ---------------------------------------------------------------------------


def test_no_endpoint_configured_is_a_clean_noop() -> None:
    """No collector endpoint resolves ⇒ OTel setup is a no-op, not a crash."""
    telemetry = TelemetryEngine()
    assert telemetry.is_otel_configured() is False
    assert telemetry._tracer_provider is None
    assert telemetry._meter_provider is None
    assert telemetry._tracer is None
    assert telemetry._meter is None
    # Calling the hooks with nothing configured must still be side-effect-free.
    telemetry.on_graph_start(run_id="r0")
    telemetry.on_response(run_id="r0", usage={"prompt": 5})
    telemetry.on_graph_end(run_id="r0", status="success")


def test_enable_otel_false_stays_unconfigured_even_with_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor opt-out wins even when a collector endpoint IS set."""
    monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", _DEAD_COLLECTOR)
    telemetry = TelemetryEngine(enable_otel=False)
    assert telemetry.is_otel_configured() is False
    assert telemetry._tracer_provider is None
    assert telemetry._meter_provider is None


# ---------------------------------------------------------------------------
# Real (non-placeholder) provider setup
# ---------------------------------------------------------------------------


def test_endpoint_configured_wires_a_real_tracer_and_meter_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core L24 assertion: enabling OTel configures REAL SDK providers."""
    monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", _DEAD_COLLECTOR)
    telemetry = TelemetryEngine()
    try:
        assert telemetry.is_otel_configured() is True

        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider

        assert isinstance(telemetry._tracer_provider, TracerProvider)
        assert isinstance(telemetry._meter_provider, MeterProvider)
        assert telemetry._tracer is not None
        assert telemetry._meter is not None
        # Real instrument objects, not placeholders/no-op counters.
        assert telemetry._token_counter is not None
        assert telemetry._graph_run_counter is not None
    finally:
        _shutdown(telemetry)


def test_generic_otel_endpoint_setting_is_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falls back to the generic ``OTEL_EXPORTER_OTLP_ENDPOINT`` when the
    engine-specific ``EPISTEMIC_GRAPH_OBS_ADDR`` (OBS-P1-1) is unset."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _DEAD_COLLECTOR)
    telemetry = TelemetryEngine()
    try:
        assert telemetry.is_otel_configured() is True
    finally:
        _shutdown(telemetry)


def test_epistemic_graph_obs_addr_wins_over_generic_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBS-P1-1's endpoint config takes priority (the engine's OWN collector),
    per the module docstring's documented priority order."""
    from agent_utilities.observability import _resolve_otel_endpoint

    monkeypatch.setenv(
        "EPISTEMIC_GRAPH_OBS_ADDR", "https://engine-collector.example/otlp"
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://langfuse.example/otel")
    assert _resolve_otel_endpoint() == "https://engine-collector.example/otlp"


# ---------------------------------------------------------------------------
# The hooks actually drive the real providers
# ---------------------------------------------------------------------------


def test_on_graph_start_and_end_drive_a_real_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace import Span as SdkSpan

    # ``tests/conftest.py`` sets ``OTEL_SDK_DISABLED=true`` by default so the
    # rest of the suite never emits real telemetry; this test explicitly
    # opts back IN (against a dead local collector — no real network
    # traffic) to prove the span this engine drives is a REAL recording
    # span, not the SDK's disabled-mode no-op.
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", _DEAD_COLLECTOR)
    telemetry = TelemetryEngine(enable_audit=False, enable_tokens=False)
    try:
        telemetry.on_graph_start(run_id="run-1", agent_id="agent-1", query="hello")
        assert "run-1" in telemetry._active_spans
        assert isinstance(telemetry._active_spans["run-1"], SdkSpan)

        telemetry.on_graph_end(run_id="run-1", status="success", duration_ms=12.5)
        # The span is closed and removed from the active-span bookkeeping —
        # never left dangling.
        assert "run-1" not in telemetry._active_spans
        assert "run-1" not in telemetry._span_tokens
    finally:
        _shutdown(telemetry)


def test_on_response_records_real_token_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", _DEAD_COLLECTOR)
    telemetry = TelemetryEngine(enable_audit=False, enable_tokens=False)
    try:
        telemetry._lazy_init()
        assert telemetry.is_otel_configured() is True
        # Must not raise — the counter is a real instrument backed by the
        # real MeterProvider constructed above.
        telemetry.on_response(
            run_id="run-1",
            usage={"prompt": 100, "response": 50, "thoughts": 0, "tool_use": 5},
            model="test-model",
        )
    finally:
        _shutdown(telemetry)


def test_shutdown_is_safe_to_call_when_never_configured() -> None:
    """``shutdown()`` on a never-configured engine must never raise."""
    telemetry = TelemetryEngine()
    telemetry.shutdown()


# ---------------------------------------------------------------------------
# annotate_epistemic — the light epistemic layer's OTel projection
# (CONCEPT:AU-KB-CURRENCY, `04-five-intersections.md` item 4)
# ---------------------------------------------------------------------------


def test_annotate_epistemic_is_a_noop_with_no_recording_span() -> None:
    """No exporter, no active span (the default suite posture, SDK disabled
    by ``tests/conftest.py``) ⇒ clean no-op, never raises."""
    telemetry = TelemetryEngine()
    telemetry.annotate_epistemic(confidence=0.5, status="confirmed")


def test_annotate_epistemic_never_requires_its_own_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``annotate_epistemic`` reads the AMBIENT current span via the OTel API
    — it must widen a span opened by a DIFFERENT pipeline (e.g. the separate
    Logfire/``custom_observability.setup_otel()`` pipeline this package also
    ships) without requiring `self._otel_configured` (this engine's OWN
    provider) to be true. A never-configured ``TelemetryEngine`` instance
    must still annotate a span some OTHER tracer opened."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("external-pipeline")

    telemetry = TelemetryEngine()  # NEVER configured — is_otel_configured() stays False
    assert telemetry.is_otel_configured() is False

    with tracer.start_as_current_span("kg.query") as span:
        telemetry.annotate_epistemic(
            confidence=0.3,
            status="contested",
            contradiction_count=1,
            policy_labels=["epistemic:contested"],
            source_count=2,
            model="test-model",
        )
        assert span.attributes["epistemic.confidence"] == 0.3
        assert span.attributes["epistemic.status"] == "contested"
        assert span.attributes["epistemic.contradiction_count"] == 1
        assert span.attributes["epistemic.policy_labels"] == ("epistemic:contested",)
        assert span.attributes["gen_ai.response.source_count"] == 2
        assert span.attributes["gen_ai.request.model"] == "test-model"


def test_annotate_epistemic_omits_unset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the fields the caller actually passed are set — no fabricated
    zero/empty defaults for fields the caller left ``None``."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("external-pipeline")

    telemetry = TelemetryEngine()
    with tracer.start_as_current_span("kg.query") as span:
        telemetry.annotate_epistemic(confidence=0.9)
        assert span.attributes["epistemic.confidence"] == 0.9
        assert "epistemic.status" not in span.attributes
        assert "epistemic.contradiction_count" not in span.attributes
        assert "gen_ai.request.model" not in span.attributes


def test_get_telemetry_engine_returns_a_process_wide_singleton() -> None:
    from agent_utilities.observability import get_telemetry_engine

    first = get_telemetry_engine()
    second = get_telemetry_engine()
    assert first is second


# ---------------------------------------------------------------------------
# X2 (W3.8) — gen_ai semconv attrs on the run span + OTEL_TRACES_EXPORTER +
# redaction. Uses an in-memory exporter (no network, real exported spans) via
# a MANUALLY-INJECTED tracer — ``on_graph_start``/``on_graph_end``/``on_response``
# only require ``self._tracer`` to be set (not ``self._otel_configured``), so
# pre-marking the engine ``_initialized`` skips ``_setup_otel()`` entirely.
# ---------------------------------------------------------------------------


def _telemetry_with_in_memory_exporter(monkeypatch: pytest.MonkeyPatch):
    # tests/conftest.py sets OTEL_SDK_DISABLED=true suite-wide so the rest of
    # the suite never emits real telemetry; TracerProvider.get_tracer() checks
    # this env var directly at call time (not read through this package's
    # ``setting()``, since it's the upstream SDK's own switch) and returns a
    # no-op tracer when it's true — opt back in for this test's manually
    # injected provider/exporter.
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
    telemetry._initialized = True  # skip _lazy_init -> _setup_otel (no endpoint needed)
    telemetry._tracer = provider.get_tracer("test-run-span")
    return telemetry, exporter


def test_on_graph_start_stamps_gen_ai_system(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``graph.run`` span is tagged with the mediating gen_ai framework
    at open time (X2) — pydantic-ai drives every model/tool call in this
    codebase, so it is the invariant ``gen_ai.system`` for the whole span."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    telemetry.on_graph_start(run_id="run-gs", agent_id="agent-1", query="hi")
    telemetry.on_graph_end(run_id="run-gs", status="success")
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.system"] == "pydantic_ai"


def test_on_graph_end_stamps_model_and_tool_call_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X2: ``on_graph_end``'s new ``model``/``tool_call_count`` kwargs land on
    the span as ``gen_ai.request.model``/``gen_ai.response.tool_call_count``
    BEFORE it closes."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    telemetry.on_graph_start(run_id="run-ge", agent_id="agent-1", query="hi")
    telemetry.on_graph_end(
        run_id="run-ge",
        status="completed",
        duration_ms=12.5,
        model="qwen2.5-72b-instruct",
        tool_call_count=3,
    )
    span = exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.request.model"] == "qwen2.5-72b-instruct"
    assert span.attributes["gen_ai.response.tool_call_count"] == 3
    assert span.attributes["status"] == "completed"
    assert span.attributes["duration_ms"] == 12.5


def test_on_graph_end_omits_model_and_tool_call_count_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X2 'if available': a run with no resolved model/tool-call tally (e.g.
    the enterprise dispatch branch) must not fabricate zero/empty values."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    telemetry.on_graph_start(run_id="run-ge2", agent_id="agent-1", query="hi")
    telemetry.on_graph_end(run_id="run-ge2", status="failed")
    span = exporter.get_finished_spans()[0]
    assert "gen_ai.request.model" not in span.attributes
    assert "gen_ai.response.tool_call_count" not in span.attributes


def test_on_response_stamps_gen_ai_usage_and_model_on_the_active_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X2: ``on_response`` now ALSO widens the run's own active span (opened by
    ``on_graph_start``) with ``gen_ai.request.model``/``gen_ai.usage.
    input_tokens``/``gen_ai.usage.output_tokens`` — in addition to the
    pre-existing opaque-ref token-tracker record and metric counter."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    telemetry.on_graph_start(run_id="run-or", agent_id="agent-1", query="hi")
    telemetry.on_response(
        run_id="run-or",
        usage={"prompt": 100, "response": 50, "thoughts": 0, "tool_use": 5},
        model="qwen2.5-72b-instruct",
    )
    telemetry.on_graph_end(run_id="run-or", status="success")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.request.model"] == "qwen2.5-72b-instruct"
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
    assert span.attributes["gen_ai.usage.output_tokens"] == 50


def test_on_response_with_no_tracked_span_is_a_clean_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``on_response`` for an untracked run_id (OTel unconfigured, or called
    outside on_graph_start/on_graph_end) must never raise."""
    telemetry, _exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    telemetry.on_response(run_id="never-started", usage={"prompt": 5})


# ---------------------------------------------------------------------------
# OTEL_TRACES_EXPORTER (X2) — the third standard env var
# ---------------------------------------------------------------------------


def test_otel_traces_exporter_none_disables_even_with_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard ``OTEL_TRACES_EXPORTER=none`` kill-switch wins even when a
    collector endpoint resolves — matching the real OTel SDK's own
    env-based auto-configuration."""
    from agent_utilities.observability import _resolve_otel_endpoint

    monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", _DEAD_COLLECTOR)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
    assert _resolve_otel_endpoint() == ""

    telemetry = TelemetryEngine()
    assert telemetry.is_otel_configured() is False


def test_otel_traces_exporter_otlp_explicit_is_equivalent_to_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``OTEL_TRACES_EXPORTER=otlp`` (the OTel spec default) behaves
    exactly like leaving it unset — export is still endpoint-driven."""
    from agent_utilities.observability import _resolve_otel_endpoint

    monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", _DEAD_COLLECTOR)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    assert _resolve_otel_endpoint() == _DEAD_COLLECTOR


def test_otel_traces_exporter_unset_preserves_current_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset (the default, zero-overhead posture) resolves purely off endpoint
    presence — X2 must not change any existing unset-var behavior."""
    from agent_utilities.observability import _resolve_otel_endpoint

    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    assert _resolve_otel_endpoint() == ""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _DEAD_COLLECTOR)
    assert _resolve_otel_endpoint() == _DEAD_COLLECTOR


# ---------------------------------------------------------------------------
# Redaction — span attributes are names/ids/counts only, never prompt text,
# secrets, or row content (X2 hard requirement).
# ---------------------------------------------------------------------------


def test_seeded_secret_like_value_never_appears_in_exported_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seeded, secret-shaped value used as the run QUERY, agent id, and run
    id must never surface verbatim on any exported span attribute — only
    ``query_length`` (a count) and opaque ``_telemetry_ref`` hashes reach the
    exporter for those fields, REGARDLESS of what's fed in. ``model`` and
    ``policy_labels`` are deliberately excluded from this sweep: both are
    plain controlled-vocabulary identifiers (names, not secrets or row
    content) per the existing, spec'd ``annotate_epistemic`` contract
    (``test_annotate_epistemic_never_requires_its_own_provider``) — this
    test's job is everything that MUST stay opaque no matter its content."""
    telemetry, exporter = _telemetry_with_in_memory_exporter(monkeypatch)
    secret = "sk-live-AKIAFAKESECRETVALUE0123456789ABCDEF"  # nosec B105 - test fixture, not a real credential

    telemetry.on_graph_start(run_id=secret, agent_id=secret, query=secret)
    telemetry.annotate_epistemic(
        confidence=0.4,
        status="contested",
        contradiction_count=1,
        policy_labels=["epistemic:contested"],
        source_count=1,
    )
    telemetry.on_response(run_id=secret, usage={"prompt": 1})
    telemetry.on_graph_end(run_id=secret, status="success", tool_call_count=1)

    spans = exporter.get_finished_spans()
    assert spans, "expected at least one exported span"
    for span in spans:
        for key, value in span.attributes.items():
            if key in ("gen_ai.request.model", "epistemic.policy_labels"):
                continue  # documented exception: plain names, not secrets
            rendered = str(value)
            assert secret not in rendered, (
                f"secret-like value leaked onto span attribute {key!r}: {rendered!r}"
            )
    # query_length is a COUNT, never the text itself; run_ref/agent_ref are
    # opaque hashes, never the raw id fed in.
    assert spans[0].attributes["query_length"] == len(secret)
    assert spans[0].attributes["run_ref"] != secret
    assert spans[0].attributes["agent_ref"] != secret
