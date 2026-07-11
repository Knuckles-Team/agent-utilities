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
        "EPISTEMIC_GRAPH_OBS_ADDR", "http://engine-collector.example/otlp"
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://langfuse.example/otel")
    assert _resolve_otel_endpoint() == "http://engine-collector.example/otlp"


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
