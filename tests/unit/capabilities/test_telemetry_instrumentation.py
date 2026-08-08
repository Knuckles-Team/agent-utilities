"""Track 3 of the pydantic-ai native-adoption program: pydantic-ai's native
``Instrumentation`` capability wired onto our own OTel pipeline.

CONCEPT:AU-OS.observability.telemetry-observability — see
``reports/program/pydantic-ai-native-adoption.md`` Track 3.

``build_fleet_instrumentation`` hands ``pydantic_ai.capabilities.Instrumentation`` the
SAME ``TracerProvider`` (built by ``TelemetryEngine._setup_otel`` — ``BatchSpanProcessor``
+ ``OTLPSpanExporter``) that the engine's own ``graph.*`` spans export through.

``TestSpanReachesARealOtlpReceiver`` proves a span produced by an ACTUAL pydantic-ai
``Agent.run()`` (with ``Instrumentation`` attached) survives the whole pipeline —
``Instrumentation`` -> ``TracerProvider`` -> ``BatchSpanProcessor`` -> ``OTLPSpanExporter``
-> a real HTTP POST — by running a real ``http.server`` OTLP-shaped receiver on
``127.0.0.1`` and asserting it actually received protobuf span bytes.

This sandboxed worktree has NO network route to the homelab's live Tempo collector (DNS
resolution for the configured collector host fails here). The loopback receiver below
proves the CODE PATH is correct end-to-end over a real socket; it does NOT prove the live
Tempo hop, which remains explicitly unverified from this environment (see the program
report for that caveat).

Two things had to be overridden explicitly to get a REAL (non-no-op) span at all, both
matching the established pattern in ``tests/unit/test_telemetry_engine.py``:

* ``tests/conftest.py`` sets ``OTEL_SDK_DISABLED=true`` suite-wide so the rest of the unit
  suite never emits real telemetry. The OTel SDK reads this live at
  ``TracerProvider.get_tracer()`` time (not at import time), so each test below opts back
  in with ``monkeypatch.setenv("OTEL_SDK_DISABLED", "false")`` — never against a live
  external network, only the loopback receiver started by the fixture.
* ``InstrumentationSettings(tracer_provider=...)`` does NOT keep the provider on a
  ``tracer_provider`` attribute — it resolves it into a bound ``Tracer`` stored as
  ``.tracer`` (verified by reading ``pydantic_ai.models.instrumented.InstrumentationSettings.__init__``).
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator

import pytest

pytest.importorskip("pydantic_ai.capabilities")
pytest.importorskip("opentelemetry.sdk.trace")

from agent_utilities.capabilities.telemetry_instrumentation import (  # noqa: E402
    build_fleet_instrumentation,
)
from agent_utilities.observability import TelemetryEngine  # noqa: E402


class _CapturingOtlpHandler(http.server.BaseHTTPRequestHandler):
    received: list[bytes] = []
    paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).received.append(body)
        type(self).paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()
        self.wfile.write(b"")

    def log_message(self, *args: object) -> None:  # silence stdlib access logging
        return None


@pytest.fixture
def loopback_otlp_receiver() -> Iterator[str]:
    """A real HTTP server on 127.0.0.1 that captures raw OTLP POST bodies."""
    _CapturingOtlpHandler.received = []
    _CapturingOtlpHandler.paths = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CapturingOtlpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestBuildFleetInstrumentation:
    def test_returns_none_when_no_collector_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EPISTEMIC_GRAPH_OBS_ADDR", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        engine = TelemetryEngine()
        assert build_fleet_instrumentation(telemetry=engine) is None

    def test_reuses_the_engines_own_tracer_provider(
        self, monkeypatch: pytest.MonkeyPatch, loopback_otlp_receiver: str
    ) -> None:
        # ``tests/conftest.py`` sets ``OTEL_SDK_DISABLED=true`` suite-wide so the
        # rest of the suite never emits real telemetry (see
        # ``tests/unit/test_telemetry_engine.py`` for the same established
        # opt-back-in pattern this test mirrors). Opt back in explicitly here —
        # against a real loopback receiver, never a live external network.
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", loopback_otlp_receiver)
        engine = TelemetryEngine()
        instrumentation = build_fleet_instrumentation(telemetry=engine)

        assert instrumentation is not None
        # `InstrumentationSettings.__init__` resolves `tracer_provider=` into a
        # bound `Tracer` (stored as `.tracer`, not the provider itself). Assert
        # it is a REAL tracer bound to a real provider (not the SDK's disabled-mode
        # `NoOpTracer`, and not `None`) — `TestSpanReachesARealOtlpReceiver` below
        # proves it is specifically OUR engine's provider by observing its spans
        # arrive at the SAME collector the engine itself is configured for.
        from opentelemetry.trace import NoOpTracer

        assert engine.tracer_provider is not None
        assert instrumentation.settings.tracer is not None
        assert not isinstance(instrumentation.settings.tracer, NoOpTracer)


class TestSpanReachesARealOtlpReceiver:
    def test_a_real_agent_run_exports_a_span_over_a_real_socket(
        self, monkeypatch: pytest.MonkeyPatch, loopback_otlp_receiver: str
    ) -> None:
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        from agent_utilities.core.contextual_model import (
            create_context_agent,
            use_grounding_policy,
        )

        # See the identical opt-back-in note in ``TestBuildFleetInstrumentation``
        # above and ``tests/unit/test_telemetry_engine.py``.
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv("EPISTEMIC_GRAPH_OBS_ADDR", loopback_otlp_receiver)
        engine = TelemetryEngine()
        instrumentation = build_fleet_instrumentation(telemetry=engine)
        assert instrumentation is not None

        def _respond(messages, info) -> ModelResponse:
            del messages, info
            return ModelResponse(parts=[TextPart(content="pong")])

        agent = create_context_agent(
            FunctionModel(_respond, model_name="fleet-instrumentation-probe"),
            capabilities=[instrumentation],
            default_capabilities=False,
        )
        with use_grounding_policy("none"):
            result = agent.run_sync("ping")
        assert result.output == "pong"

        # Flush the BatchSpanProcessor synchronously so the export happens before
        # this test asserts on it, instead of racing the background export thread.
        flushed = engine.tracer_provider.force_flush(timeout_millis=10_000)
        assert flushed is True

        assert _CapturingOtlpHandler.received, (
            "no POST reached the loopback OTLP receiver — Instrumentation -> "
            "TracerProvider -> BatchSpanProcessor -> OTLPSpanExporter pipeline "
            "did not export a span"
        )
        assert any(path.endswith("/v1/traces") for path in _CapturingOtlpHandler.paths)
        # Real protobuf-encoded span bytes, not an empty/placeholder body.
        assert all(len(body) > 0 for body in _CapturingOtlpHandler.received)

        engine.shutdown()
