"""Tests for self-ingest telemetry (CONCEPT:AU-KG.ingest.attaching-this-root-logger).

Covers:
* the opt-in / default-off no-op behavior (no env / no endpoint),
* OTLP + ``_bulk`` record formatting into the exact wire shapes the engine
  expects (no live engine required),
* a mock endpoint receiving batched records through the sink + log handler,
* RunTrace / ToolCall provenance emission.

@pytest.mark.concept("AU-KG.ingest.attaching-this-root-logger")
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.observability.self_ingest import (
    SelfIngestConfig,
    SelfIngestLogHandler,
    SelfIngestSink,
    emit_run_trace,
    emit_tool_call,
    get_self_ingest_sink,
    install_self_ingest_logging,
    reset_self_ingest_sink,
    set_self_ingest_sink,
)

pytestmark = pytest.mark.concept("AU-KG.ingest.attaching-this-root-logger")


@pytest.fixture(autouse=True)
def _reset():
    reset_self_ingest_sink()
    yield
    reset_self_ingest_sink()


class _CapturingTransport:
    """A fake transport that records (url, payload) and reports success."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, payload: dict) -> bool:
        self.calls.append((url, payload))
        return self.ok


def _enabled_config(**over) -> SelfIngestConfig:
    base = dict(
        enabled=True,
        endpoint="http://obs.test:4318",
        mode="otlp",
        batch_size=100,
        flush_interval=0.05,
    )
    base.update(over)
    return SelfIngestConfig(**base)


# ---------------------------------------------------------------------------
# Opt-in / default-off no-op behavior
# ---------------------------------------------------------------------------
class TestNoOp:
    def test_unconfigured_singleton_is_none(self, monkeypatch):
        monkeypatch.delenv("AGENT_UTILITIES_SELF_INGEST", raising=False)
        monkeypatch.delenv("EPISTEMIC_GRAPH_OBS_ADDR", raising=False)
        reset_self_ingest_sink()
        assert get_self_ingest_sink() is None

    def test_on_but_no_endpoint_disabled(self, monkeypatch):
        monkeypatch.setenv("AGENT_UTILITIES_SELF_INGEST", "1")
        monkeypatch.delenv("EPISTEMIC_GRAPH_OBS_ADDR", raising=False)
        reset_self_ingest_sink()
        assert get_self_ingest_sink() is None

    def test_disabled_sink_emit_is_noop(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(SelfIngestConfig(enabled=False), transport=transport)
        assert sink.enabled is False
        sink.emit_log(body="hello")
        assert sink.flush() == 0
        assert transport.calls == []

    def test_emit_helpers_return_false_when_disabled(self, monkeypatch):
        monkeypatch.delenv("AGENT_UTILITIES_SELF_INGEST", raising=False)
        monkeypatch.delenv("EPISTEMIC_GRAPH_OBS_ADDR", raising=False)
        reset_self_ingest_sink()
        assert emit_run_trace(run_id="r1") is False
        assert emit_tool_call(run_id="r1", tool_name="t") is False

    def test_install_logging_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("AGENT_UTILITIES_SELF_INGEST", raising=False)
        reset_self_ingest_sink()
        assert install_self_ingest_logging(logging.getLogger("au.test.noop")) is False


# ---------------------------------------------------------------------------
# Wire-shape formatting (the shape the engine expects)
# ---------------------------------------------------------------------------
class TestFormatting:
    def test_otlp_shape(self):
        sink = SelfIngestSink(_enabled_config(service_name="graph-os"))
        payload = sink.format_otlp(
            [
                {
                    "timestamp_ns": 1_700_000_000_000_000_000,
                    "severity_text": "ERROR",
                    "body": "boom",
                    "attributes": {"run.id": "r-1", "duration_ms": 12},
                    "event_type": "run_trace",
                }
            ]
        )
        rl = payload["resourceLogs"][0]
        # resource service.name
        res_attrs = {a["key"]: a["value"] for a in rl["resource"]["attributes"]}
        assert res_attrs["service.name"]["stringValue"] == "graph-os"
        rec = rl["scopeLogs"][0]["logRecords"][0]
        assert rec["timeUnixNano"] == "1700000000000000000"
        assert rec["severityNumber"] == 17  # ERROR
        assert rec["severityText"] == "ERROR"
        assert rec["body"]["stringValue"] == "boom"
        attrs = {a["key"]: a["value"] for a in rec["attributes"]}
        assert attrs["run.id"]["stringValue"] == "r-1"
        # 64-bit ints proto-JSON encode as strings
        assert attrs["duration_ms"]["intValue"] == "12"
        assert attrs["event.type"]["stringValue"] == "run_trace"

    def test_bulk_shape(self):
        sink = SelfIngestSink(_enabled_config(mode="bulk", service_name="au"))
        payload = sink.format_bulk(
            [
                {
                    "timestamp_ns": 5,
                    "severity_text": "INFO",
                    "body": "hi",
                    "attributes": {"tool.name": "grep"},
                    "event_type": "tool_call",
                }
            ]
        )
        assert "records" in payload
        item = payload["records"][0]
        assert item["service"] == "au"
        assert item["body"] == "hi"
        assert item["event_type"] == "tool_call"
        assert item["tool.name"] == "grep"

    def test_url_derivation(self):
        assert SelfIngestConfig(endpoint="http://x:1/").url == "http://x:1/v1/logs"
        assert (
            SelfIngestConfig(endpoint="http://x:1", mode="bulk").url
            == "http://x:1/_bulk"
        )
        # already-suffixed endpoints are not double-suffixed
        assert (
            SelfIngestConfig(endpoint="http://x:1/v1/logs").url == "http://x:1/v1/logs"
        )


# ---------------------------------------------------------------------------
# Batched delivery to a mock endpoint
# ---------------------------------------------------------------------------
class TestDelivery:
    def test_flush_sends_batched_records(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        for i in range(3):
            sink.emit_log(body=f"m{i}", level="INFO")
        sent = sink.flush()
        assert sent == 3
        # One batch (all 3 under batch_size) → one POST to /v1/logs
        assert len(transport.calls) == 1
        url, payload = transport.calls[0]
        assert url == "http://obs.test:4318/v1/logs"
        records = payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        assert [r["body"]["stringValue"] for r in records] == ["m0", "m1", "m2"]
        assert sink.sent == 3

    def test_batch_size_splits_into_multiple_sends(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(batch_size=2), transport=transport)
        for i in range(5):
            sink.emit_log(body=f"m{i}")
        assert sink.flush() == 5
        assert len(transport.calls) == 3  # 2 + 2 + 1

    def test_failure_counts_and_no_raise(self):
        transport = _CapturingTransport(ok=False)
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        sink.emit_log(body="x")
        assert sink.flush() == 0  # nothing counted as sent on failure
        assert sink.failures == 1
        assert sink.sent == 0

    def test_queue_overflow_drops_without_blocking(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(queue_max=2), transport=transport)
        for i in range(5):
            sink.emit_log(body=f"m{i}")
        assert sink.dropped == 3
        assert sink.emitted == 5

    def test_log_handler_forwards_records(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        handler = SelfIngestLogHandler(sink, level=logging.INFO)
        lg = logging.getLogger("au.test.selfingest")
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        try:
            lg.info("hello from logger")
        finally:
            lg.removeHandler(handler)
        assert sink.flush() == 1
        rec = transport.calls[0][1]["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
        assert rec["body"]["stringValue"] == "hello from logger"
        attrs = {a["key"] for a in rec["attributes"]}
        assert "logger.name" in attrs


# ---------------------------------------------------------------------------
# RunTrace / ToolCall provenance stream
# ---------------------------------------------------------------------------
class TestProvenanceStream:
    def test_emit_run_trace_and_tool_call(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        set_self_ingest_sink(sink)

        assert emit_run_trace(run_id="run-9", status="success", duration_ms=3.0)
        assert emit_tool_call(run_id="run-9", tool_name="graph_query", status="error")

        assert sink.flush() == 2
        recs = transport.calls[0][1]["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        by_type = {
            next(
                a["value"]["stringValue"]
                for a in r["attributes"]
                if a["key"] == "event.type"
            ): r
            for r in recs
        }
        assert "run_trace" in by_type
        assert "tool_call" in by_type
        # error status maps to ERROR severity
        assert by_type["tool_call"]["severityNumber"] == 17
