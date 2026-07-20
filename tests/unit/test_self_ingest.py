"""Tests for self-ingest telemetry (CONCEPT:AU-KG.ingest.attaching-this-root-logger).

Covers:
* the opt-in / default-off no-op behavior (no env / no endpoint),
* OTLP + ``_bulk`` record formatting into the exact wire shapes the engine
  expects (no live engine required),
* a mock endpoint receiving batched records through the sink + log handler,
* RunTrace / ToolCall provenance emission,
* write-ahead/ack-after-send durability: every emitted record is persisted
  before network delivery and remains replayable until acknowledgment,
* opaque per-tenant identity plus PII/local-path sanitization.

@pytest.mark.concept("AU-KG.ingest.attaching-this-root-logger")
"""

from __future__ import annotations

import contextvars
import logging
from io import StringIO

import pytest

from agent_utilities.observability.self_ingest import (
    SelfIngestConfig,
    SelfIngestLogHandler,
    SelfIngestSink,
    SpillBuffer,
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


class _FlakyTransport:
    """Fails the first ``fail_times`` calls, then succeeds — for requeue tests."""

    def __init__(self, fail_times: int = 1):
        self.fail_times = fail_times
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, payload: dict) -> bool:
        self.calls.append((url, payload))
        if len(self.calls) <= self.fail_times:
            return False
        return True


def _enabled_config(**over) -> SelfIngestConfig:
    base = dict(
        enabled=True,
        endpoint="http://obs.test:4318",
        mode="otlp",
        batch_size=100,
        flush_interval=0.05,
        # Every enabled emission is now write-ahead. Keep ordinary unit tests
        # in-memory; crash/restart cases override this with tmp_path.
        spill_path=":memory:",
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
        assert [r["body"]["stringValue"] for r in records] == [
            "event:log",
            "event:log",
            "event:log",
        ]
        refs = [
            next(
                attr["value"]["stringValue"]
                for attr in record["attributes"]
                if attr["key"] == "message.ref"
            )
            for record in records
        ]
        assert len(set(refs)) == 3
        assert all(ref.startswith("pref_telemetry_message_") for ref in refs)
        assert sink.sent == 3
        assert sink.persisted == 3
        assert sink.spill_depth() == 0

    def test_batch_size_splits_into_multiple_sends(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(batch_size=2), transport=transport)
        for i in range(5):
            sink.emit_log(body=f"m{i}")
        assert sink.flush() == 5
        assert len(transport.calls) == 3  # 2 + 2 + 1

    def test_failure_counts_and_no_raise(self, tmp_path):
        """A permanently-failing transport must never silently lose the record.

        CONCEPT:AU-OS.observability.durable-telemetry-pipeline — after exhausting its
        in-process retries, the record is durably spilled (not dropped).
        """
        transport = _CapturingTransport(ok=False)
        sink = SelfIngestSink(
            _enabled_config(max_retries=2, spill_path=str(tmp_path / "spill.db")),
            transport=transport,
        )
        sink.emit_log(body="x")
        assert sink.flush() == 0  # nothing counted as sent on failure
        assert sink.failures >= 1
        assert sink.sent == 0
        # Not silently lost: it lands in the durable buffer, not the void.
        assert sink.dropped == 0
        assert sink.spilled == 1
        assert sink.spill_depth() == 1

    def test_queue_overflow_spills_without_blocking(self, tmp_path):
        """A saturated in-process queue is backpressure, not a reason to lose data.

        CONCEPT:AU-OS.observability.durable-telemetry-pipeline — overflow spills to the
        durable buffer; ``dropped`` stays 0.
        """
        transport = _CapturingTransport()
        sink = SelfIngestSink(
            _enabled_config(queue_max=2, spill_path=str(tmp_path / "spill.db")),
            transport=transport,
        )
        for i in range(5):
            sink.emit_log(body=f"m{i}")
        assert sink.dropped == 0
        assert sink.spilled == 3
        assert sink.emitted == 5
        # All five records are write-ahead; three could not enter the delivery
        # accelerator but remain in the same durable backlog.
        assert sink.spill_depth() == 5

    def test_dropped_only_when_spill_buffer_itself_saturated(self, tmp_path):
        """The one true-loss case: durable buffer AND queue are both full.

        Even then it must be loudly counted + logged, never silent.
        """
        transport = _CapturingTransport()
        sink = SelfIngestSink(
            _enabled_config(
                queue_max=1,
                spill_path=str(tmp_path / "spill.db"),
                spill_max_records=1,
            ),
            transport=transport,
        )
        from agent_utilities.observability import self_ingest as module

        emergency = StringIO()
        handler = logging.StreamHandler(emergency)
        previous = list(module._emergency_logger.handlers)
        module._emergency_logger.handlers[:] = [handler]
        try:
            for i in range(4):
                sink.emit_log(body=f"m{i}")
        finally:
            module._emergency_logger.handlers[:] = previous
        assert sink.dropped >= 1
        assert "write_ahead_rejected" in emergency.getvalue()

    def test_requeue_then_success_never_drops(self):
        """A transient failure retries in-process and eventually sends — no loss."""
        transport = _FlakyTransport(fail_times=1)
        sink = SelfIngestSink(_enabled_config(max_retries=3), transport=transport)
        sink.emit_log(body="retry-me")
        sent = sink.flush()
        assert sent == 1
        assert sink.sent == 1
        assert sink.requeued >= 1
        assert sink.dropped == 0
        assert sink.spilled == 0
        # First call failed, second (requeued) call succeeded.
        assert len(transport.calls) == 2

    def test_redeem_spill_resends_durable_backlog(self, tmp_path):
        """Once the endpoint recovers, the worker drains the durable backlog."""
        transport = _CapturingTransport(ok=False)
        spill_path = str(tmp_path / "spill.db")
        sink = SelfIngestSink(
            _enabled_config(max_retries=0, spill_path=spill_path),
            transport=transport,
        )
        sink.emit_log(body="stuck")
        assert sink.flush() == 0
        assert sink.spilled == 1
        assert sink.spill_depth() == 1

        # Endpoint recovers.
        transport.ok = True
        assert sink._redeem_spill(10) == 1
        assert sink.sent == 1
        assert sink.spill_depth() == 0

    def test_failed_redeem_keeps_row_until_ack(self, tmp_path):
        transport = _CapturingTransport(ok=False)
        sink = SelfIngestSink(
            _enabled_config(queue_max=1, spill_path=str(tmp_path / "spill.db")),
            transport=transport,
        )
        sink.emit_log(body="durable")
        # Remove the accelerator entry to exercise direct WAL redemption.
        sink._drain(10)
        assert sink._redeem_spill(10) == 0
        assert sink.spill_depth() == 1

        transport.ok = True
        assert sink._redeem_spill(10) == 1
        assert sink.spill_depth() == 0

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
        assert rec["body"]["stringValue"] == "event:log"
        attrs = {a["key"] for a in rec["attributes"]}
        assert "logger.name" in attrs
        assert "message.ref" in attrs
        assert "hello from logger" not in str(rec)


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
        assert "run-9" not in str(recs)


# ---------------------------------------------------------------------------
# Per-tenant identity stamping (CONCEPT:AU-OS.identity.authenticated-identity-enforcement)
# ---------------------------------------------------------------------------
class TestTenantStamping:
    def _attrs_of(self, transport: _CapturingTransport, index: int = 0) -> dict:
        recs = transport.calls[0][1]["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        return {a["key"]: a["value"] for a in recs[index]["attributes"]}

    def test_log_stamped_with_ambient_actor(self):
        from agent_utilities.models.company_brain import ActorType
        from agent_utilities.security.brain_context import ActorContext, use_actor

        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)

        with use_actor(
            ActorContext(
                actor_id="agent:marketing",
                actor_type=ActorType.AI_AGENT,
                tenant_id="acme",
            )
        ):
            sink.emit_log(body="tenant-scoped log")

        assert sink.flush() == 1
        attrs = self._attrs_of(transport)
        assert attrs["tenant.ref"]["stringValue"].startswith("pref_tenant_")
        assert attrs["actor.ref"]["stringValue"].startswith("pref_actor_")
        assert "acme" not in str(transport.calls)
        assert "agent:marketing" not in str(transport.calls)

    def test_run_trace_and_tool_call_are_stamped_too(self):
        """Stamping happens at the single ``emit`` choke-point, not just logs."""
        from agent_utilities.models.company_brain import ActorType
        from agent_utilities.security.brain_context import ActorContext, use_actor

        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        set_self_ingest_sink(sink)

        with use_actor(
            ActorContext(
                actor_id="agent:ops", actor_type=ActorType.AI_AGENT, tenant_id="acme-2"
            )
        ):
            emit_run_trace(run_id="run-42", status="success")

        assert sink.flush() == 1
        attrs = self._attrs_of(transport)
        assert attrs["tenant.ref"]["stringValue"].startswith("pref_tenant_")
        assert attrs["actor.ref"]["stringValue"].startswith("pref_actor_")
        assert "acme-2" not in str(transport.calls)
        assert "agent:ops" not in str(transport.calls)

    def test_missing_actor_stamps_no_identity(self):
        """No ambient actor means no identity reference is synthesized."""
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        def emit_unscoped():
            sink.emit_log(body="unscoped")
            assert sink.flush() == 1

        contextvars.Context().run(emit_unscoped)
        attrs = self._attrs_of(transport)
        assert "tenant.ref" not in attrs
        assert "actor.ref" not in attrs

    def test_local_path_and_personal_identity_are_redacted(self):
        transport = _CapturingTransport()
        sink = SelfIngestSink(_enabled_config(), transport=transport)
        sink.emit_log(
            body="failed under /home/example/private/file.txt",
            attributes={"user_name": "Example Person", "endpoint": "https://private"},
        )
        assert sink.flush() == 1
        payload = str(transport.calls)
        assert "/home/example" not in payload
        assert "Example Person" not in payload
        assert "https://private" not in payload


# ---------------------------------------------------------------------------
# SpillBuffer — durable, crash-safe overflow store
# ---------------------------------------------------------------------------
class TestSpillBuffer:
    def test_peek_requires_explicit_ack(self, tmp_path):
        buf = SpillBuffer(str(tmp_path / "spill.db"))
        assert buf.available
        assert buf.append({"body": "a"})
        assert buf.append({"body": "b"})
        assert buf.count() == 2

        pending = buf.peek_batch(10)
        assert [row.record["body"] for row in pending] == ["a", "b"]
        assert buf.count() == 2
        assert buf.ack([row.durable_id for row in pending]) == 2
        assert buf.count() == 0
        buf.close()

    def test_survives_across_instances(self, tmp_path):
        """The buffer is durable: a fresh instance sees a prior instance's data."""
        path = str(tmp_path / "spill.db")
        buf1 = SpillBuffer(path)
        buf1.append({"body": "durable"})
        buf1.close()

        buf2 = SpillBuffer(path)
        assert buf2.count() == 1
        assert buf2.peek_batch(10)[0].record["body"] == "durable"
        buf2.close()

    def test_bounded_capacity_rejects_beyond_max(self, tmp_path):
        buf = SpillBuffer(str(tmp_path / "spill.db"), max_records=1)
        assert buf.append({"body": "first"})
        assert buf.append({"body": "second"}) is None
        assert buf.count() == 1

    def test_unwritable_path_degrades_to_unavailable(self, tmp_path):
        # A path whose parent cannot be created (a file standing where a
        # directory is needed) must degrade gracefully, never raise.
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        buf = SpillBuffer(str(blocker / "nested" / "spill.db"))
        assert buf.available is False
        assert buf.append({"body": "x"}) is None
        assert buf.peek_batch(10) == []
        assert buf.count() == 0
