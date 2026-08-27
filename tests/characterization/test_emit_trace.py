"""Characterization tests for ``agent_utilities.harness.tracing._emit_trace``.

CX-AU-10. Pins OBSERVED behaviour of the unmodified function before a
complexity-reduction refactor (CCN 43 -> target <= 10). These tests must stay
byte-identical across the refactor commit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.core.config import config
from agent_utilities.harness import tracing
from agent_utilities.harness.trace_backend import LangfuseTraceBackend
from agent_utilities.harness.tracing import _emit_trace


@pytest.fixture(autouse=True)
def _reset_kg_sink():
    tracing.set_kg_trace_sink(None)
    yield
    tracing.set_kg_trace_sink(None)


@pytest.fixture
def enable_export(monkeypatch):
    monkeypatch.setattr(config, "trace_export_enabled", True)
    monkeypatch.setattr(config, "langfuse_secret_key_ref", "env://TEST_SECRET")
    monkeypatch.setattr(config, "langfuse_capture_content", True)


@pytest.fixture
def disable_export(monkeypatch):
    monkeypatch.setattr(config, "trace_export_enabled", False)


def _fake_backend(monkeypatch):
    """A real LangfuseTraceBackend whose network-touching _get_api is stubbed."""
    backend = LangfuseTraceBackend()
    api = MagicMock()
    monkeypatch.setattr(backend, "_get_api", lambda: api)
    monkeypatch.setattr(
        "agent_utilities.harness.trace_backend.create_trace_backend",
        lambda backend_type="langfuse": backend,
    )
    return backend, api


def test_unsafe_identifier_short_circuits_before_any_sink(monkeypatch, enable_export):
    """An identifier the privacy guard would redact aborts emission entirely --
    no KG sink call, no Langfuse backend construction."""
    monkeypatch.setattr(
        tracing._TRACE_PRIVACY,
        "sanitize_text",
        lambda value: (value, SimpleNamespace(changed=True)),
    )
    sink = MagicMock()
    tracing.set_kg_trace_sink(sink)
    create_backend = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.harness.trace_backend.create_trace_backend", create_backend
    )

    _emit_trace(
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data={"a": 1},
        output_data={"b": 2},
        is_root=True,
    )

    sink.record_event.assert_not_called()
    create_backend.assert_not_called()


def test_kg_sink_receives_expected_kwargs_for_root_generation(
    monkeypatch, enable_export
):
    sink = MagicMock()
    tracing.set_kg_trace_sink(sink)
    monkeypatch.setattr(
        config, "trace_export_enabled", False
    )  # isolate the KG-sink path

    _emit_trace(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id="parent-1",
        name="llm_call",
        trace_type="generation-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data={"prompt": "hi"},
        output_data={"text": "hello"},
        level="DEFAULT",
        tags=["x"],
        metadata={
            "model": "m1",
            "provider": "p1",
            "input_tokens": 3,
            "output_tokens": 5,
        },
        session_id="sess-1",
        is_root=True,
    )

    sink.record_event.assert_called_once()
    kwargs = sink.record_event.call_args.kwargs
    assert kwargs["trace_id"] == "trace-1"
    assert kwargs["span_id"] == "span-1"
    assert kwargs["name"] == "llm_call"
    assert kwargs["is_root"] is True
    assert kwargs["kind"] == "llm"  # "generation" substring in trace_type
    assert kwargs["parent_span_id"] == "parent-1"
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["error"] is None  # level != ERROR
    assert kwargs["model"] == "m1"
    assert kwargs["provider"] == "p1"
    assert kwargs["input_tokens"] == 3
    assert kwargs["output_tokens"] == 5
    assert kwargs["tags"] == ["x"]
    # capture_content True + is_root True -> real text, truncated to 4000 chars
    assert kwargs["input_text"] == str({"prompt": "hi"})
    assert kwargs["output_text"] == str({"text": "hello"})


def test_kg_sink_kind_is_general_for_non_generation_trace_type(
    monkeypatch, enable_export
):
    sink = MagicMock()
    tracing.set_kg_trace_sink(sink)
    monkeypatch.setattr(config, "trace_export_enabled", False)

    _emit_trace(
        trace_id="t2",
        span_id="s2",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        is_root=False,
    )

    kwargs = sink.record_event.call_args.kwargs
    assert kwargs["kind"] == "general"
    # is_root False -> input_text/output_text always empty regardless of capture_content
    assert kwargs["input_text"] == ""
    assert kwargs["output_text"] == ""


def test_kg_sink_metadata_only_mode_suppresses_content_even_when_root(
    monkeypatch, enable_export
):
    monkeypatch.setattr(config, "langfuse_capture_content", False)
    monkeypatch.setattr(config, "trace_export_enabled", False)
    sink = MagicMock()
    tracing.set_kg_trace_sink(sink)

    _emit_trace(
        trace_id="t3",
        span_id="s3",
        parent_span_id=None,
        name="op",
        trace_type="generation-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data={"secret": "value"},
        output_data={"secret": "value"},
        is_root=True,
    )

    kwargs = sink.record_event.call_args.kwargs
    assert kwargs["input_text"] == ""
    assert kwargs["output_text"] == ""


def test_kg_sink_error_level_reports_metadata_only_error_string(
    monkeypatch, enable_export
):
    monkeypatch.setattr(config, "langfuse_capture_content", False)
    monkeypatch.setattr(config, "trace_export_enabled", False)
    sink = MagicMock()
    tracing.set_kg_trace_sink(sink)

    _emit_trace(
        trace_id="t4",
        span_id="s4",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        level="ERROR",
        status_message="boom: secret leaked",
        is_root=False,
    )

    kwargs = sink.record_event.call_args.kwargs
    # capture_content False + level ERROR -> exported_status_message forced to "error"
    assert kwargs["error"] == "error"


def test_kg_sink_exception_is_swallowed(monkeypatch, enable_export):
    monkeypatch.setattr(config, "trace_export_enabled", False)
    sink = MagicMock()
    sink.record_event.side_effect = RuntimeError("sink is down")
    tracing.set_kg_trace_sink(sink)

    # Must not raise -- tracing must never break the traced caller.
    _emit_trace(
        trace_id="t5",
        span_id="s5",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        is_root=False,
    )


def test_sink_without_record_event_attribute_is_skipped(monkeypatch, enable_export):
    monkeypatch.setattr(config, "trace_export_enabled", False)
    tracing.set_kg_trace_sink(object())  # no record_event attribute

    # Must not raise.
    _emit_trace(
        trace_id="t6",
        span_id="s6",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        is_root=False,
    )


def test_langfuse_export_skipped_when_export_disabled(monkeypatch, disable_export):
    create_backend = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.harness.trace_backend.create_trace_backend", create_backend
    )

    _emit_trace(
        trace_id="t7",
        span_id="s7",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        is_root=False,
    )

    create_backend.assert_not_called()


def test_langfuse_root_trace_create_batches_a_single_trace_event(
    monkeypatch, enable_export
):
    _backend, api = _fake_backend(monkeypatch)

    _emit_trace(
        trace_id="trace-x",
        span_id="span-x",
        parent_span_id=None,
        name="root_op",
        trace_type="trace-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data={"in": 1},
        output_data={"out": 2},
        tags=["a", "b"],
        metadata={"k": "v"},
        session_id="sess-x",
        is_root=True,
    )

    api.ingestion_batch.assert_called_once()
    batch = api.ingestion_batch.call_args.kwargs["batch"]
    assert len(batch) == 1  # trace-create root with no separate span
    event = batch[0]
    assert event["type"] == "trace-create"
    assert event["body"]["id"] == "trace-x"
    assert event["body"]["name"] == "root_op"
    assert event["body"]["tags"] == ["a", "b"]
    assert event["body"]["sessionId"] == "sess-x"
    assert event["body"]["input"] == {"in": 1}
    assert event["body"]["output"] == {"out": 2}


def test_langfuse_root_span_create_batches_trace_plus_span(monkeypatch, enable_export):
    _backend, api = _fake_backend(monkeypatch)

    _emit_trace(
        trace_id="trace-y",
        span_id="span-y",
        parent_span_id=None,
        name="root_span_op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data={"in": 1},
        output_data={"out": 2},
        is_root=True,
    )

    batch = api.ingestion_batch.call_args.kwargs["batch"]
    assert len(batch) == 2
    assert batch[0]["type"] == "trace-create"
    assert (
        batch[1]["type"] == "span-create"
    )  # is_root forces actual_type to span-create
    assert batch[1]["body"]["id"] == "span-y"
    assert batch[1]["body"]["traceId"] == "trace-y"


def test_langfuse_non_root_span_includes_parent_and_status(monkeypatch, enable_export):
    _backend, api = _fake_backend(monkeypatch)
    monkeypatch.setattr(config, "langfuse_capture_content", False)

    _emit_trace(
        trace_id="trace-z",
        span_id="span-z",
        parent_span_id="parent-z",
        name="child_op",
        trace_type="generation-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data={"secret": 1},
        output_data={"secret": 2},
        level="ERROR",
        status_message="boom",
        is_root=False,
    )

    batch = api.ingestion_batch.call_args.kwargs["batch"]
    assert len(batch) == 1
    event = batch[0]
    assert event["type"] == "generation-create"  # non-root keeps actual trace_type
    assert event["body"]["parentObservationId"] == "parent-z"
    assert event["body"]["statusMessage"] == "error"  # metadata-only ERROR mapping
    assert "input" not in event["body"]  # capture_content False
    assert "output" not in event["body"]
    assert event["body"]["metadata"] == {"content_retention": "metadata"}


def test_langfuse_api_exception_is_swallowed(monkeypatch, enable_export):
    _backend, api = _fake_backend(monkeypatch)
    api.ingestion_batch.side_effect = RuntimeError("network down")

    # Must not raise.
    _emit_trace(
        trace_id="trace-w",
        span_id="span-w",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        is_root=False,
    )


def test_backend_not_langfuse_instance_skips_export(monkeypatch, enable_export):
    """create_trace_backend can return a non-Langfuse backend (e.g. under a
    different configured backend_type); _emit_trace must no-op rather than
    assume Langfuse-shaped attributes exist."""
    monkeypatch.setattr(
        "agent_utilities.harness.trace_backend.create_trace_backend",
        lambda backend_type="langfuse": object(),
    )

    # Must not raise (would AttributeError on ._get_api() if the isinstance
    # guard were removed).
    _emit_trace(
        trace_id="trace-v",
        span_id="span-v",
        parent_span_id=None,
        name="op",
        trace_type="span-create",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        input_data=None,
        output_data=None,
        is_root=False,
    )
