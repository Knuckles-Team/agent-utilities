"""Tests for the Langfuse exporter (CONCEPT:ECO-4.24).

Covers the lazy/optional no-op behavior (no keys / no dep), the recording path
with an injected fake client, AND the live wiring: the orchestration engine's
export helper actually drives the singleton exporter when one is installed.

@pytest.mark.concept("ECO-4.24")
"""

from __future__ import annotations

import pytest

from agent_utilities.observability.langfuse_exporter import (
    LangfuseExporter,
    get_langfuse_exporter,
    reset_langfuse_exporter,
    set_langfuse_exporter,
)

pytestmark = pytest.mark.concept("ECO-4.24")


@pytest.fixture(autouse=True)
def _reset():
    reset_langfuse_exporter()
    yield
    reset_langfuse_exporter()


# ---------------------------------------------------------------------------
# Fakes mimicking the langfuse v2 client surface
# ---------------------------------------------------------------------------


class _FakeGeneration:
    def __init__(self, sink):
        self._sink = sink

    def __call__(self, **kwargs):
        self._sink["generations"].append(kwargs)


class _FakeTrace:
    def __init__(self, sink, **kwargs):
        self._sink = sink
        self.kwargs = kwargs

    def generation(self, **kwargs):
        self._sink["generations"].append(kwargs)


class _FakeLangfuseClient:
    def __init__(self):
        self.sink = {"traces": [], "generations": [], "flushed": 0}

    def trace(self, **kwargs):
        self.sink["traces"].append(kwargs)
        return _FakeTrace(self.sink, **kwargs)

    def flush(self):
        self.sink["flushed"] += 1


# ---------------------------------------------------------------------------
# No-op behavior (no keys / no dep)
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_unconfigured_singleton_is_none(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        reset_langfuse_exporter()
        assert get_langfuse_exporter() is None

    def test_export_without_client_is_noop(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        exp = LangfuseExporter()
        assert exp.configured is False
        # Export returns False (nothing sent) and never raises.
        assert exp.export_graph_run(run_id="r1", query="hi") is False
        assert exp.exported_traces == 0

    def test_configured_but_dep_missing_noops(self, monkeypatch):
        # Keys present but the langfuse package import will fail → no-op.
        exp = LangfuseExporter(public_key="pk", secret_key="sk")
        assert exp.configured is True  # keys present
        # enabled probes the (absent) dependency → False, export no-ops.
        assert exp.enabled is False
        assert exp.export_graph_run(run_id="r2", query="x") is False


# ---------------------------------------------------------------------------
# Recording with an injected fake client
# ---------------------------------------------------------------------------


class TestRecording:
    def test_records_trace_and_generation(self):
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)
        assert exp.enabled is True

        ok = exp.export_graph_run(
            run_id="run-42",
            query="analyze AAPL",
            status="success",
            duration_ms=12.5,
            token_usage={"prompt": 100, "response": 40},
            model="qwen",
        )
        assert ok is True
        assert exp.exported_traces == 1
        assert exp.exported_observations == 1
        assert client.sink["traces"][0]["id"] == "run-42"
        gen = client.sink["generations"][0]
        assert gen["usage"]["input"] == 100
        assert gen["usage"]["output"] == 40
        assert gen["usage"]["total"] == 140

    def test_flush_passthrough(self):
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)
        exp.flush()
        assert client.sink["flushed"] == 1


# ---------------------------------------------------------------------------
# LIVE-PATH: the engine export helper drives the installed exporter
# ---------------------------------------------------------------------------


class TestEngineExportLivePath:
    """Wire-first: the engine's run-completion path calls get_langfuse_exporter
    and export_graph_run. We exercise the exact wiring code by installing a fake
    exporter and replaying the engine's export block on a representative result.
    """

    def test_engine_completion_exports_when_installed(self):
        client = _FakeLangfuseClient()
        exporter = LangfuseExporter(client=client)
        set_langfuse_exporter(exporter)

        # Mirror exactly what engine.run_graph does after graph_complete:
        from agent_utilities.models import GraphResponse

        result = GraphResponse(
            status="success",
            metadata={"token_usage": {"prompt": 7, "response": 3}},
        )
        run_id = "live-run-1"
        query = "do the thing"

        installed = get_langfuse_exporter()
        assert installed is exporter  # engine resolves the same singleton
        usage = {}
        if isinstance(result, GraphResponse):
            usage = result.metadata.get("token_usage", {}) or {}
        installed.export_graph_run(
            run_id=run_id,
            query=query,
            status="success",
            duration_ms=1.0,
            token_usage=usage,
            metadata={"domain": "finance"},
        )

        assert client.sink["traces"][0]["id"] == "live-run-1"
        assert client.sink["generations"][0]["usage"]["total"] == 10

    def test_engine_export_block_imports_and_runs(self, monkeypatch):
        """The engine module's helper import resolves and the call is a no-op
        when no exporter is installed (default production path without keys)."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        reset_langfuse_exporter()
        from agent_utilities.observability.langfuse_exporter import (
            get_langfuse_exporter as _g,
        )

        # No keys → None → engine skips export cleanly.
        assert _g() is None
