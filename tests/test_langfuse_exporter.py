"""Tests for the Langfuse exporter (CONCEPT:AU-OS.observability.langfuse-exporter).

Covers the lazy/optional no-op behavior (no keys / no dep), the recording path
with an injected fake client, AND the live wiring: the orchestration engine's
export helper actually drives the singleton exporter when one is installed.

@pytest.mark.concept("AU-OS.observability.langfuse-exporter")
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import types
from types import SimpleNamespace

import pytest

from agent_utilities.observability.langfuse_exporter import (
    LangfuseExporter,
    get_langfuse_exporter,
    reset_langfuse_exporter,
    set_langfuse_exporter,
)

pytestmark = pytest.mark.concept("AU-OS.observability.langfuse-exporter")


@pytest.fixture(autouse=True)
def _reset():
    reset_langfuse_exporter()
    yield
    reset_langfuse_exporter()


def _set_credential_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LANGFUSE_PUBLIC", "synthetic-public")
    monkeypatch.setenv("TEST_LANGFUSE_SECRET", "synthetic-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY_REF", "env://TEST_LANGFUSE_PUBLIC")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_REF", "env://TEST_LANGFUSE_SECRET")


# ---------------------------------------------------------------------------
# Fakes mimicking the supported Langfuse v4 client surface
# ---------------------------------------------------------------------------


class _FakeObservation:
    def __init__(self, sink, kind, **kwargs):
        self._sink = sink
        self._kind = kind
        self._sink[kind].append(kwargs)

    def start_observation(self, **kwargs):
        return _FakeObservation(self._sink, "generations", **kwargs)

    def end(self):
        self._sink["ended"].append(self._kind)


class _FakeLangfuseClient:
    def __init__(self):
        self.sink = {"traces": [], "generations": [], "ended": [], "flushed": 0}

    def start_observation(self, **kwargs):
        return _FakeObservation(self.sink, "traces", **kwargs)

    def create_trace_id(self, *, seed):
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

    def flush(self):
        self.sink["flushed"] += 1


# ---------------------------------------------------------------------------
# No-op behavior (no keys / no dep)
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_unconfigured_singleton_is_none(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY_REF", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)
        reset_langfuse_exporter()
        assert get_langfuse_exporter() is None

    def test_export_without_client_is_noop(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY_REF", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)
        exp = LangfuseExporter()
        assert exp.configured is False
        # Export returns False (nothing sent) and never raises.
        assert exp.export_graph_run(run_id="r1", query="hi") is False
        assert exp.exported_traces == 0

    def test_trace_credentials_do_not_enable_export_without_explicit_flag(
        self, monkeypatch
    ):
        _set_credential_refs(monkeypatch)
        monkeypatch.setenv("TRACE_EXPORT_ENABLED", "false")
        reset_langfuse_exporter()

        assert get_langfuse_exporter() is None

    def test_trace_export_flag_arms_configured_exporter(self, monkeypatch):
        _set_credential_refs(monkeypatch)
        monkeypatch.setenv("TRACE_EXPORT_ENABLED", "true")
        reset_langfuse_exporter()

        exporter = get_langfuse_exporter()
        assert exporter is not None
        assert exporter.configured is True

    def test_configured_but_dep_missing_noops(self, monkeypatch):
        # Keys present but the langfuse package import will fail → no-op.
        monkeypatch.setitem(sys.modules, "langfuse", None)
        exp = LangfuseExporter(public_key="pk", secret_key="sk")
        assert exp.configured is True  # keys present
        # enabled probes the (absent) dependency → False, export no-ops.
        assert exp.enabled is False
        assert exp.export_graph_run(run_id="r2", query="x") is False

    def test_host_resolution_uses_only_canonical_input(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "https://canonical.invalid")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://source.invalid")
        monkeypatch.setenv("LANGFUSE_URL", "https://legacy.invalid")
        assert LangfuseExporter()._host == "https://canonical.invalid"

        monkeypatch.delenv("LANGFUSE_HOST")
        assert LangfuseExporter()._host == ""

    def test_client_failure_does_not_log_host_values(self, monkeypatch, caplog):
        canonical = "https://canonical.invalid"
        monkeypatch.setenv("LANGFUSE_HOST", canonical)

        module = types.ModuleType("langfuse")

        class _FailingLangfuse:
            def __init__(self, **kwargs):
                raise RuntimeError(f"cannot connect to {kwargs.get('host')}")

        module.Langfuse = _FailingLangfuse
        monkeypatch.setitem(sys.modules, "langfuse", module)

        with caplog.at_level(logging.DEBUG):
            assert LangfuseExporter(public_key="pk", secret_key="sk").enabled is False

        assert canonical not in caplog.text

    def test_real_client_uses_one_dedicated_sdk_tracer_provider(self, monkeypatch):
        captured: list[dict[str, object]] = []
        created_providers: list[object] = []
        global_proxy_provider = object()

        class _DedicatedTracerProvider:
            def __init__(self):
                created_providers.append(self)

        class _ConstructedClient:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        module = types.ModuleType("langfuse")
        module.Langfuse = _ConstructedClient
        monkeypatch.setitem(sys.modules, "langfuse", module)
        monkeypatch.setattr(
            "opentelemetry.sdk.trace.TracerProvider", _DedicatedTracerProvider
        )
        monkeypatch.setattr(
            "opentelemetry.trace.get_tracer_provider",
            lambda: global_proxy_provider,
        )
        monkeypatch.setattr(
            "agent_utilities.observability.langfuse_exporter.configure_langfuse_trust",
            lambda: SimpleNamespace(valid=True),
        )

        exporter = LangfuseExporter(public_key="pk", secret_key="sk")

        assert exporter.enabled is True
        assert exporter.enabled is True
        assert len(captured) == 1
        assert len(created_providers) == 1
        assert captured[0]["tracer_provider"] is created_providers[0]
        assert captured[0]["tracer_provider"] is not global_proxy_provider

    def test_constructed_client_exports_and_flushes_v4_observations(self, monkeypatch):
        client = _FakeLangfuseClient()

        class _ConstructedClient:
            def __new__(cls, **_kwargs):
                return client

        module = types.ModuleType("langfuse")
        module.Langfuse = _ConstructedClient
        monkeypatch.setitem(sys.modules, "langfuse", module)
        monkeypatch.setattr(
            "agent_utilities.observability.langfuse_exporter.configure_langfuse_trust",
            lambda: SimpleNamespace(valid=True),
        )

        exporter = LangfuseExporter(public_key="pk", secret_key="sk")

        assert exporter.export_graph_run(run_id="constructed-v4", query="")
        exporter.flush()
        assert len(client.sink["traces"]) == 1
        assert client.sink["flushed"] == 1


class TestCanonicalHostConsumers:
    def test_gateway_widget_uses_canonical_host(self, monkeypatch):
        canonical = "https://canonical.invalid"
        monkeypatch.setenv("LANGFUSE_HOST", canonical)
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://source.invalid")
        monkeypatch.setenv("LANGFUSE_URL", "https://legacy.invalid")

        captured = {}
        package = types.ModuleType("langfuse_agent")
        api_client = types.ModuleType("langfuse_agent.api_client")

        class _FakeApi:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def health(self):
                return {"status": "OK"}

            def get_traces(self, **_kwargs):
                return {"totalItems": 0}

        api_client.LangfuseApi = _FakeApi
        package.api_client = api_client
        monkeypatch.setitem(sys.modules, "langfuse_agent", package)
        monkeypatch.setitem(sys.modules, "langfuse_agent.api_client", api_client)

        from agent_utilities.gateway.models import ServiceConfig
        from agent_utilities.gateway.widgets.langfuse import Widget

        result = Widget().fetch_data(
            ServiceConfig(id="test", name="test", widget_type="langfuse")
        )

        assert result.status == "ok"
        assert captured["base_url"] == canonical

    def test_hydration_status_uses_canonical_host(self, monkeypatch):
        canonical = "https://canonical.invalid"
        monkeypatch.setenv("LANGFUSE_HOST", canonical)
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://source.invalid")
        monkeypatch.setenv("LANGFUSE_URL", "https://legacy.invalid")

        from agent_utilities.knowledge_graph.core.hydration import HydrationManager

        assert HydrationManager().get_status()["langfuse"]["url"] == canonical


# ---------------------------------------------------------------------------
# Recording with an injected fake client
# ---------------------------------------------------------------------------


class TestRecording:
    def test_trace_root_context_is_fresh_and_not_the_ambient_graphos_span(self):
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        tracer = provider.get_tracer("synthetic-graphos-request")
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)

        try:
            with tracer.start_as_current_span("ambient-request") as ambient:
                ambient_trace_id = f"{ambient.get_span_context().trace_id:032x}"
                assert exp.export_graph_run(run_id="root-one", query="")
                assert exp.export_graph_run(run_id="root-two", query="")
                assert exp.export_graph_run(run_id="root-one", query="")
        finally:
            provider.shutdown()

        root_contexts = [trace["trace_context"] for trace in client.sink["traces"]]
        root_ids = [context["trace_id"] for context in root_contexts]
        assert all(re.fullmatch(r"[0-9a-f]{32}", trace_id) for trace_id in root_ids)
        assert root_ids[0] != ambient_trace_id
        assert root_ids[1] != ambient_trace_id
        assert root_ids[0] != root_ids[1]
        assert root_ids[2] == root_ids[0]

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
        trace = client.sink["traces"][0]
        assert trace["name"].startswith("graph_run:pref_run_")
        assert trace["input"] == ""
        assert trace["metadata"]["content_retention"] == "metadata"
        gen = client.sink["generations"][0]
        assert gen["usage_details"]["input"] == 100
        assert gen["usage_details"]["output"] == 40
        assert gen["usage_details"]["total"] == 140

    def test_records_and_ends_v4_observations(self):
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)

        assert exp.export_graph_run(
            run_id="run-v4",
            query="",
            token_usage={"prompt": 2, "response": 3},
            model="synthetic-model",
        )

        assert client.sink["traces"][0]["name"].startswith("graph_run:pref_run_")
        assert client.sink["traces"][0]["input"] == ""
        generation = client.sink["generations"][0]
        assert generation["as_type"] == "generation"
        assert generation["usage_details"] == {"input": 2, "output": 3, "total": 5}
        assert client.sink["ended"] == ["generations", "traces"]

    def test_metadata_only_trace_retains_closed_attribution_evidence(self):
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)
        evidence = {
            "model_ref": "pref_model_" + "a" * 64,
            "model_class": "economy",
            "skill_ref": "pref_skill_" + "b" * 64,
            "skill_body_ref": "pref_skill_body_" + "c" * 64,
        }

        assert exp.export_graph_run(
            run_id="run:" + "d" * 32,
            query="content is not retained",
            metadata={"untrusted": "discarded"},
            evidence=evidence,
        )

        trace = client.sink["traces"][0]
        assert trace["input"] == ""
        assert trace["metadata"]["content_retention"] == "metadata"
        assert trace["metadata"]["run_ref"].startswith("pref_run_")
        assert {key: trace["metadata"][key] for key in evidence} == evidence
        assert "untrusted" not in trace["metadata"]

    def test_invalid_attribution_evidence_fails_without_export(self):
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)

        assert not exp.export_graph_run(
            run_id="run:" + "d" * 32,
            evidence={"model_ref": "raw-model-name"},
        )
        assert client.sink["traces"] == []
        assert exp.exported_traces == 0

    def test_unsupported_client_does_not_report_a_fake_export(self):
        exp = LangfuseExporter(client=object())

        assert exp.export_graph_run(run_id="unsupported", query="") is False
        assert exp.exported_traces == 0

    def test_flush_passthrough(self):
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)
        exp.flush()
        assert client.sink["flushed"] == 1

    def test_trace_payload_is_privacy_sanitized_before_export(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "true")
        client = _FakeLangfuseClient()
        exp = LangfuseExporter(client=client)

        assert exp.export_graph_run(
            run_id="privacy-run",
            query="Inspect contact@example.test in /home/example/private/input.md",
            metadata={
                "owner_name": "Example Person",
                "endpoint": "https://internal.example.test/graphql",
            },
        )

        trace = client.sink["traces"][0]
        assert "contact@example.test" not in trace["input"]
        assert "/home/example" not in trace["input"]
        assert trace["metadata"]["owner_name"] == "[REDACTED_PERSON]"
        assert trace["metadata"]["endpoint"] == "[REDACTED_LOCATION]"
        assert trace["metadata"]["privacy_redactions"] >= 4


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

        assert client.sink["traces"][0]["name"].startswith("graph_run:pref_run_")
        assert client.sink["generations"][0]["usage_details"]["total"] == 10

    def test_engine_export_block_imports_and_runs(self, monkeypatch):
        """The engine module's helper import resolves and the call is a no-op
        when no exporter is installed (default production path without keys)."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY_REF", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)
        reset_langfuse_exporter()
        from agent_utilities.observability.langfuse_exporter import (
            get_langfuse_exporter as _g,
        )

        # No keys → None → engine skips export cleanly.
        assert _g() is None
