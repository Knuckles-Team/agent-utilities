"""Static security contracts for metadata-only OTLP export."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.observability import custom_observability as observability


@pytest.mark.parametrize(
    "value",
    [
        "Authorization=secret\r\nInjected=value",
        "Authorization=first,authorization=second",
        "missing-separator",
    ],
)
def test_otlp_headers_reject_ambiguous_or_injectable_values(value: str) -> None:
    with pytest.raises(ValueError):
        observability.parse_otlp_headers(value)


def test_metadata_only_attribute_projection_contains_no_raw_strings() -> None:
    raw = {
        "user.name": "person@example.test",
        "http.url": "https://example.test/private?token=secret",
        "tool.arguments": '{"path":"/private/location"}',
        "duration_ms": 12.5,
        "status": "success",
    }
    projected = observability._metadata_only_attributes(raw)
    rendered = repr(projected)

    assert "person@example.test" not in rendered
    assert "token=secret" not in rendered
    assert "/private/location" not in rendered
    assert "user.name" not in projected
    assert "http.url" not in projected
    assert "tool.arguments" not in projected
    assert projected["duration_ms"] == 12.5
    assert projected["status"] == "success"


def test_metadata_only_exporter_drops_events_links_resources_and_status_text() -> None:
    trace = pytest.importorskip("opentelemetry.trace")
    status_module = pytest.importorskip("opentelemetry.trace.status")
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    resources = pytest.importorskip("opentelemetry.sdk.resources")

    context = trace.SpanContext(
        trace_id=1,
        span_id=2,
        is_remote=False,
        trace_flags=trace.TraceFlags(1),
        trace_state=trace.TraceState([("vendor", "private-value")]),
    )
    raw = sdk_trace.ReadableSpan(
        name="execute_tool environment-specific-name",
        context=context,
        resource=resources.Resource({"host.name": "private-host"}),
        attributes={"tool.arguments": "top-secret", "status": "success"},
        events=(SimpleNamespace(name="exception", attributes={"message": "secret"}),),
        links=(SimpleNamespace(context=context, attributes={"token": "secret"}),),
        status=status_module.Status(
            status_module.StatusCode.ERROR, "private error description"
        ),
        start_time=1,
        end_time=2,
    )
    inner = MagicMock()
    inner.export.return_value = "ok"
    exporter = observability._MetadataOnlySpanExporter(
        inner, service_ref="pref_service_safe"
    )

    assert exporter.export([raw]) == "ok"
    exported = inner.export.call_args.args[0][0]
    rendered = repr(
        {
            "name": exported.name,
            "resource": dict(exported.resource.attributes),
            "attributes": dict(exported.attributes or {}),
            "events": exported.events,
            "links": exported.links,
            "status": exported.status.description,
            "trace_state": list(exported.context.trace_state.items()),
        }
    )
    for forbidden in (
        "environment-specific-name",
        "private-host",
        "top-secret",
        "private error description",
        "private-value",
    ):
        assert forbidden not in rendered
    assert exported.events == ()
    assert exported.links == ()


def test_setup_installs_explicit_no_content_agent_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        _instrument_default = None

        @classmethod
        def instrument_all(cls, value=True):
            cls._instrument_default = value

    class FakeTrust:
        @staticmethod
        def child_env() -> dict[str, str]:
            return {}

    from agent_utilities.core import transport_security

    fake_logfire = MagicMock()
    monkeypatch.setattr(observability, "HAS_LOGFIRE", True)
    monkeypatch.setattr(observability, "logfire", fake_logfire)
    monkeypatch.setattr(
        observability,
        "instrument_context_agents",
        lambda settings: (
            FakeAgent.instrument_all(settings) or FakeAgent._instrument_default
        ),
    )
    monkeypatch.setattr(
        observability,
        "disable_context_agent_instrumentation",
        lambda: FakeAgent.instrument_all(False),
    )
    monkeypatch.setattr(observability, "_otel_initialized", False)
    monkeypatch.setattr(observability, "_agent_instrumented_metadata_only", False)
    monkeypatch.setattr(
        observability, "_create_otlp_span_processor", lambda **_: object()
    )
    monkeypatch.setattr(
        transport_security,
        "resolve_configured_tls_profile",
        lambda *_args, **_kwargs: FakeTrust(),
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "before-test")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://before.example.test")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=before")

    observability.setup_otel(
        service_name="environment-specific-service",
        endpoint="https://telemetry.example.test/otel",
        headers="Authorization=Basic opaque",
    )

    call = fake_logfire.instrument_pydantic_ai.call_args
    assert call.kwargs["include_content"] is False
    assert call.kwargs["include_binary_content"] is False
    assert FakeAgent._instrument_default.include_content is False
    assert FakeAgent._instrument_default.include_binary_content is False
    assert (
        "environment-specific-service"
        not in observability.os.environ["OTEL_SERVICE_NAME"]
    )


def test_reference_auth_and_langfuse_tls_are_reused_only_for_same_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://telemetry.example.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY_REF", "env://TEST_LANGFUSE_PUBLIC")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_REF", "env://TEST_LANGFUSE_SECRET")
    monkeypatch.setenv("TEST_LANGFUSE_PUBLIC", "synthetic-public")
    monkeypatch.setenv("TEST_LANGFUSE_SECRET", "synthetic-secret")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS_REF", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_SECRET_KEY_REF", raising=False)

    headers, reused = observability._resolve_otel_headers(
        endpoint="https://telemetry.example.test/api/public/otel",
        headers=None,
        public_key=None,
        secret_key=None,
    )
    assert reused is True
    assert observability.parse_otlp_headers(headers)["Authorization"].startswith(
        "Basic "
    )

    cross_origin_headers, cross_origin_reused = observability._resolve_otel_headers(
        endpoint="https://collector.example.test/otel",
        headers=None,
        public_key=None,
        secret_key=None,
    )
    assert cross_origin_headers == ""
    assert cross_origin_reused is False


def test_endpoint_derivation_and_origin_policy_share_canonical_loopback_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:8080")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY_REF", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)

    endpoint = observability._resolve_otel_endpoint(
        None,
        public_key="synthetic-public",
        secret_key="synthetic-" + "secret",
    )

    assert endpoint == "http://localhost:8080/api/public/otel"
    assert observability._same_origin(endpoint, "http://localhost:8080") is True
    assert observability._is_langfuse_otel_endpoint(endpoint, "http://localhost:8080")
    assert not observability._is_langfuse_otel_endpoint(
        "http://localhost:8080/custom-collector", "http://localhost:8080"
    )
    assert observability._same_origin(
        "http://localhost/api/public/otel", "http://localhost:80"
    )
    with pytest.raises(ValueError):
        observability._resolve_otel_endpoint("http://collector.example.test/otel")


def test_same_origin_langfuse_transport_reuses_tls_independent_of_auth_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://telemetry.example.test")
    config = SimpleNamespace(
        otel_tls_profile=None,
        otel_tls_profile_ref=None,
        langfuse_tls_profile="private-trust",
        langfuse_tls_profile_ref=None,
    )
    resolved = object()
    with (
        patch.object(observability, "AgentConfig", return_value=config),
        patch(
            "agent_utilities.core.transport_security.resolve_configured_tls_profile",
            return_value=resolved,
        ) as resolver,
    ):
        result = observability._resolve_otel_transport(
            "https://telemetry.example.test/api/public/otel"
        )

    assert result is resolved
    resolver.assert_called_once_with(
        "OTEL",
        profile_name="private-trust",
        profile_ref=None,
        config=config,
    )


def test_runtime_materialized_headers_override_their_configured_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF", "env://OTEL_PUBLIC")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_SECRET_KEY_REF", "env://OTEL_SECRET")
    monkeypatch.setenv("OTEL_PUBLIC", "synthetic-public")
    monkeypatch.setenv("OTEL_SECRET", "synthetic-secret")

    headers, reused = observability._resolve_otel_headers(
        endpoint="https://collector.example.test/otel",
        headers="Authorization=Basic materialized",
        public_key=None,
        secret_key=None,
    )

    assert headers == "Authorization=Basic materialized"
    assert reused is False


@pytest.mark.parametrize(
    ("status_code", "payload", "expected_ok", "expected_error"),
    [
        (200, {"data": []}, True, None),
        (401, {}, False, "authentication_failed"),
    ],
)
def test_otel_health_uses_authenticated_langfuse_api_semantics(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    payload: dict,
    expected_ok: bool,
    expected_error: str | None,
) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://telemetry.example.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY_REF", "env://LANGFUSE_TEST_PUBLIC")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_REF", "env://LANGFUSE_TEST_SECRET")
    monkeypatch.setenv("LANGFUSE_TEST_PUBLIC", "synthetic-public")
    monkeypatch.setenv("LANGFUSE_TEST_SECRET", "synthetic-secret")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS_REF", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_SECRET_KEY_REF", raising=False)
    response = SimpleNamespace(status_code=status_code, json=lambda: payload)
    client = MagicMock()
    client.get.return_value = response
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    trust = SimpleNamespace(httpx_kwargs=lambda: {}, cleanup=lambda: None)

    with (
        patch.object(observability, "_resolve_otel_transport", return_value=trust),
        patch(
            "agent_utilities.core.http_client.create_http_client",
            return_value=client_context,
        ),
    ):
        report = observability.verify_otel_pipeline()

    assert report["exporter_ok"] is expected_ok
    assert report.get("endpoint_error") == expected_error
    if status_code == 200:
        request = client.get.call_args
        assert request.kwargs["headers"]["Authorization"].startswith("Basic ")
        assert request.kwargs["params"] == {"limit": 1}


def test_otel_env_header_ref_does_not_initialize_graph_secret_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OTEL_HEADERS", "Authorization=Basic synthetic")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS_REF", "env://TEST_OTEL_HEADERS")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_SECRET_KEY_REF", raising=False)

    with patch(
        "agent_utilities.security.secrets_client.create_secrets_client",
        side_effect=AssertionError("env refs must not initialize graph secrets"),
    ) as create_secrets_client:
        headers, reused = observability._resolve_otel_headers(
            endpoint="https://collector.example.test/otel",
            headers=None,
            public_key=None,
            secret_key=None,
        )

    create_secrets_client.assert_not_called()
    assert headers == "Authorization=Basic synthetic"
    assert reused is False
