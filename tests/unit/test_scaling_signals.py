"""Scaling signal providers (CONCEPT:OS-5.29).

Covers: the ScalingSignalProvider protocol, the zero-infra
LocalMetricsProvider (in-process gauge read, graceful None without
prometheus-client, None for unknown families), the PrometheusHttpProvider
against a fake httpx transport (vector sum, empty/error/HTTP-failure ⇒ None,
PromQL templating incl. the {service} placeholder), and the
set_scaling_signal_provider deployment seam + config-driven resolution.

@pytest.mark.concept("OS-5.29")
"""

from __future__ import annotations

import json
import sys
import types

import httpx
import pytest

from agent_utilities.orchestration.scaling_signals import (
    LocalMetricsProvider,
    PrometheusHttpProvider,
    ScalingSignalProvider,
    get_scaling_signal_provider,
    set_scaling_signal_provider,
)

pytestmark = pytest.mark.concept("OS-5.29")


@pytest.fixture(autouse=True)
def _reset_seam():
    yield
    set_scaling_signal_provider(None)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_builtins_satisfy_the_protocol():
    assert isinstance(LocalMetricsProvider(), ScalingSignalProvider)
    assert isinstance(PrometheusHttpProvider("http://prom:9090"), ScalingSignalProvider)


# ---------------------------------------------------------------------------
# LocalMetricsProvider
# ---------------------------------------------------------------------------


class _Sample:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value


class _Family:
    def __init__(self, name: str, values: list[float]):
        self.name = name
        self.samples = [_Sample(name, v) for v in values]


def _fake_prometheus(monkeypatch, families: list[_Family]):
    registry = types.SimpleNamespace(collect=lambda: iter(families))
    module = types.ModuleType("prometheus_client")
    module.REGISTRY = registry
    monkeypatch.setitem(sys.modules, "prometheus_client", module)


def test_local_provider_sums_the_mapped_gauge_family(monkeypatch):
    _fake_prometheus(
        monkeypatch,
        [
            _Family("agent_utilities_kg_ingest_queue_depth", [120.0, 30.0]),
            _Family("agent_utilities_kg_ingest_consumer_lag", [7.0]),
        ],
    )
    provider = LocalMetricsProvider()
    assert provider.signal_value("any-svc", "queue_depth") == 150.0
    assert provider.signal_value("any-svc", "consumer_lag") == 7.0


def test_local_provider_resolves_custom_family_names(monkeypatch):
    _fake_prometheus(monkeypatch, [_Family("my_custom_gauge", [4.0])])
    assert LocalMetricsProvider().signal_value("svc", "my_custom_gauge") == 4.0


def test_local_provider_unknown_signal_is_no_data(monkeypatch):
    _fake_prometheus(monkeypatch, [_Family("something_else", [9.0])])
    assert LocalMetricsProvider().signal_value("svc", "queue_depth") is None


def test_local_provider_without_prometheus_client_is_no_data(monkeypatch):
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    assert LocalMetricsProvider().signal_value("svc", "queue_depth") is None


# ---------------------------------------------------------------------------
# PrometheusHttpProvider (fake transport — no server, no client lib)
# ---------------------------------------------------------------------------


def _prom_transport(payload, status_code: int = 200, capture: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status_code, content=json.dumps(payload).encode())

    return httpx.MockTransport(handler)


def _vector(*values: float) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": {}, "value": [1718000000.0, str(v)]} for v in values],
        },
    }


def test_prometheus_provider_sums_vector_results():
    provider = PrometheusHttpProvider(
        "http://prom:9090/", transport=_prom_transport(_vector(120.0, 30.0))
    )
    assert provider.signal_value("svc", "queue_depth") == 150.0


def test_prometheus_provider_empty_result_is_no_data():
    provider = PrometheusHttpProvider(
        "http://prom:9090", transport=_prom_transport(_vector())
    )
    assert provider.signal_value("svc", "queue_depth") is None


def test_prometheus_provider_error_status_is_no_data():
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        transport=_prom_transport({"status": "error", "error": "bad query"}),
    )
    assert provider.signal_value("svc", "queue_depth") is None


def test_prometheus_provider_http_failure_is_no_data():
    provider = PrometheusHttpProvider(
        "http://prom:9090", transport=_prom_transport({}, status_code=500)
    )
    assert provider.signal_value("svc", "queue_depth") is None


def test_prometheus_provider_query_templating():
    captured: list[httpx.Request] = []
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        transport=_prom_transport(_vector(1.0), capture=captured),
    )
    provider.signal_value("vector-mcp", "cpu")
    cpu_query = captured[-1].url.params["query"]
    assert 'container_label_com_docker_swarm_service_name="vector-mcp"' in cpu_query

    provider.signal_value("vector-mcp", 'sum(my_metric{svc="{service}"})')
    custom_query = captured[-1].url.params["query"]
    assert custom_query == 'sum(my_metric{svc="vector-mcp"})'

    assert captured[-1].url.path == "/api/v1/query"


# ---------------------------------------------------------------------------
# Deployment seam + config resolution
# ---------------------------------------------------------------------------


def test_injected_provider_wins():
    class _Custom:
        name = "custom"

        def signal_value(self, service: str, signal: str) -> float | None:
            return 42.0

    custom = _Custom()
    set_scaling_signal_provider(custom)
    assert get_scaling_signal_provider() is custom
    set_scaling_signal_provider(None)
    assert get_scaling_signal_provider() is not custom


def test_default_resolution_without_url_is_local(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "scaling_prometheus_url", None)
    assert isinstance(get_scaling_signal_provider(), LocalMetricsProvider)


def test_url_flag_selects_prometheus_provider(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "scaling_prometheus_url", "http://prom:9090")
    provider = get_scaling_signal_provider()
    assert isinstance(provider, PrometheusHttpProvider)
    assert provider.base_url == "http://prom:9090"
