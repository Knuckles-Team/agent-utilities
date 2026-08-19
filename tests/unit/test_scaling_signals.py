"""Typed and bounded autoscaling signal providers.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from dataclasses import replace

import httpx
import pytest

from agent_utilities.orchestration.scaling_signals import (
    _PROMETHEUS_INFLIGHT,
    KG_INGEST_CONSUMER_LAG_OBSERVED_AT,
    KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
    KG_INGEST_WORKER_SERVICE,
    MAX_QUERY_LENGTH,
    MAX_SIGNAL_AGE_S,
    BulkScalingSignalProvider,
    LocalMetricsProvider,
    PrometheusHttpProvider,
    ScalingSignalProvider,
    ScalingSignalSample,
    SignalDefinition,
    _reap_prometheus_inflight,
    get_scaling_signal_provider,
    get_signal_definition,
    read_scaling_signal_samples,
    set_scaling_signal_provider,
    validate_scaling_signal_sample,
)

pytestmark = pytest.mark.concept("AU-OS.scaling.reactive-replica-autoscaling")


@pytest.fixture(autouse=True)
def _reset_seam():
    yield
    set_scaling_signal_provider(None)


def _now() -> float:
    return time.time()


def test_typed_sample_is_immutable_and_rejects_bad_values():
    sample = ScalingSignalSample(
        value=4,
        source="test",
        service="svc",
        signal="cpu",
        aggregation="per_replica",
        observed_at=_now(),
        unit="percent",
        scope="service",
    )
    assert sample.value == 4.0
    with pytest.raises((AttributeError, TypeError)):
        sample.value = 5.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        ScalingSignalSample(
            value=float("nan"),
            source="test",
            service="svc",
            signal="cpu",
            aggregation="per_replica",
            observed_at=_now(),
            unit="percent",
            scope="service",
        )


def test_builtins_satisfy_the_typed_protocol():
    assert isinstance(LocalMetricsProvider(), ScalingSignalProvider)
    assert isinstance(PrometheusHttpProvider("http://prom:9090"), ScalingSignalProvider)
    assert isinstance(LocalMetricsProvider(), BulkScalingSignalProvider)
    assert isinstance(
        PrometheusHttpProvider("http://prom:9090"), BulkScalingSignalProvider
    )


def test_builtin_signal_metadata_is_truthful_and_scoped():
    queue = get_signal_definition("queue_depth")
    lag = get_signal_definition("consumer_lag")
    cpu = get_signal_definition("cpu")
    assert queue is not None and (queue.unit, queue.scope) == ("items", "fleet")
    assert lag is not None and (lag.unit, lag.scope) == ("messages", "fleet")
    assert cpu is not None and (cpu.unit, cpu.scope) == ("percent", "service")
    assert queue.service_binding == KG_INGEST_WORKER_SERVICE
    assert lag.service_binding == KG_INGEST_WORKER_SERVICE
    assert queue.local_observed_at_metric_family == KG_INGEST_QUEUE_DEPTH_OBSERVED_AT
    assert lag.local_observed_at_metric_family == KG_INGEST_CONSUMER_LAG_OBSERVED_AT
    assert queue.binds_service("svc") is False
    assert queue.binds_service(KG_INGEST_WORKER_SERVICE) is True
    assert cpu.binds_service("svc") is True


def test_metadata_mismatch_is_no_data_at_consumption():
    sample = ScalingSignalSample(
        value=1.0,
        source="test",
        service="svc",
        signal="queue_depth",
        aggregation="fleet_total",
        observed_at=_now(),
        unit="items",
        scope="fleet",
    )
    assert (
        validate_scaling_signal_sample(
            sample,
            service="svc",
            signal="queue_depth",
            aggregation="fleet_total",
            unit="messages",
            scope="fleet",
        )
        is None
    )


class _Sample:
    _TIMESTAMP_UNSET = object()

    def __init__(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
        timestamp: float | None | object = _TIMESTAMP_UNSET,
    ):
        self.name = name
        self.value = value
        self.labels = labels or {}
        self.timestamp = _now() if timestamp is self._TIMESTAMP_UNSET else timestamp


class _Family:
    def __init__(self, name: str, samples: list[_Sample]):
        self.name = name
        self.samples = samples


def _fake_prometheus(monkeypatch, families: list[_Family]):
    registry = types.SimpleNamespace(collect=lambda: iter(families))
    module = types.ModuleType("prometheus_client")
    module.REGISTRY = registry
    monkeypatch.setitem(sys.modules, "prometheus_client", module)


def test_local_provider_preserves_fleet_total_semantics(monkeypatch):
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "agent_utilities_kg_ingest_queue_depth",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        120.0,
                        labels={"backend": "sqlite"},
                    ),
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        30.0,
                        labels={"backend": "kafka"},
                    ),
                ],
            ),
            _Family(
                KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                        _now(),
                        labels={"backend": "sqlite"},
                    ),
                    _Sample(
                        KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                        _now(),
                        labels={"backend": "kafka"},
                    ),
                ],
            ),
            _Family(
                "agent_utilities_kg_ingest_consumer_lag",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_consumer_lag",
                        7.0,
                        labels={"topic": "kg_tasks", "group": "kg-ingest"},
                    )
                ],
            ),
            _Family(
                KG_INGEST_CONSUMER_LAG_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_CONSUMER_LAG_OBSERVED_AT,
                        _now(),
                        labels={"topic": "kg_tasks", "group": "kg-ingest"},
                    )
                ],
            ),
        ],
    )
    queue_definition = replace(
        get_signal_definition("queue_depth"), service_binding="any-svc"
    )
    lag_definition = replace(
        get_signal_definition("consumer_lag"), service_binding="any-svc"
    )
    provider = LocalMetricsProvider(
        {"queue_depth": queue_definition, "consumer_lag": lag_definition}
    )
    queue = provider.signal_value("any-svc", "queue_depth")
    lag = provider.signal_value("any-svc", "consumer_lag")
    assert queue is not None and queue.value == 150.0
    assert queue.aggregation == "fleet_total"
    assert lag is not None and lag.value == 7.0
    assert lag.aggregation == "fleet_total"


def test_local_bulk_provider_collects_registry_once(monkeypatch):
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "agent_utilities_kg_ingest_queue_depth",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        10.0,
                        labels={"backend": "sqlite"},
                    )
                ],
            ),
            _Family(
                KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                        _now(),
                        labels={"backend": "sqlite"},
                    )
                ],
            ),
            _Family(
                "agent_utilities_kg_ingest_consumer_lag",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_consumer_lag",
                        4.0,
                        labels={"topic": "kg_tasks", "group": "kg-ingest"},
                    )
                ],
            ),
            _Family(
                KG_INGEST_CONSUMER_LAG_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_CONSUMER_LAG_OBSERVED_AT,
                        _now(),
                        labels={"topic": "kg_tasks", "group": "kg-ingest"},
                    )
                ],
            ),
        ],
    )
    queue = replace(get_signal_definition("queue_depth"), service_binding="svc")
    lag = replace(get_signal_definition("consumer_lag"), service_binding="svc")
    provider = LocalMetricsProvider({"queue_depth": queue, "consumer_lag": lag})
    import prometheus_client

    original_collect = prometheus_client.REGISTRY.collect
    calls = 0

    def counted_collect():
        nonlocal calls
        calls += 1
        return original_collect()

    monkeypatch.setattr(prometheus_client.REGISTRY, "collect", counted_collect)
    values = provider.signal_values([("svc", "queue_depth"), ("svc", "consumer_lag")])
    assert calls == 1
    assert values[("svc", "queue_depth")] is not None
    assert values[("svc", "consumer_lag")] is not None


def test_local_provider_rejects_missing_observed_at_companion(monkeypatch):
    definition = replace(get_signal_definition("queue_depth"), service_binding="svc")
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "agent_utilities_kg_ingest_queue_depth",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        10.0,
                        timestamp=None,
                    )
                ],
            )
        ],
    )
    assert (
        LocalMetricsProvider({"queue_depth": definition}).signal_value(
            "svc", "queue_depth"
        )
        is None
    )


def test_unlabelled_global_builtin_cannot_be_relabelled_to_a_service(monkeypatch):
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "agent_utilities_kg_ingest_queue_depth",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        10.0,
                        labels={"backend": "sqlite"},
                    )
                ],
            ),
            _Family(
                KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                        _now(),
                        labels={"backend": "sqlite"},
                    )
                ],
            ),
        ],
    )
    provider = LocalMetricsProvider()
    assert provider.signal_value("svc-a", "queue_depth") is None
    assert provider.signal_value("svc-b", "queue_depth") is None
    assert provider.signal_value(KG_INGEST_WORKER_SERVICE, "queue_depth") is not None

    requests: list[httpx.Request] = []
    prometheus = PrometheusHttpProvider(
        "http://prom:9090",
        transport=_prom_transport(_vector(10.0), capture=requests),
    )
    assert prometheus.signal_value(KG_INGEST_WORKER_SERVICE, "queue_depth") is not None
    assert prometheus.signal_value("svc-a", "queue_depth") is None
    assert len(requests) == 1


def test_local_provider_requires_allowlisted_service_bound_metric(monkeypatch):
    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        local_metric_family="service_requests",
        local_observed_at_metric_family="service_requests_observed_at",
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "service_requests",
                [
                    _Sample(
                        "service_requests",
                        4.0,
                        labels={"service": "svc-a", "pod": "p1"},
                    ),
                    _Sample(
                        "service_requests",
                        6.0,
                        labels={"service": "svc-a", "pod": "p2"},
                    ),
                    _Sample(
                        "service_requests",
                        9.0,
                        labels={"service": "svc-b", "pod": "p1"},
                    ),
                ],
            ),
            _Family(
                "service_requests_observed_at",
                [
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc-a", "pod": "p1"},
                    ),
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc-a", "pod": "p2"},
                    ),
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc-b", "pod": "p1"},
                    ),
                ],
            ),
        ],
    )
    provider = LocalMetricsProvider({"requests": definition})
    sample = provider.signal_value("svc-a", "requests")
    assert sample is not None and sample.value == 5.0
    assert provider.signal_value("svc-c", "requests") is None


def test_local_provider_rejects_malformed_or_high_cardinality_metrics(monkeypatch):
    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        local_metric_family="service_requests",
        local_observed_at_metric_family="service_requests_observed_at",
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "service_requests",
                [
                    _Sample(
                        "service_requests",
                        1.0,
                        labels={"service": "svc", "pod": "p1"},
                    ),
                    _Sample(
                        "service_requests",
                        2.0,
                        labels={"service": "svc", "pod": "p2"},
                    ),
                    _Sample(
                        "service_requests",
                        3.0,
                        labels={"service": "svc", "pod": "p3"},
                    ),
                ],
            ),
            _Family(
                "service_requests_observed_at",
                [
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc", "pod": "p1"},
                    ),
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc", "pod": "p2"},
                    ),
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc", "pod": "p3"},
                    ),
                ],
            ),
        ],
    )
    assert (
        LocalMetricsProvider({"requests": definition}, max_series=2).signal_value(
            "svc", "requests"
        )
        is None
    )

    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "agent_utilities_kg_ingest_queue_depth",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        -1.0,
                        labels={"backend": "sqlite"},
                    )
                ],
            ),
            _Family(
                KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                        _now(),
                        labels={"backend": "sqlite"},
                    )
                ],
            ),
        ],
    )
    assert LocalMetricsProvider().signal_value("svc", "queue_depth") is None


def test_local_provider_requires_exact_unique_value_observed_at_pairs(monkeypatch):
    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        local_metric_family="service_requests",
        local_observed_at_metric_family="service_requests_observed_at",
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "service_requests",
                [_Sample("service_requests", 1.0, labels={"service": "svc"})],
            ),
            _Family(
                "service_requests_observed_at",
                [
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "other"},
                    )
                ],
            ),
        ],
    )
    provider = LocalMetricsProvider({"requests": definition})
    assert provider.signal_value("svc", "requests") is None

    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "service_requests",
                [
                    _Sample("service_requests", 1.0, labels={"service": "svc"}),
                    _Sample("service_requests", 2.0, labels={"service": "svc"}),
                ],
            ),
            _Family(
                "service_requests_observed_at",
                [
                    _Sample(
                        "service_requests_observed_at",
                        _now(),
                        labels={"service": "svc"},
                    )
                ],
            ),
        ],
    )
    assert provider.signal_value("svc", "requests") is None


def test_local_provider_uses_durable_companion_timestamp(monkeypatch):
    observed_at = _now() - 5.0
    _fake_prometheus(
        monkeypatch,
        [
            _Family(
                "agent_utilities_kg_ingest_queue_depth",
                [
                    _Sample(
                        "agent_utilities_kg_ingest_queue_depth",
                        10.0,
                        labels={"backend": "sqlite"},
                        timestamp=None,
                    )
                ],
            ),
            _Family(
                KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                [
                    _Sample(
                        KG_INGEST_QUEUE_DEPTH_OBSERVED_AT,
                        observed_at,
                        labels={"backend": "sqlite"},
                        timestamp=None,
                    )
                ],
            ),
        ],
    )
    sample = LocalMetricsProvider().signal_value(
        KG_INGEST_WORKER_SERVICE, "queue_depth"
    )
    assert sample is not None
    assert sample.observed_at == observed_at


def test_local_provider_unknown_signal_is_no_data(monkeypatch):
    _fake_prometheus(
        monkeypatch,
        [_Family("unapproved_metric", [_Sample("unapproved_metric", 4.0)])],
    )
    assert LocalMetricsProvider().signal_value("svc", "unapproved_metric") is None


def test_local_provider_without_prometheus_client_is_no_data(monkeypatch):
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    assert LocalMetricsProvider().signal_value("svc", "queue_depth") is None


def _prom_transport(payload, status_code: int = 200, capture: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status_code, content=json.dumps(payload).encode())

    return httpx.MockTransport(handler)


def _vector(
    *values: float,
    observed_at: float | None = None,
    metrics: list[dict[str, str]] | None = None,
) -> dict:
    timestamp = _now() if observed_at is None else observed_at
    metric_rows = metrics or [{} for _ in values]
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": metric, "value": [timestamp, str(value)]}
                for metric, value in zip(metric_rows, values, strict=True)
            ],
        },
    }


def test_prometheus_provider_returns_typed_fleet_total_sample():
    definition = replace(get_signal_definition("queue_depth"), service_binding="svc")
    provider = PrometheusHttpProvider(
        "http://prom:9090/",
        signal_definitions={"queue_depth": definition},
        transport=_prom_transport(_vector(120.0, 30.0)),
    )
    sample = provider.signal_value("svc", "queue_depth")
    assert sample is not None
    assert sample.value == 150.0
    assert sample.source == "prometheus"
    assert sample.aggregation == "fleet_total"


def test_prometheus_provider_averages_per_replica_series():
    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='requests_rate{service="{service}"}',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090/",
        signal_definitions={"requests": definition},
        transport=_prom_transport(
            _vector(
                4.0,
                6.0,
                metrics=[{"service": "svc"}, {"service": "svc"}],
            )
        ),
    )

    sample = provider.signal_value("svc", "requests")

    assert sample is not None
    assert sample.value == 5.0


def test_prometheus_provider_rejects_stale_replay_and_cross_service_samples():
    queue_definition = replace(
        get_signal_definition("queue_depth"), service_binding="svc"
    )
    stale = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"queue_depth": queue_definition},
        transport=_prom_transport(
            _vector(1.0, observed_at=_now() - MAX_SIGNAL_AGE_S - 1)
        ),
    )
    assert stale.signal_value("svc", "queue_depth") is None

    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='sum(rate(requests_total{service="{service}"}[1m]))',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    wrong_service = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"requests": definition},
        transport=_prom_transport(_vector(1.0, metrics=[{"service": "other"}])),
    )
    assert wrong_service.signal_value("svc", "requests") is None

    sample = ScalingSignalSample(
        value=1.0,
        source="test",
        service="svc",
        signal="queue_depth",
        aggregation="fleet_total",
        observed_at=_now(),
        unit="items",
        scope="fleet",
    )
    assert (
        validate_scaling_signal_sample(
            sample,
            service="svc",
            signal="queue_depth",
            aggregation="fleet_total",
            unit="items",
            scope="fleet",
            previous_observed_at=sample.observed_at,
        )
        is None
    )


def test_prometheus_provider_rejects_malformed_high_cardinality_and_bad_values():
    queue_definition = replace(
        get_signal_definition("queue_depth"), service_binding="svc"
    )
    malformed = {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1]}]},
    }
    assert (
        PrometheusHttpProvider(
            "http://prom:9090",
            signal_definitions={"queue_depth": queue_definition},
            transport=_prom_transport(malformed),
        ).signal_value("svc", "queue_depth")
        is None
    )

    high_cardinality = _vector(*range(3))
    assert (
        PrometheusHttpProvider(
            "http://prom:9090",
            signal_definitions={"queue_depth": queue_definition},
            max_series=2,
            transport=_prom_transport(high_cardinality),
        ).signal_value("svc", "queue_depth")
        is None
    )

    assert (
        PrometheusHttpProvider(
            "http://prom:9090",
            signal_definitions={"queue_depth": queue_definition},
            transport=_prom_transport(_vector(float("nan"))),
        ).signal_value("svc", "queue_depth")
        is None
    )


def test_prometheus_provider_timeout_and_unknown_symbolic_signal_are_no_data():
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    queue_definition = replace(
        get_signal_definition("queue_depth"), service_binding="svc"
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"queue_depth": queue_definition},
        transport=httpx.MockTransport(timeout),
    )
    assert provider.signal_value("svc", "queue_depth") is None
    assert provider.signal_value("svc", 'sum(raw_metric{service="svc"})') is None


def test_provider_resource_limits_are_strictly_typed_and_bounded():
    with pytest.raises(ValueError):
        PrometheusHttpProvider("http://prom:9090", timeout=True)
    with pytest.raises(ValueError):
        PrometheusHttpProvider("http://prom:9090", max_series=1.5)


def test_prometheus_provider_uses_only_deployment_allowlisted_query():
    captured: list[httpx.Request] = []
    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='sum(rate(requests_total{service="{service}"}[1m]))',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"requests": definition},
        transport=_prom_transport(
            _vector(1.0, metrics=[{"service": "svc"}]), capture=captured
        ),
    )
    sample = provider.signal_value("svc", "requests")
    assert sample is not None and sample.aggregation == "per_replica"
    assert captured[-1].url.params["query"] == (
        'sum(rate(requests_total{service="svc"}[1m]))'
    )
    with pytest.raises(ValueError):
        SignalDefinition(
            name="unsafe",
            aggregation="per_replica",
            query_template="x" * (MAX_QUERY_LENGTH + 1),
        )


def test_builtin_cpu_query_is_service_bound_and_well_formed():
    captured: list[httpx.Request] = []
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        transport=_prom_transport(
            _vector(
                0.75,
                metrics=[{"container_label_com_docker_swarm_service_name": "svc-a"}],
            ),
            capture=captured,
        ),
    )
    sample = provider.signal_value("svc-a", "cpu")
    assert sample is not None and sample.aggregation == "per_replica"
    assert captured[-1].url.params["query"] == (
        "100 * avg(rate(container_cpu_usage_seconds_total"
        '{container_label_com_docker_swarm_service_name="svc-a"}[5m]))'
    )


def test_prometheus_bulk_deduplicates_queries_and_bounds_concurrency():
    captured: list[httpx.Request] = []
    lock = threading.Lock()
    active = 0
    max_active = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        query = request.url.params["query"]
        service = "svc-a" if "svc-a" in query else "svc-b"
        with lock:
            active += 1
            max_active = max(max_active, active)
            captured.append(request)
        time.sleep(0.01)
        with lock:
            active -= 1
        return httpx.Response(
            200,
            content=json.dumps(_vector(1.0, metrics=[{"service": service}])).encode(),
        )

    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='sum(requests_total{service="{service}"})',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"requests": definition},
        transport=httpx.MockTransport(handler),
        max_concurrency=1,
        overall_timeout=1.0,
    )
    values = provider.signal_values(
        [
            ("svc-a", "requests"),
            ("svc-a", "requests"),
            ("svc-b", "requests"),
        ]
    )
    assert len(captured) == 2
    assert max_active <= 1
    assert all(
        values[request] is not None
        for request in (("svc-a", "requests"), ("svc-b", "requests"))
    )


def test_prometheus_bulk_uses_rolling_waves_for_all_allowlisted_queries():
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        service = query.split('service="', 1)[1].split('"', 1)[0]
        captured.append(service)
        time.sleep(0.005)
        return httpx.Response(
            200,
            content=json.dumps(_vector(1.0, metrics=[{"service": service}])).encode(),
        )

    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='sum(requests_total{service="{service}"})',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"requests": definition},
        transport=httpx.MockTransport(handler),
        transport_thread_safe=True,
        max_concurrency=4,
        overall_timeout=1.0,
    )
    requests = [(f"svc-{index}", "requests") for index in range(6)]
    values = provider.signal_values(requests)
    assert len(captured) == 6
    assert set(captured) == {f"svc-{index}" for index in range(6)}
    assert all(values[request] is not None for request in requests)


def test_prometheus_bulk_returns_at_the_overall_deadline():
    started = threading.Event()
    release = threading.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        started.set()
        release.wait(1.0)
        return httpx.Response(200, content=json.dumps(_vector(1.0)).encode())

    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='sum(requests_total{service="{service}"})',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"requests": definition},
        transport=httpx.MockTransport(handler),
        max_concurrency=1,
        timeout=1.0,
        overall_timeout=0.05,
    )
    started_at = time.monotonic()
    values = provider.signal_values([("svc-a", "requests"), ("svc-b", "requests")])
    elapsed = time.monotonic() - started_at
    release.set()
    for _ in range(20):
        _reap_prometheus_inflight()
        if not _PROMETHEUS_INFLIGHT:
            break
        time.sleep(0.001)
    assert started.is_set()
    assert elapsed < 0.5
    assert all(value is None for value in values.values())


def test_prometheus_timeout_slot_is_bounded_across_repeated_ticks():
    from agent_utilities.orchestration import scaling_signals as signal_module

    started = threading.Event()
    release = threading.Event()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(1.0)
        return httpx.Response(200, content=json.dumps(_vector(1.0)).encode())

    definition = SignalDefinition(
        name="requests",
        aggregation="per_replica",
        query_template='sum(requests_total{service="{service}"})',
        service_label="service",
        unit="requests_per_second",
        scope="service",
    )
    provider = PrometheusHttpProvider(
        "http://prom:9090",
        signal_definitions={"requests": definition},
        transport=httpx.MockTransport(handler),
        max_concurrency=1,
        overall_timeout=0.03,
    )
    first = provider.signal_values([("svc-a", "requests")])
    assert started.is_set()
    second = provider.signal_values([("svc-a", "requests")])
    assert first[("svc-a", "requests")] is None
    assert second[("svc-a", "requests")] is None
    assert calls == 1
    assert signal_module._PROMETHEUS_EXECUTOR is not None  # noqa: SLF001
    assert all(
        thread.daemon
        for thread in signal_module._PROMETHEUS_EXECUTOR._threads  # noqa: SLF001
    )
    release.set()
    for _ in range(20):
        _reap_prometheus_inflight()
        if not _PROMETHEUS_INFLIGHT:
            break
        time.sleep(0.001)
    assert not _PROMETHEUS_INFLIGHT


def test_untrusted_single_read_provider_is_not_fanned_out():
    class RemoteSingleRead:
        name = "remote-single"
        calls = 0

        def signal_definition(self, signal, service=None):
            return replace(
                get_signal_definition("cpu"),
                name=signal,
                service_binding=service,
            )

        def signal_value(self, service, signal):
            self.calls += 1
            raise AssertionError("remote single reads must not be fanned out")

    provider = RemoteSingleRead()
    values = read_scaling_signal_samples(provider, [("svc-a", "cpu"), ("svc-b", "cpu")])
    assert values == {("svc-a", "cpu"): None, ("svc-b", "cpu"): None}
    assert provider.calls == 0


def test_injected_provider_requires_the_typed_definition_and_sample_contract():
    class _Custom:
        name = "custom"

        def signal_definition(self, signal: str) -> SignalDefinition | None:
            return SignalDefinition(
                name=signal,
                aggregation="per_replica",
                query_template='sum(custom_metric{service="{service}"})',
                service_label="service",
                unit="units",
                scope="service",
            )

        def signal_value(self, service: str, signal: str) -> ScalingSignalSample:
            return ScalingSignalSample(
                value=42.0,
                source=self.name,
                service=service,
                signal=signal,
                aggregation="per_replica",
                observed_at=_now(),
                unit="units",
                scope="service",
            )

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
    assert provider.timeout <= 10.0
