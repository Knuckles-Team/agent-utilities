#!/usr/bin/python
from __future__ import annotations

"""Scaling signal providers — the injectable load sense of the autoscaler.

CONCEPT:OS-5.29 — Reactive replica autoscaling (signal seam).

The fleet autoscaler never talks to a metrics system directly; it reads a
:class:`ScalingSignalProvider` — ``signal_value(service, signal) -> float |
None``. ``None`` means "no data": the autoscaler then takes NO scaling action
for that service (the same never-act-on-zero-evidence rule the reconciler
applies to unobserved services).

Built-ins:

* :class:`LocalMetricsProvider` — the DEFAULT, zero-infra provider: reads the
  in-process Prometheus gauges this package already maintains
  (CONCEPT:OS-5.23 gateway metrics + CONCEPT:KG-2.55/KG-2.57 ingest
  backpressure): ``queue_depth`` → ``agent_utilities_kg_ingest_queue_depth``,
  ``consumer_lag`` → ``agent_utilities_kg_ingest_consumer_lag``. Any other
  signal name is looked up verbatim as a metric family in the local registry.
  Without the optional ``metrics`` extra (``prometheus-client``) every lookup
  is ``None`` — and therefore no scaling ever happens.
* :class:`PrometheusHttpProvider` — instant queries against a configured
  Prometheus base URL (``SCALING_PROMETHEUS_URL``); a small ``httpx`` GET to
  ``/api/v1/query``, no client-library dependency. agent-utilities does not
  hard-depend on a Prometheus deployment: the provider only activates when
  the URL flag is set.
* :func:`set_scaling_signal_provider` — the deployment injection seam
  (mirrors ``set_fleet_observer`` / ``set_fleet_actuator``): a deployment
  with a richer source (lgtm-mcp, Thanos, a pushgateway, …) registers its
  own implementation.

Signal-value convention (matters for the target-tracking math): the
well-known signals ``queue_depth`` and ``consumer_lag`` are FLEET-TOTAL
values (the autoscaler divides by current replicas); every other signal —
``cpu`` and custom metrics — is treated as a PER-REPLICA average, so custom
PromQL should aggregate accordingly (e.g. ``avg(...)``, not ``sum(...)``).
"""

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Well-known signal names → the shipped metric families they read.
SIGNAL_QUEUE_DEPTH = "queue_depth"
SIGNAL_CONSUMER_LAG = "consumer_lag"
SIGNAL_CPU = "cpu"

_LOCAL_FAMILIES = {
    SIGNAL_QUEUE_DEPTH: "agent_utilities_kg_ingest_queue_depth",
    SIGNAL_CONSUMER_LAG: "agent_utilities_kg_ingest_consumer_lag",
}

# Default PromQL per well-known signal. ``{service}`` is substituted with the
# (validated) service name. cpu reads the swarm service-name label cAdvisor
# stamps on containers; deployments with different labeling use a custom
# signal (verbatim PromQL) instead.
_PROMQL_TEMPLATES = {
    SIGNAL_QUEUE_DEPTH: "sum(agent_utilities_kg_ingest_queue_depth)",
    SIGNAL_CONSUMER_LAG: "sum(agent_utilities_kg_ingest_consumer_lag)",
    SIGNAL_CPU: (
        "100 * avg(rate(container_cpu_usage_seconds_total"
        '{{container_label_com_docker_swarm_service_name="{service}"}}[5m]))'
    ),
}


@runtime_checkable
class ScalingSignalProvider(Protocol):
    """Anything that can report the current load signal for a service."""

    name: str

    def signal_value(self, service: str, signal: str) -> float | None:
        """Current value of ``signal`` for ``service``; ``None`` = no data.

        Never raises — a provider failure is "no data" (and no scaling).
        """
        ...  # ABSTRACT-OK


class LocalMetricsProvider:
    """Zero-infra provider over this process's own Prometheus registry.

    Sums every sample of the resolved metric family (gauges here are
    label-partitioned by backend/topic, and the autoscaler wants the total
    backlog). Returns ``None`` when ``prometheus_client`` is absent or the
    family has no samples — unknown signals therefore never scale anything.
    """

    name = "local"

    def signal_value(self, service: str, signal: str) -> float | None:  # noqa: ARG002
        family = _LOCAL_FAMILIES.get(signal, signal)
        try:
            from prometheus_client import REGISTRY
        except ImportError:
            return None
        try:
            total = 0.0
            seen = False
            for metric in REGISTRY.collect():
                if metric.name != family:
                    continue
                for sample in metric.samples:
                    if sample.name == family:
                        total += float(sample.value)
                        seen = True
            return total if seen else None
        except Exception as e:  # noqa: BLE001 — no data beats a crashed tick
            logger.debug("LocalMetricsProvider: %s read failed: %s", family, e)
            return None


class PrometheusHttpProvider:
    """Instant-query provider against an external Prometheus base URL.

    Well-known signals use the shipped PromQL templates; any other signal is
    sent verbatim as PromQL with an optional ``{service}`` placeholder. The
    values of a vector result are summed. Any transport/parse problem (and an
    empty result) is ``None`` — no data, no scaling.
    """

    name = "prometheus"

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        transport: Any = None,  # injectable for tests (httpx.MockTransport)
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def _query_for(self, service: str, signal: str) -> str:
        template = _PROMQL_TEMPLATES.get(signal)
        if template is not None:
            return template.format(service=service)
        # Custom signal = verbatim PromQL; plain substring substitution so
        # label-selector braces in the query never break the placeholder.
        return signal.replace("{service}", service)

    def signal_value(self, service: str, signal: str) -> float | None:
        from agent_utilities.core.http_client import create_http_client

        query = self._query_for(service, signal)
        try:
            with create_http_client(
                timeout=self.timeout, transport=self.transport
            ) as client:
                resp = client.get(
                    f"{self.base_url}/api/v1/query", params={"query": query}
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:  # noqa: BLE001 — provider failures are "no data"
            logger.debug("PrometheusHttpProvider: query %r failed: %s", query, e)
            return None
        if not isinstance(payload, dict) or payload.get("status") != "success":
            return None
        result = (payload.get("data") or {}).get("result") or []
        total = 0.0
        seen = False
        for entry in result:
            try:
                total += float(entry["value"][1])
                seen = True
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        return total if seen else None


# ── registry (deployment injection point) ───────────────────────────

_PROVIDER: ScalingSignalProvider | None = None


def set_scaling_signal_provider(provider: ScalingSignalProvider | None) -> None:
    """Register the process-wide provider (``None`` resets to config default)."""
    global _PROVIDER
    _PROVIDER = provider


def get_scaling_signal_provider() -> ScalingSignalProvider:
    """Resolve the active provider: injected > Prometheus URL flag > local."""
    if _PROVIDER is not None:
        return _PROVIDER
    url: str | None = None
    try:
        from agent_utilities.core.config import config as _cfg

        url = getattr(_cfg, "scaling_prometheus_url", None) or None
    except Exception:  # noqa: BLE001
        url = None
    if url:
        return PrometheusHttpProvider(url)
    return LocalMetricsProvider()
