"""Python-tier Prometheus metrics for the API gateway.

CONCEPT:OS-5.23 — Gateway Middle-Tier Hardening (metrics, per-tenant rate
limiting, engine circuit breaker, multi-worker readiness).

The Rust engine already exposes ``epistemic_graph_*`` Prometheus metrics; this
module mirrors that naming style for the Python tier with an
``agent_utilities_gateway_*`` prefix so dashboards read coherently:

* ``agent_utilities_gateway_requests_total{route,method,status}``
* ``agent_utilities_gateway_request_duration_seconds{route}`` (histogram)
* ``agent_utilities_gateway_in_flight_requests`` (gauge)
* ``agent_utilities_gateway_rate_limited_total{tenant}``
* ``agent_utilities_gateway_engine_requests_total{op,outcome}``
* ``agent_utilities_gateway_engine_breaker_state{endpoint}`` (0=closed,
  1=half-open, 2=open)

``prometheus_client`` is an OPTIONAL dependency (the ``metrics`` extra). When
absent every metric degrades to a shared no-op so the middleware costs ~nothing
and ``GET /metrics`` returns a self-describing placeholder.

Cardinality discipline: the ``route`` label is always a route TEMPLATE
(``/api/graph/{name}``), never a raw path — unmatched requests collapse into
the single ``unmatched`` bucket so internet scanners cannot mint series.

Multi-worker note: metrics live in the default per-process registry. With
``GATEWAY_WORKERS>1`` (shared listen socket) a scrape samples ONE worker;
aggregate across replicas at the Prometheus level or run one worker per
container. See ``docs/architecture/gateway_scaling.md``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GATEWAY_IN_FLIGHT",
    "GATEWAY_RATE_LIMITED",
    "GATEWAY_REQUEST_DURATION",
    "GATEWAY_REQUESTS",
    "DB_CALLS",
    "DISPATCH_QUEUE_DEPTH",
    "DISPATCH_TURNS",
    "DISPATCH_WORKERS",
    "ENGINE_BREAKER_STATE",
    "ENGINE_REQUESTS",
    "ENGINE_SHARD_REQUESTS",
    "ENGINE_SHARD_UP",
    "KG_INGEST_CONSUMER_LAG",
    "KG_INGEST_QUEUE_DEPTH",
    "MCP_CHILD_BREAKER_STATE",
    "MCP_CHILD_CALLS",
    "MCP_CHILD_QUEUE_DEPTH",
    "MCP_CHILD_RESTARTS",
    "PROMETHEUS_AVAILABLE",
    "SKILL_CALLS",
    "TOOL_CALLS",
    "GatewayMetricsMiddleware",
    "metrics_asgi_endpoint",
    "metrics_endpoint",
    "render_metrics",
]


class _NoopMetric:
    """Shared no-op stand-in for Counter/Gauge/Histogram (metrics extra absent)."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:  # noqa: ARG002
        return self

    def inc(self, amount: float = 1.0) -> None:  # noqa: ARG002
        return None

    def dec(self, amount: float = 1.0) -> None:  # noqa: ARG002
        return None

    def observe(self, value: float) -> None:  # noqa: ARG002
        return None

    def set(self, value: float) -> None:  # noqa: ARG002
        return None


try:  # pragma: no cover - exercised only when the extra is installed
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # graceful no-op fallback (optional `metrics` extra)
    PROMETHEUS_AVAILABLE = False


def _counter(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    try:
        return Counter(name, doc, labelnames=labelnames)
    except ValueError:  # duplicate registration (module re-import in tests)
        return _NoopMetric()


def _gauge(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    try:
        return Gauge(name, doc, labelnames=labelnames)
    except ValueError:
        return _NoopMetric()


def _histogram(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    try:
        return Histogram(
            name,
            doc,
            labelnames=labelnames,
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
            ),
        )
    except ValueError:
        return _NoopMetric()


GATEWAY_REQUESTS = _counter(
    "agent_utilities_gateway_requests_total",
    "Gateway HTTP requests by route template, method and status code.",
    ("route", "method", "status"),
)
GATEWAY_REQUEST_DURATION = _histogram(
    "agent_utilities_gateway_request_duration_seconds",
    "Gateway HTTP request duration by route template.",
    ("route",),
)
GATEWAY_IN_FLIGHT = _gauge(
    "agent_utilities_gateway_in_flight_requests",
    "Gateway HTTP requests currently being handled by this process.",
)
GATEWAY_RATE_LIMITED = _counter(
    "agent_utilities_gateway_rate_limited_total",
    "Requests rejected (429) by the per-tenant token-bucket rate limiter.",
    ("tenant",),
)
ENGINE_REQUESTS = _counter(
    "agent_utilities_gateway_engine_requests_total",
    "epistemic-graph engine client calls by operation and outcome "
    "(ok | connection_error | error | short_circuited).",
    ("op", "outcome"),
)
ENGINE_BREAKER_STATE = _gauge(
    "agent_utilities_gateway_engine_breaker_state",
    "Engine circuit-breaker state per endpoint (0=closed, 1=half-open, 2=open).",
    ("endpoint",),
)
# Tenant-partitioned engine sharding visibility (CONCEPT:KG-2.58 / OS-5.28):
# one series per configured GRAPH_SERVICE_ENDPOINTS entry. The gauge is
# refreshed on every real client connect attempt and by the daemon's
# shard_topology_status probe; the counter splits the existing engine-call
# outcomes per shard so a hot or failing shard is visible at a glance.
# Cardinality is bounded by the configured endpoint list (a handful).
ENGINE_SHARD_UP = _gauge(
    "agent_utilities_engine_shard_up",
    "Per-shard engine reachability by endpoint (1=reachable, 0=unreachable).",
    ("endpoint",),
)
ENGINE_SHARD_REQUESTS = _counter(
    "agent_utilities_engine_shard_requests_total",
    "Engine client calls per shard endpoint and outcome "
    "(ok | connection_error | error | short_circuited).",
    ("endpoint", "outcome"),
)
# Ingest scale-out backpressure visibility (CONCEPT:KG-2.57): sampled by the
# KG maintenance scheduler on the leader host. Depth is uniform across queue
# backends (sqlite/postgres row count, kafka = kg-ingest consumer-group lag);
# the lag series exists separately so Kafka dashboards/alerts read naturally.
KG_INGEST_QUEUE_DEPTH = _gauge(
    "agent_utilities_kg_ingest_queue_depth",
    "Pending KG ingest tasks in the selected durable task queue.",
    ("backend",),
)
KG_INGEST_CONSUMER_LAG = _gauge(
    "agent_utilities_kg_ingest_consumer_lag",
    "Total kg-ingest consumer-group lag (unconsumed messages) per topic.",
    ("topic", "group"),
)
# MCP multiplexer per-child resilience (CONCEPT:ECO-4.34): one series per
# aggregated child server (~50, bounded by mcp_config.json). The multiplexer
# runs standalone; like every metric here these degrade to no-ops when
# prometheus_client (the optional ``metrics`` extra) is absent.
MCP_CHILD_CALLS = _counter(
    "agent_utilities_mcp_child_calls_total",
    "Multiplexer tool calls per child server and outcome "
    "(ok | error | transport_error | timeout | busy | unavailable | "
    "short_circuited).",
    ("server", "outcome"),
)
MCP_CHILD_BREAKER_STATE = _gauge(
    "agent_utilities_mcp_child_breaker_state",
    "Per-child circuit-breaker state (0=closed, 1=half-open, 2=open).",
    ("server",),
)
MCP_CHILD_RESTARTS = _counter(
    "agent_utilities_mcp_child_restarts_total",
    "Automatic restarts of crashed multiplexer child servers.",
    ("server",),
)
MCP_CHILD_QUEUE_DEPTH = _gauge(
    "agent_utilities_mcp_child_queue_depth",
    "Tool calls queued behind a child's concurrency limit right now.",
    ("server",),
)

# Runtime usage granularity (CONCEPT:OS-5.31): per-tool / per-skill / per-db
# call counters, siblings of MCP_CHILD_CALLS. Emitted alongside the usage-store
# rows so the same data is visible in both Prometheus and /api/observability.
TOOL_CALLS = _counter(
    "agent_utilities_tool_calls_total",
    "Tool invocations by coarse category (read|edit|bash|search|mcp|tool|db).",
    ("category",),
)
SKILL_CALLS = _counter(
    "agent_utilities_skill_calls_total",
    "Skill invocations by skill name.",
    ("skill",),
)
DB_CALLS = _counter(
    "agent_utilities_db_calls_total",
    "Database/graph-engine calls by store (usage|kg|state).",
    ("store",),
)

# Queue-driven agent dispatch visibility (CONCEPT:ORCH-1.45): sampled by the
# dispatch workers on their heartbeat tick. Depth is uniform across queue
# transports (kafka = agent-dispatch consumer-group lag, postgres/sqlite =
# row count); the workers gauge counts fresh heartbeats in the fleet
# registry; the turns counter splits processed turns by outcome
# (completed | failed | skipped | expired).
DISPATCH_QUEUE_DEPTH = _gauge(
    "agent_utilities_dispatch_queue_depth",
    "Unclaimed dispatched agent turns in the agent_turns queue.",
    ("backend",),
)
DISPATCH_TURNS = _counter(
    "agent_utilities_dispatch_turns_total",
    "Dispatched agent turns processed by this worker process, by outcome.",
    ("outcome",),
)
DISPATCH_WORKERS = _gauge(
    "agent_utilities_dispatch_workers",
    "Live agent-dispatch workers (fresh heartbeats in the fleet registry).",
)


def render_metrics() -> tuple[bytes, str]:
    """Render the process metric registry in Prometheus exposition format.

    Returns ``(body, content_type)``. Without ``prometheus_client`` installed
    a self-describing placeholder is returned (still a 200 — scrapers treat
    the target as up, just empty).
    """
    if not PROMETHEUS_AVAILABLE:
        return (
            b"# agent-utilities gateway metrics unavailable: install the "
            b"'metrics' extra (prometheus-client). (CONCEPT:OS-5.23)\n",
            "text/plain; version=0.0.4; charset=utf-8",
        )
    from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


async def metrics_endpoint() -> Any:
    """``GET /metrics`` handler for FastAPI ``add_api_route`` (no params)."""
    from starlette.responses import Response

    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


async def metrics_asgi_endpoint(request: Any) -> Any:  # noqa: ARG001
    """``GET /metrics`` handler for plain Starlette ``add_route``."""
    return await metrics_endpoint()


def _route_template(scope: dict[str, Any], status: int) -> str:
    """Resolve a BOUNDED-cardinality route label for ``scope``.

    FastAPI's ``APIRoute.matches`` records the matched route object in
    ``scope["route"]`` — its ``path_format`` is the template (``/api/x/{id}``).
    Requests that matched no APIRoute (mounted sub-apps, 404s) collapse into a
    first-segment bucket, and outright misses (404/405) into ``unmatched`` so
    scanner traffic cannot mint unbounded series.
    """
    route = scope.get("route")
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    if template:
        return str(template)
    if status in (404, 405):
        return "unmatched"
    parts = [p for p in scope.get("path", "/").split("/") if p]
    return f"/{parts[0]}/*" if parts else "/"


class GatewayMetricsMiddleware:
    """Pure-ASGI middleware recording request count/duration/in-flight metrics.

    CONCEPT:OS-5.23. Mounted OUTERMOST by
    :func:`agent_utilities.gateway.graph_api.register_graph_routes` so auth
    rejections (401) and rate-limit rejections (429) are counted too. No
    heavy deps: with ``prometheus_client`` absent every record is a no-op.
    ``/metrics`` itself is not instrumented (self-scrape noise).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") == "/metrics":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET"))
        status_box = {"status": 500}  # app crash before response.start → 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_box["status"] = int(message.get("status", 500))
            await send(message)

        GATEWAY_IN_FLIGHT.inc()
        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            GATEWAY_IN_FLIGHT.dec()
            status = status_box["status"]
            route = _route_template(scope, status)
            GATEWAY_REQUESTS.labels(
                route=route, method=method, status=str(status)
            ).inc()
            GATEWAY_REQUEST_DURATION.labels(route=route).observe(duration)
