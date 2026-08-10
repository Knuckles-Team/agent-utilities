"""Python-tier Prometheus metrics for the API gateway.

CONCEPT:AU-OS.observability.no-op-without-metrics — Gateway Middle-Tier Hardening (metrics, per-tenant rate
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

import asyncio
import contextlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GATEWAY_IN_FLIGHT",
    "GATEWAY_RATE_LIMITED",
    "GATEWAY_REQUEST_DURATION",
    "GATEWAY_REQUESTS",
    "BUS_DISPATCH",
    "BUS_MESSAGES",
    "BUS_PARTICIPANTS",
    "BUS_SEND_DURATION",
    "CONTEXT_COMPILER_ITEMS",
    "CONTEXT_COMPILER_KV_CACHE",
    "CONTEXT_COMPILER_TOKENS",
    "CONTEXT_COMPILER_TTFT",
    "DB_CALLS",
    "DELEGATION_DURATION",
    "DELEGATION_IN_FLIGHT",
    "DELEGATION_RUNS",
    "DISPATCH_QUEUE_DEPTH",
    "DISPATCH_TURNS",
    "DISPATCH_WORKERS",
    "ENGINE_BREAKER_STATE",
    "ENGINE_REQUESTS",
    "ENGINE_REQUEST_LATENCY",
    "ENGINE_SHARD_REQUESTS",
    "ENGINE_SHARD_UP",
    "KG_INGEST_CONSUMER_LAG",
    "KG_INGEST_QUEUE_DEPTH",
    "KVCACHE_CHECKPOINT_OPS",
    "KVCACHE_CHECKPOINT_RECOMMENDATIONS",
    "KVCACHE_CHECKPOINT_TIER_OPS",
    "KVCACHE_CLIENT_REQUESTS",
    "KVCACHE_FORK_EVENTS",
    "KVCACHE_FORK_SHARED_BYTES",
    "PROMPT_CACHE_OPS",
    "PROMPT_CACHE_TOKENS_SAVED",
    "SEMANTIC_CACHE_OPS",
    "SEMANTIC_CACHE_BYTES_SAVED",
    "SEMANTIC_CACHE_STALENESS_SECONDS",
    "LANE_IN_FLIGHT",
    "LANE_QUEUE_DEPTH",
    "MCP_CHILD_BREAKER_STATE",
    "MCP_CHILD_CALLS",
    "MCP_CHILD_PROCESS_RUNNING",
    "MCP_CHILD_QUEUE_DEPTH",
    "MCP_CHILD_RESTARTS",
    "MCP_CHILD_TOOLS_DISPATCHABLE",
    "MCP_CHILD_TOOLS_MOUNTED",
    "MCP_SERVERS_DISPATCHABLE",
    "MCP_SERVERS_RUNNING",
    "MCP_TOOL_CALLS",
    "MCP_TOOL_DURATION",
    "MCP_TOOL_IN_FLIGHT",
    "PROMETHEUS_AVAILABLE",
    "REPOSITORY_BISECTION_RUNS",
    "REPOSITORY_BUILD_CACHE_OPS",
    "REPOSITORY_CAPACITY_RESERVATIONS",
    "REPOSITORY_JOB_QUEUE_AGE",
    "REPOSITORY_JOB_QUEUE_DEPTH",
    "REPOSITORY_JOB_RETRIES",
    "REPOSITORY_LANDING_DRIFT",
    "REPOSITORY_LANE_AGE",
    "REPOSITORY_LANE_COUNT",
    "REPOSITORY_LANE_DISK_BYTES",
    "REPOSITORY_RELEASE_DAG_STAGE_DURATION",
    "REPOSITORY_VALIDATION_DURATION",
    "REPOSITORY_WORKER_HEALTH",
    "SCHEDULED_JOB_DURATION",
    "SCHEDULED_JOB_RUNS",
    "SKILL_CALLS",
    "TOOL_CALLS",
    "GatewayMetricsMiddleware",
    "delegation_span",
    "metrics_asgi_endpoint",
    "metrics_endpoint",
    "publish_multiplexer_child_gauges",
    "register_scrape_sampler",
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


_DEFAULT_LATENCY_BUCKETS = (
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
)


def _histogram(
    name: str,
    doc: str,
    labelnames: tuple[str, ...] = (),
    *,
    buckets: tuple[float, ...] = _DEFAULT_LATENCY_BUCKETS,
) -> Any:
    if not PROMETHEUS_AVAILABLE:
        return _NoopMetric()
    try:
        return Histogram(name, doc, labelnames=labelnames, buckets=buckets)
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
ENGINE_REQUEST_LATENCY = _histogram(
    "agent_utilities_gateway_engine_request_latency_seconds",
    "epistemic-graph engine client call latency (seconds) by operation. The "
    "per-op round-trip time — so a contended engine (slow point-reads like "
    "GetNodeProperties) or one heavy op dominating the shared connection is "
    "visible at a glance without a profiler. Pairs with a slow-call WARN log in "
    "engine_breaker so the same signal reaches logs and metrics.",
    ("op",),
)
ENGINE_BREAKER_STATE = _gauge(
    "agent_utilities_gateway_engine_breaker_state",
    "Engine circuit-breaker state per endpoint (0=closed, 1=half-open, 2=open).",
    ("endpoint",),
)
# Tenant-partitioned engine sharding visibility (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw / OS-5.28):
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
# Ingest scale-out backpressure visibility (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer): sampled by the
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
# Unified scheduler + 8-lane task-queue observability (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent /
# CONCEPT:AU-ORCH.execution.two-level-fair-rotation): the one seam every recurring/background job
# (schedule_engine.run_scheduler_tick -> the task_lanes.py queue) runs through, so a daemon leak
# shows up as a counter/gauge query instead of a grep through N tick bodies. ``schedule`` is
# bounded (~33 named :Schedule nodes); ``lane`` is bounded (the 8 functional lanes in
# task_lanes.TASK_LANES) — both safe cardinalities.
SCHEDULED_JOB_RUNS = _counter(
    "agent_utilities_scheduled_job_runs_total",
    "Scheduled/recurring job runs by schedule name and outcome (ok|failed|skipped).",
    ("schedule", "outcome"),
)
SCHEDULED_JOB_DURATION = _histogram(
    "agent_utilities_scheduled_job_duration_seconds",
    "Scheduled/recurring job execution duration by schedule name.",
    ("schedule",),
)
LANE_QUEUE_DEPTH = _gauge(
    "agent_utilities_lane_queue_depth",
    "Pending KG task-queue depth by functional lane (task_lanes.TASK_LANES).",
    ("lane",),
)
LANE_IN_FLIGHT = _gauge(
    "agent_utilities_lane_in_flight",
    "In-flight (currently claimed/running) KG tasks by functional lane.",
    ("lane",),
)
# MCP multiplexer per-child resilience (CONCEPT:AU-ECO.mcp.profile-differences-from-client): one series per
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

# Running vs dispatchable — the ONE distinction these series exist to preserve
# (CONCEPT:AU-ECO.multiplexer.running-vs-dispatchable-metrics). A child process
# being up is NOT the same fact as its tools being callable: a mounted child
# whose circuit breaker is open, or one whose tools were never registered as
# live FastMCP forwarders, is `running` yet dispatches nothing. Conflating the
# two is precisely the lie the multiplexer's health surface removed, so the
# metric names keep them apart and a dashboard must never sum them together:
#
#   * ``*_process_running`` / ``*_servers_running``  → PROCESS-level fact only.
#   * ``*_tools_dispatchable`` / ``*_servers_dispatchable`` → can a tool call
#     actually be served right now (child up AND breaker not open).
#   * ``*_tools_mounted`` → registered as live forwarding tools (dynamic mode
#     mounts on demand, so mounted <= aggregated and mounted != dispatchable).
#
# Sampled by ``MCPMultiplexer.status_snapshot()`` — the same call that renders
# the ``multiplexer_status`` tool — so the gauge and the tool can never drift.
# ``server`` is bounded by mcp_config.json (~50 children).
MCP_CHILD_PROCESS_RUNNING = _gauge(
    "agent_utilities_mcp_child_process_running",
    "Per-child PROCESS liveness only (1 = the child runtime reports state 'up', "
    "0 = starting/restarting/failed). This is NOT dispatchability — see "
    "agent_utilities_mcp_child_tools_dispatchable.",
    ("server",),
)
MCP_CHILD_TOOLS_MOUNTED = _gauge(
    "agent_utilities_mcp_child_tools_mounted",
    "Per-child tools currently registered as live forwarding tools on this "
    "multiplexer (dynamic mode mounts on demand, so this is <= the child's "
    "aggregated catalog size).",
    ("server",),
)
MCP_CHILD_TOOLS_DISPATCHABLE = _gauge(
    "agent_utilities_mcp_child_tools_dispatchable",
    "Per-child mounted tools that a call could actually be dispatched to right "
    "now (child process up AND its circuit breaker not open). Always <= "
    "agent_utilities_mcp_child_tools_mounted; the gap is the degraded set.",
    ("server",),
)
MCP_SERVERS_RUNNING = _gauge(
    "agent_utilities_mcp_servers_running",
    "Child servers whose PROCESS is up. Fleet-level twin of "
    "agent_utilities_mcp_child_process_running — a process-liveness count, "
    "never a dispatchability claim.",
)
MCP_SERVERS_DISPATCHABLE = _gauge(
    "agent_utilities_mcp_servers_dispatchable",
    "Child servers that can serve a tool call right now (process up, breaker "
    "not open, at least one tool mounted). The truthful fleet-capability "
    "number; always <= agent_utilities_mcp_servers_running.",
)

# Runtime usage granularity (CONCEPT:AU-OS.observability.persist-this-graph-run): per-tool / per-skill / per-db
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

# Repository-development provenance/metrics bridge (CONCEPT:AU-KG.audit.repository-job-provenance-bridge,
# RMDD-19): bounded-cardinality instruments for repository job queue/capacity/
# build-cache/validation/bisection/lane/worker/retry/landing/release-DAG
# state, read by ``agent_utilities.observability.repository_metrics`` and
# emitted by repository-manager's domain provenance adapter
# (``repository_manager.provenance``). Every label is a bounded enum — a
# repository alias, a resource/priority/retry/drift class, a host inventory
# alias, a validation stage (C-06), or a build-cache outcome (C-05) — never a
# job id, filesystem path, or credential, matching the cardinality discipline
# every other block in this module already follows.
REPOSITORY_JOB_QUEUE_DEPTH = _gauge(
    "agent_utilities_repository_job_queue_depth",
    "Pending repository jobs by repository alias and priority class.",
    ("repo", "priority_class"),
)
REPOSITORY_JOB_QUEUE_AGE = _histogram(
    "agent_utilities_repository_job_queue_age_seconds",
    "Age of a repository job at the moment it leaves the queue, by repository alias.",
    ("repo",),
)
REPOSITORY_CAPACITY_RESERVATIONS = _gauge(
    "agent_utilities_repository_capacity_reservations",
    "Active resource reservations by resource class and host alias.",
    ("resource_class", "host_alias"),
)
REPOSITORY_BUILD_CACHE_OPS = _counter(
    "agent_utilities_repository_build_cache_ops_total",
    "Build-cache operations by repository alias and outcome (C-05: "
    "hit|waited_hit|produced_miss|degraded_uncacheable|corrupted_entry|"
    "refused|failed).",
    ("repo", "outcome"),
)
REPOSITORY_VALIDATION_DURATION = _histogram(
    "agent_utilities_repository_validation_duration_seconds",
    "Validation gate duration by repository alias and stage (C-06: "
    "feedback|integration|certification|smoke|release).",
    ("repo", "stage"),
)
REPOSITORY_BISECTION_RUNS = _counter(
    "agent_utilities_repository_bisection_runs_total",
    "Generation failure-bisection runs by repository alias and outcome.",
    ("repo", "outcome"),
)
REPOSITORY_LANE_COUNT = _gauge(
    "agent_utilities_repository_lane_count",
    "Active development lanes by repository alias.",
    ("repo",),
)
REPOSITORY_LANE_DISK_BYTES = _gauge(
    "agent_utilities_repository_lane_disk_bytes",
    "Observed lane worktree disk usage (bytes) by repository alias.",
    ("repo",),
)
REPOSITORY_LANE_AGE = _histogram(
    "agent_utilities_repository_lane_age_seconds",
    "Lane age at close/reclaim time by repository alias.",
    ("repo",),
)
REPOSITORY_WORKER_HEALTH = _gauge(
    "agent_utilities_repository_worker_health",
    "Repository worker health by host inventory alias (1=healthy, 0=unhealthy).",
    ("host_alias",),
)
REPOSITORY_JOB_RETRIES = _counter(
    "agent_utilities_repository_job_retries_total",
    "Repository job retry/dead-letter transitions by repository alias and retry class.",
    ("repo", "retry_class"),
)
REPOSITORY_LANDING_DRIFT = _counter(
    "agent_utilities_repository_landing_drift_total",
    "Landing base-moved / re-target events by repository alias and drift kind.",
    ("repo", "drift_kind"),
)
REPOSITORY_RELEASE_DAG_STAGE_DURATION = _histogram(
    "agent_utilities_repository_release_dag_stage_duration_seconds",
    "Workspace release-DAG stage duration by repository alias and stage.",
    ("repo", "stage"),
)

# Per-tool instrumentation for STANDALONE MCP servers built by
# ``create_mcp_server`` (CONCEPT:AU-OS.observability.no-op-without-metrics). One series per tool this server
# process exposes (bounded by the server's own tool set); Prometheus adds the
# job/instance labels so the same tool name on different servers stays distinct.
# These are the server-side siblings of MCP_CHILD_* (the multiplexer's view).
# Recorded by ``ToolMetricsMiddleware`` and scraped from the server's own
# ``GET /metrics`` route; no-ops when the optional ``metrics`` extra is absent.
MCP_TOOL_CALLS = _counter(
    "agent_utilities_mcp_tool_calls_total",
    "MCP tool invocations on this server by tool name and outcome (ok|error).",
    ("tool", "outcome"),
)
MCP_TOOL_DURATION = _histogram(
    "agent_utilities_mcp_tool_duration_seconds",
    "MCP tool execution duration on this server by tool name.",
    ("tool",),
)
MCP_TOOL_IN_FLIGHT = _gauge(
    "agent_utilities_mcp_tool_in_flight",
    "MCP tool calls currently executing on this server.",
)

# Queue-driven agent dispatch visibility (CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch): sampled by the
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

# In-process delegation visibility (CONCEPT:AU-OS.observability.delegation-run-metrics).
# The DISPATCH_* family above measures the QUEUE-consumed path
# (agent_dispatch_worker.execute_agent_turn); these measure the synchronous
# ``Orchestrator.execute_agent``/``execute_workflow`` calls that
# ``graph_orchestrate`` and every in-process delegate actually take — the path
# that had no telemetry at all, which is why delegation reliability could only
# be debugged by reading run traces after the fact. ``target`` is the agent or
# workflow name (bounded by the registered fleet/workflow set); ``kind``
# separates the two entry points so an agent named like a workflow cannot
# merge series.
DELEGATION_RUNS = _counter(
    "agent_utilities_delegation_runs_total",
    "In-process delegated runs by kind (agent|workflow), target name and "
    "outcome (ok|error|cancelled).",
    ("kind", "target", "outcome"),
)
DELEGATION_DURATION = _histogram(
    "agent_utilities_delegation_duration_seconds",
    "Wall-clock duration of one in-process delegated run "
    "(Orchestrator.execute_agent / execute_workflow), by kind and target.",
    ("kind", "target"),
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
)
DELEGATION_IN_FLIGHT = _gauge(
    "agent_utilities_delegation_in_flight",
    "Delegated runs currently executing in this process, by kind.",
    ("kind",),
)

# Agent-to-agent communication bus visibility (CONCEPT:AU-ECO.bus.operator-view-agentbus, the AgentBus of
# ECO-4.84). Participants are sampled by ``AgentBus.status``/``roster``; message and
# dispatch counters are bumped on the send/dispatch path; send duration measures the
# server-side bus write latency (the cross-host wire is durable-store polling on top).
# Labels are bounded (status/kind/outcome are small enums) so cardinality stays flat.
BUS_PARTICIPANTS = _gauge(
    "agent_utilities_bus_participants",
    "Registered bus participants by computed presence (online|offline).",
    ("status",),
)
BUS_MESSAGES = _counter(
    "agent_utilities_bus_messages_total",
    "Bus messages by kind (direct|topic) and outcome (delivered|denied|no_recipient).",
    ("kind", "outcome"),
)
BUS_SEND_DURATION = _histogram(
    "agent_utilities_bus_send_seconds",
    "Server-side latency of one AgentBus.send (gate + per-recipient durable write).",
)
BUS_DISPATCH = _counter(
    "agent_utilities_bus_dispatch_total",
    "Bus message->fleet-work dispatches by outcome (submitted|denied|failed).",
    ("outcome",),
)

# ContextCompiler answer-path efficiency (CONCEPT:AU-KG.retrieval.context-compiler /
# CONCEPT:AU-KG.retrieval.context-compiler-kv-seam — WS-4). The mature ContextCompiler
# + Seam-6 KV-cache wiring had no measurement of its claimed efficiency; these mirror
# the existing gateway/MCP metric families (bounded, small label sets) rather than
# standing up a new metrics system. Recorded by ``ContextCompiler.compile`` and
# ``context_compiler_serving.bundle_chat_completion`` — no-op wherever the ``metrics``
# extra (prometheus-client) is absent, same as every metric above.
CONTEXT_COMPILER_ITEMS = _counter(
    "agent_utilities_context_compiler_items_total",
    "ContextCompiler candidate items per compile() call by outcome "
    "(selected|dropped_policy|dropped_redundant|dropped_budget).",
    ("outcome",),
)
CONTEXT_COMPILER_TOKENS = _histogram(
    "agent_utilities_context_compiler_tokens",
    "ContextCompiler per-compile() token counts by kind: 'in' is the MMR-selected "
    "pool handed to the token-budget fit, 'selected' is tokens actually kept "
    "(ContextBundle.tokens_used) — the selection-efficiency ratio is selected/in.",
    ("kind",),
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
)
CONTEXT_COMPILER_KV_CACHE = _counter(
    "agent_utilities_context_compiler_kv_cache_requests_total",
    "ContextCompiler Seam-6 bundle-level KV-cache lookups (compile(kv_backend=...)) "
    "by outcome (hit|miss) — hit-rate = hit / (hit + miss).",
    ("outcome",),
)
CONTEXT_COMPILER_TTFT = _histogram(
    "agent_utilities_context_compiler_ttft_seconds",
    "Wall-clock latency of a governed model call using a compiled ContextCompiler "
    "bundle — a time-to-first-token PROXY (see scripts/measure_bundle_kv_reuse.py "
    "for the token-level vLLM prefix-cache signal this approximates), split by "
    "whether the bundle itself was served from the Seam-6 KV cache (kv_cache_hit) "
    "and by call-site population (path='bundle_chat_completion': background "
    "enrichment/extraction calls via context_compiler_serving.bundle_chat_completion; "
    "path='delegated_run': interactive execute_agent/execute_workflow model calls "
    "via the mandatory contextual_model wrapper, CONCEPT:AU-KG.retrieval.context-compiler, W3.7) "
    "so the two latency populations are never averaged together.",
    ("kv_cache_hit", "path"),
)
KVCACHE_CLIENT_REQUESTS = _counter(
    "agent_utilities_kvcache_client_requests_total",
    "EpistemicGraphKVBackend.get() lookups by outcome (hit|miss|error) — the "
    "client-side half of the EG-187 remote KV-cache contract; GET /kv/stats on "
    "the engine exposes the aggregate server-side occupancy/dedup counters.",
    ("outcome",),
)
KVCACHE_FORK_EVENTS = _counter(
    "agent_utilities_kvcache_fork_events_total",
    "Zero-copy KV-fork rung outcomes (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork) by "
    "result and reason. result='forked' when the CoW snapshot/fork rung engaged for a "
    "branch fan-out; result='fallback' when it transparently degraded to the per-branch "
    "copy path (reason one of backend_unavailable|fork_unsupported|snapshot_failed|"
    "no_derivable_pages) — never a hard failure, the cohort still runs either way.",
    ("result", "reason"),
)
KVCACHE_FORK_SHARED_BYTES = _gauge(
    "agent_utilities_kvcache_fork_shared_bytes",
    "Bytes shared (Arc'd across branches, not copied) as of the most recent KV-fork "
    "rung engagement — the zero-copy proof (CONCEPT:EG-KG.memory.zero-copy-snapshot-fork) "
    "surfaced as a scrapeable gauge alongside GET /kv/fork/stats on the engine.",
)
KVCACHE_CHECKPOINT_OPS = _counter(
    "agent_utilities_kvcache_checkpoint_ops_total",
    "KVCheckpoint operations (CONCEPT:AU-KG.memory.kv-checkpoint-resource) by op "
    "(create|instantiate_agent|restore_conversation) and outcome (ok|not_found|"
    "cross_tenant_denied|stale|cold_start_fallback) — the fail-closed/traced-fallback "
    "proof for checkpoint load.",
    ("op", "outcome"),
)
KVCACHE_CHECKPOINT_RECOMMENDATIONS = _counter(
    "agent_utilities_kvcache_checkpoint_recommendations_total",
    "Checkpoint-worthiness verdicts (CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring) "
    "by recommended tier (none|ram|disk) — how often the scorer set thinks the current "
    "context is worth freezing, independent of whether anything acted on it.",
    ("tier",),
)
KVCACHE_CHECKPOINT_TIER_OPS = _counter(
    "agent_utilities_kvcache_checkpoint_tier_ops_total",
    "Tiered-checkpoint actions (CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring) by "
    "trigger (user|agent|system), tier (ram|disk) and outcome (taken|declined|"
    "eligibility_denied|error) — the proof of which of the three trigger paths actually "
    "took a checkpoint, and why a disk promotion was refused.",
    ("trigger", "tier", "outcome"),
)
PROMPT_CACHE_OPS = _counter(
    "agent_utilities_prompt_cache_ops_total",
    "Provider-native prompt-cache requests (CONCEPT:AU-ORCH.optimization.provider-prompt-cache — "
    "Anthropic cache_control / OpenAI prompt_cache_key) by op (fold|response) and outcome "
    "(anthropic|openai|other|hit|miss) — hit-rate = hit / (hit + miss) over 'response' events.",
    ("op", "outcome"),
)
PROMPT_CACHE_TOKENS_SAVED = _counter(
    "agent_utilities_prompt_cache_tokens_saved_total",
    "Cumulative input tokens served from a provider prompt cache (RequestUsage.cache_read_tokens) "
    "— the measurable savings signal for CONCEPT:AU-ORCH.optimization.provider-prompt-cache, labeled "
    "by provider.",
    ("provider",),
)
SEMANTIC_CACHE_OPS = _counter(
    "agent_utilities_semantic_cache_ops_total",
    "Semantic-cache operations (CONCEPT:AU-KG.memory.semantic-response-cache) by op "
    "(lookup|store|invalidate|evict) and outcome (hit|miss|disabled|refused_side_effect|"
    "refused_freshness|stale|below_threshold|embed_unavailable|error|stored|skipped) — "
    "default-OFF, opt-in per request/policy; hit-rate = hit / (hit + miss) over 'lookup' events.",
    ("op", "outcome"),
)
SEMANTIC_CACHE_BYTES_SAVED = _counter(
    "agent_utilities_semantic_cache_bytes_saved_total",
    "Cumulative response bytes served from the semantic cache instead of a live model call "
    "(CONCEPT:AU-KG.memory.semantic-response-cache) — the measurable savings signal.",
)
SEMANTIC_CACHE_STALENESS_SECONDS = _histogram(
    "agent_utilities_semantic_cache_staleness_seconds",
    "Age of the served entry at the moment of a semantic-cache HIT (CONCEPT:AU-KG.memory."
    "semantic-response-cache) — the staleness distribution required to keep a stale-serving "
    "policy honest; entries older than the caller's freshness_tolerance_seconds never hit.",
    buckets=(1, 5, 15, 30, 60, 300, 900, 1800, 3600, 21600, 86400),
)

# ── Universal-ingestion program: embedding/retrieval policy + evaluation
#    (reports/program/universal-ingestion.md, tracks 8/9) ───────────────────
EMBEDDING_VERSION_MISMATCHES = _counter(
    "agent_utilities_embedding_version_mismatches_total",
    "EmbeddingVersionMismatchError raisals (CONCEPT:AU-KG.retrieval.embedding-version-identity) "
    "by site (capability_index_add|capability_index_designate) — a vector from one embedding "
    "model was about to be compared against another's; each increment is a refused comparison, "
    "never a silently degraded ranking. Sustained non-zero indicates a mid-flight model swap "
    "that needs the generation-swap reindex described in embedding_versioning.py.",
    ("site",),
)
RETRIEVAL_ACL_ENFORCEMENT = _counter(
    "agent_utilities_retrieval_acl_enforcement_total",
    "Per-node ACL/owner-scope enforcement outcomes on vector/hybrid retrieval "
    "(CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval, search_hybrid) by outcome "
    "(admitted|denied|enforcement_failed) — 'denied' is a node the actor was not permitted to "
    "read being filtered out (fail-closed, not a silent empty result); 'enforcement_failed' is "
    "a PermissionError from the enforcement plumbing itself (fails closed, never falls back to "
    "unfiltered results).",
    ("outcome",),
)
RETRIEVAL_CITATION_RESOLUTION = _counter(
    "agent_utilities_retrieval_citation_resolution_total",
    "Evidence-citation resolution outcomes (CONCEPT:AU-KG.retrieval.mandatory-evidence-citation) "
    "by status (current|moved|stale|lost) — see evidence_spine.citation_status. A retrieval "
    "result whose citation cannot be resolved to real evidence ('lost') is a defect, not a "
    "degraded result; sustained 'stale'/'lost' indicates the source drifted since the citing "
    "chunk was indexed and the affected artifact needs reprocessing.",
    ("status",),
)


#: Callbacks run immediately before a ``/metrics`` scrape is rendered, for
#: gauges whose truth lives in a live object rather than in an event stream
#: (e.g. multiplexer child health). Each runs exception-isolated so one bad
#: sampler can never blank the whole scrape. Registration is idempotent by
#: identity, and the list is tiny and bounded by the number of subsystems that
#: own sampled state — this is not a plugin system.
_SCRAPE_SAMPLERS: list[Any] = []


def register_scrape_sampler(sampler: Any) -> None:
    """Register a zero-arg callable refreshed on every ``/metrics`` render."""
    if sampler not in _SCRAPE_SAMPLERS:
        _SCRAPE_SAMPLERS.append(sampler)


def _run_scrape_samplers() -> None:
    """Refresh sampled gauges, never letting one sampler fail the scrape."""
    for sampler in tuple(_SCRAPE_SAMPLERS):
        try:
            sampler()
        except Exception as exc:
            logger.warning(
                "Metrics scrape sampler %r failed (exception_type=%s): %s — its "
                "gauges are serving their last known values.",
                getattr(sampler, "__qualname__", sampler),
                type(exc).__name__,
                exc,
            )


@contextlib.contextmanager
def delegation_span(kind: str, target: str) -> Any:
    """Record one in-process delegated run (CONCEPT:AU-OS.observability.delegation-run-metrics).

    Wraps the body in the in-flight gauge, the duration histogram and the
    runs counter, classifying the outcome as ``ok``, ``cancelled``
    (:class:`asyncio.CancelledError`, which is NOT a delegation failure) or
    ``error``. The exception always propagates unchanged — this records, it
    never handles.
    """
    DELEGATION_IN_FLIGHT.labels(kind=kind).inc()
    start = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "error"
        raise
    finally:
        DELEGATION_IN_FLIGHT.labels(kind=kind).dec()
        DELEGATION_DURATION.labels(kind=kind, target=target).observe(
            time.perf_counter() - start
        )
        DELEGATION_RUNS.labels(kind=kind, target=target, outcome=outcome).inc()


def publish_multiplexer_child_gauges(snapshot: dict[str, Any]) -> None:
    """Publish the running-vs-dispatchable child gauges from a status snapshot.

    CONCEPT:AU-ECO.multiplexer.running-vs-dispatchable-metrics. Takes the exact
    dict ``MCPMultiplexer.status_snapshot()`` returns so the gauges and the
    ``multiplexer_status`` tool can never disagree — one producer, two
    renderings. A child is *running* when its runtime state is ``up``; it is
    *dispatchable* only when it is also not circuit-broken AND has at least one
    mounted tool, because a call to an open-breaker child is rejected before it
    ever reaches the child.
    """
    children = snapshot.get("children")
    if not isinstance(children, dict):
        return
    running_total = 0
    dispatchable_total = 0
    for name, child in children.items():
        if not isinstance(child, dict):
            continue
        server = str(name)
        running = 1 if str(child.get("state", "")) == "up" else 0
        mounted = int(child.get("mounted_tools", 0) or 0)
        breaker_open = str(child.get("breaker", "")) == "open"
        dispatchable = mounted if (running and not breaker_open) else 0
        MCP_CHILD_PROCESS_RUNNING.labels(server=server).set(running)
        MCP_CHILD_TOOLS_MOUNTED.labels(server=server).set(mounted)
        MCP_CHILD_TOOLS_DISPATCHABLE.labels(server=server).set(dispatchable)
        running_total += running
        dispatchable_total += 1 if dispatchable else 0
    MCP_SERVERS_RUNNING.set(running_total)
    MCP_SERVERS_DISPATCHABLE.set(dispatchable_total)


def _collection_registry() -> Any:
    """Resolve the registry a scrape must collect from.

    CONCEPT:AU-OS.observability.multiprocess-registry-guard — the classic
    ``prometheus_client`` footgun. Every metric in this module is created in the
    default per-process ``REGISTRY``, which is the CORRECT mode for how we
    actually deploy: graph-os is one FastMCP process per pod and the gateway
    daemon is a separate container on its own port, so Prometheus already scrapes
    them as distinct targets and aggregates across replicas with ``sum by``.
    Multiprocess mode is deliberately NOT the default — it needs a shared
    writable directory plus dead-worker file cleanup, and it silently changes
    metric semantics (``Gauge`` needs an explicit ``multiprocess_mode``,
    in-flight gauges become fleet sums, and ``_created`` series disappear).

    The failure mode we guard against is the *silent* one: if an operator ever
    puts this process behind a pre-fork server and sets
    ``PROMETHEUS_MULTIPROC_DIR``, collecting the default ``REGISTRY`` would
    report only the one worker that happened to serve the scrape while looking
    perfectly healthy. So when that variable is set we honour it and collect
    through ``MultiProcessCollector`` instead. A set-but-unusable value is
    logged loudly and falls back to the single-process registry rather than
    failing the scrape — a scrape that 500s hides the numbers entirely.
    """
    from prometheus_client import REGISTRY

    from agent_utilities.core.config import setting

    # Read through the central accessor (never bare os.environ) — and note this
    # is prometheus_client's OWN standard variable, set by whatever supervises
    # the workers, not a flag this package invents.
    multiproc_dir = str(setting("PROMETHEUS_MULTIPROC_DIR", "") or "").strip()
    if not multiproc_dir:
        return REGISTRY
    try:
        from prometheus_client import CollectorRegistry, multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
        return registry
    except Exception as exc:
        logger.error(
            "PROMETHEUS_MULTIPROC_DIR=%r is set but multiprocess collection "
            "failed (exception_type=%s: %s); falling back to this worker's "
            "registry only — the scrape UNDER-REPORTS the other workers. "
            "Either make the directory writable and empty at boot or unset the "
            "variable. (CONCEPT:AU-OS.observability.multiprocess-registry-guard)",
            multiproc_dir,
            type(exc).__name__,
            exc,
        )
        return REGISTRY


def render_metrics() -> tuple[bytes, str]:
    """Render the process metric registry in Prometheus exposition format.

    Returns ``(body, content_type)``. Without ``prometheus_client`` installed
    a self-describing placeholder is returned (still a 200 — scrapers treat
    the target as up, just empty). Honours ``PROMETHEUS_MULTIPROC_DIR`` — see
    :func:`_collection_registry`.
    """
    if not PROMETHEUS_AVAILABLE:
        return (
            b"# agent-utilities gateway metrics unavailable: install the "
            b"'metrics' extra (prometheus-client). (CONCEPT:AU-OS.observability.no-op-without-metrics)\n",
            "text/plain; version=0.0.4; charset=utf-8",
        )
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    _run_scrape_samplers()
    return generate_latest(_collection_registry()), CONTENT_TYPE_LATEST


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

    CONCEPT:AU-OS.observability.no-op-without-metrics. Mounted OUTERMOST by
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
            try:
                from agent_utilities.observability.gateway_health import (
                    record_request_duration,
                )

                # Same `duration` value the histogram above just consumed —
                # feeds the shared health-trend/anomaly kernel (CONCEPT:
                # AU-OS.observability.unified-health-kernels /
                # AU-KG.identity.evidence-spine-convergence). Best-effort.
                record_request_duration(duration)
            except Exception as e:  # noqa: BLE001 — feeds an in-memory trend sample only (record_request_duration is itself documented as must-never-affect-the-response-path); a dropped sample just thins one window's anomaly-detection density, the response and the histogram above are already recorded
                logger.debug("gateway health sample failed: %s", e)
