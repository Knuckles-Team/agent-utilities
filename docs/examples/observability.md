# Worked Example: Scraping and Dashboarding the Platform

## What this demonstrates

A working Prometheus scrape configuration for the two metric surfaces — the
gateway's Python-tier `/metrics` endpoint (CONCEPT:OS-5.23) and the engine
shards' native listeners — plus ready-to-paste Grafana panel queries over the
headline series.

Deep dives: [gateway_scaling.md](../architecture/gateway_scaling.md) and
[engine_sharding.md](../architecture/engine_sharding.md). Full metric
reference: [metrics.md](../reference/metrics.md).

## Prerequisites (ladder rung)

Any rung of the [deployment ladder](../guides/deployment-configurations.md)
that runs the gateway. Python-tier metrics need the optional `metrics` extra:

```bash
pip install "agent-utilities[metrics]"   # prometheus-client
```

Without it the middleware degrades to a no-op and `GET /metrics` returns a
self-describing placeholder (still HTTP 200). The endpoint is controlled by
`GATEWAY_METRICS` (default on) and is **exempt from the identity middleware**
— scrapers cannot mint JWTs (CONCEPT:OS-5.23).

## 1. Scrape targets

Two distinct surfaces:

1. **Gateway** — `GET /metrics` on the gateway HTTP port, serving every
   `agent_utilities_*` series registered in
   `agent_utilities/observability/gateway_metrics.py`. Multi-worker note:
   metrics live in the per-process default registry; with
   `GATEWAY_WORKERS>1` a scrape samples ONE worker — run one worker per
   container or aggregate across replicas in Prometheus.
2. **Engine shards** — each `epistemic-graph-server` process exposes its own
   native `epistemic_graph_*` series on its `--metrics-addr` listener. In
   the shipped `docker/engine-shards.compose.yml` the **RPC ports are
   9101–9103 and the metrics listeners are 9111–9113** (one scrape target
   per shard) — see the [sharding walkthrough](sharding-walkthrough.md).

```yaml
# prometheus.yml
scrape_configs:
  - job_name: agent-utilities-gateway
    metrics_path: /metrics
    static_configs:
      - targets: ["gateway.example.com:8000"]   # your gateway host:port

  - job_name: epistemic-graph-shards
    metrics_path: /metrics
    static_configs:
      - targets:
          - "shards.example.com:9111"   # engine-shard-1 --metrics-addr
          - "shards.example.com:9112"   # engine-shard-2
          - "shards.example.com:9113"   # engine-shard-3
        labels:
          tier: engine
```

## 2. Headline series (compact)

The full table, including labels and producers, lives in the
[metrics reference](../reference/metrics.md). The ones most dashboards start
with:

| Series | Type | What it tells you |
|---|---|---|
| `agent_utilities_gateway_requests_total{route,method,status}` | counter | Traffic + error ratio per route template |
| `agent_utilities_gateway_request_duration_seconds{route}` | histogram | Latency distribution per route |
| `agent_utilities_gateway_in_flight_requests` | gauge | Concurrency right now |
| `agent_utilities_gateway_rate_limited_total{tenant}` | counter | 429s from the per-tenant token bucket |
| `agent_utilities_gateway_engine_requests_total{op,outcome}` | counter | Engine client calls + failure mix |
| `agent_utilities_gateway_engine_breaker_state{endpoint}` | gauge | Engine circuit breaker (0=closed, 1=half-open, 2=open) |
| `agent_utilities_engine_shard_up{endpoint}` | gauge | Per-shard reachability (1/0) |
| `agent_utilities_engine_shard_requests_total{endpoint,outcome}` | counter | Engine calls split per shard |
| `agent_utilities_kg_ingest_queue_depth{backend}` | gauge | Pending KG ingest tasks (autoscaler signal `queue_depth`) |
| `agent_utilities_kg_ingest_consumer_lag{topic,group}` | gauge | Kafka kg-ingest lag (autoscaler signal `consumer_lag`) |
| `agent_utilities_dispatch_queue_depth{backend}` | gauge | Unclaimed dispatched agent turns |
| `agent_utilities_dispatch_turns_total{outcome}` | counter | Processed agent turns by outcome |
| `agent_utilities_dispatch_workers` | gauge | Live dispatch workers (fresh heartbeats) |
| `agent_utilities_mcp_child_calls_total{server,outcome}` | counter | Multiplexer tool calls per child |
| `agent_utilities_mcp_child_breaker_state{server}` | gauge | Per-child breaker (0/1/2) |
| `agent_utilities_mcp_child_restarts_total{server}` | counter | Child crash-restarts |
| `agent_utilities_mcp_child_queue_depth{server}` | gauge | Calls queued behind a child's concurrency limit |

Cardinality discipline worth knowing: `route` is always a route TEMPLATE
(`/api/graph/{name}`); unmatched scanner traffic collapses into a single
`unmatched` bucket.

## 3. Grafana panel queries

Request rate by route (req/s):

```promql
sum by (route) (rate(agent_utilities_gateway_requests_total[5m]))
```

p95 latency per route — `agent_utilities_gateway_request_duration_seconds`
**is** a Prometheus `Histogram` (verified in
`observability/gateway_metrics.py`; buckets 5 ms – 60 s), so the standard
quantile recipe applies to its `_bucket` series:

```promql
histogram_quantile(
  0.95,
  sum by (le, route) (rate(agent_utilities_gateway_request_duration_seconds_bucket[5m]))
)
```

Error ratio (5xx share of all requests):

```promql
sum(rate(agent_utilities_gateway_requests_total{status=~"5.."}[5m]))
/
sum(rate(agent_utilities_gateway_requests_total[5m]))
```

Ingest backlog and Kafka consumer lag (the same series the OS-5.29
autoscaler target-tracks — see the
[autoscaling example](autoscaling-signals.md)):

```promql
sum(agent_utilities_kg_ingest_queue_depth)
sum by (topic) (agent_utilities_kg_ingest_consumer_lag)
```

Agent-dispatch queue depth vs live workers (two queries, one panel):

```promql
sum(agent_utilities_dispatch_queue_depth)
agent_utilities_dispatch_workers
```

Shard availability and breaker states (alert when a shard is down or any
breaker is open, i.e. value 2):

```promql
agent_utilities_engine_shard_up
max by (endpoint) (agent_utilities_gateway_engine_breaker_state)
```

Multiplexer child restarts over the last hour (a climbing line = a
crash-looping child MCP server):

```promql
sum by (server) (increase(agent_utilities_mcp_child_restarts_total[1h]))
```

## 4. Alert sketches

```yaml
groups:
  - name: agent-utilities
    rules:
      - alert: EngineShardDown
        expr: agent_utilities_engine_shard_up == 0
        for: 2m
        labels: {severity: critical}
        annotations:
          summary: "Engine shard {{ $labels.endpoint }} unreachable"
      - alert: EngineBreakerOpen
        expr: agent_utilities_gateway_engine_breaker_state == 2
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: "Engine circuit breaker open for {{ $labels.endpoint }}"
```

Point the Alertmanager receiver for these at the fleet-events ingress and the
platform will triage its own incidents — that loop is the
[fleet events example](fleet-events-wiring.md).

## Verification

```bash
curl -sS http://localhost:8000/metrics | grep -E '^agent_utilities_' | head
python3 -m pytest tests/unit/test_gateway_metrics.py -q
```

---

*Smoke-run against this tree (2026-06-11): every `agent_utilities_*` series
name and label set above was verified against
`agent_utilities/observability/gateway_metrics.py`, and `python3 -m pytest
tests/unit/test_gateway_metrics.py -q` passed as part of a 99-test green run.
The prometheus.yml/Grafana/alert snippets target external tooling and were
reviewed against code only (no Prometheus was deployed for this doc).*
