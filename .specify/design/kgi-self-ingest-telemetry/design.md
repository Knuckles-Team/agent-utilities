# Design Document: agent-utilities ships its OWN telemetry into the engine's observability store — the engine becomes its own observability backend

> `agent_utilities/observability/self_ingest.py`.

CONCEPT:AU-KG.ingest.attaching-this-root-logger ·
CONCEPT:AU-KG.ingest.self-ingest

## Decision — self-ingest telemetry over the engine's own OTLP endpoint, opt-in and write-ahead

`self_ingest.py:1-25`, `375-385`.

**The problem**: structured log records and `RunTrace`/`:ToolCall`
provenance events are exactly the kind of data the epistemic-graph engine's
observability store is designed to hold and reason over — but by default
they were emitted only to conventional logging/tracing sinks, outside the
graph the rest of the system dogfoods for everything else.

**The rejected alternative**: keeping agent-utilities/graph-os's own
telemetry in a SEPARATE observability system from the engine's own store —
the conventional choice (a dedicated logging backend, a separate tracing
service), and explicitly the status quo this design changes. That would mean
the system's own operational history is not queryable through the same
graph interface as everything else it ingests.

**The design chosen** (mirroring the existing Langfuse exporter pattern):

- **`AU-KG.ingest.attaching-this-root-logger`** — the harness-side attachment
  point: `agent-utilities` and `graph-os` attach a root-logger-level sink
  (`SelfIngestSink`, `self_ingest.py:378`) so structured log records plus
  `RunTrace`/`:ToolCall` events are captured at the source, across every
  entrypoint that logs (`__main__.py:28`, `gateway/daemon.py:425`,
  `orchestration/engine.py:823`).
- **`AU-KG.ingest.self-ingest`** — the engine-side receiving contract: the
  harness posts over the engine's own OTLP/HTTP log-ingestion endpoint
  (`EPISTEMIC_GRAPH_OBS_ADDR` + `POST /v1/logs` OTLP, or a `_bulk`
  endpoint) — the SAME ingestion surface any other OTLP-speaking source
  would use, not a bespoke self-only API. The engine becomes its own
  observability backend for its own telemetry.

Two properties are load-bearing, not incidental:

1. **Opt-in, default-off.** Nothing happens unless
   `AGENT_UTILITIES_SELF_INGEST` is truthy AND `EPISTEMIC_GRAPH_OBS_ADDR` is
   set; every method is a clean no-op when disabled, so the LIVE request path
   is never affected by turning this off — the rejected alternative
   (always-on self-ingest) would make every deployment pay the cost/risk of
   self-telemetry even when nobody consumes it.
2. **Write-ahead, never network-blocking.** `SelfIngestSink.emit`
   synchronously appends a SANITIZED record to a local WAL before it becomes
   eligible to send; a background daemon thread batches network delivery.
   The rejected alternative — emit-then-network-send inline — would make the
   hot path (every log call, every tool call) block on network I/O to the
   observability store, an unacceptable coupling between "did I log
   something" and "is the obs store currently reachable."

## Risk Assessment

- **Blast Radius**: `agent_utilities/observability/self_ingest.py`,
  `agent_utilities/__main__.py`, `agent_utilities/gateway/daemon.py`,
  `agent_utilities/orchestration/engine.py`.
- **Backward Compatible**: Yes — default-off; a deployment that never sets
  `AGENT_UTILITIES_SELF_INGEST`/`EPISTEMIC_GRAPH_OBS_ADDR` is completely
  unaffected.
- **Breaking Changes**: None.
- **Known weak point**: the WAL is local-disk — a host that loses its local
  disk (or runs read-only/ephemeral) before the background thread flushes a
  batch loses that batch of self-telemetry permanently; the write-ahead
  guarantee is against network unavailability, not disk loss.
