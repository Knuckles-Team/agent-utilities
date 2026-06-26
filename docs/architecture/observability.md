# Observability — Metrics, Logs, Traces, Alerts

How every MCP service **and** every Portainer stack's container is monitored,
built on the existing LGTM stack (`services/lgtm/`).

## Topology

```
                    ┌──────────── Prometheus (15s) ──────────┐
 node-exporter ─────┤  hosts (global, every node)            │
 cAdvisor ──────────┤  containers (global, every container)  │── rules.yml ─► Alertmanager ─► Mattermost
 MCP /metrics ──────┤  mcp-fleet (file-SD, 52 targets)       │
 blackbox /health ──┤  blackbox-mcp (synthetic probe)        │
                    └────────────────┬───────────────────────┘
                                     ▼
 promtail (docker SD) ─► Loki        Grafana (grafana.arpa, Keycloak OIDC)
 app traces ─► Tempo + Langfuse      provisioned datasources + dashboards
```

| Signal | Collector | Store | Notes |
|--------|-----------|-------|-------|
| Host metrics | node-exporter (global) | Prometheus | CPU/mem/disk/net per node |
| Container metrics | cAdvisor (global) | Prometheus | per-container, labelled `com.docker.stack.namespace` |
| MCP app metrics | each MCP `GET /metrics` | Prometheus | per-tool count/latency/error (CONCEPT:OS-5.23) |
| Synthetic health | blackbox-exporter | Prometheus | `GET /health` per MCP |
| Logs | promtail (docker SD) | Loki | container stdout/stderr, labelled stack/service |
| Traces | OTEL | Tempo + Langfuse | Langfuse for LLM traces |

## Per-MCP metrics (one change, whole fleet)

`create_mcp_server` mounts an unauthenticated `GET /metrics` and a
`ToolMetricsMiddleware` recording, per server:

- `agent_utilities_mcp_tool_calls_total{tool,outcome}`
- `agent_utilities_mcp_tool_duration_seconds_bucket{tool}` (histogram)
- `agent_utilities_mcp_tool_in_flight`

These are the server-side complement to the multiplexer's `agent_utilities_mcp_child_*`
metrics. All metrics no-op without the optional `metrics` extra.

## Scrape coverage (auto-maintained)

`scripts/gen_prometheus_mcp_targets.py` reads `deploy/mcp-fleet.registry.yml` and
writes `services/lgtm/targets/mcp-fleet.json` — one target per MCP at
`<stack>_<service>:8000` with `stack`/`service` labels. Two Prometheus jobs reuse
that file-SD: `mcp-fleet` (scrape `/metrics`) and `blackbox-mcp` (rewrite each
target into a `/health` probe). Re-run the generator on fleet change.

## Dashboards (provisioned as code)

`scripts/gen_grafana_dashboards.py` emits three dashboards into
`services/lgtm/grafana/provisioning/dashboards/json/`:

- **MCP Fleet Overview** — every stack up/probe/req/error/p95 + per-stack
  container CPU/mem (the "all Portainer stacks" view).
- **MCP Per-Service** — templated by `$stack`: tool rate/latency/errors,
  in-flight, container CPU/mem, and a Loki logs panel.
- **Host & Infra** — node-exporter CPU/mem/disk per host.

Datasources (Prometheus/Loki/Tempo) are provisioned in
`grafana/provisioning/datasources/`.

## Alerts

`services/lgtm/rules.yml` groups (→ Alertmanager → Mattermost):

- **infra-availability** — `InstanceDown` (non-MCP jobs).
- **mcp-fleet** — `McpServiceDown`, `McpProbeFailed`, `McpHighToolErrorRate`,
  `McpHighToolLatencyP95`, `McpChildBreakerOpen`.
- **containers** — `ContainerOOMKilled`, `ContainerHighMemory`, `ContainerRestarting`.
- **hosts** — `HostHighCpu`, `HostHighMemory`, `HostLowDisk`.

## What lights up when

The config is in git; activation is two deploys:

1. **Redeploy the LGTM stack** → the new jobs, blackbox, promtail, dashboards and
   rules go live.
2. **Rebuild the agent-utilities image** (or mount its source) → MCP `/metrics`
   starts returning data; until then `mcp-fleet` targets read "down" (the
   `McpServiceDown` rule has a 10-minute fuse to stay quiet during the rollout).

## KG-native agent observability & evaluation (Graphiti + Opik absorption)

The LGTM stack above is *infrastructure* observability (host/container/fleet metrics,
logs, synthetic health). Complementary to it is **KG-native application observability +
evaluation** — every agent run is captured as a first-class graph subgraph and scored by
the same engine, so traces are *queryable and reasoned over* rather than buried in an
opaque store (the moat over Opik's ClickHouse). Concepts: OS-5.68 (capture),
AHE-3.64/3.65/3.66/3.67/3.68 (online-scoring / G-Eval / tool-judge / sandboxed metrics /
dataset-prompt loop), KG-2.257 (moat queries).

### 1. Always-on capture → the trace subgraph

```mermaid
flowchart LR
    subgraph capture["Always-on capture (no vendor key needed)"]
        dec["@trace / @generation\ndecorators"]
        mw["create_model wrap\n(WrapperModel, per-LLM-call)"]
    end
    dec -->|"_emit_trace"| sink
    mw -->|"record_event"| sink["KGTraceBackend\n(default sink)"]
    daemon["host daemon startup\nset_kg_trace_sink()"] -.installs.-> sink
    sink -->|"add_node + link"| kg[("epistemic-graph")]
    sink -. fan-out (optional) .-> lf["Langfuse / OTel"]
    subgraph tracegraph["Trace subgraph in the KG"]
        T["TraceNode\ninput/output/cost/status"]
        S["SpanNode"]
        G["GenerationNode\nmodel/tokens/cost/latency"]
        T -->|HAS_SPAN| S
        T -->|HAS_GENERATION| G
    end
    kg --- tracegraph
    pricing["pricing catalog\n(ECO-4.40)"] -.cost.-> G
```

### 2. Online-scoring + evaluation (one judge path for prod + regression)

```mermaid
flowchart TD
    T["root trace completes"] -->|on_trace_complete hook| pool["OnlineScoringSampler\n(off hot-path thread pool)"]
    pool --> sel{"trace large?\n(>12 spans)"}
    sel -->|yes| tj["tool-judge\n(navigates spans via tools)"]
    sel -->|no| ij["inline LLM judge\n(EvalRunner._assertion_judge)"]
    pool --> rules["automation rules"]
    pool --> regs["regression assertions\n(EvalCorpus.load_cases)"]
    pool --> metrics["sandboxed Python metrics\n(SandboxedExecutor)"]
    rules & regs & metrics --> verdict
    tj & ij --> verdict["OnlineScoreNode /\nAssertionResultNode"]
    verdict -->|SCORED_BY| T
    verdict -->|FAILED| corpus["EvalCorpus.add_from_trace\n→ DatasetItemNode(source=trace)"]
    corpus -->|re-checked on future traces| regs
    geval["G-Eval\n(logprob-weighted + cached CoT)"] -.alt scorer.-> verdict
    pv["StructuredPrompt.version()\n→ PromptVersionNode"] -.prompt_version_id.-> verdict
    llm["vLLM / OpenAI-style"] -.judge calls.-> tj & ij & geval
```

### 3. Moat queries + the focused graph-os tool suite

Because traces/scores/generations/prompt-versions are KG nodes, the engine answers
questions an opaque trace store cannot — exposed through **focused, intent-scoped MCP
tools** (the `graph_analyze` 30-action wall was split so an agent selects by intent):

```mermaid
flowchart LR
    subgraph tools["graph-os analyze suite (focused tools)"]
        gobs["graph_observe"]
        gcode["graph_code"]
        gres["graph_research"]
        gev["graph_evaluate"]
        gexp["graph_explain"]
        gana["graph_analyze\n(residual ops/structural)"]
    end
    gcode & gres & gev & gexp -->|delegate| core["_execute_tool\n(one action core)"]
    gobs --> ta["trace_analytics"]
    ta --> q1["trace_rootcause\n(failures → agent)"]
    ta --> q2["prompt_regression\n(score per prompt version)"]
    ta --> q3["failure_cluster\n(systemic breaks)"]
    q1 & q2 & q3 --> kg[("trace subgraph")]
    core --> kg
```

> Engine-side wirings (Track A bi-temporal `AS OF`, Track C1 hybrid search / rerankers,
> Track C2 dedup) are documented in the epistemic-graph repo
> (`docs/uql.md`, `docs/architecture/engine.md`).
