# Gateway daemon — the one host process and everything it runs

The agent-utilities **API gateway daemon** (`python -m agent_utilities.gateway.daemon`,
`start_host_daemon`) is the single authoritative host process (`KG_DAEMON_ROLE=host`).
Every other entry point — the MCP server, CLI, scripts — runs as a `client` and enqueues
work to the durable queue this daemon drains. This page is the **complete map of what runs
inside it** (kept in sync with `daemon_status()`).

```mermaid
flowchart TB
    subgraph GW["API Gateway Daemon (single host process — KG_DAEMON_ROLE=host)"]
        direction TB

        subgraph THREADS["Consolidated daemon threads"]
            T1[submission]
            T2[graph_writer]
            T3[maintenance scheduler]
            T4[embed_backfill]
            T5[task_workers pool]
        end

        subgraph JOBS["Maintenance jobs (scheduler tick)"]
            J1[analysis] --- J2[loop_cycle] --- J3[sai_factory]
            J4[failure_ingest] --- J5[skill_scheduler] --- J6[anomaly_consumer]
            J7[fuseki_publish] --- J8[compaction] --- J9[evolution]
            J10[reconcile_durable] --- J11[enrichment] --- J12[usage_log_sync]
            J13[usage_pricing_refresh] --- J14[file_watch] --- J15[hygiene]
            J16[task_reaper] --- J17[tenant_gc]
        end

        subgraph MSG["Messaging inbound router (ECO-4.51, thread w/ own loop)"]
            R[InboundRouter] --> B1[(Telegram backend)]
            R --> B2[(Slack / Teams / Mattermost / … when configured)]
            R --> H[planner handler]
            H --> AGENT[dedicated messaging agent\nlean; local-default / Claude]
            AGENT -. delegates .-> MCPOS
            H --> ING[KG auto-ingest chat memory]
        end

        REST["REST API (/graph/*, /daemon/*, /fleet/*, /metrics, /api/...)"]
    end

    subgraph SVC["Separate served processes"]
        MCPOS[graph-os MCP server\nsse 127.0.0.1:8100\ngraph_orchestrate / graph_search / graph_reach]
        MUX[mcp-multiplexer\ndynamic find_tools/load_tools → fleet]
        ENG[(epistemic-graph engine\nUDS /tmp/epistemic-graph.sock)]
    end

    subgraph STORE["State & queues"]
        PG[(Postgres\nqueue_backend + state_store)]
        SNAP[(engine snapshots / persist-dir)]
    end

    T1 --> PG
    T5 --> PG
    T2 --> ENG
    T3 --> JOBS
    REST --> ENG
    AGENT --> ENG
    ING --> ENG
    MCPOS --> ENG
    MUX --> MCPOS
    ENG --> SNAP
    B1 -->|poll/getUpdates + send| TG((User on Telegram))

    classDef ext fill:#533483,stroke:#7b2cbf,color:#fff
    class TG,ENG,PG ext
```

## What each group is

- **Daemon threads** — the consolidated background workers: `submission` (queue submit),
  `graph_writer` (durable writes to the engine), `maintenance` (the scheduler that fires
  the jobs below), `embed_backfill` (embeddings catch-up), `task_workers` (on-demand work
  pool draining the queue).
- **Maintenance jobs** — declarative scheduled work (`deploy/schedules.yml` + built-ins):
  KG `analysis`, the **`loop_cycle`** Loop engine, `sai_factory`, `failure_ingest`,
  `skill_scheduler`, `anomaly_consumer`, `fuseki_publish`, `compaction`, `evolution`,
  `reconcile_durable`, `enrichment`, `usage_log_sync`, `usage_pricing_refresh`,
  `file_watch`, `hygiene`, `task_reaper`, `tenant_gc`.
- **Messaging inbound router** (ECO-4.51) — runs on its own event loop in a daemon thread;
  connects every configured backend, ingests chat to the KG, and routes to the dedicated
  messaging agent, which **delegates** heavy work to graph-os (ECO-4.59).
- **REST API** — the gateway HTTP surface (`/graph/*`, `/daemon/*`, `/fleet/*`, `/metrics`).
- **Separate served processes** — the graph-os **MCP server** (sse :8100), the
  **mcp-multiplexer** (dynamic fleet tools), and the Rust **epistemic-graph engine** (the
  daemon connects to it over UDS; the engine is not in-process).
- **State & queues** — Postgres backs the task queue + externalized state; the engine
  persists snapshots to its persist-dir.

Run `agent-utilities-doctor` or `GET /daemon` (`daemon_status()`) for the live status that
this diagram mirrors.
