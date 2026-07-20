# Recipe — Enterprise multi-host

> Ladder position: this recipe combines **rung (d) — Scaled multi-host** and
> **rung (e) — Autonomous operations** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-d-scaled-multi-host)
> guide, which carries the complete AgentConfig surface for both
> rungs and their verification steps. Note both rungs are marked
> **not exercised in CI** there — validate in staging.

This recipe describes an orchestrator-neutral multi-host deployment with a selected
connector fleet. The **`agent-utilities-deployment`** skill-workflow drives it from
an operator-owned runtime profile. The repository contains no site inventory,
endpoint, credential, certificate, or environment-specific placement profile.

## What runs

| Layer | Components |
|---|---|
| Edge | Operator-managed TLS ingress and service discovery |
| Core | OIDC identity provider · runtime secret provider · deployment controller · OpenTelemetry-compatible observability |
| Engine | **shared/remote epistemic-graph engine** (the one authority/SoR), reached by every client via `GRAPH_SERVICE_ENDPOINTS`; shard it when one host saturates |
| Data (mirrors) | **optional** Postgres/pg-age **mirror** (write-only fan-out for SQL-side querying/BI) · Kafka (event backbone) |
| agent-utilities | REST gateway + KG host daemon, replicated; graph-os over streamable-http |
| Connectors | the approved `*-mcp` fleet selected by the external runtime profile |
| UIs | agent-webui (Fleet Supervisor), agent-terminal-ui, geniusbot |

## Deploy (skill-workflow)

The `agent-utilities-deployment` workflow runs an ordered bootstrap: discover and
validate the operator-supplied inventory, generate placement, prepare the selected
orchestrator and workload identity, configure secret and TLS references, deploy the engine and GraphOS,
attach optional mirrors and the connector fleet, then certify the live topology.
Every phase consumes an external runtime profile; no generated site profile is added
to this repository.

Select the **enterprise** profile when the workflow's Step-0 questionnaire asks,
and toggle the integrations you want.

## AgentConfig (neutral enterprise shape)

Persist this shape under `$XDG_CONFIG_HOME/agent-utilities/config.json`. Keep the
resolved inventory and secret material outside AgentConfig.

```json
{
  "GRAPH_SERVICE_ENDPOINTS": ["tls://engine.example.invalid:9100"],
  "ENGINE_TLS_PROFILE_REF": "secret://tls/engine-profile",
  "GRAPH_DB_CONNECTION_PROFILE_REF": "secret://graph/mirror-profile",
  "KG_DAEMON_ROLE": "host",
  "TASK_QUEUE_BACKEND": "kafka",
  "KG_IDENTITY_OAUTH2": {
    "token_url": "https://identity.example.invalid/oauth2/token",
    "client_id": "graph-client",
    "client_secret": "secret://identity/graph-os-client-secret",
    "audience": "graph-services"
  },
  "AUTH_JWT_JWKS_URI": "https://identity.example.invalid/.well-known/jwks.json",
  "AUTH_JWT_AUDIENCE": "graph-services",
  "KG_POLICY_VERSION": "current",
  "LANGFUSE_HOST": "https://observability.example.invalid",
  "LANGFUSE_PUBLIC_KEY_REF": "secret://observability/langfuse-public-key",
  "LANGFUSE_SECRET_KEY_REF": "secret://observability/langfuse-secret-key",
  "LANGFUSE_TLS_PROFILE_REF": "secret://tls/langfuse-profile",
  "LANGFUSE_CAPTURE_CONTENT": false,
  "LANGFUSE_KG_AUTO_INGEST": false
}
```

Endpoints above are reserved documentation values. Operators inject their own
topology and resolve every secret/TLS reference at runtime. Langfuse MCP and
propose-only failure evolution auto-enable when both credential references resolve;
trace export, content capture, and KG auto-ingestion remain explicit opt-ins.

## Scale note

The connector fleet is stateless and scales horizontally through the selected
orchestrator. The KG host daemon is a singleton per host per the
`KG_DAEMON_ROLE=host` lock; running a large agent fleet additionally needs
multiple gateway workers (`GATEWAY_WORKERS`) + a durable queue (Kafka, above) +
shared state storage and, when selected, a graph mirror — see the
[capacity model](../scaling/capacity_model.md). Durable execution (idempotency
+ at-least-once) is already in place to make that safe. The work itself scales
through the two consumer fleets — `kg-ingest-worker` (ingest, `kg_tasks`
partitions) and `agent-dispatch-worker` (agent turns, `agent_turns`
session-keyed partitions) — on any host that reaches Kafka, Postgres, and the
engine; invocations are in
[rung (d) of the ladder](../guides/deployment-configurations.md#rung-d-scaled-multi-host).

### Engine shards (Stage 2 — tenant-partitioned engine authority)

When one engine host saturates, run N engine shards (each a slice of the one
authority) and add to `config.json`:

```json
{
  "GRAPH_SERVICE_ENDPOINTS": [
    "tls://engine.example.invalid:9100"
  ],
  "ENGINE_TLS_PROFILE_REF": "secret://tls/engine-profile"
}
```

The stable coordinator returns each tenant graph's authoritative MultiRaft group,
epoch, and fence. The client never hashes endpoint names. An unreachable authority
or ambiguous group topology fails closed; governed movement advances the epoch.
Coordinator-contact health is available from the doctor and
`agent_utilities_engine_shard_up{endpoint}`. Full semantics:
[engine sharding](../architecture/engine_sharding.md).

## Operate

The **agent-webui Fleet Supervisor** (`/api/fleet/*`) is your single pane of
glass: per-domain health/error-rates, live topology, one-click pause/kill
containment, and the mutation/risk approval queue.

To let the platform operate on itself — golden loop, failure-driven evolution,
the desired-state fleet reconciler (`FLEET_RECONCILER` + a real
`FLEET_ACTUATOR`), the replica autoscaler (`FLEET_AUTOSCALER`), ActionPolicy
postures, and the `POST /api/fleet/events` monitoring webhook
(`FLEET_EVENTS_TOKEN_REF`) — follow
[rung (e) of the ladder](../guides/deployment-configurations.md#rung-e-autonomous-operations).
The shipped defaults are deliberately inert: `FLEET_ACTUATOR=dryrun` and an
ActionPolicy that queues every mutating action for human approval.
