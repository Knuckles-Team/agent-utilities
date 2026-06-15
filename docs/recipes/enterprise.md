# Recipe — Enterprise (swarm)

> Ladder position: this recipe combines **rung (d) — Scaled multi-host** and
> **rung (e) — Autonomous operations** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-d-scaled-multi-host)
> guide, which carries the complete flag-by-flag `.env`/`config.json` for both
> rungs and their verification steps. Note both rungs are marked
> **not exercised in CI** there — validate in staging.

Multi-node Docker Swarm with the full integration set and the complete `*-mcp`
connector fleet. This is the "run the enterprise" tier. It is driven by the
**`agent-os-genesis` (alias `day0`)** skill-workflow rather than by hand.

## What runs

| Layer | Components |
|---|---|
| Edge | Caddy (HTTPS ingress) · Technitium DNS (authoritative `.arpa`) |
| Core | Keycloak (SSO) · OpenBao (secrets) · Portainer (stack GitOps) · LGTM (Prometheus/Loki/Grafana/Tempo) |
| Data | Postgres/pg-age (durable KG L2) · Kafka (event backbone) |
| agent-utilities | REST gateway + KG host daemon, replicated; graph-os over streamable-http |
| Connectors | the **entire** `*-mcp` fleet (`enterprise` profile) via Portainer GitOps |
| UIs | agent-webui (Fleet Supervisor), agent-terminal-ui, geniusbot |

## Deploy (skill-workflow)

The `agent-os-genesis` (alias `day0`) workflow runs the ordered bootstrap:

1. `ssh-bootstrap` → full-mesh SSH across inventory hosts.
2. `network-topology-sweep` + `hardware-profile-sweep` → discovery.
3. `deployment-planner` → tiered placement manifest.
4. `swarm-mesh-provisioner` → swarm + overlay networks.
5. core-edge deploy → registry → DNS → Caddy → Portainer.
6. `secret-vault-manager` → OpenBao + Keycloak.
7. `gitlab-repository-seeder` + `portainer-gitops-bind` → stacks bound to Git.
8. **agent-utilities** → install deps, start graph-os + multiplexer, deploy the
   `*-mcp` fleet from `deploy/mcp-fleet.registry.yml`, wire pg-age + Kafka +
   OpenBao + Langfuse + Keycloak.
9. `graph-os` → materialize the full topology in the KG.

Select the **enterprise** profile when the workflow's Step-0 questionnaire asks,
and toggle the integrations you want.

## `config.json` (generalized, enterprise switches)

```jsonc
{
  "graph_backend": "tiered",
  "graph_db_uri": "postgresql://agent:REDACTED@pg-age.example.arpa:5432/agent_kg",
  "kg_daemon_role": "host",

  // Durable platform state (sessions/goals/checkpoints/queues) on shared
  // Postgres — enables fleet-wide leader election for daemon ticks
  "state_db_uri": "postgresql://agent:REDACTED@pg-age.example.arpa:5432/agent_state",

  "task_queue_backend": "kafka",
  "kafka_bootstrap_servers": "kafka.example.arpa:9092",

  // Agent turns via the session-keyed queue, executed by the
  // agent-dispatch-worker fleet (default "inline" = in-process)
  "agent_dispatch_backend": "queue",

  "secrets_vault_url": "https://openbao.example.arpa",
  "vault_auth_method": "approle",

  "kg_auth_required": true,
  "auth_jwt_jwks_uri": "https://keycloak.example.arpa/realms/agents/protocol/openid-connect/certs",
  "auth_jwt_issuer": "https://keycloak.example.arpa/realms/agents",

  "enable_otel": true,
  "otel_exporter_otlp_endpoint": "https://langfuse.example.arpa/api/public/otel",
  "langfuse_host": "https://langfuse.example.arpa"
}
```

(Keys in `~/.config/agent-utilities/config.json` are upper-cased to their env
aliases and applied only where the env var is unset — environment always wins.)

## Scale note

The connector fleet is stateless and scales horizontally on the swarm. The KG
host daemon is a singleton per host per the `KG_DAEMON_ROLE=host` flock; running
the agent swarm at very large scale (the 100k+ target) additionally needs
multiple gateway workers (`GATEWAY_WORKERS`) + a durable queue (Kafka, above) +
shared pg-age/state-store Postgres — see the
[capacity model](../scaling/capacity_model.md). Durable execution (idempotency
+ at-least-once) is already in place to make that safe. The work itself scales
through the two consumer fleets — `kg-ingest-worker` (ingest, `kg_tasks`
partitions) and `agent-dispatch-worker` (agent turns, `agent_turns`
session-keyed partitions) — on any host that reaches Kafka, Postgres, and the
engine; invocations are in
[rung (d) of the ladder](../guides/deployment-configurations.md#rung-d-scaled-multi-host).

### Engine shards (Stage 2 — tenant-partitioned L0)

When one engine host saturates, run N engine shards and add to `config.json`:

```jsonc
{
  "graph_service_endpoints": [
    "tcp://kg-shard-1.example.arpa:9101",
    "tcp://kg-shard-2.example.arpa:9102",
    "tcp://kg-shard-3.example.arpa:9103"
  ],
  "graph_service_auth_secret": "ONE shared secret across shards + clients"
}
```

Graphs (and therefore tenants — `tenant → named graph → HRW → shard`) are
routed client-side by rendezvous hashing; an unreachable shard fails loud, and
per-shard health is on the gateway's `GET /api/dashboard/daemon/shards` +
`agent_utilities_engine_shard_up{endpoint}`. Worked single-host 3-shard compose:
`docker/engine-shards.compose.yml`; full semantics (including the manual
snapshot migration caveat when re-sharding):
[engine sharding](../architecture/engine_sharding.md).

## Operate

The **agent-webui Fleet Supervisor** (`/api/fleet/*`) is your single pane of
glass: per-domain health/error-rates, live topology, one-click pause/kill
containment, and the mutation/risk approval queue.

To let the platform operate on itself — golden loop, failure-driven evolution,
the desired-state fleet reconciler (`FLEET_RECONCILER` + a real
`FLEET_ACTUATOR`), the replica autoscaler (`FLEET_AUTOSCALER`), ActionPolicy
postures, and the `POST /api/fleet/events` monitoring webhook
(`FLEET_EVENTS_TOKEN`) — follow
[rung (e) of the ladder](../guides/deployment-configurations.md#rung-e-autonomous-operations).
The shipped defaults are deliberately inert: `FLEET_ACTUATOR=dryrun` and an
ActionPolicy that queues every mutating action for human approval.
