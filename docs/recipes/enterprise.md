# Recipe — Enterprise (swarm)

Multi-node Docker Swarm with the full integration set and the complete `*-mcp`
connector fleet. This is the "run the enterprise" tier. It is driven by the
**`day0_bootstrap_orchestrator`** skill-workflow rather than by hand.

## What runs

| Layer | Components |
|---|---|
| Edge | Caddy (HTTPS ingress) · Technitium DNS (authoritative `.arpa`) |
| Core | Keycloak (SSO) · OpenBao (secrets) · Portainer (stack GitOps) · LGTM (Prometheus/Loki/Grafana/Tempo) |
| Data | Postgres/pggraph (durable KG L2) · Kafka (event backbone) |
| agent-utilities | REST gateway + KG host daemon, replicated; graph-os over streamable-http |
| Connectors | the **entire** `*-mcp` fleet (`enterprise` profile) via Portainer GitOps |
| UIs | agent-webui (Fleet Supervisor), agent-terminal-ui, geniusbot |

## Deploy (skill-workflow)

The `day0_bootstrap_orchestrator` workflow runs the ordered bootstrap:

1. `ssh-bootstrap` → full-mesh SSH across inventory hosts.
2. `network-topology-sweep` + `hardware-profile-sweep` → discovery.
3. `deployment-planner` → tiered placement manifest.
4. `swarm-mesh-provisioner` → swarm + overlay networks.
5. core-edge deploy → registry → DNS → Caddy → Portainer.
6. `secret-vault-manager` → OpenBao + Keycloak.
7. `gitlab-repository-seeder` + `portainer-gitops-bind` → stacks bound to Git.
8. **agent-utilities** → install deps, start graph-os + multiplexer, deploy the
   `*-mcp` fleet from `deploy/mcp-fleet.registry.yml`, wire pggraph + Kafka +
   OpenBao + Langfuse + Keycloak.
9. `graph-os` → materialize the full topology in the KG.

Select the **enterprise** profile when the workflow's Step-0 questionnaire asks,
and toggle the integrations you want.

## `config.json` (generalized, enterprise switches)

```jsonc
{
  "graph_backend": "tiered",
  "graph_db_uri": "postgresql://agent:REDACTED@pggraph.example.arpa:5432/agent_kg",
  "kg_daemon_role": "host",

  "task_queue_backend": "kafka",
  "kafka_bootstrap_servers": "kafka.example.arpa:9092",

  "secrets_vault_url": "https://openbao.example.arpa",
  "vault_auth_method": "approle",

  "auth_jwt_jwks_uri": "https://keycloak.example.arpa/realms/agents/protocol/openid-connect/certs",
  "auth_jwt_issuer": "https://keycloak.example.arpa/realms/agents",

  "enable_otel": true,
  "otel_exporter_otlp_endpoint": "https://langfuse.example.arpa/api/public/otel",
  "langfuse_host": "https://langfuse.example.arpa"
}
```

## Scale note

The connector fleet is stateless and scales horizontally on the swarm. The KG
host daemon is a singleton per host per the `KG_DAEMON_ROLE=host` flock; running
the agent swarm at very large scale (the 100k+ target) additionally needs
multiple gateway workers + a durable queue (Kafka, above) + shared pggraph state
— see the [capacity model](../scaling/capacity_model.md). Durable execution
(idempotency + at-least-once) is already in place to make that safe.

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
per-shard health is on `GET /daemon/shards` +
`agent_utilities_engine_shard_up{endpoint}`. Worked single-host 3-shard compose:
`docker/engine-shards.compose.yml`; full semantics (including the manual
snapshot migration caveat when re-sharding):
[engine sharding](../architecture/engine_sharding.md).

## Operate

The **agent-webui Fleet Supervisor** (`/api/fleet/*`) is your single pane of
glass: per-domain health/error-rates, live topology, one-click pause/kill
containment, and the mutation/risk approval queue.
