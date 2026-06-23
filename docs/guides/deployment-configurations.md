# Supported Deployment Configurations

This is the flagship configuration ladder for agent-utilities: five complete,
copy-paste-ready configurations, from a zero-infrastructure laptop to an
autonomous multi-host fleet. Each rung builds on the previous one and states
its delta explicitly.

Every flag name, default, and behavior on this page is verified against
`agent_utilities/core/config.py` (`AgentConfig`), the shipped
`docker/*.compose.yml` files, and the module that reads the flag. The
authoritative flag inventory (with per-flag verdicts) is
[Configuration Reference & Flag Audit](../architecture/configuration.md).
Configurations the CI pipeline cannot stand up (live Kafka, live Postgres,
multi-shard engines, live actuators) are explicitly marked
**not exercised in CI** below, with a pointer to the unit suites that cover
their logic against injected fakes.

## Summary matrix

| Rung | Durable state | Task queue | Agent dispatch | Auth | Engines | Autonomy |
|---|---|---|---|---|---|---|
| [(a) Zero-infra dev](#rung-a-zero-infra-dev) | per-host SQLite | `sqlite` (auto) | `inline` (default) | none (open, startup warning) | 1 local engine (UDS) | off; ActionPolicy = approval-required default |
| [(b) Secured single node](#rung-b-secured-single-node) | per-host SQLite | `sqlite` (auto) | `inline` | engine HMAC + required JWT identity + brain enforcement | 1 local engine, HMAC | off |
| [(c) Durable single node](#rung-c-durable-single-node) | shared Postgres (`STATE_DB_URI`) | `postgres` (auto) | `inline` | as (b) | 1 local engine | off |
| [(d) Scaled multi-host](#rung-d-scaled-multi-host) | shared Postgres | `kafka` (explicit) | `queue` + worker fleet | as (b) | 3 TCP shards (HRW-routed) | off |
| [(e) Autonomous operations](#rung-e-autonomous-operations) | shared Postgres | `kafka` | `queue` | as (b) | 3 TCP shards | golden loop, failure-driven evolution, fleet reconciler, autoscaler, event webhook |

## How configuration is loaded

Three layers, in precedence order:

1. **Environment variables / `.env`** — every flag is a typed field on
   `AgentConfig` (`agent_utilities/core/config.py`); the env alias is the
   flag's canonical name. The CI gate `scripts/check_no_env_sprawl.py` keeps
   new flags on `AgentConfig` instead of scattered `os.environ` reads.
2. **XDG `config.json`** — discovered at
   `~/.config/agent-utilities/config.json` (override the directory with
   `AGENT_UTILITIES_CONFIG_DIR`; resolution lives in
   `agent_utilities/core/paths.py`). Each key is upper-cased to its env name
   and applied **only if that env var is not already set** — so the
   environment always wins. Lists/dicts are JSON-encoded (this is how
   `chat_models` / `embedding_models` registries are normally configured).
   A template `config.json` is written on first run if none exists.
   `GRAPH_SERVICE_ENDPOINTS` and other list flags accept either a JSON list
   (natural in `config.json`) or a comma-separated string (natural in `.env`).
3. **Defaults** — the values baked into `AgentConfig`. The zero-infra rung
   below runs entirely on these.

Data lives under XDG paths: config `~/.config/agent-utilities/`, data
`~/.local/share/agent-utilities/` (`AGENT_UTILITIES_DATA_DIR` override), cache
`~/.cache/agent-utilities/`.

The relevant processes (console scripts from `pyproject.toml
[project.scripts]`):

| Command | Role |
|---|---|
| `graph-os` | MCP tool surface (stdio default; `--transport streamable-http --port <p>` for HTTP; compose: `docker/mcp.compose.yml`, port 8004) |
| `graph-os-daemon` | Standalone KG host daemon: holds the flock host lock, drains the durable task queue, runs all maintenance/autonomy ticks. `--status` and `--drain-queue` flags. It serves no HTTP. |
| `python -m agent_utilities` | The REST gateway (FastAPI agent server): `/health`, `/api/graph/*`, `/api/sessions`, `/api/goals`, `/api/fleet/*`, `/api/dashboard/*`, `/metrics`. Binds `HOST`:`PORT` (defaults `0.0.0.0`:`9000`). `GATEWAY_WORKERS` pre-forks it. When it runs, it hosts the KG daemon itself (flock-elected) — you do not also need `graph-os-daemon`. |
| `kg-ingest-worker` | Decoupled ingest consumer (`kg-ingest` group), engine client only (KG-2.57) |
| `agent-dispatch-worker` | Stateless agent-turn consumer (`agent-dispatch` group), engine client only (ORCH-1.45) |

---

## Rung (a): Zero-infra dev

**What you get:** the full platform on one machine with no external services.
The knowledge graph runs on the default `epistemic_graph` backend — the
`epistemic-graph` Rust engine is the one database (the authority), providing
durable persistence, in-memory cache, graph compute, and ontology reasoning in a
single self-contained process with no mirrors — all platform state in per-host
SQLite files under the XDG data dir, agent turns
executed in-process (`AGENT_DISPATCH_BACKEND=inline` is the default), and no
authentication. **What you don't get:** identity (callers are trusted as-is
and a one-time startup warning says so), durability beyond this host, more
than one host, and any autonomous operations (all autonomy flags default off;
the shipped ActionPolicy marks every mutating operational action
`approval_required`).

### `.env`

An **empty `.env` is a valid rung-(a) configuration** — every line below is
either the shipped default (shown commented, for visibility) or optional:

```dotenv
# ---- Everything below the first block is the shipped default ----

# Spawn the local Rust engine on first connect if it is not already running.
# The `epistemic-graph-server` binary ships with the epistemic-graph wheel.
# Without this flag, the engine daemon must already be running (there is no
# in-process fallback); autostart only ever applies to the local endpoint.
EPISTEMIC_GRAPH_AUTOSTART=1

# Model provider — only needed for LLM-backed features (agents, enrichment).
# Graph operations run without it.
OPENAI_API_KEY=sk-your-key
# ...or any other provider key on AgentConfig (ANTHROPIC_API_KEY,
# GEMINI_API_KEY, ...), or OPENAI_BASE_URL for a local vLLM/Ollama endpoint.

# -- Defaults, listed explicitly (do not need to be set) --
#GRAPH_BACKEND=epistemic_graph     # the engine IS the database; self-contained, no mirrors
#KG_DAEMON_ROLE=auto               # flock host-lock elects ONE host per machine
#STATE_DB_URI=                     # unset = per-host SQLite state files
#TASK_QUEUE_BACKEND=               # unset = auto: sqlite (no STATE_DB_URI)
#AGENT_DISPATCH_BACKEND=inline     # agent turns execute in-process
#KG_AUTH_REQUIRED=false            # open access + one-time startup warning
#KG_BRAIN_ENFORCE=false            # ACL/permission checks are no-ops
#GATEWAY_WORKERS=1
#GATEWAY_METRICS=true              # /metrics on the gateway
#KG_DEV_MODE=false                 # false = all maintenance daemons on
#FLEET_RECONCILER=false
#FLEET_AUTOSCALER=false
#KG_LOOP=false
#KG_FAILURE_EVOLUTION=false
#ACTION_POLICY_PATH=               # empty = shipped conservative default policy
```

### `config.json`

None required. On first run a template is written to
`~/.config/agent-utilities/config.json`; the one thing most dev setups put in
it is the model registry (`chat_models` / `embedding_models`) — see
[`docs/examples/config.json`](../examples/config.json).

### Engine auth on this rung

Even with zero configuration, engine traffic is authenticated: when
`GRAPH_SERVICE_AUTH_SECRET` is unset, a per-install HMAC secret is minted once
and persisted at `~/.local/share/agent-utilities/engine_secret` (mode 0600,
`knowledge_graph/core/graph_compute.py`), and every local process — including
any engine this install spawns — agrees on it (CONCEPT:OS-5.14).
`KG_ENGINE_INSECURE=1` is the explicit dev opt-out.

### Verify

```bash
# 1. KG host daemon status (host lock holder + live daemon threads)
uv run graph-os-daemon --status

# 2. Python facade round-trip (no services needed beyond the local engine)
python3 -c "
from agent_utilities.knowledge_graph.facade import KnowledgeGraph
kg = KnowledgeGraph()
print(kg.query('MATCH (n) RETURN count(n) AS n'))
"

# 3. MCP surface over stdio (register in your IDE's mcp_config.json)
#    { "mcpServers": { "graph-os": { "command": "uv", "args": ["run", "graph-os"] } } }

# 4. REST gateway (also hosts the KG daemon when it runs)
python -m agent_utilities &        # binds 0.0.0.0:9000 by default (HOST/PORT)
curl -s localhost:9000/health
curl -s -X POST localhost:9000/api/graph/query \
  -H 'content-type: application/json' \
  -d '{"cypher": "MATCH (n) RETURN count(n) AS n"}'
```

Recipe form: [Tiny](../recipes/tiny.md). MCP consumption patterns:
[mcp-consumption](../examples/mcp-consumption.md).

---

## Rung (b): Secured single node

**What you get:** everything from (a) plus a closed identity perimeter:
engine HMAC (already automatic — now made explicit), server-minted JWT
identity required on the KG surface, and fail-closed node-level permission
enforcement. **What you don't get:** durability beyond this host, scale-out,
autonomy.

The pieces (all CONCEPT:OS-5.14, validated in
`agent_utilities/security/auth.py` and
`agent_utilities/security/request_identity.py`):

- **Engine HMAC** — automatic per-install secret as in rung (a); set
  `GRAPH_SERVICE_AUTH_SECRET` explicitly only when multiple installs/hosts
  must share one engine.
- **`KG_AUTH_REQUIRED=1`** — `ActorIdentityMiddleware` validates
  `Authorization: Bearer <JWT>` against `AUTH_JWT_JWKS_URI` (JWKS cached 5
  minutes), checks `AUTH_JWT_ISSUER` / `AUTH_JWT_AUDIENCE` when set, and mints
  the server-side `ActorContext` from the claims (`sub`/`client_id`/`azp` →
  actor, `roles`/`realm_access.roles`/`scope` → roles,
  `tenant_id`/`tenant`/`org_id`/`tid` → tenant). An invalid token is always a
  401 — even with `KG_AUTH_REQUIRED` off. With it on, requests without a valid
  token get 401; only health paths (`/health`, `/healthz`, `/api/health`,
  `/api/healthz`) and `/metrics` stay open. Caller-supplied
  `_actor`/`_roles`/`_tenant` kwargs are ignored.
- **`KG_AUTH_TOKEN`** — a JWT minting the identity of a *stdio* MCP process
  (stdio has no Authorization header); validated against the same JWKS.
- **`KG_BRAIN_ENFORCE=1`** — turns the node-ACL / ontology-permission checks
  from no-ops into real fail-closed enforcement (read in
  `knowledge_graph/core/company_brain_runtime.py`; truthy values `1`, `true`,
  `yes`, `on`; default `false`). With enforcement on, nodes **without** an ACL
  are denied by default; `KG_ACL_DEFAULT_ALLOW=1` is the explicit escape hatch
  that allows un-ACL'd nodes.

### `.env`

```dotenv
# ---- Everything from rung (a) ----
EPISTEMIC_GRAPH_AUTOSTART=1
OPENAI_API_KEY=sk-your-key

# ---- plus: identity & enforcement (OS-5.14) ----

# Required server-validated JWT identity on the KG REST/MCP surface
KG_AUTH_REQUIRED=1
AUTH_JWT_JWKS_URI=https://keycloak.example.internal/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://keycloak.example.internal/realms/agents
AUTH_JWT_AUDIENCE=agent-utilities

# Identity for stdio MCP processes (no Authorization header on stdio)
#KG_AUTH_TOKEN=eyJhbGciOi...

# Fail-closed node-level permissioning
KG_BRAIN_ENFORCE=1
#KG_ACL_DEFAULT_ALLOW=0            # default: nodes without an ACL are denied

# Engine HMAC is automatic (per-install secret at data_dir()/engine_secret).
# Set explicitly only to share one engine across installs:
#GRAPH_SERVICE_AUTH_SECRET=<openssl rand -hex 32>
#KG_ENGINE_INSECURE=false          # default: secure; 1 = dev opt-out
```

### Verify

```bash
python -m agent_utilities &

# Health stays open
curl -s localhost:9000/health

# Without a token: 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:9000/api/graph/query \
  -H 'content-type: application/json' -d '{"cypher":"MATCH (n) RETURN n LIMIT 1"}'
# -> 401

# With a valid JWT from your issuer: 200
TOKEN=$(curl -s -X POST "$AUTH_JWT_ISSUER/protocol/openid-connect/token" \
  -d grant_type=client_credentials -d client_id=agent-utilities \
  -d client_secret=REDACTED | jq -r .access_token)
curl -s -X POST localhost:9000/api/graph/query \
  -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"cypher":"MATCH (n) RETURN count(n) AS n"}'

# Engine secret was minted automatically
ls -l ~/.local/share/agent-utilities/engine_secret    # mode 0600
```

JWT validation (JWKS fetch/cache, claim mapping, 401 paths) is unit-tested;
**a live identity provider is not exercised in CI** — validate the issuer
wiring against your IdP once per environment. Worked example:
[identity-jwt](../examples/identity-jwt.md). Background:
[Secrets & auth](secrets-auth.md).

---

## Rung (c): Durable single node

**What you get:** everything from (b) plus durability that survives the host
process — and the schema/locking groundwork for multi-host. One flag,
`STATE_DB_URI`, externalizes ALL durable platform state (durable-execution
checkpoints, sessions/turns/goals, the KG task + staging queue) onto one
shared Postgres through a single connection pool (CONCEPT:OS-5.16–5.18,
[state externalization](../architecture/state_externalization.md)). The engine
remains the graph authority; turning on `GRAPH_BACKEND=fanout` with a
`GRAPH_DB_URI` adds an asynchronous Postgres/pg-age **mirror** for interop and DR
(never on the read path).
**What you don't get:** horizontal scale-out of ingest or agent execution
(still one host doing the work), autonomy.

What turns on, with no further flags:

- **`TASK_QUEUE_BACKEND` auto-resolution** — unset means auto: `postgres`
  when `STATE_DB_URI` is set, else `sqlite`
  (`knowledge_graph/core/queue_backend.py: create_task_queue`). The Postgres
  queue claims with `FOR UPDATE SKIP LOCKED` (at-least-once, 600 s visibility
  timeout). Setting `TASK_QUEUE_BACKEND=postgres` *explicitly* makes an
  unreachable database a fail-loud startup error instead of a logged SQLite
  fallback (CONCEPT:KG-2.55).
- **Daemon leadership** — `DaemonLeadership` (`core/leadership.py`) holds a
  Postgres session advisory lock per role; maintenance ticks (analysis,
  golden loop, failure ingest, fuseki publish, reconciler, autoscaler) become
  leader-only fleet-wide. A crashed leader's lock releases server-side and a
  follower takes over within one tick (CONCEPT:OS-5.17). Under SQLite this is
  a no-op (single host).

### Compose

`docker/pg-age.compose.yml` provisions a graph-enabled Postgres (ParadeDB)
on host port **5433**, database `agent_kg`, user/password `agent`/`agent`,
with init scripts from `docker/pg-age-init/`:

```bash
docker compose -f docker/pg-age.compose.yml up -d
# A separate logical DB for platform state keeps it apart from the graph mirror:
docker exec agent-pg-age psql -U agent -d agent_kg -c 'CREATE DATABASE agent_state'
```

Any Postgres you already run works the same way; the compose file is the
worked single-host example.

### `.env`

```dotenv
# ---- Everything from rung (b) ----
EPISTEMIC_GRAPH_AUTOSTART=1
OPENAI_API_KEY=sk-your-key
KG_AUTH_REQUIRED=1
AUTH_JWT_JWKS_URI=https://keycloak.example.internal/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://keycloak.example.internal/realms/agents
AUTH_JWT_AUDIENCE=agent-utilities
KG_BRAIN_ENFORCE=1

# ---- plus: durable state (OS-5.16) + async graph mirror ----

# ONE flag moves checkpoints + sessions/turns/goals + the KG task queue
# onto shared Postgres (unset = the per-host SQLite files of rungs a/b)
STATE_DB_URI=postgresql://agent:agent@localhost:5433/agent_state
#STATE_DB_POOL_SIZE=8              # default: max connections in the ONE shared pool

# Async Postgres/pg-age MIRROR of the engine graph (interop/BI/DR; off the read path)
GRAPH_BACKEND=fanout
GRAPH_MIRROR_TARGETS=postgresql
GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg

# Task queue: leave unset — auto resolves to postgres because STATE_DB_URI is
# set. Set explicitly to make a missing database fail loud at startup:
#TASK_QUEUE_BACKEND=postgres
```

### Verify

```bash
uv run graph-os-daemon --status     # or run the gateway; queue backend should be postgres

# State survives a restart: create a goal, restart, list goals
curl -s -X POST localhost:9000/api/goals -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"description":"durability probe"}'
# restart the gateway process, then:
curl -s localhost:9000/api/goals -H "authorization: Bearer $TOKEN"

# The live-Postgres integration pass (skipped without STATE_DB_URI):
STATE_DB_URI=postgresql://agent:agent@localhost:5433/agent_state \
  python -m pytest tests/integration/test_state_postgres_live.py
```

**Not exercised in CI:** the live-Postgres paths. CI runs the unit suites
(`tests/unit/test_state_store.py`, `tests/unit/test_durable_state_postgres.py`,
`tests/unit/knowledge_graph/test_queue_backend.py`) against fake
pools/connections; the live suite `tests/integration/test_state_postgres_live.py`
exercises the real SKIP LOCKED claims, advisory leadership, and schema, but
only runs when `STATE_DB_URI` points at a reachable Postgres — run it once
against your database.

Recipe form: [Single-node prod](../recipes/single-node-prod.md).

---

## Rung (d): Scaled multi-host

**What you get:** everything from (c) plus horizontal scale-out of all three
work planes — ingest (Kafka-partitioned task queue + `kg-ingest-worker`
fleet), agent execution (session-keyed `agent_turns` queue +
`agent-dispatch-worker` fleet), and the graph engine itself (N shards,
HRW-routed by tenant/graph) — with N gateway workers/replicas behind a load
balancer and Prometheus scraping every tier. **What you don't get:**
autonomy (rung e); automatic data re-sharding (adding/removing a shard
reassigns ~1/N of graphs but moves no data — migration is a manual snapshot
export/import, see [engine sharding](../architecture/engine_sharding.md)).

**Not exercised in CI.** No live Kafka broker, multi-shard engine topology,
or cross-host worker fleet runs in CI. The selection/routing/partitioning/
delivery contracts are covered by unit suites with injected transport fakes:
`tests/unit/knowledge_graph/test_kafka_ingest_scaleout.py`,
`tests/unit/knowledge_graph/test_engine_sharding.py`,
`tests/unit/test_agent_dispatch.py`. Validate this rung end-to-end in a
staging environment before relying on it.

### Compose

```bash
# Kafka (KRaft, single broker).
# Minimal — topics are created by the app's idempotent ensure-topic:
docker compose -f docker/docker-compose.kafka.yml up -d
# ...or fully provisioned (kg.* event topics, retention policies, tunable
# partitions via KG_TASKS_PARTITIONS):
docker compose -f docker/kafka-kraft.compose.yml up -d

# 3 engine shards, one shared HMAC secret (the engine binary refuses to start
# without one — there is deliberately no baked-in default):
export GRAPH_SERVICE_AUTH_SECRET="$(openssl rand -hex 32)"   # SAME everywhere
docker compose -f docker/engine-shards.compose.yml up -d
# shard RPC: tcp://:9101 :9102 :9103 — Prometheus /metrics: :9111 :9112 :9113
```

### `.env`

```dotenv
# ---- Everything from rung (c) ----
OPENAI_API_KEY=sk-your-key
KG_AUTH_REQUIRED=1
AUTH_JWT_JWKS_URI=https://keycloak.example.internal/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://keycloak.example.internal/realms/agents
AUTH_JWT_AUDIENCE=agent-utilities
KG_BRAIN_ENFORCE=1
STATE_DB_URI=postgresql://agent:agent@pg.example.internal:5433/agent_state
GRAPH_BACKEND=fanout
GRAPH_MIRROR_TARGETS=postgresql
GRAPH_DB_URI=postgresql://agent:agent@pg.example.internal:5433/agent_kg
# (EPISTEMIC_GRAPH_AUTOSTART dropped: remote tcp:// shards are never
#  auto-spawned — an unreachable shard is a fail-loud ConnectionError.)

# ---- plus: Kafka task queue (KG-2.55/2.56) ----
TASK_QUEUE_BACKEND=kafka            # explicit = fail-loud if the broker is down
KAFKA_BOOTSTRAP_SERVERS=kafka.example.internal:9092
#KG_TASKS_PARTITIONS=6              # default; ensured at startup, grow-only;
                                    # bounds kg-ingest consumer parallelism

# ---- plus: queue-driven agent dispatch (ORCH-1.45) ----
AGENT_DISPATCH_BACKEND=queue        # turns return a job handle; workers execute
#AGENT_TURNS_PARTITIONS=6           # default; bounds concurrent-session parallelism

# ---- plus: engine shards (KG-2.58 / OS-5.14) ----
# Comma-separated or JSON list. The list must be IDENTICAL (verbatim strings)
# on every client; order does not matter (HRW hashing).
GRAPH_SERVICE_ENDPOINTS=tcp://kg-shard-1.example.internal:9101,tcp://kg-shard-2.example.internal:9102,tcp://kg-shard-3.example.internal:9103
GRAPH_SERVICE_AUTH_SECRET=<the one shared secret>
#KG_DEFAULT_GRAPH=__bus__           # default; tenant-mapped before HRW routing

# ---- plus: gateway scale + observability (OS-5.23) ----
GATEWAY_WORKERS=4                   # pre-forked workers on ONE listen socket;
                                    # the flock elects ONE KG host among them
#GATEWAY_METRICS=true               # default: /metrics on the gateway
#GATEWAY_RATE_LIMIT=0               # per-tenant req/s; 0 = off (buckets per process)
#GATEWAY_RATE_BURST=0               # 0 = 2x rate
#ENGINE_BREAKER_THRESHOLD=5         # default; 0 = off
#ENGINE_BREAKER_COOLDOWN=15         # default, seconds
```

The same settings in `config.json` form (JSON list for the endpoints):

```jsonc
{
  "task_queue_backend": "kafka",
  "kafka_bootstrap_servers": "kafka.example.internal:9092",
  "agent_dispatch_backend": "queue",
  "graph_service_endpoints": [
    "tcp://kg-shard-1.example.internal:9101",
    "tcp://kg-shard-2.example.internal:9102",
    "tcp://kg-shard-3.example.internal:9103"
  ],
  "graph_service_auth_secret": "<the one shared secret>"
}
```

### Worker fleets

Run on any host that can reach Kafka, Postgres, and the shards. Both force
`KG_DAEMON_ROLE=client` (never contend for the KG host flock) and fail loud at
startup if they cannot reach the engine with the shared HMAC secret:

```bash
# Ingest workers (consumer group "kg-ingest"); worker count autosized from
# CPU/memory when --workers is omitted:
kg-ingest-worker --workers 4 --bootstrap-servers kafka.example.internal:9092

# Agent dispatch workers (consumer group "agent-dispatch"); default 1 thread —
# turns are LLM-bound:
agent-dispatch-worker --workers 2
```

### Load balancer (Caddy)

N gateway replicas (each `GATEWAY_WORKERS=1`) or one multi-worker gateway —
both are supported; see [gateway scaling](../architecture/gateway_scaling.md)
for the per-process state table (metrics registries, rate-limit buckets).

```caddyfile
agents.example.internal {
	reverse_proxy gw-1:9000 gw-2:9000 gw-3:9000 {
		health_uri /health
	}
}
```

### Prometheus scrape

`/metrics` registries are per-process — scrape each gateway replica directly
(not through the load balancer), plus each engine shard's metrics listener
(ports from `docker/engine-shards.compose.yml`):

```yaml
scrape_configs:
  - job_name: agent-utilities-gateway
    static_configs:
      - targets: ["gw-1:9000", "gw-2:9000", "gw-3:9000"]
  - job_name: epistemic-graph-shards
    static_configs:
      - targets: ["kg-shard-1.example.internal:9111",
                  "kg-shard-2.example.internal:9112",
                  "kg-shard-3.example.internal:9113"]
```

Key series: `agent_utilities_gateway_requests_total`,
`agent_utilities_engine_shard_up{endpoint}`,
`agent_utilities_dispatch_queue_depth`,
`agent_utilities_dispatch_turns_total{outcome}`,
`agent_utilities_dispatch_workers`. Full walkthrough:
[observability](../examples/observability.md).

### Verify

```bash
# Kafka topics exist with the right partition counts (created/grown at startup)
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --describe \
  --topic kg_tasks    # docker-compose.kafka.yml image; the kafka-kraft.compose.yml
                      # image uses /opt/kafka/bin/kafka-topics.sh

# Shard topology: per-shard reachability + breaker state
curl -s localhost:9000/api/dashboard/daemon/shards -H "authorization: Bearer $TOKEN"

# Dispatch fleet is visible (worker heartbeats in the topology)
curl -s localhost:9000/api/fleet/topology -H "authorization: Bearer $TOKEN"

# Metrics are flowing
curl -s gw-1:9000/metrics | grep agent_utilities_engine_shard_up
curl -s kg-shard-1.example.internal:9111/metrics | head
```

Worked examples: [sharding-walkthrough](../examples/sharding-walkthrough.md),
[queue-dispatch-walkthrough](../examples/queue-dispatch-walkthrough.md).
Deep dives: [engine sharding](../architecture/engine_sharding.md),
[agent dispatch](../architecture/agent_dispatch.md),
[event backbone](../architecture/event_backbone_architecture.md),
[capacity model](../scaling/capacity_model.md).

---

## Rung (e): Autonomous operations

**What you get:** everything from (d) plus the platform operating on itself:
the golden-loop research/remediation cycle, failure-driven evolution from
Langfuse telemetry, the desired-state fleet reconciler with a real actuator,
the reactive replica autoscaler, and webhook ingress for monitoring events.
Every mutating action still flows through the ONE ActionPolicy gate — the
shipped default policy queues all of it for human approval, so "autonomous"
is opt-in per action kind. **What you don't get:** unattended mutation out of
the box (you must relax the policy rule-by-rule), and auto-merge of evolution
proposals unless you explicitly enable `KG_GOLDEN_AUTO_MERGE`.

All ticks below run in the KG host daemon and are **leader-only** under
`STATE_DB_URI` (rung c) — exactly one host in the fleet runs them.

**Not exercised in CI.** The control logic (policy gate, reconciler diff,
autoscaler bounds, golden-loop stages) is unit-tested, but CI never runs a
live Langfuse, a real Docker actuator, or a live Prometheus signal source.
The shipped defaults are deliberately inert (`FLEET_ACTUATOR=dryrun`,
approval-required policy); treat every relaxation as a production change.

### `.env`

```dotenv
# ---- Everything from rung (d), plus: ----

# -- Golden loop: propose-only self-evolution (intake -> acquire -> distill) --
KG_LOOP=true
#KG_LOOP_INTERVAL=3600       # default, seconds
#KG_LOOP_TOPICS=5            # default: hot topics per tick
#KG_LOOP_DISTILL=false            # default; opt-in distillation stage
#KG_LOOP_BREADTH=false            # default; opt-in breadth scan
#KG_LOOP_STANDARDIZE=false        # default; opt-in standardization stage
KG_GOLDEN_AUTO_MERGE=false          # default; keep merges human-gated
#EVOLUTION_WORKTREE_ROOT=           # default: data_dir()/evolution_worktrees

# -- Failure-driven evolution (AHE-3.18): Langfuse failures -> remediation --
KG_FAILURE_EVOLUTION=true
#KG_FAILURE_EVOLUTION_INTERVAL=3600 # default, seconds
#KG_FAILURE_EVOLUTION_WINDOW=86400  # default: telemetry look-back, seconds
#KG_FAILURE_REGRESSION_DATASET=false # default; dataset-based regression path
LANGFUSE_HOST=https://langfuse.example.internal
LANGFUSE_PUBLIC_KEY=pk-lf-REDACTED
LANGFUSE_SECRET_KEY=sk-lf-REDACTED

# -- Fleet reconciler (OS-5.25): desired state vs observed, policy-gated --
FLEET_RECONCILER=true
#FLEET_RECONCILER_INTERVAL=120      # default, seconds
#FLEET_RECONCILER_MAX_ACTIONS=5     # default: storm guard per tick
#FLEET_REGISTRY_PATH=               # empty = shipped deploy/mcp-fleet.registry.yml
#FLEET_DESIRED_STATE_PATH=          # optional per-service replicas/version overlay
FLEET_ACTUATOR=docker               # default "dryrun" records intent, mutates NOTHING;
                                    # "docker" = reference CLI actuator.
                                    # Portainer/Swarm: set_fleet_actuator() seam (below)
#DEPLOY_WATCH_WINDOW=300            # default: post-deploy health watch, seconds
#DEPLOY_WATCH_POLL=15               # default: probe interval inside the watch

# -- Autoscaler (OS-5.29): load signal -> registry min/max -> policy gate --
FLEET_AUTOSCALER=true
#FLEET_AUTOSCALER_INTERVAL=60       # default, seconds
SCALING_PROMETHEUS_URL=http://prometheus.example.internal:9090
                                    # unset = zero-infra in-process gauges

# -- ActionPolicy (OS-5.24): the single autonomy decision point --
ACTION_POLICY_PATH=/etc/agent-utilities/action-policy.yml
# Empty (default) = the shipped conservative policy
# (deploy/action-policy.default.yml): every mutating kind approval_required,
# only diagnose/observe/notify/record_dry_run auto. KG governance_rule
# overrides (scope: action_policy) win over file rules either way.

# -- Monitoring webhook ingress (OS-5.15) --
FLEET_EVENTS_TOKEN=<openssl rand -hex 32>
# Shared secret for POST /api/fleet/events (header X-Fleet-Events-Token,
# constant-time compare, re-read per request so rotation needs no restart).
# Default unset = the endpoint accepts unauthenticated posts.
```

### Custom actuator (Portainer/Swarm)

`FLEET_ACTUATOR` selects between `dryrun` and the reference `docker` CLI
actuator; anything else is deployment-wired through the seam:

```python
from agent_utilities.orchestration.fleet_actuation import set_fleet_actuator

set_fleet_actuator(MyPortainerActuator())   # real actuation behind the policy gate
```

### Verify

```bash
# Reconciler/autoscaler proposals land in the approval queue (default policy
# queues every mutating action):
curl -s localhost:9000/api/fleet/approvals -H "authorization: Bearer $TOKEN"

# Webhook ingress: rejected without the token...
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:9000/api/fleet/events \
  -H 'content-type: application/json' -d '{"alerts":[]}'
# -> 401
# ...accepted with it:
curl -s -X POST 'localhost:9000/api/fleet/events?source=alertmanager' \
  -H "x-fleet-events-token: $FLEET_EVENTS_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"alerts":[{"status":"firing","labels":{"alertname":"probe"}}]}'

# Daemon ticks are registered (leader host):
uv run graph-os-daemon --status
```

Worked examples:
[action-policy-postures](../examples/action-policy-postures.md) (relaxing the
default posture rule-by-rule),
[fleet-events-wiring](../examples/fleet-events-wiring.md) (Alertmanager /
Uptime Kuma / Portainer payloads),
[autoscaling-signals](../examples/autoscaling-signals.md),
[evolution-publication](../examples/evolution-publication.md) (how promoted
proposals become reviewable local branches). Deep dives:
[fleet autonomy](../architecture/fleet_autonomy.md),
[failure-driven evolution](../architecture/failure_driven_evolution.md),
[autonomous evolution guide](autonomous-evolution.md).

Recipe form: [Enterprise](../recipes/enterprise.md).

---

## Where to go next

- Flag-by-flag inventory and the configuration-discipline rule:
  [Configuration Reference & Flag Audit](../architecture/configuration.md)
- Worked end-to-end examples: [ontology-to-workflow](../examples/ontology-to-workflow.md),
  [identity-jwt](../examples/identity-jwt.md),
  [observability](../examples/observability.md),
  [sharding-walkthrough](../examples/sharding-walkthrough.md),
  [queue-dispatch-walkthrough](../examples/queue-dispatch-walkthrough.md),
  [fleet-events-wiring](../examples/fleet-events-wiring.md),
  [action-policy-postures](../examples/action-policy-postures.md),
  [autoscaling-signals](../examples/autoscaling-signals.md),
  [evolution-publication](../examples/evolution-publication.md),
  [mcp-consumption](../examples/mcp-consumption.md)
- Architecture deep dives:
  [state externalization](../architecture/state_externalization.md),
  [engine sharding](../architecture/engine_sharding.md),
  [gateway scaling](../architecture/gateway_scaling.md),
  [agent dispatch](../architecture/agent_dispatch.md),
  [fleet autonomy](../architecture/fleet_autonomy.md),
  [graph service layer](../architecture/graph_service_layer.md)
- Sizing: [capacity model](../scaling/capacity_model.md)
