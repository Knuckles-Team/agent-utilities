# Supported Deployment Configurations

This is the flagship configuration ladder for agent-utilities: five complete,
copy-paste-ready configurations, from a laptop with no separately managed graph
infrastructure to an autonomous multi-host fleet. Each rung builds on the
previous one and states its delta explicitly.

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

| Rung | Durable state | WorkItem transport | Agent dispatch | Auth | Engines | Autonomy |
|---|---|---|---|---|---|---|
| [(a) Zero-infra dev](#rung-a-zero-infra-dev) | per-host SQLite | `sqlite` (auto) | `inline` (default) | neutral one-time in-memory proof for exact tiny local stdio; external identity for every other boundary; engine HMAC | 1 supervised local engine | off; ActionPolicy = approval-required default |
| [(b) Secured single node](#rung-b-secured-single-node) | per-host SQLite | `sqlite` (auto) | `inline` | process JWT + per-request Bearer JWT + engine HMAC + brain enforcement | 1 local engine, HMAC | off |
| [(c) Durable single node](#rung-c-durable-single-node) | shared Postgres (`STATE_DB_URI`) | `postgres` (auto) | `inline` | as (b) | 1 local engine | off |
| [(d) Scaled multi-host](#rung-d-scaled-multi-host) | shared Postgres | `kafka` (explicit) | `queue` + worker fleet | as (b) | 3-member cell, catalog-routed MultiRaft groups | off |
| [(e) Autonomous operations](#rung-e-autonomous-operations) | shared Postgres | `kafka` | `queue` | as (b) | production cell | golden loop, failure-driven evolution, fleet reconciler, autoscaler, event webhook |

## How configuration is loaded

Three layers, in precedence order:

1. **Explicit process environment** — every flag is a typed field on
   `AgentConfig` (`agent_utilities/core/config.py`); the env alias is the
   flag's canonical name. The CI gate `scripts/check_no_env_sprawl.py` keeps
   new flags on `AgentConfig` instead of scattered `os.environ` reads. AgentConfig
   never searches a repository or launch directory for a dotenv file.
2. **XDG `config.json`** — discovered under
   `$XDG_CONFIG_HOME/agent-utilities/config.json` (override the directory with
   `AGENT_UTILITIES_CONFIG_DIR`; resolution lives in
   `agent_utilities/core/paths.py`). Each key is upper-cased to its env name
   and applied **only if that env var is not already set** — so the
   environment always wins. Lists/dicts are JSON-encoded (this is how
   `chat_models` / `embedding_models` registries are normally configured).
   The optional fixed sibling `runtime-secrets.json` projects only keys targeted
   by exact `env://` references; it never overrides an explicit process value.
   A template `config.json` is written on first run if none exists.
   `GRAPH_SERVICE_ENDPOINTS` and other list flags accept either a JSON list
   (natural in `config.json`) or a comma-separated explicit process value.
3. **Defaults** — the values baked into `AgentConfig`. The zero-infra rung uses
   defaults for graph topology and local persistence. Only tiny packaged-local
   GraphOS over stdio may derive its neutral process authority in memory; every
   other boundary requires explicit external identity.

Data lives under XDG paths: config `$XDG_CONFIG_HOME/agent-utilities/`, data
`$XDG_DATA_HOME/agent-utilities/` (`AGENT_UTILITIES_DATA_DIR` override), and
cache `$XDG_CACHE_HOME/agent-utilities/`. The runtime supplies platform defaults
when an XDG variable is unset.

The relevant processes (console scripts from `pyproject.toml
[project.scripts]`):

| Command | Role |
|---|---|
| `graph-os` | MCP tool surface (stdio default; `--transport streamable-http --port <p>` for HTTP; compose: `docker/mcp.compose.yml`, port 8004) |
| `graph-os-daemon` | Standalone KG host daemon: holds the flock host lock, drains the durable task queue, runs all maintenance/autonomy ticks. `--status` and `--drain-queue` flags. It serves no HTTP. |
| `python -m agent_utilities` | The REST gateway (FastAPI agent server): `/health`, `/api/graph/*`, `/api/sessions`, `/api/goals`, `/api/fleet/*`, `/api/dashboard/*`, `/metrics`. Binds `HOST`:`PORT` (defaults `0.0.0.0`:`9000`). `GATEWAY_WORKERS` pre-forks it. When it runs, it hosts the KG daemon itself (flock-elected) — you do not also need `graph-os-daemon`. |
| `kg-ingest-worker` | Decoupled ingest consumer (`kg-ingest` group), engine client only (AU-KG.ingest.decoupled-kg-ingest-consumer) |
| `agent-dispatch-worker` | Stateless agent-turn consumer (`agent-dispatch` group), engine client only (ORCH-1.45) |

---

## Rung (a): Zero-infra dev

**What you get:** the full platform on one machine with no separately managed
graph database or engine service. The knowledge graph runs on the default
`epistemic_graph` backend — the
`epistemic-graph` Rust engine is the one database (the authority), providing
durable persistence, in-memory cache, graph compute, and ontology reasoning in a
packaged, supervised out-of-process engine with no mirrors — all platform state in per-host
SQLite files under the XDG data dir, agent turns queued to a local dispatch
worker, a verified neutral process session for the exact local stdio boundary,
and authenticated engine traffic. GraphOS signs and validates a short-lived JWT
with an in-memory key as a one-time proof, destroys both, and returns a
process-lifetime session. It is not anonymous and contains no personal, host,
endpoint, filesystem, credential, or proof identity. **What you don't get:** durability
beyond this host, more than one host, or autonomous operations (all autonomy flags default off;
the shipped ActionPolicy marks every mutating operational action
`approval_required`).

### AgentConfig projection

Generate the XDG `config.json` with the explicit tiny deployment profile. The
zero-infrastructure exception is only `graph-os --transport stdio` with no
engine endpoints and no external process-identity source. The
`KEY=value` notation below is documentation shorthand for AgentConfig aliases,
not a file to create in the repository. Put non-secret policy values and secret
references in AgentConfig; concrete token and client-secret material belongs only
in the runtime secret system:

```text
DEPLOYMENT_PROFILE=tiny
# Keep GRAPH_SERVICE_ENDPOINTS unset so GraphOS owns the packaged engine.
# Keep KG_AUTH_TOKEN_REF and KG_IDENTITY_OAUTH2 unset so exact stdio startup
# uses the neutral in-memory authority. This exception applies to no other entrypoint.

# With GRAPH_SERVICE_ENDPOINTS unset, graph-os owns the packaged local Rust
# engine lifecycle. The `epistemic-graph-server` binary ships with the wheel;
# there is no in-process fallback.

# Model provider — only needed for LLM-backed features (agents, enrichment).
# Graph operations run without it.
CHAT_MODELS=[{"id":"chat-model","provider":"openai","api_key_ref":"vault://platform/llm#api_key"}]
# Other providers use the same per-model api_key_ref shape. Raw provider keys
# are explicit process-only inputs and cannot be durable AgentConfig values.

# -- Graph/runtime defaults, listed explicitly (do not need to be set) --
# The engine IS the fixed database authority; no selector is required.
#KG_DAEMON_ROLE=auto               # flock host-lock elects ONE host per machine
#STATE_DB_URI=                     # unset = per-host SQLite state files
#TASK_QUEUE_BACKEND=               # unset = auto: sqlite (no STATE_DB_URI)
# Agent dispatch is queue-only; TASK_QUEUE_BACKEND selects its transport.
# Tenant, ACL, session, and single-client enforcement are baked in.
#GATEWAY_WORKERS=1
#GATEWAY_METRICS=true              # /metrics on the gateway
#KG_DEV_MODE=false                 # false = all maintenance daemons on
#FLEET_RECONCILER=false
#FLEET_AUTOSCALER=false
#KG_LOOP=false
#KG_FAILURE_EVOLUTION=false
#ACTION_POLICY_PATH=               # empty = shipped conservative default policy
```

Before starting a daemon, library graph surface, REST gateway, network MCP
transport, non-tiny profile, or any GraphOS connected to explicit engine
endpoints, configure exactly one `KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2`
plus `AUTH_JWT_JWKS_URI`, audience, issuer, and `KG_POLICY_VERSION`. An explicit
source that cannot be acquired or validated never falls back to the local
authority.

### `config.json`

Generate the XDG file with `setup-config`; do not add an external process identity
for the exact tiny packaged-local stdio path. A model registry
(`chat_models` / `embedding_models`) is optional for graph-only operations; see
[`docs/examples/config.json`](../examples/config.json). Run
`agent-utilities-doctor --only graph_identity auth` before launch. Add `secrets`
to that doctor selection when validating an external identity source.

### Process and engine auth on this rung

For the exact tiny packaged-local stdio boundary, GraphOS generates an asymmetric
key in memory, signs a short-lived JWT with fixed neutral service claims as a
one-time proof, validates it through the normal decoder, destroys the private key
and token, and returns the immutable process-lifetime graph session. Every other
boundary resolves exactly one runtime process identity and validates it against
the configured JWKS/issuer/audience. Missing, ambiguous, unresolved, or invalid
external identity aborts startup without local fallback.
Separately, engine traffic is authenticated: when
`GRAPH_SERVICE_AUTH_SECRET` is unset, a per-install HMAC secret is minted once
and persisted as `engine_secret` in the discovered XDG data directory (mode 0600,
`knowledge_graph/core/graph_compute.py`), and every local process — including
any engine this install spawns — agrees on it (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).
The native transport has no unauthenticated development mode.

### Verify

```bash
# 1. Prove the neutral local authority boundary before launch.
agent-utilities-doctor --only graph_identity auth

# 2. MCP surface over stdio (register the already-installed command in your IDE).
#    { "mcpServers": { "graph-os": { "command": "graph-os", "args": ["--transport", "stdio"] } } }

# Daemon, library, REST, network MCP, and explicit-endpoint verification belongs
# to rung (b): configure its external process identity first.
```

Recipe form: [Tiny](../recipes/tiny.md). MCP consumption patterns:
[mcp-consumption](../examples/mcp-consumption.md).

---

## Rung (b): Secured single node

**What you get:** everything from (a) plus a network-serving identity perimeter.
The process JWT remains mandatory, and every REST or HTTP MCP graph request must
present its own validated Bearer JWT; the server never substitutes process
authority for a network caller. This rung also enables brain enforcement and
fail-closed node-level permissions. **What you don't get:** durability beyond
this host, scale-out, or autonomy.

The pieces (all CONCEPT:AU-OS.identity.authenticated-identity-enforcement, validated in
`agent_utilities/security/auth.py` and
`agent_utilities/security/request_identity.py`):

- **Engine HMAC** — automatic per-install secret as in rung (a); set
  `GRAPH_SERVICE_AUTH_SECRET` explicitly only when multiple installs/hosts
  must share one engine.
- **Verified GraphSession identity** — `ActorIdentityMiddleware` validates
  `Authorization: Bearer <JWT>` against `AUTH_JWT_JWKS_URI` (JWKS cached 5
  minutes), checks the required `AUTH_JWT_AUDIENCE` plus the pinned issuer, and mints
  the server-side `ActorContext` from the claims (`sub`/`client_id`/`azp` →
  actor, `roles`/`realm_access.roles`/`scope` → roles,
  `tenant_id`/`tenant`/`org_id`/`tid` → tenant). Missing or invalid credentials
  get 401; only health paths (`/health`, `/healthz`, `/api/health`,
  `/api/healthz`) stay open, returning only `{"status":"ok"}` with
  `Cache-Control: no-store`; readiness and topology detail remain authenticated.
  Caller-supplied `_actor`/`_roles`/`_tenant` kwargs are rejected.
  `KG_POLICY_VERSION` is required and stamped into the session. Only an explicit
  or identity-mapped `kg:admin` capability grants graph administration; a generic
  `admin` application role does not.
- **`KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2`** — exactly one stdio process
  identity source. The first resolves a provisioned JWT from the secret manager;
  the second mints a short-lived JWT with client credentials whose secret is
  itself a runtime secret reference. Both are validated against the same JWKS.
  The validated expiry is shared with all process workers in an in-memory lease
  that contains no token or identity. Renewal accepts only the same subject,
  actor type, capabilities, tenant, authentication state, and groups. Drift is
  rejected; failed renewal retries without extending the lease, and all graph
  work fails closed at expiry.
- **Mandatory graph authority** — every operation inherits the server-verified
  actor, tenant, scopes, audience, and policy revision from its ambient
  `GraphSession`. Tenant/ACL enforcement is always active, missing ACLs deny,
  and authorization failures never return unfiltered data. These are compiled
  contracts, not deployment flags.

Declaring `APP_PROFILE=production` also makes transport and execution bypasses
fatal at startup. Certificate verification must remain enabled, remote engine
TCP must be TLS protected, and unauthenticated/wildcard-host/developer escape
hatches must remain disabled. Private PKI is supported through a named TLS
profile or secret-backed CA bundle; production does not require public roots and
must not use a hardcoded `verify=false` exception.

`APP_PROFILE` is only that runtime safety posture; it does not identify a
deployment topology. `DEPLOYMENT_PROFILE` must name exactly `tiny`,
`single-node-prod`, or `enterprise`, and every generated configuration persists
it. Because both production topologies use `APP_PROFILE=production`, the config
doctor refuses a production posture with no explicit deployment identity instead
of guessing that it is enterprise.

### AgentConfig projection

```text
# ---- Everything from rung (a) ----
CHAT_MODELS=[{"id":"chat-model","provider":"openai","api_key_ref":"vault://platform/llm#api_key"}]

# ---- plus: identity & enforcement (OS-5.14) ----

# Required server-validated JWT identity on the KG REST/MCP surface
AUTH_JWT_JWKS_URI=https://identity.example.test/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://identity.example.test/realms/agents
AUTH_JWT_AUDIENCE=agent-utilities
KG_POLICY_VERSION=policy-v1

# Identity for stdio MCP processes (no Authorization header on stdio).
# Configure exactly one source; this example resolves a provisioned JWT.
KG_AUTH_TOKEN_REF=secret://graph-os/stdio-token

# Alternatively configure KG_IDENTITY_OAUTH2 as a JSON client-credentials block
# whose client_secret is a runtime secret reference.

# Engine HMAC is automatic (per-install secret at data_dir()/engine_secret).
# Set explicitly only to share one engine across installs:
#GRAPH_SERVICE_AUTH_SECRET=${GRAPH_SERVICE_AUTH_SECRET_FROM_SUPERVISOR}
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
ls -l "$XDG_DATA_HOME/agent-utilities/engine_secret"    # mode 0600
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
shared Postgres through a single connection pool (CONCEPT:AU-OS.state.unified-durable-state-externalization–5.18,
[state externalization](../architecture/state_externalization.md)). The engine
remains the graph authority; declaring `GRAPH_MIRROR_TARGETS` with a
`GRAPH_DB_CONNECTION_PROFILE_REF` adds an asynchronous Postgres/pg-age **projection** for interop and DR
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
  fallback (CONCEPT:AU-KG.backend.selectable-queue-backend).
- **Daemon leadership** — `DaemonLeadership` (`core/leadership.py`) holds a
  Postgres session advisory lock per role; maintenance ticks (analysis,
  golden loop, failure ingest, fuseki publish, reconciler, autoscaler) become
  leader-only fleet-wide. A crashed leader's lock releases server-side and a
  follower takes over within one tick (CONCEPT:AU-OS.state.cross-host-daemon-leadership). Under SQLite this is
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

### AgentConfig projection

```text
# ---- Everything from rung (b) ----
CHAT_MODELS=[{"id":"chat-model","provider":"openai","api_key_ref":"vault://platform/llm#api_key"}]
AUTH_JWT_JWKS_URI=https://identity.example.test/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://identity.example.test/realms/agents
AUTH_JWT_AUDIENCE=agent-utilities
KG_POLICY_VERSION=policy-v1

# ---- plus: durable state (AU-OS.state.unified-durable-state-externalization) + async graph mirror ----

# ONE flag moves session/turn/fleet metadata + queue delivery onto shared
# Postgres (unset = per-host SQLite support stores). WorkItem checkpoints stay
# in epistemic-graph.
STATE_DB_URI=vault://platform/state#profile
#STATE_DB_POOL_SIZE=8              # default: max connections in the ONE shared pool

# Async Postgres/pg-age MIRROR of the engine graph (interop/BI/DR; off the read path)
GRAPH_MIRROR_TARGETS=postgresql
GRAPH_DB_CONNECTION_PROFILE_REF=vault://platform/graph#profile

# WorkItem notification transport: leave unset — auto resolves to postgres because STATE_DB_URI is
# set. Set explicitly to make a missing database fail loud at startup:
#TASK_QUEUE_BACKEND=postgres
```

### Verify

```bash
graph-os-daemon --status     # or run the gateway; queue backend should be postgres

# State survives a restart: create a goal, restart, list goals
curl -s -X POST localhost:9000/api/goals -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"description":"durability probe"}'
# restart the gateway process, then:
curl -s localhost:9000/api/goals -H "authorization: Bearer $TOKEN"

# The live-Postgres integration pass (skipped without STATE_DB_URI):
STATE_DB_URI="$STATE_DB_URI" \
  python -m pytest tests/integration/test_state_postgres_live.py
```

**Not exercised in CI:** the live-Postgres paths. CI runs the unit suites
(`tests/unit/test_state_store.py`,
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
`agent-dispatch-worker` fleet), and the graph engine itself (a replicated cell
whose catalog assigns tenant graphs to fenced MultiRaft groups) — with N gateway workers/replicas behind a load
balancer and Prometheus scraping every tier. **What you don't get:**
autonomy (rung e). Placement movement is governed by the engine catalog and
online reshard workflow; clients never reassign data from an endpoint-list hash.

**Not exercised in CI.** No live Kafka broker, multi-member engine cell,
or cross-host worker fleet runs in CI. The selection/routing/partitioning/
delivery contracts are covered by unit suites with injected transport fakes:
`tests/unit/knowledge_graph/test_kafka_ingest_scaleout.py`,
`tests/unit/knowledge_graph/test_engine_sharding.py`,
`tests/unit/test_agent_dispatch.py`. Validate this rung end-to-end in a
staging environment before relying on it.

### Production cell

```bash
# Kafka (KRaft, single broker).
# Minimal — topics are created by the app's idempotent ensure-topic:
docker compose -f docker/docker-compose.kafka.yml up -d
# ...or fully provisioned (kg.* event topics, retention policies, tunable
# partitions via KG_TASKS_PARTITIONS):
docker compose -f docker/kafka-kraft.compose.yml up -d

# Render only an exact, signed, compatibility-checked release manifest.
check-graphos-compatibility --manifest RELEASE_MANIFEST
python scripts/release/render_production_cell.py \
  --manifest RELEASE_MANIFEST --output RENDERED_DIRECTORY
python scripts/deployment/check_production_assets.py \
  --directory RENDERED_DIRECTORY --rendered
kubectl apply -k RENDERED_DIRECTORY
```

`docker/engine-shards.compose.yml` remains a compatibility fixture for independent
development engines. It does not provide Raft replication, authoritative movement,
or production availability and must not be used as this rung's data plane.

### AgentConfig projection

```text
# ---- Everything from rung (c) ----
CHAT_MODELS=[{"id":"chat-model","provider":"openai","api_key_ref":"vault://platform/llm#api_key"}]
AUTH_JWT_JWKS_URI=https://identity.example.test/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://identity.example.test/realms/agents
AUTH_JWT_AUDIENCE=agent-utilities
KG_POLICY_VERSION=policy-v1
STATE_DB_URI=vault://platform/state#profile
GRAPH_MIRROR_TARGETS=postgresql
GRAPH_DB_CONNECTION_PROFILE_REF=vault://platform/graph#profile
# ---- plus: Kafka task queue (KG-2.55/2.56) ----
TASK_QUEUE_BACKEND=kafka            # explicit = fail-loud if the broker is down
KAFKA_BOOTSTRAP_SERVERS=kafka.example.test:9092
#KG_TASKS_PARTITIONS=6              # default; ensured at startup, grow-only;
                                    # bounds kg-ingest consumer parallelism

# ---- plus: queue-driven agent dispatch (ORCH-1.45) ----
# Turns always return a job handle; deploy agent-dispatch-worker processes.
#AGENT_TURNS_PARTITIONS=6           # default; bounds concurrent-session parallelism
#AGENT_DISPATCH_MAX_DEPTH=100000    # default; fail-closed durable admission bound
#AGENT_DISPATCH_CLAIM_TTL_S=120     # default; capped at the 300s recovery objective
#AGENT_DISPATCH_RENEW_INTERVAL_S=30 # default; periodic fenced renewal

# ---- plus: engine placement ----
# All clients use one stable coordinator. Placement group, endpoint, epoch, and
# fencing token come from the engine catalog and stale routes refresh once.
GRAPH_SERVICE_ENDPOINTS=tls://kg-coordinator.example.test:9100
ENGINE_TLS_PROFILE_REF=vault://platform/engine/tls-profile
# Only a non-production topology exposing separate group listeners sets this
# explicit JSON map; an endpointless route without a map fails closed.
#GRAPH_RAFT_GROUP_ENDPOINTS={"0":"tls://kg-group-0.example.test:9100"}
# A concrete value is injected by the deployment supervisor at process start.
GRAPH_SERVICE_AUTH_SECRET=${GRAPH_SERVICE_AUTH_SECRET_FROM_SUPERVISOR}
#KG_DEFAULT_GRAPH=__bus__           # default; policy-mapped before catalog lookup

# ---- plus: gateway scale + observability (AU-OS.observability.no-op-without-metrics) ----
GATEWAY_WORKERS=4                   # pre-forked workers on ONE listen socket;
                                    # the flock elects ONE KG host among them
#GATEWAY_METRICS=true               # default: /metrics on the gateway
#GATEWAY_RATE_LIMIT=0               # per-tenant req/s; 0 = off (buckets per process)
#GATEWAY_RATE_BURST=0               # 0 = 2x rate
#ENGINE_BREAKER_THRESHOLD=5         # default; 0 = off
#ENGINE_BREAKER_COOLDOWN=15         # default, seconds
```

The same settings in `config.json` form:

```jsonc
{
  "task_queue_backend": "kafka",
  "kafka_bootstrap_servers": "kafka.example.test:9092",
  "agent_dispatch_backend": "queue",
  "graph_service_endpoints": [
    "tls://kg-coordinator.example.test:9100"
  ],
  "engine_tls_profile_ref": "vault://platform/engine/tls-profile",
  "graph_service_auth_secret": "<the one shared secret>"
}
```

### Worker fleets

Run on any host that can reach Kafka, Postgres, and the cell coordinator. Both force
`KG_DAEMON_ROLE=client` (never contend for the KG host flock) and fail loud at
startup if they cannot reach the engine with the shared HMAC secret:

```bash
# Ingest workers (consumer group "kg-ingest"); worker count autosized from
# CPU/memory when --workers is omitted:
kg-ingest-worker --workers 4 --bootstrap-servers kafka.example.test:9092

# Agent dispatch workers (consumer group "agent-dispatch"); default 1 thread —
# turns are LLM-bound:
agent-dispatch-worker --workers 2
```

### Load balancer (Caddy)

N gateway replicas (each `GATEWAY_WORKERS=1`) or one multi-worker gateway —
both are supported; see [gateway scaling](../architecture/gateway_scaling.md)
for the per-process state table (metrics registries, rate-limit buckets).

```caddyfile
agents.example {
	reverse_proxy gw-1:9000 gw-2:9000 gw-3:9000 {
		health_uri /health
	}
}
```

### Prometheus scrape

`/metrics` registries are per-process — scrape each gateway replica directly
(not through the load balancer), plus every engine member's native metrics listener:

```yaml
scrape_configs:
  - job_name: agent-utilities-gateway
    static_configs:
      - targets: ["gw-1:9000", "gw-2:9000", "gw-3:9000"]
  - job_name: epistemic-graph-members
    static_configs:
      - targets: ["kg-member-0.example.test:9110",
                  "kg-member-1.example.test:9110",
                  "kg-member-2.example.test:9110"]
```

Key series: `agent_utilities_gateway_requests_total`, native
`epistemic_graph_*` Raft/WAL/checkpoint health,
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

# Client endpoint topology: coordinator reachability + breaker state. This legacy
# view is not the Raft member inventory; use native engine placement/Raft health.
curl -s localhost:9000/api/dashboard/daemon/shards -H "authorization: Bearer $TOKEN"

# Dispatch fleet is visible (worker heartbeats in the topology)
curl -s localhost:9000/api/fleet/topology -H "authorization: Bearer $TOKEN"

# Metrics are flowing
curl -s gw-1:9000/metrics | grep agent_utilities_engine_shard_up
curl -s https://kg-member-0.example.test:9110/metrics | grep epistemic_graph_ | head
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

### AgentConfig projection

```text
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

# -- Failure-driven evolution (AU-AHE.harness.failure-evolution): Langfuse failures -> remediation --
# Auto-enabled by the two Langfuse credentials below; set false to opt out.
#KG_FAILURE_EVOLUTION=true
#KG_FAILURE_EVOLUTION_INTERVAL=3600 # default, seconds
#KG_FAILURE_EVOLUTION_WINDOW=86400  # default: telemetry look-back, seconds
#KG_FAILURE_REGRESSION_DATASET=false # default; dataset-based regression path
LANGFUSE_HOST=https://langfuse.example.test
LANGFUSE_PUBLIC_KEY_REF=vault://platform/langfuse#public_key
LANGFUSE_SECRET_KEY_REF=vault://platform/langfuse#secret_key

# -- Fleet reconciler (AU-OS.config.desired-state-fleet-reconciler): desired state vs observed, policy-gated --
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
SCALING_PROMETHEUS_URL=https://prometheus.example.test:9090
                                    # unset = zero-infra in-process gauges

# -- ActionPolicy (OS-5.24): the single autonomy decision point --
ACTION_POLICY_PATH=$XDG_CONFIG_HOME/agent-utilities/action-policy.yml
# Empty (default) = the shipped conservative policy
# (deploy/action-policy.default.yml): every mutating kind approval_required,
# only diagnose/observe/notify/record_dry_run auto. KG governance_rule
# overrides (scope: action_policy) win over file rules either way.

# -- Monitoring webhook ingress (AU-OS.config.fleet-event-ingress) --
FLEET_EVENTS_TOKEN_REF=<secret-provider-reference>
# Shared secret for POST /api/fleet/events (header X-Fleet-Events-Token,
# constant-time compare, re-read per request so rotation needs no restart).
# Default unset = unauthenticated callers are rejected.
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
  -H "x-fleet-events-token: <runtime-resolved-token>" \
  -H 'content-type: application/json' \
  -d '{"alerts":[{"status":"firing","labels":{"alertname":"probe"}}]}'

# Daemon ticks are registered (leader host):
graph-os-daemon --status
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
