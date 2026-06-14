# Enterprise Enablement Runbook — turning the built platform on

> **Why this exists.** The enterprise scale-out and autonomy capabilities are **already
> built and merged**, but they ship **off by default** so the zero-infra laptop
> experience stays byte-for-byte unchanged. Getting to "running enterprise agent OS" is
> therefore an **operational** sequence — push → deploy → enable flags under a chosen
> autonomy posture — not an engineering project. This runbook is that sequence.

Each stage is independently reversible (unset the flag → previous behavior). Enable them
**in order**: security first (so multi-tenant is safe before you scale it), then state
externalization (so the stateless tiers can multiply), then sharding, then the brain,
then autonomy. Verify each stage before the next.

---

## Stage A — Publish (push the locally-merged work)

Everything is merged to local `main` across the repos but **not pushed**. Releases MUST
use the phased path so dependents never build before an upstream version is on PyPI.

```bash
# from the workspace-validator / repository-manager
auto_push --phased            # waits per workspace.yml: each dep phase's CI → PyPI → next
```

- **Do not** bare-`git push` every repo at once — that races publish order (dependents
  build against a not-yet-published upstream version).
- Order is driven by `workspace.yml` dependency phases; `agent-utilities` and
  `epistemic-graph` publish before the connector/surface repos that depend on them.
- **Verify:** each phase's CI is green and the new version is visible on PyPI / the image
  is on the registry before the next phase starts.

## Stage B — Deploy the engine + gateway

1. **Swap the `epistemic-graph` engine binary** on the canonical host and restart the
   daemon. The new binary **refuses an empty auth secret** — set
   `GRAPH_SERVICE_AUTH_SECRET` (or the gateway mints/sets it) *before* restart or the
   engine will not start.
2. **Restart the gateway** (`python -m agent_utilities`) so it picks up the new
   `agent_utilities` release and middleware (identity, rate-limit, metrics).
3. **Verify:** `GET /metrics` returns `agent_utilities_*` series; the engine answers a
   `graph_query` over the socket; `multiplexer_status` is healthy.

## Stage C — Security gate (enable FIRST of the flags) — OS-5.14

This is the 06-10 review's #1 blocker. The platform supports exactly **one trust domain**
until these are on.

| Flag | Effect |
|---|---|
| `GRAPH_SERVICE_AUTH_SECRET` | HMAC engine auth (already required by the new binary) |
| `KG_AUTH_REQUIRED=true` | gateway rejects unauthenticated requests with 401 |
| `KG_AUTH_TOKEN` / JWKS (`auth_jwt_jwks_uri`, `auth_jwt_issuer`, `auth_jwt_audience`) | server mints `ActorContext` from a validated JWT |
| `KG_ACL_DEFAULT_ALLOW=false` | permission checks **fail closed** |

- **Verify:** an unauthenticated request → 401; a request with a valid JWT scopes to the
  right tenant; a cross-tenant read is denied (not silently allowed).

## Stage D — Externalize state (multiply stateless tiers) — OS-5.16–18, KG-2.54

```bash
export STATE_DB_URI=postgresql://agent:agent@pg-age.arpa:5432/agent_kg
```

Moves durable-execution checkpoints, sessions/turns/goals, and the KG task queue off
per-host SQLite onto shared Postgres — with `SKIP LOCKED` queue claims and advisory-lock
daemon leadership (singleton ticks run on exactly one host). Now you can run **N gateway
replicas** behind Caddy and they share state.

- **Verify:** two gateway replicas claim queue items without double-processing; kill the
  leader → another host acquires the advisory lock and resumes background ticks; a
  running goal rehydrates as `orphaned` (not lost) after a restart.

## Stage E — Shard the engines (scale the graph) — KG-2.58 / OS-5.28

```bash
export GRAPH_SERVICE_ENDPOINTS=unix:///run/eg-0.sock,unix:///run/eg-1.sock,unix:///run/eg-2.sock
```

With 2+ endpoints, clients route each named graph to its owning shard via HRW
(rendezvous) hashing. Use the `docker/engine-shards.compose.yml` 3-shard recipe.

- **Verify:** the gateway `daemon/shards` route shows all shards reachable; a query to a
  graph lands on its owning shard; an unreachable shard fails loud (not silent).

## Stage F — Company Brain (governed retrieval) — KG_BRAIN_ENFORCE

```bash
export KG_BRAIN_ENFORCE=true
```

Turns on the 6-layer brain: source-authority conflict resolution with trust decay,
field-level survivorship, data-level ACLs + tenant scoping + read audit, and the
human-correction → durable rule → eval feedback loop. **Off, the brain is legacy
byte-for-byte** — so enable only after Stage C (identity) is verified.

- **Verify:** conflicting source values resolve to the higher-trust source; an ACL-marked
  field is dropped for an unauthorized reader; a human correction persists as a rule.

## Stage G — Autonomy loops (under an ActionPolicy posture) — AHE-3.18–21, OS-5.15/5.24–29

The control plane is built; these flags let it *act*. **Gate everything mutating through
an ActionPolicy posture** — start locked-down, graduate deliberately.

1. **Choose a posture** (`examples/action-policies/`): `locked-down.yml` (approve
   everything) → `supervised.yml` (auto-notify, human-approves mutations) →
   `scoped-autonomous.yml` (auto within blast-radius caps + rate limits + windows). The
   shipped `deploy/action-policy.default.yml` requires approval for everything mutating.
2. **Wire monitoring in:** point alertmanager / uptime-kuma / Portainer webhooks at
   `POST /api/fleet/events` so the pull-only monitoring stack becomes a wake-up source.
3. **Enable the propose-only learning loops** (still off by default):
   ```bash
   export KG_GOLDEN_LOOP=true            # research assimilation tick (propose-only)
   export KG_FAILURE_EVOLUTION=true      # Langfuse-failure-driven remediation (propose-only)
   ```
   These publish reviewable **local branches** via the ActionPolicy-gated `ChangePublisher`
   — **never auto-pushed**. Auto-merge stays off.
4. **Optional, later:** `AGENT_DISPATCH_BACKEND=queue` (session-keyed dispatch, ORCH-1.45)
   and `TASK_QUEUE_BACKEND=kafka` (fail-loud ingest scale-out, KG-2.55–57) once the queue
   infra is deployed.

- **Verify:** a synthetic unhealthy-service event ingests as a `FleetEvent`; the
  reconciler proposes (dry-run) a remediation; the ActionPolicy decision is recorded and
  requires approval under the chosen posture; a golden-loop tick produces a proposal
  branch, not a push.

## Stage H — Deploy the semantic plane (ontology-driven, optional) — KG-2.52

Turn on the Fuseki publish tick (`KG_FUSEKI_PUBLISH` + a Fuseki dataset with OWL-mini
inference) so the authoritative TBox is SPARQL-queryable by every agent via jena-mcp, and
`graph_orchestrate(action="compile_process")` lifts descriptive processes into executable
plans against it.

- **Verify:** the published TBox answers a SPARQL query in Fuseki; a harvested
  `BusinessProcess` compiles to a `WorkflowDefinition` with a `REALIZES` edge.

---

## Rollback

Every stage is a flag. To revert a stage, unset its flag(s) and restart the affected tier;
behavior returns to the prior stage. The only ordering constraint on rollback is the
inverse of enablement (disable autonomy before brain before sharding before state before
security) so you never widen blast radius while a higher layer is still trusting it.

## Reference

- Per-flag inventory & audit: [docs/architecture/configuration.md](../architecture/configuration.md)
- Postures: `examples/action-policies/{locked-down,supervised,scoped-autonomous}.yml`
- Scale-out deep dives: [state_externalization](../architecture/state_externalization.md) ·
  [engine_sharding](../architecture/engine_sharding.md) ·
  [fleet_autonomy](../architecture/fleet_autonomy.md) ·
  [gateway_scaling](../architecture/gateway_scaling.md)
- Thin/scalable frontends: [scalable-frontends.md](scalable-frontends.md)

---

## Where each flag and secret lives

Three homes, by kind:

- **Secrets** (engine HMAC secret, the state-DB DSN with credentials) → **OpenBao**,
  `kv2` mount `apps/`, path **`apps/agent-utilities/deployment`**. Mirrored into the
  gitignored deployment `.env`. Pull at runtime by setting `SECRETS_BACKEND=vault` +
  `SECRETS_VAULT_URL=http://openbao.arpa` + `SECRETS_VAULT_MOUNT=apps`.
- **Flags / endpoints** (non-secret booleans + URIs) → the deployment **`.env`**
  (`agent-utilities/.env`, gitignored) or the XDG `config.json` key.
- **Defaults** → `agent_utilities/core/config.py` (`AgentConfig`).

| Capability | Env / config key | Default | Secret? (OpenBao key) | Auto-config |
|---|---|---|---|---|
| Engine HMAC auth | `GRAPH_SERVICE_AUTH_SECRET` | unset | **yes** (`GRAPH_SERVICE_AUTH_SECRET`) | engine refuses empty |
| Request auth (OS-5.14) | `KG_AUTH_REQUIRED` | `false` | no | **auto-on** when `AUTH_JWT_ISSUER`/`AUTH_JWT_JWKS_URI` set |
| Fail-closed ACL | `KG_ACL_DEFAULT_ALLOW` | `false` | no | deny-by-default |
| Shared state (OS-5.16) | `STATE_DB_URI` | unset | **yes** (`STATE_DB_URI`) | uses Postgres when set, else SQLite |
| Sharding (KG-2.58) | `GRAPH_SERVICE_ENDPOINTS` | unset | no | 2+ endpoints → HRW sharding |
| Company Brain | `KG_BRAIN_ENFORCE` | `false` | no | explicit |
| Golden loop | `KG_GOLDEN_LOOP` | `false` | no | explicit (propose-only) |
| Failure evolution | `KG_FAILURE_EVOLUTION` | `false` | no | explicit (propose-only) |
| Queue dispatch | `AGENT_DISPATCH_BACKEND` | `inline` | no | explicit |
| Kafka ingest | `TASK_QUEUE_BACKEND` | unset | no | fail-loud when set |
| Fuseki publish (KG-2.52) | `KG_FUSEKI_PUBLISH` | `false` | no | **auto-on** when `KG_FUSEKI_ENDPOINT`/`JENA_FUSEKI_URL` set |
| Thin frontend | `KG_DAEMON_ROLE` | `auto` (host) | no | `client` → reach shared host |

### Configure-by-default (and how to opt out)

Two flags **engage automatically once their deployment dependency is configured**, so a
real deployment does not have to remember a second knob — while the zero-infra laptop
default stays byte-for-byte unchanged (no dependency → nothing turns on). This is the
`AgentConfig` model validator `_auto_enable_from_dependencies` (`core/config.py`):

- Configure a JWT issuer/JWKS → **`KG_AUTH_REQUIRED` engages**. Opt out with an explicit
  `KG_AUTH_REQUIRED=false` (an explicit value always wins — it lands in `model_fields_set`).
- Configure a Fuseki endpoint → **`KG_FUSEKI_PUBLISH` engages**. Opt out with `KG_FUSEKI_PUBLISH=false`.

### Storing / rotating the secrets

Write or rotate a secret in OpenBao, then mirror it into the deployment `.env`:

```bash
TOKEN=$(grep BAO_ROOT_TOKEN services/openbao/.env | cut -d= -f2)
# read current deployment secrets
curl -s -H "X-Vault-Token: $TOKEN" \
  http://openbao.arpa/v1/apps/data/agent-utilities/deployment | jq '.data.data | keys'
# rotate the engine secret
curl -s -X POST -H "X-Vault-Token: $TOKEN" \
  -d '{"data":{"GRAPH_SERVICE_AUTH_SECRET":"<new>","STATE_DB_URI":"<dsn>"}}' \
  http://openbao.arpa/v1/apps/data/agent-utilities/deployment
```

> **Local dev vs deployment.** The local/laptop instance resolves its config from the
> *workspace* path and stays zero-infra; the per-repo `agent-utilities/.env` is the
> *deployment* config (read when agent-utilities runs from its own directory or a
> container). Keeping auth/secrets in the deployment `.env` + OpenBao therefore does not
> arm the local dev instance — exactly the configure-by-default / opt-out split.
