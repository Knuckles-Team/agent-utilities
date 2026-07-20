# Self-Setup: deploy agent-utilities with every option considered

The one runbook to take agent-utilities from nothing to running — the path **Claude
follows to set itself up**, and the path an operator follows to stand up a host. It is
**config-complete**: one command generates the current `AgentConfig` surface, and
doctor validates the deployment.

It composes existing pieces rather than duplicating them — the `setup-config` and
`setup-databases` commands, the `database-environment-setup` skill, and the
`agent-utilities-deployment` workflow for the multi-node deployment profile. The guided version is the
same **`agent-utilities-deployment`** skill.

## Pick a profile

| Profile | For | Externals |
|---|---|---|
| **tiny** | local self-setup / edge | none — GraphOS supervises the packaged epistemic-graph engine out of process |
| **single-node-prod** | one durable host | optional Postgres/pg-age mirror, optional OpenBao/Langfuse |
| **enterprise** | multi-node fleet | swarm, Postgres/Neo4j mirrors, Kafka, Keycloak, observability |

These are the rungs of [deployment-configurations.md](deployment-configurations.md);
the per-flag detail lives in [configuration.md](../architecture/configuration.md).

## 1. Install

```bash
pip install "agent-utilities[serving]"   # GraphOS, full engine, headless agent, Langfuse, auth, metrics
# Use "agent-utilities[all]" only when this host needs every optional integration.
```

## 2. Generate the complete config (all options)

Don't hand-write `config.json`. Generate a full, profile-seeded one — every option at
a sensible default, with the handful of deployment-varying keys pre-filled:

```bash
setup-config generate --profile single-node-prod      # writes the XDG AgentConfig
setup-config reference                                 # current options by subsystem
```

Secret-bearing fields are blanked. Fill them with runtime secret references, never
resolved values, certificate paths, endpoints containing credentials, or machine
locations. Equivalent MCP/REST surface:
`graph_configure(action="generate_config", config_key="single-node-prod")`.

## 3. Identity and runtime secret references

The tiny packaged-local stdio path is self-authorizing only when all four
conditions hold: `graph-os --transport stdio`, `DEPLOYMENT_PROFILE=tiny`, no
`GRAPH_SERVICE_ENDPOINTS`, and neither external process identity is configured.
It signs and validates a short-lived JWT with an in-memory key as a one-time
proof, uses fixed neutral service claims, destroys the key and token, and returns
a process-lifetime session. No personal identity, host name, endpoint,
filesystem path, credential, or proof material is represented or persisted.
Validate it with:

```bash
agent-utilities-doctor --only graph_identity auth
```

Every other boundary requires external authority. For example, the
`single-node-prod` configuration generated above can use these reference-only
settings:

```json
{
  "SECRETS_BACKEND": "vault",
  "KG_AUTH_TOKEN_REF": "secret://identity/graph-os-token",
  "GRAPH_DB_CONNECTION_PROFILE_REF": "secret://graph/mirror-profile",
  "TLS_PROFILES_REF": "secret://tls/profile-catalog"
}
```

Configure exactly one `KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2`; the OAuth2 block
must reference its client secret. This applies to every network transport,
non-tiny profile, explicit engine endpoint, and non-GraphOS-stdio entry point. An
invalid configured source never falls back to local authority. The
`agent-utilities-deployment` skill can prepare the selected secret backend and the
doctor can validate resolution without printing the material:

```bash
agent-utilities-doctor --only graph_identity auth secrets transport_security
```

## 4. Databases (single-node-prod / enterprise)

Run the [databases recipe](../recipes/databases.md) / `database-environment-setup`
skill: Stardog (prod) or local `/api/sparql` (dev) + a Postgres with AGE + pgvector +
pg_search, projection fan-out wiring (`GRAPH_MIRROR_TARGETS`),
and graph backfill into the AGE mirror. The **tiny** profile skips this entirely —
the packaged, supervised epistemic-graph engine is the authority, with no mirror.

External Neo4j/openCypher, AGE, LadybugDB/Kuzu, remote epistemic-graph, and
GraphQL sources instead use the governed
[universal connector lifecycle](../architecture/universal-external-graph-connectors.md):
declare reference-only aliases in `EXTERNAL_GRAPH_CONNECTORS`, discover, propose,
approve, run `external_graph_doctor`, and ingest. No source-specific schema or
ontology is bundled.

## 5. Launch

```bash
graph-os            # MCP server; choose this for MCP clients
graph-os-daemon     # optional headless work host; no HTTP API
python -m agent_utilities --host 127.0.0.1 --port 9000  # agent + REST/API gateway
```

Containerized: `docker compose -f docker/mcp.compose.yml up -d` (plus
`docker/pg-age-full.compose.yml` for an optional pg-age mirror).

## 6. Auth & observability (enterprise)

Configure JWT issuer/JWKS, audience, `KG_POLICY_VERSION`, and policy through
AgentConfig. For Langfuse, persist only `LANGFUSE_HOST`, credential references, and
a TLS-profile reference. Both credential references automatically make the native
Langfuse MCP child and propose-only failure evolution available unless explicitly
disabled; trace export, content capture, and KG auto-ingestion remain explicit
opt-ins. Auto-ingestion additionally requires an independent persistence HMAC-key
reference.

## 7. Multi-node → agent-utilities-deployment

For a full swarm (SSH mesh, hardware placement, overlay networks, ingress, GitOps,
fleet deploy), hand off to the **`agent-utilities-deployment`** skill. This guide
generates and validates the config *around* that bootstrap; it doesn't reimplement it.

## 8. Verify

Run the holistic doctor — one sweep across config, engine, backend, secrets, auth,
the MCP fleet, hooks, and observability, each line carrying a remediation + the skill
that fixes it (brew/flutter-doctor style):

```bash
agent-utilities-doctor                 # human-readable; --json for machines, --fix for safe auto-remediation
agent-utilities-doctor --live          # prove MCP, Langfuse, and native optimizer capabilities
```

The normal sweep is static. `--live` performs real, bounded operations: when the
corresponding features are enabled, it mounts the current Langfuse MCP child,
requires its metadata-only runtime posture, executes a one-row trace read through
that child, calls the Langfuse API directly, emits and reads back one metadata-only
diagnostic trace, and submits one content-free `ProgramOptimize` job to an
already-active engine. It does not autostart an engine, and its report contains no
endpoint, credential, identity, or local-path material.

It composes the focused checks too, which you can still run directly:

```bash
setup-config doctor --profile single-node-prod        # config: required keys, durability, secret refs
python scripts/validate_mcp_config.py --live          # MCP reachability (catch 502s)
```

The config check reuses the production-safety rules (`collect_production_violations`)
so a config that pins you to a single host or in-memory broker is flagged before you
ship. Also reachable as `graph_configure(action="system_doctor")` (MCP/REST).
A green doctor + a `graph_write`/`graph_query` round-trip = you're up.

## See also
- [Day-0 overview](day0.md) · [Deployment configurations](deployment-configurations.md) ·
  [Configuration reference](configuration.md) ·
  [Universal connectors](../architecture/universal-external-graph-connectors.md)
- Recipes: [tiny](../recipes/tiny.md) · [single-node-prod](../recipes/single-node-prod.md) ·
  [enterprise](../recipes/enterprise.md) · [databases](../recipes/databases.md)
