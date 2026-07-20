# Day-0 deployment

Day 0 establishes the packaged Rust graph engine, a GraphOS surface, optional
connectors, and runtime integrations without persisting deployment-specific paths,
endpoints, identities, or secrets in the repository.

Choose a supported shape:

- [Tiny](../recipes/tiny.md): supervised local engine and no required external
  database.
- [Single-node production](../recipes/single-node-prod.md): durable host with optional
  mirrors and connectors.
- [Enterprise multi-host](../recipes/enterprise.md): remote engine authority,
  workload identity, shared state, and horizontally scaled workers.

## Tiny bootstrap

From a trusted source checkout:

```bash
./scripts/bootstrap.sh
```

The bootstrap installs the project, creates the tiny profile only when no config is
present, and runs a graph smoke test. Its configuration target is the XDG AgentConfig
document at `$XDG_CONFIG_HOME/agent-utilities/config.json`; it does not create or
consume a repository `.env` file.

For an installed package, generate the same neutral profile directly:

```bash
setup-config generate --profile tiny
```

Review the generated AgentConfig, configure one governed client identity where the
selected surface requires it, and run preflight checks before starting GraphOS:

If local `env://` references are needed, place their values in the optional fixed
`runtime-secrets.json` beside the XDG `config.json`. On POSIX, set exact mode `0600`
or `0400`. Only referenced keys are projected, and doctor reports aggregate source
readiness without paths, names, references, or values. On native Windows, inject
the referenced values through the process environment; private file sources fail
closed until descriptor-level ACL validation is available.

```bash
agent-utilities doctor --only config auth secrets transport_security graph_connections
```

## Four steps for every profile

### 1. Install the required composition

```bash
uv sync
# Published package alternative:
uvx --refresh --from "agent-utilities[serving]>=1.27.1,<2.0.0" graph-os --help
```

The package always carries the full epistemic-graph engine capability. Add only the
Agent Utilities feature groups required by the selected deployment; see
[Installation](installation.md).

### 2. Start one surface

```bash
# MCP for an IDE or delegating agent
graph-os

# Loopback MCP HTTP
graph-os --transport streamable-http --host 127.0.0.1 --port 8004

# Headless KG host without an HTTP API
graph-os-daemon

# REST/API gateway for UIs and application clients
python -m agent_utilities
```

GraphOS supervises the packaged Rust engine as an out-of-process child when
`GRAPH_SERVICE_ENDPOINTS` is absent. When endpoints are configured, it is connect-only
and never creates a local substitute. Non-loopback bindings require verified JWT/OIDC
identity and trusted TLS termination.

The only identity exception is `graph-os --transport stdio` with
`DEPLOYMENT_PROFILE=tiny`, no endpoints, and neither external process-identity
source. It creates and validates a neutral bootstrap JWT with an in-memory key,
then discards the key and token. All other surfaces require exactly one external
process identity, and failure never falls back locally.

See [Consumption models](consumption-models.md) for surface selection.

### 3. Attach approved connectors

The generated fleet registry describes available `*-mcp` packages, but an external
runtime profile selects which connectors are enabled and where they run. Use the
`agent-utilities-deployment` workflow to validate tool schemas, workload identity,
TLS profiles, and runtime secret references before registration.

Do not commit the selected inventory, discovered endpoints, resolved certificates,
or source-system credentials. Universal external graph sources are configured through
`EXTERNAL_GRAPH_CONNECTORS`; see
[Universal external graph connectors](../architecture/universal-external-graph-connectors.md).

### 4. Configure integrations in XDG AgentConfig

Use reference-only settings and neutral service discovery:

| Integration | AgentConfig boundary |
|---|---|
| Optional graph projection | `GRAPH_MIRROR_TARGETS` and `GRAPH_DB_CONNECTION_PROFILE_REF=secret://graph/mirror-profile` |
| Remote engine | `GRAPH_SERVICE_ENDPOINTS` plus `ENGINE_TLS_PROFILE_REF` |
| Queue scale-out | `TASK_QUEUE_BACKEND`; broker topology, credentials, and TLS stay in the external runtime profile |
| Client identity | tiny packaged-local GraphOS stdio: neither external source; every other boundary: exactly one of `KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2`, plus server JWKS/audience/policy settings |
| Langfuse | `LANGFUSE_HOST`, both credential references, and `LANGFUSE_TLS_PROFILE_REF` |
| OpenTelemetry | `ENABLE_OTEL`, `TRACE_EXPORT_ENABLED`, and the configured exporter TLS/credential references |

The engine remains the authority when a mirror is enabled. Mirror writes are
governed fan-out for external query and reporting; they do not become a second source
of truth.

Langfuse MCP and propose-only failure evolution auto-enable when both Langfuse
credential references resolve. Trace export, content capture, and KG auto-ingestion
remain explicit choices. Auto-ingestion also requires a persistence HMAC-key
reference.

## Automated Day 0

The `agent-utilities-deployment` workflow consumes an operator-owned profile, performs
discovery, writes reference-only XDG AgentConfig, deploys the selected surfaces and
connectors, and certifies the live topology. It does not add a site profile to this
repository.

## Validate

```bash
agent-utilities doctor --only config auth secrets transport_security graph_connections langfuse
```

Then run the profile's authenticated graph, delegation, trace, and connector smoke
tests before enabling autonomous mutations.
