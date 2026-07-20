# Configuration

Agent Utilities has one durable configuration boundary: the XDG
`agent-utilities/config.json` document validated by `AgentConfig`. Generate that
document instead of copying a hand-maintained schema:

```bash
setup-config generate --profile tiny
setup-config reference
agent-utilities-doctor
```

The generated [runtime configuration catalog](../reference/runtime-configuration.md)
is the field-by-field source of truth. The
[configuration and flag audit](../architecture/configuration.md) explains why each
setting exists and which safety checks apply. This guide covers only the choices an
operator normally makes.

## Configuration rules

- Commit neutral aliases, bounded policy values, and runtime secret references only.
- Keep credentials, bearer tokens, private trust material, concrete connection
  profiles, source schemas, source queries, and machine filesystem locations outside
  the repository.
- Select TLS with `TLS_PROFILES_REF` and a purpose-specific profile or profile
  reference. Certificate verification is mandatory; there is no supported
  `verify=false` configuration.
- Supply runtime overrides through the deployment environment only when a value must
  differ by process. `AgentConfig` remains the typed validation boundary.
- Run doctor after every configuration change. Doctor reports aliases, readiness,
  counts, and digests without returning resolved secrets, endpoints, identities, or
  paths.

## MCP fleet secret aliases

Keep a persistent MCP child catalog portable by placing a neutral uppercase
alias in each credential field:

```json
{
  "mcpServers": {
    "example-child": {
      "command": "example-child",
      "env": {
        "ACCESS_TOKEN": "env://CHILD_ACCESS_TOKEN"
      }
    }
  }
}
```

The runtime may project `CHILD_ACCESS_TOKEN` directly through its environment
or runtime-secrets source. When the direct alias is absent, durable AgentConfig
may contain a reference-only fallback:

```json
{
  "MCP_FLEET_SECRET_REFS": {
    "CHILD_ACCESS_TOKEN": "env://CHILD_ACCESS_TOKEN"
  }
}
```

Aliases use `A-Z`, `0-9`, and `_`, beginning with a letter. Mapping values are
limited to `env://`, `vault://`, and `secret://` runtime references; inline
material, unsupported schemes, and traversal are rejected. A same-alias
`env://` mapping selects that key from the runtime-secrets source; a different
`env://` target or a store reference provides an alternate runtime source.
Direct alias material always wins, so rotation can be projected without
rewriting the catalog or fallback map. Validate aggregate readiness without
disclosing aliases or references:

```bash
agent-utilities-doctor --only mcp_fleet_secrets mcp_fleet
```

## GraphOS process authority

The only boundary with no external process-identity configuration is
`graph-os --transport stdio` with all of the following:

- `DEPLOYMENT_PROFILE=tiny`;
- no `GRAPH_SERVICE_ENDPOINTS`;
- no `KG_AUTH_TOKEN_REF`; and
- no `KG_IDENTITY_OAUTH2`.

GraphOS creates an asymmetric key in memory, signs and validates a short-lived JWT
with fixed neutral service claims as a one-time proof, destroys the key and token,
and returns a process-lifetime session. It persists no user name, host name,
endpoint, filesystem path, credential, proof material, or other local identity.

Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point requires exactly one external process identity source:

- `KG_AUTH_TOKEN_REF` references a JWT provisioned by the runtime; or
- `KG_IDENTITY_OAUTH2` describes a client-credentials flow whose `client_secret`
  value is itself a runtime secret reference.

The token is validated against the configured issuer/JWKS, audience, tenant
authority, and `KG_POLICY_VERSION`. Raw tokens and client secrets are not durable
configuration values. External stdio sessions share an in-memory expiry-only
lease with background workers. Renewal must preserve subject, actor type,
capabilities, tenant, authentication state, and groups. Drift is rejected;
failure never extends the lease, and graph work fails closed at expiry while
renewal retries. A neutral provisioned-token shape is:

```json
{
  "KG_AUTH_TOKEN_REF": "secret://identity/graph-os-token",
  "AUTH_JWT_JWKS_URI": "https://identity.example.invalid/.well-known/jwks.json",
  "AUTH_JWT_AUDIENCE": "graph-services",
  "KG_POLICY_VERSION": "current"
}
```

Validate local authority, or external identity and secret resolution, before
launch:

```bash
agent-utilities-doctor --only graph_identity auth secrets
```

Network requests carry their own validated bearer identity and never inherit
process authority implicitly. A configured, unresolved, or invalid external source
never falls back to the local authority. See
[Identity and JWT](../examples/identity-jwt.md) for both authority modes.

## External provider runtime profiles

`PROVIDER_CONFIGS` is the common connection boundary for provider MCP packages.
It replaces provider-specific durable endpoint and credential fields with a
neutral profile whose values are runtime references:

```json
{
  "PROVIDER_CONFIGS": {
    "example-provider": {
      "enabled": true,
      "endpoint_ref": "env://EXAMPLE_PROVIDER_ENDPOINT",
      "credential_refs": {
        "EXAMPLE_PROVIDER_TOKEN": "secret://providers/example/token"
      },
      "selector_refs": {
        "EXAMPLE_PROVIDER_SCOPE": "vault://providers/example/scope"
      },
      "tls_profile_ref": "secret://tls/example-provider"
    }
  }
}
```

Profile names are neutral lowercase aliases. Credential and selector keys are
bounded uppercase runtime aliases. Values use only `env://`, `vault://`, or
`secret://` references. An endpoint reference resolves to credential-free HTTPS;
cleartext HTTP is accepted only for an exact loopback host. Every endpoint must
select exactly one named TLS profile or TLS-profile reference, and certificate and
hostname verification cannot be disabled.

A local MCP child selects its profile without copying deployment data into the
child catalog:

```json
{
  "mcpServers": {
    "example-provider": {
      "command": "example-provider-mcp",
      "provider_profile": "example-provider"
    }
  }
}
```

GraphOS resolves the complete profile in the trusted parent before spawning the
child. It rewrites only that profile's resolved values to bounded, child-private
ephemeral aliases and retains any temporary TLS material only for the child
session. Original references, the parent configuration root, ambient Vault/engine
authority, and unrelated process variables never cross the boundary. The child
consumes the isolated projection at its provider boundary. Remote MCP children are
independently deployed and cannot select a parent-local provider profile. Raw
values, references, profile names, endpoints, identities, and paths are excluded
from doctor output:

```bash
agent-utilities-doctor --only provider_profiles
```

Providers with independently trusted endpoints use one profile per connection.
Customized source schemas and ontologies do not belong in these profiles; graph and
GraphQL schema discovery remains governed by `EXTERNAL_GRAPH_CONNECTORS`.

## GraphOS topology and entry points

`GRAPH_SERVICE_ENDPOINTS` is the sole topology selector:

- When it is absent, GraphOS supervises the packaged Rust
  `epistemic-graph-server` as a durable, out-of-process local child over a private
  transport.
- When it is configured, GraphOS is connect-only and never starts a local substitute.
  Remote engine trust belongs in `ENGINE_TLS_PROFILE_REF`.

The installed entry points have distinct jobs:

| Entry point | Responsibility |
| --- | --- |
| `graph-os` | MCP server and on-demand MCP fleet gateway over stdio or streamable HTTP |
| `graph-os-daemon` | Headless queue, maintenance, and background-work host; it serves no HTTP API |
| `python -m agent_utilities` | Agent and centralized REST/API gateway |

For streamable HTTP outside loopback, configure JWT/OIDC authentication and trusted
TLS termination. An unauthenticated non-loopback listener fails closed. See
[Deployment](deployment.md) and [Consumption Models](consumption-models.md).

## Models

`CHAT_MODELS` and `EMBEDDING_MODELS` are typed registries in AgentConfig. Each entry
selects a provider, model identifier, optional endpoint, bounded concurrency, and
capabilities. Authentication is either `api_key_ref` or an OAuth2
client-credentials block with a referenced client secret. Optional gateway headers
use `headers_ref`, whose resolved value is a bounded JSON object. Literal `api_key`
and `headers` fields are not part of the current model schema.

```json
{
  "CHAT_MODELS": [
    {
      "id": "chat-model",
      "provider": "openai",
      "base_url": "https://model.example.invalid/v1",
      "api_key_ref": "secret://models/chat-api-key",
      "headers_ref": "env://CHAT_MODEL_HEADERS",
      "tools_enabled": true,
      "can_route": true,
      "can_kg": true
    }
  ],
  "EMBEDDING_MODELS": [
    {
      "id": "embedding-model",
      "provider": "openai",
      "base_url": "https://embedding.example.invalid/v1",
      "api_key_ref": "secret://models/embedding-api-key"
    }
  ],
  "MODEL_TLS_PROFILE_REF": "secret://tls/model-profile",
  "EMBEDDING_TLS_PROFILE_REF": "secret://tls/embedding-profile"
}
```

Use independent TLS profiles when model endpoints have different trust boundaries.
For `env://CHAT_MODEL_HEADERS`, inject a JSON object such as
`{"X-Client-Id":"runtime-supplied"}` through the explicit process environment or
the fixed XDG runtime-secret source. It is parsed and bounded only while the client
is constructed; it is never copied into AgentConfig, doctor output, or a generated
configuration.
See [Model Registries](models.md) for routing and capacity configuration.

## External graphs and GraphQL

`EXTERNAL_GRAPH_CONNECTORS` declares neutral source aliases and governance bounds for
Neo4j/openCypher, Apache AGE, LadybugDB/Kuzu, remote epistemic-graph, and GraphQL
sources. Source endpoints, credentials, database names, queries, variables, TLS
material, discovered schemas, and custom ontologies remain in separately resolved
runtime documents.

```json
{
  "EXTERNAL_GRAPH_CONNECTORS": [
    {
      "name": "external-knowledge",
      "source_alias": "external-domain",
      "backend": "graphql",
      "connection_profile_ref": "secret://external/connection",
      "mapping_policy_ref": "secret://external/mapping-policy",
      "auth_profile_ref": "secret://external/auth",
      "tls_profile_ref": "secret://external/tls",
      "variables_ref": "secret://external/variables",
      "ingest_operation": "document_read",
      "discovery_max_types": 200,
      "discovery_max_depth": 6,
      "ingest_max_records": 1000,
      "require_approval": true,
      "schema_drift_policy": "fail_closed"
    }
  ]
}
```

Every connector follows the same current lifecycle:

1. `add_connection`
2. `discover_connection_schema`
3. `propose_connection_mapping`
4. `approve_connection_mapping`
5. `external_graph_doctor`
6. `ingest_connection`

Discovery is bounded and read-only. Ingestion re-discovers the schema, verifies the
approved schema and mapping-policy digests, and writes governed `ChangeEnvelope`
transactions to the local epistemic-graph authority. No connector ships a customized
environment profile. See [Universal External Graph Connectors](../architecture/universal-external-graph-connectors.md)
and [Privacy-safe External Graph Ingestion](../architecture/privacy-safe-external-ingestion.md).

## TLS and private trust

Store a TLS-profile catalog behind `TLS_PROFILES_REF`, then select the appropriate
entry with a purpose-specific setting such as:

- `ENGINE_TLS_PROFILE_REF`
- `MODEL_TLS_PROFILE_REF`
- `EMBEDDING_TLS_PROFILE_REF`
- `LANGFUSE_TLS_PROFILE_REF`
- `OTEL_TLS_PROFILE_REF`

The shared resolver projects the verified profile to Requests, HTTPX, SSL, database
drivers, and supervised MCP children. A profile may supply a complete private CA
chain, mTLS identity, and proxy policy. Files created while resolving that material
are private runtime artifacts and are never copied into AgentConfig, reports, traces,
or source control.

```bash
agent-utilities-doctor --only transport_security
```

## Langfuse and failure evolution

Persist only Langfuse connection metadata and references:

```json
{
  "LANGFUSE_HOST": "https://observability.example.invalid",
  "LANGFUSE_PUBLIC_KEY_REF": "secret://observability/langfuse-public-key",
  "LANGFUSE_SECRET_KEY_REF": "secret://observability/langfuse-secret-key",
  "LANGFUSE_TLS_PROFILE_REF": "secret://tls/langfuse-profile",
  "LANGFUSE_CAPTURE_CONTENT": false,
  "LANGFUSE_KG_AUTO_INGEST": false
}
```

When both credential references resolve, `LANGFUSE_MCP_ENABLED` and the propose-only
`KG_FAILURE_EVOLUTION` capability enable automatically unless either is explicitly
set to `false`. Credential availability does **not** authorize trace export, prompt or
response capture, or graph persistence:

- `TRACE_EXPORT_ENABLED` remains an explicit export gate.
- `LANGFUSE_CAPTURE_CONTENT` remains `false` and production remains metadata-only.
- `LANGFUSE_KG_AUTO_INGEST` remains `false`; enabling it also requires an independent
  `LANGFUSE_PERSISTENCE_HMAC_KEY_REF`.

```bash
agent-utilities-doctor --only langfuse
agent-utilities-doctor --live
```

The live check performs bounded, metadata-only connectivity and trace round-trip
proofs when configured. See [Failure-Driven Evolution](../architecture/failure_driven_evolution.md)
and [Usage, Cost, and Observability](observability-usage-tracking.md).

## Native program optimization

Program optimization is supplied by the mandatory full epistemic-graph engine. It
uses the durable `ProgramOptimize` jobs plane and has no second Python optimizer or
model-provider dependency. The `avatar` family accepts opaque `tool_refs` and
positive/negative governed trace references; its comparator returns a reference-only
`tool_policy` through the existing model transport rather than introducing another
provider setting.

| Setting | Default | Purpose |
| --- | --- | --- |
| `KG_OPTIMIZATION_ENABLED` | `true` | Enable propose-only scheduled optimization |
| `KG_OPTIMIZATION_INTERVAL` | `10800` | Sweep interval in seconds |

`agent-utilities-doctor --live` can submit one content-free optimizer probe to an
already active engine. The probe never autostarts an engine or retains source content.

## Universal connector and policy checks

Focused static checks are available without contacting configured endpoints:

```bash
agent-utilities-doctor --only config transport_security graph_connections langfuse
```

Use `agent-utilities-doctor --live` only when bounded network and engine operations
are authorized. The complete current setting inventory remains the generated
[runtime configuration catalog](../reference/runtime-configuration.md); do not copy
its tables into another manually maintained schema.
