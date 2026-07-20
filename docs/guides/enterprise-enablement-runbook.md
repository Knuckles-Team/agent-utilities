# Enterprise enablement runbook

This runbook enables the current enterprise deployment posture in a deliberate
order: publish, establish identity and transport security, externalize state,
scale the engine, and then enable governed autonomy.

## Configuration boundary

Keep deployment state outside the source tree:

- Store non-secret settings and runtime references in
  `$XDG_CONFIG_HOME/agent-utilities/config.json`.
- Resolve `secret://`, `vault://`, or `env://` references through the configured
  secret provider at process start. The reference name is stable; the resolved
  value is never written to AgentConfig, logs, traces, or reports.
- Fields that consume resolved connection material, including `STATE_DB_URI` and
  a distributed engine's HMAC secret, must be injected into process memory by the
  deployment secret resolver. Do not mirror them into a repository `.env` file.
- Store CA bundles, client certificates, server names, and verification policy in
  reusable TLS profiles. Reference those profiles from AgentConfig; never disable
  verification in application code.
- Keep host inventory, placement, and provider-specific secret paths in the
  operator's external deployment system.

Run the identity and transport preflight before starting a graph surface:

```bash
agent-utilities doctor --only config auth secrets transport_security graph_connections
```

## Stage A — Publish in dependency order

Use repository-manager with the ecosystem `workspace.yml` so upstream packages
publish before their consumers:

```bash
auto_push --phased
```

Verify each dependency phase before advancing. The workspace manifest, rather than
machine-specific paths or a handwritten repository list, defines the order.

## Stage B — Establish identity and secure transport

Configure one client identity mode and the server JWT policy in XDG AgentConfig.
This neutral shape illustrates the boundary; substitute runtime-discovered service
names and operator-owned reference identifiers:

```json
{
  "GRAPH_SERVICE_ENDPOINTS": ["tls://engine.example.invalid:9100"],
  "ENGINE_TLS_PROFILE_REF": "secret://tls/engine-client-profile",
  "KG_IDENTITY_OAUTH2": {
    "token_url": "https://identity.example.invalid/oauth2/token",
    "client_id": "graph-client",
    "client_secret": "secret://identity/graph-client-secret",
    "audience": "graph-services"
  },
  "AUTH_JWT_JWKS_URI": "https://identity.example.invalid/.well-known/jwks.json",
  "AUTH_JWT_AUDIENCE": "graph-services",
  "KG_POLICY_VERSION": "current"
}
```

`KG_AUTH_TOKEN_REF` is the alternative for deployments that issue a governed static
runtime token. Configure exactly one of `KG_AUTH_TOKEN_REF` and
`KG_IDENTITY_OAUTH2`. Missing identity, invalid JWTs, absent ACL infrastructure,
and ambiguous engine placement fail closed.

For a supervised local engine, AgentConfig may omit `GRAPH_SERVICE_ENDPOINTS`; the
packaged engine and its per-install authentication material are managed under the
XDG runtime/data boundary. A distributed engine receives its shared authentication
material from the deployment secret resolver at launch.

Verify that unauthenticated access is rejected, valid identities resolve the intended
tenant, cross-tenant reads are denied, and every remote certificate chains through
the selected TLS profile.

## Stage C — Deploy the engine and surfaces

Deploy pinned artifacts through the operator-owned orchestrator, then start only the
surfaces required by the topology:

- `graph-os` for MCP and fleet delegation;
- `graph-os-daemon` for a headless KG host without an HTTP API;
- `python -m agent_utilities` for the REST/API gateway.

Hot-swap one compatible deployment unit at a time behind health checks. Verify
metrics, an authenticated graph query, multiplexer health, and a clean doctor report
before advancing.

## Stage D — Externalize durable state

For multi-host gateways, configure the secret provider to inject the resolved
PostgreSQL DSN as `STATE_DB_URI` at process start. The DSN is secret material and is
not an AgentConfig literal. Non-secret pool controls remain in XDG AgentConfig:

```json
{
  "STATE_DB_POOL_SIZE": 8,
  "TASK_QUEUE_BACKEND": "postgres"
}
```

This externalizes session/turn/fleet metadata and queue-delivery state. It does
not move execution checkpoints or create a second goal lifecycle: the
engine-native WorkItem remains authoritative for claim, lease, fencing,
`checkpoint_id`, idempotency, and terminal result. Verify that two gateway
replicas cannot process the same delivery claim, leadership moves after a
replica exits, and an interrupted WorkItem resumes only through its current
native lease.

## Stage E — Scale engine authority

Clients contact a stable coordinator and never infer placement from endpoint names:

```json
{
  "GRAPH_SERVICE_ENDPOINTS": ["tls://engine.example.invalid:9100"],
  "ENGINE_TLS_PROFILE_REF": "secret://tls/engine-client-profile"
}
```

If a deployment exposes Raft groups separately, add the strict
`GRAPH_RAFT_GROUP_ENDPOINTS` map generated from the external placement inventory.
The coordinator remains authoritative for graph ownership, epochs, and fences.
Unreachable or ambiguous authority fails closed.

Verify coordinator health, route a graph to its authoritative group, and confirm
that a failed group produces an explicit error rather than a local substitute.

## Stage F — Verify governed retrieval

The Company Brain boundary applies source-authority arbitration, confidence decay,
field-level survivorship, tenant scoping, data ACLs, read audit, and durable human
corrections. Identity from Stage B is mandatory.

Verify a source conflict, an ACL-protected field, and a human correction end to end.

## Stage G — Enable propose-only learning and autonomy

Start with the locked-down ActionPolicy and graduate only after recorded approvals
and failure drills. Keep mutation, development, and merge gates review-first:

```json
{
  "KG_LOOP": true,
  "KG_LOOP_BREADTH": true,
  "KG_LOOP_MINE_DISCOVERY": true,
  "KG_LOOP_BELIEF_REVISION": true,
  "KG_LOOP_INSIGHT_VALIDATION": true,
  "KG_LOOP_TRACE_MINING": true,
  "KG_OPTIMIZATION_ENABLED": true,
  "ENABLE_OTEL": true,
  "TRACE_EXPORT_ENABLED": true,
  "LANGFUSE_CAPTURE_CONTENT": false,
  "KG_FAILURE_REGRESSION_DATASET": false,
  "KG_GOLDEN_AUTO_MERGE": false,
  "KG_AGENT_AUTO_APPLY": false,
  "KG_LOOP_AUTO_DEVELOP": false,
  "KG_INSIGHT_AUTONOMY": false
}
```

Configure Langfuse with `LANGFUSE_PUBLIC_KEY_REF`, `LANGFUSE_SECRET_KEY_REF`, and
`LANGFUSE_TLS_PROFILE_REF`. When both credential references resolve, the Langfuse
MCP child and propose-only failure evolution auto-enable unless explicitly disabled.
Trace export, content capture, and KG auto-ingestion remain explicit opt-ins. Enabling
auto-ingestion also requires `LANGFUSE_PERSISTENCE_HMAC_KEY_REF`.

Feed monitoring events through the authenticated fleet-event endpoint using
`FLEET_EVENTS_TOKEN_REF`. A synthetic event must create a reviewable proposal and an
ActionPolicy decision; it must not silently mutate infrastructure or push code.

Kafka is an optional queue transport for larger fleets. Its brokers, credentials,
and TLS material belong in the external deployment profile and runtime secret/TLS
references.

## Stage H — Enable the semantic publication plane

To publish the authoritative TBox to Fuseki, configure the discovered HTTPS dataset
endpoint, a runtime password reference, and the explicit publish gate:

```json
{
  "KG_FUSEKI_ENDPOINT": "https://semantic.example.invalid/dataset",
  "GRAPH_FUSEKI_USER": "publisher",
  "GRAPH_FUSEKI_PASSWORD_REF": "secret://semantic/fuseki-publisher-password",
  "KG_FUSEKI_PUBLISH": true
}
```

Verify that the dataset answers a SPARQL query and that a harvested business process
compiles to a governed workflow with provenance.

## Capability shutdown

Disable autonomous and mutating capabilities before removing their dependent
identity, state, or transport services. Preserve audit records and confirm quiescence
before scaling down a tier.

## Secret rotation

Rotation is provider-neutral:

1. Write the new value under the operator-owned secret identifier.
2. Keep the AgentConfig reference unchanged.
3. restart or hot-reload the affected workload through the orchestrator.
4. Re-run `doctor` and an authenticated smoke test.
5. Revoke the superseded value after all consumers report healthy.

Do not use a root token, embed a provider API path, retrieve secrets with shell
history-visible commands, or copy resolved values into a repository file.

## References

- [Configuration reference](configuration.md)
- [Configuration architecture audit](../architecture/configuration.md)
- [Deployment configurations](deployment-configurations.md)
- [Engine sharding](../architecture/engine_sharding.md)
- [State externalization](../architecture/state_externalization.md)
- [Fleet autonomy](../architecture/fleet_autonomy.md)
- [Scalable frontends](scalable-frontends.md)
