# MCP fleet authentication and monitoring runbook

This runbook applies the provider-neutral trust model described in
[MCP authentication and network trust](../architecture/mcp_auth.md). Keep all
endpoints, identities, credentials, certificate material, and policy documents
in runtime configuration or a secret manager—not in the repository.

## Prepare a network service

1. Configure JWT verification directly or through an OIDC/OAuth identity proxy.
2. Configure an exact issuer, audience, algorithm, required scopes, and JWKS
   source when using JWT.
3. Configure direct TLS, or trusted TLS termination plus exact ingress-peer
   CIDRs.
4. Configure exact host authorities and, only if browser access is needed,
   exact browser/WebSocket origins.
5. Size the request, connection, backlog, rate, and tool-call limits for the
   service.
6. If an external policy decision point is used, load a default-deny policy
   before exposing the listener.

The listener must fail startup if authentication or TLS is incomplete. There is
no unauthenticated remote-listener mode.

## Configure multiplexer-to-child identity

For OAuth 2.0 client credentials, supply this contract through AgentConfig,
environment projection, and runtime secret references:

```text
MCP_CLIENT_AUTH=oidc-client-credentials
OIDC_CLIENT_ID=<service-client-id>
OIDC_CLIENT_SECRET_REF=<runtime-secret-reference>
OIDC_ISSUER=https://<identity-provider>/<issuer>
OIDC_AUDIENCE=<child-audience>
OIDC_SCOPE=<space-separated-scopes>
OIDC_TLS_PROFILE_REF=<runtime-tls-profile-reference>
```

`OIDC_TOKEN_URL` may pin a token endpoint when discovery is unavailable. The
endpoint must match the egress and TLS policies. For HTTP Basic children, use
`MCP_CLIENT_AUTH=basic`, `MCP_BASIC_AUTH_USERNAME`, and
`MCP_BASIC_AUTH_PASSWORD_REF`.

Never put bearer tokens or resolved secrets in child configuration. Token
minting is bounded, cached, refreshed before expiry, and retried once after a
401. If credential production fails, the delegated call fails closed.

For catalog-owned credential fields, persist only `env://ALIAS`. Prefer a
direct runtime-secrets or environment projection. If one portable catalog must
resolve through different secret backends, configure
`MCP_FLEET_SECRET_REFS` as an alias-to-reference JSON mapping in AgentConfig.
The direct alias wins; the fallback is consulted only when it is unavailable.
Both paths resolve in memory at the child boundary.

## Phased rollout

1. Establish the multiplexer service identity and verify its non-secret issuer,
   audience, and scopes.
2. Load authorization policy and verify the service principal has only its
   intended child/tool permissions.
3. Enable authentication on one read-only child.
4. Verify a direct unauthenticated call returns 401 and a delegated authorized
   call succeeds.
5. Verify a caller lacking delegation or child-required scopes receives 403.
6. Expand in small waves, leaving high-impact infrastructure tools until the
   lower-risk fleet is verified.

Rollback must preserve the network boundary. Restore the last known-good auth
or policy configuration; do not expose a remote no-auth listener.

## Monitoring

- `/health` is a generic readiness probe and must be reachable only through the
  intended network boundary.
- For remote metrics, set `MCP_METRICS_TOKEN_REF` to a runtime secret reference
  and configure the scraper to send that bearer. Without a valid reference,
  `/metrics` is not registered remotely.
- Alert on authentication failures, policy denials, child circuit breakers,
  resource-limit rejections, process restarts, memory pressure, and disk
  pressure. Avoid labels containing identities, URLs, paths, tool input, or
  model content.
- Run `agent-utilities-doctor --only mcp_fleet_secrets` after changing alias
  projections or fallback references. Its output contains aggregate counts only.

## Rotation

1. Rotate the credential in the identity provider or secret manager.
2. Update the referenced runtime value without changing repository files.
3. Restart or hot-reload the owning process according to the deployment policy.
4. Verify a new token can be minted and an authorized child call succeeds.
5. Revoke the previous credential and verify it is rejected.

Do not print tokens, secret values, endpoints, certificate contents, local paths,
or trace payloads during verification.

## Private trust and trace verification

Use an OIDC or telemetry TLS profile reference containing the required CA chain
and optional client identity. Resolve it into owner-readable, ephemeral runtime
material and remove that material when the process exits. Never set TLS
verification to false.

For metadata-only trace verification:

1. Record only the current trace count.
2. Emit a synthetic trace with a random, non-sensitive label.
3. Flush the exporter and read the trace through the configured observability
   integration.
4. Report only before/after counts and a boolean readback result.

The trace must not contain prompts, completions, arguments, results, headers,
tokens, identities, URLs, or filesystem paths.
