# MCP authentication and network trust

All MCP servers created by `create_mcp_server` share the same inbound network,
authentication, authorization, and resource boundaries. Configuration is
provider-neutral and contains no deployment-specific endpoints or identities.

## Trust model

- `stdio` and loopback-only HTTP may use local process trust.
- A non-loopback HTTP or SSE listener requires an authentication provider.
- A non-loopback listener also requires direct TLS, or explicit trusted-ingress
  mode with an exact ingress-peer CIDR allowlist.
- Network listeners require exact `Host` values. Browser and WebSocket clients
  additionally require exact origins; an unset origin allowlist blocks requests
  that carry `Origin`.
- Request bodies, connections, listener backlog, authentication values, and
  token responses are bounded. Duplicate credential headers are rejected.
- Authentication runs outside route matching, so mounted routes, WebSockets,
  health surfaces, and future routes cannot accidentally bypass the boundary.

There is no remote no-auth escape hatch.

## Inbound authentication

Use one of these modes:

| Mode | Configuration contract |
|---|---|
| JWT | Set `AUTH_TYPE=jwt`, issuer, audience, JWKS URI, asymmetric algorithm, and required scopes. |
| Static | Set `AUTH_TYPE=static` and `FASTMCP_SERVER_AUTH_STATIC_TOKENS_REF` to a runtime secret reference containing the token map. |
| OIDC/OAuth proxy | Configure the selected provider through its standard discovery metadata and runtime secret references. |

JWT verification pins the issuer and audience, permits only the configured
algorithm, validates token time claims, and retrieves bounded JWKS through the
shared DNS-pinned HTTP client. Plain HTTP is accepted only for loopback
development. Private identity-provider hosts require an exact entry in
`OIDC_HTTP_ALLOWED_PRIVATE_HOSTS`.

Private trust and mutual TLS use `OIDC_TLS_PROFILE` or
`OIDC_TLS_PROFILE_REF`. TLS verification cannot be disabled at a call site.
Profiles and certificate material are runtime configuration and must not be
written into repository files.

Authorization is separate from token validity. Configure required scopes and,
when used, an external policy decision point with a default-deny policy. Tool
delegation requires explicit discovery/delegation capabilities; child-required
scopes can only narrow the caller's authority.

## Network listener configuration

For direct TLS, configure `MCP_TLS_CERTFILE` and `MCP_TLS_KEYFILE` using paths to
runtime-projected material. For trusted TLS termination, set
`MCP_TLS_TERMINATED=true` and `MCP_TRUSTED_PROXY_CIDRS` to the exact ingress
peers. In both cases configure:

- `MCP_ALLOWED_HOSTS`: exact host authorities accepted by the listener.
- `MCP_ALLOWED_ORIGINS`: exact browser/WebSocket origins, when browser access is
  required.
- `MCP_MAX_REQUEST_BYTES`, `MCP_MAX_CONNECTIONS`, and
  `MCP_LISTEN_BACKLOG`: deployment-appropriate resource ceilings.

Do not trust forwarding headers from arbitrary peers. Trusted-ingress mode
rejects connections whose immediate peer is outside the configured CIDRs.

## Multiplexer outbound authentication

`MCP_CLIENT_AUTH` controls the service identity used for remote child MCP
servers:

- `oidc-client-credentials`: mint and refresh a bearer using
  `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET_REF`, `OIDC_ISSUER` or
  `OIDC_TOKEN_URL`, `OIDC_AUDIENCE`, and optional `OIDC_SCOPE`.
- `basic`: use `MCP_BASIC_AUTH_USERNAME` and
  `MCP_BASIC_AUTH_PASSWORD_REF`.
- `none`: attach no service credential. Use only when the child does not require
  service authentication.

Secret values may be injected ephemerally for compatibility, but secret
references are preferred. References are resolved only at the outbound request
boundary. Selecting a service-auth mode with incomplete configuration, failed
discovery, or a failed token mint aborts the child request; it never retries
anonymously. An explicitly configured child authorization header is never
overwritten.

Remote child URLs must be HTTPS outside loopback, contain no user information,
query, or fragment, and pass the shared DNS-pinned egress policy. Local child
processes receive a narrow environment allowlist rather than the parent process
environment.

## Operational endpoints

`/health` exposes only a generic readiness result and is suitable for a trusted
network probe. On loopback, `/metrics` is available locally. On a non-loopback
listener, `/metrics` is registered only when `MCP_METRICS_TOKEN_REF` resolves to
a valid bearer; otherwise the route is absent. Both surfaces return
`Cache-Control: no-store` and disclose no endpoint, identity, or filesystem
information.

## Verification checklist

1. Confirm startup rejects a non-loopback no-auth listener.
2. Confirm startup rejects remote plaintext without a trusted TLS ingress.
3. Confirm wrong `Host`, browser origin, token issuer, audience, scope, and
   ingress peer are rejected.
4. Confirm duplicate authorization headers and oversized requests are rejected.
5. Confirm a child token-mint failure fails the delegated call without an
   anonymous retry.
6. Confirm `/metrics` is absent remotely without its runtime token reference and
   returns 401 for a wrong bearer when enabled.
7. Confirm logs and traces contain only opaque identity references and bounded,
   metadata-only attributes.
