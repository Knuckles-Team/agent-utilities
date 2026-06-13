# MCP Fleet Authentication (JWT + Eunomia)

How every MCP server we deploy authenticates callers and authorizes tool calls,
and how the multiplexer reaches JWT-protected children. The whole fleet is built
from one factory, so this is configured once and applies everywhere.

## The model

```
Claude Code ──► mcp-multiplexer ──►  child MCP servers (57×, *.arpa/mcp)
                     │                     │  AUTH_TYPE=jwt
                     │ client_credentials  │  ├─ JWTVerifier (Keycloak JWKS, aud=agent-services)
                     │ service token       │  └─ Eunomia middleware (policy, fail-closed)
                     ▼                     ▼
                 Keycloak  ◄────────  validates bearer (JWKS)
              (realm master)
```

Every `-mcp` service is built by `create_mcp_server`
(`agent_utilities/mcp/server_factory.py`), so all of them honor the same env:

| Env | Meaning |
|-----|---------|
| `AUTH_TYPE=jwt` | Verify a Keycloak-issued bearer with `JWTVerifier`. |
| `FASTMCP_SERVER_AUTH_JWT_ISSUER` | `http://keycloak.arpa/realms/master` |
| `FASTMCP_SERVER_AUTH_JWT_JWKS_URI` | `.../protocol/openid-connect/certs` |
| `FASTMCP_SERVER_AUTH_JWT_AUDIENCE` | `agent-services` |
| `EUNOMIA_TYPE=remote` + `EUNOMIA_REMOTE_URL` | Authorize each tool call against the policy server. |

These are non-secret internal URLs and live in the compose template
(`scripts/gen_mcp_service_stacks.py` `COMPOSE_TMPL`), so newly generated stacks
are auth-on by default and `compose.dev.yml` inherits them via `make_editable`.

## Two properties you must design around

1. **Eunomia fails *closed*.** With `default_effect: deny`
   (`agent_utilities/mcp/eunomia_principal.py`), a JWT service with **no policy**
   for the caller's principal denies *every* call. So a baseline policy that
   allows the multiplexer's service principal must exist at `eunomia.arpa`
   **before** a service is flipped to jwt.

2. **The multiplexer must present a token.** Children are configured per-entry in
   `mcp_config.json`; historically none carried an `Authorization` header, so a
   child flipped to jwt became unreachable (401) through the aggregator. (Local
   *stdio* children like `graph-os` are exempt — stdio has no HTTP auth.)

## Multiplexer outbound auth (client-credentials)

`agent_utilities/mcp/client_credentials.py` gives the multiplexer one service
identity. When `MCP_CLIENT_AUTH=oidc-client-credentials`, it mints a Keycloak
service-account token (OAuth2 `client_credentials`, audience `agent-services` —
the same audience children verify), caches and refreshes it, and the multiplexer
attaches `Authorization: Bearer <token>` to every **remote** child that doesn't
declare its own header. It never overrides an explicit header, and a mint failure
degrades to no header (the child then 401s — visible in metrics/logs, not a crash).

Configuration (multiplexer service):

| Env | Value |
|-----|-------|
| `MCP_CLIENT_AUTH` | `oidc-client-credentials` |
| `OIDC_CLIENT_ID` | `mcp-multiplexer` (Keycloak confidential client) |
| `OIDC_CLIENT_SECRET` | injected from OpenBao at deploy |
| `OIDC_AUDIENCE` | `agent-services` (default) |
| `OIDC_TOKEN_URL` | derived from the JWT issuer if unset |

## /metrics and /health are unauthenticated

`create_mcp_server` registers `GET /metrics` and `GET /health` as custom routes
**outside** the auth/eunomia path (same pattern as graph-os `/health`), so
Prometheus and blackbox probes need no token. They are overlay-network-scoped
(no Caddy route). See [Observability](observability.md).

## Rollout

Auth is rolled out in phased waves so each flip is verified before the next; the
two gates above (token provisioning + baseline policy) come first. The end-to-end
procedure — creating the Keycloak client, loading the policy, flipping services,
and rolling back — is in the
[MCP Fleet Auth & Monitoring runbook](../guides/mcp-fleet-auth-and-monitoring-runbook.md).
