# Worked example: verified GraphOS authority

Every served graph operation requires a server-minted `GraphSession`. There is
no anonymous or caller-asserted identity mode: REST and network MCP requests use
a validated Bearer JWT. Stdio normally uses a validated external process token,
with one narrow zero-infrastructure exception for tiny packaged-local GraphOS.
Every resulting session contains an authenticated subject, tenant, audience, and
policy revision.

Caller payload fields named `_actor`, `_roles`, or `_tenant` are rejected. They
cannot override identity or tenancy.

## Tiny packaged-local stdio

The only graph process that needs no external identity provider is:

```bash
setup-config generate --profile tiny
agent-utilities-doctor --only graph_identity auth
graph-os --transport stdio
```

This contract also requires `GRAPH_SERVICE_ENDPOINTS`, `KG_AUTH_TOKEN_REF`, and
`KG_IDENTITY_OAUTH2` to remain unset. GraphOS generates an asymmetric key in
memory, signs a short-lived JWT containing fixed neutral service claims, and
validates it through the same decoder used for external tokens. That ephemeral
key and JWT are a one-time proof only: both are destroyed before GraphOS returns
the process-lifetime session. It does not place a user name, host name, endpoint,
filesystem path, credential, or other local identifier in the claims or
persistent state.

This is not a fallback. Selecting a network transport, a non-tiny profile, an
explicit engine endpoint, or either external process-identity source disables the
local authority. Missing, ambiguous, unresolved, or invalid external authority
then aborts startup.

## External authority configuration

All other GraphOS shapes require exactly one external process identity source and
the server validation policy:

```bash
AUTH_JWT_JWKS_URI=https://idp.example.test/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://idp.example.test/realms/agents
AUTH_JWT_AUDIENCE=graph-os
KG_POLICY_VERSION=policy-v1

# Configure exactly one runtime source; no token is stored in AgentConfig.
# Network calls still require their own per-request Bearer identity.
KG_AUTH_TOKEN_REF=secret://graph-os/stdio-token
# Or: KG_IDENTITY_OAUTH2={"token_url":"https://identity.example.test/token",...}
```

For external authority, `AUTH_JWT_JWKS_URI`, `AUTH_JWT_AUDIENCE`, and
`KG_POLICY_VERSION` are mandatory inputs. `AUTH_JWT_ISSUER` should also be
pinned. Private PKI is supported through the runtime TLS profile or CA bundle;
certificate verification is never disabled in code.

The validated claims map as follows:

- subject: `sub`, `client_id`, or `azp`
- capabilities: role, scope, and configured group mappings
- tenant: `tenant_id`, `tenant`, `org_id`, `tid`, or `org`
- audience: the server-configured expected JWT audience
- policy revision: the server-configured `KG_POLICY_VERSION`

Only the literal effective capability `kg:admin` grants graph administration.
It may be supplied directly by a validated token or produced by
`IDENTITY_GROUP_CAPABILITY_MAP`. A generic application role named `admin` is not
an alias and never grants graph administration.

## External stdio authority lifetime

An external stdio process session never outlives its validated token silently.
GraphOS records the bounded expiry in a thread-safe in-memory lease shared by the
main tool boundary and captured background workers; the lease contains no token
or identity material. Before expiry, GraphOS reacquires and validates a token
from the configured source. Renewal succeeds only when subject, actor type,
capabilities, tenant, authentication state, and groups exactly match the original
authority. Identity drift is rejected.

Renewal failures are retried without extending the old lease. Once that lease
expires, tool dispatch and background graph work fail closed until the same
authority is successfully renewed. A static `KG_AUTH_TOKEN_REF` must therefore be
rotated by its runtime secret provider; `KG_IDENTITY_OAUTH2` can reacquire through
the configured client-credentials flow.

## Call the gateway

In production, obtain the token from the IdP using the appropriate interactive
or client-credentials flow.

```python
import httpx

token = "<validated IdP token>"
response = httpx.post(
    "https://graph.example.test/api/graph/query",
    headers={"Authorization": f"Bearer {token}"},
    json={"cypher": "MATCH (n) RETURN count(n) AS count"},
    timeout=30,
)
print(response.status_code, response.json())
```

The same request with curl:

```bash
curl https://graph.example.test/api/graph/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"cypher": "MATCH (n) RETURN count(n) AS count"}'
```

## Failure contract

| Condition | Result |
| --- | --- |
| Missing Bearer token | `401` with a generic verified-identity error |
| Malformed, forged, expired, wrong-issuer, or wrong-audience token | `401` without validator detail |
| Verified token without tenant | `403` |
| Missing configured audience or policy revision | request/session mint fails closed |
| Caller supplies `_actor`, `_roles`, or `_tenant` | tool dispatch rejects the request |
| Generic `admin` role without effective `kg:admin` | no graph-administration scope is granted |
| External process-authority renewal changes identity or capabilities | renewal is rejected; the old lease is not extended |
| External process-authority renewal fails through lease expiry | tool and background graph work fail closed; renewal continues retrying |
| Exact tiny, packaged-local stdio boundary with both external sources unset | one neutral in-memory proof is validated and destroyed; a process-lifetime session is returned |
| Missing/ambiguous external process identity, acquisition failure, or token validation failure | graph-os startup aborts; there is no local fallback |
| Unauthenticated health probe | allowed on `/health`, `/healthz`, `/api/health`, and `/api/healthz` only; status-only JSON with `Cache-Control: no-store`, never readiness/topology detail |
| Unauthenticated `/metrics` | `401` on the gateway; standalone remote metrics require their configured bearer token |

All authentication failures use privacy-safe messages. They do not include token
content, subject, tenant, endpoint, local paths, or policy values. Resource-level
authorization still applies after identity validation through GraphSession scopes,
tenant isolation, and the ontology/ACL policy layer.

See [Identity inheritance](../architecture/identity-inheritance.md) for role and
group normalization and [Configuration](../architecture/configuration.md) for the
complete runtime contract.
