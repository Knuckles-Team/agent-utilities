# Worked Example: Minting and Validating a JWT for KG_AUTH_REQUIRED Mode

**What this demonstrates.** CONCEPT:OS-5.14, authenticated identity
enforcement: with `KG_AUTH_REQUIRED=1` the gateway mints the request's
`ActorContext` (actor / roles / tenant) **server-side** from a validated JWT —
caller-supplied `_actor`/`_roles`/`_tenant` kwargs are ignored — and requests
without a valid Bearer token are rejected 401, fail-closed, with only health
probes and `/metrics` exempt. You mint a token, call the gateway with it, and
see every failure mode. Deep dives:
[Autonomous governance and zero trust](../architecture/autonomous_governance_and_zero_trust.md),
[Configuration](../architecture/configuration.md).

**Prerequisites (ladder rung).** The secured rung of
[Deployment configurations](../guides/deployment-configurations.md): the REST gateway
(`python -m agent_utilities`) plus an identity provider that serves a JWKS endpoint
(Keycloak, Azure AD, Okta, Auth0 — anything OIDC). For a self-contained demo,
any static HTTP server hosting a JWKS JSON works; that is exactly what the
smoke run below did.

---

## 1. Server configuration

The relevant typed flags (`agent_utilities/core/config.py`):

```bash
# .env on the gateway
KG_AUTH_REQUIRED=1                     # enforce server-validated identity (default 0)
AUTH_JWT_JWKS_URI=https://idp.example.test/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://idp.example.test/realms/agents     # optional but recommended
AUTH_JWT_AUDIENCE=graph-os                                  # optional but recommended

# stdio MCP servers have no Authorization header; their process identity comes
# from a validated token instead (only consulted when KG_AUTH_REQUIRED is on):
# KG_AUTH_TOKEN=<jwt>
```

Facts to know, verified against `agent_utilities/security/auth.py` and
`agent_utilities/security/request_identity.py`:

- Validation is **JWKS-based** (`AUTH_JWT_JWKS_URI`, fetched over HTTP and
  cached for 5 minutes). There is no shared-secret (HS256) server option on
  this path — keys come from the JWKS document, and the `joserfc` library
  (the `auth` extra) verifies the signature.
- Accepted claims: `exp`/`iat` are validated with 30s leeway; `iss` must equal
  `AUTH_JWT_ISSUER` and `aud` must equal/contain `AUTH_JWT_AUDIENCE` when those
  are configured.
- The claims-to-actor mapping (`actor_from_claims`, first match wins):
  - `actor_id` ← `sub` | `client_id` | `azp`
  - `roles` ← `roles` | `realm_access.roles` (Keycloak) | space-separated
    `scope`/`scp`
  - `tenant_id` ← `tenant_id` | `tenant` | `org_id` | `tid`
  - `actor_type` ← HUMAN when an `email` claim is present, else
    AUTOMATED_SERVICE (provenance only)
  - the minted actor carries `authenticated=True`
- An **invalid** token is always 401 — even when `KG_AUTH_REQUIRED=0`.
- With `KG_AUTH_REQUIRED=1`, requests with **no** token are 401 except the
  exempt paths: `/health`, `/healthz`, `/api/health`, `/api/healthz` and
  `/metrics` (scrapers cannot mint JWTs; `/metrics` carries only aggregate
  counters).
- With `KG_AUTH_REQUIRED=0` (the default), unauthenticated requests pass
  through with a one-time prominent startup warning — the legacy honor-system
  mode.

## 2. Mint a token and call the gateway (Python)

The token below was minted with **PyJWT** and validated through the tree's real
code path (`actor_from_bearer_token`) in the smoke run:

```python
import time

import httpx
import jwt  # pyjwt

PRIVATE_KEY_PEM = open("demo_rsa_private.pem").read()  # pair of the JWKS key
now = int(time.time())
token = jwt.encode(
    {
        "sub": "agent:harvest-runner",
        "iss": "https://idp.example.test/realms/agents",   # must match AUTH_JWT_ISSUER
        "aud": "graph-os",                                 # must match AUTH_JWT_AUDIENCE
        "iat": now,
        "exp": now + 3600,
        "roles": ["kg.writer", "workflow.executor"],
        "tenant_id": "acme",
    },
    PRIVATE_KEY_PEM,
    algorithm="RS256",
    headers={"kid": "demo-key-1"},  # kid should match a key in the JWKS
)

resp = httpx.post(
    "http://localhost:9000/api/graph/query",
    headers={"Authorization": f"Bearer {token}"},
    json={"cypher": "MATCH (n) RETURN count(n) AS c"},
    timeout=30,
)
print(resp.status_code, resp.json())
```

(In production the IdP mints the token — client-credentials grant for service
actors — and you never hold the private key yourself.)

**Expected actor** minted server-side from this token (captured from the smoke
run):

```python
{"actor_id": "agent:harvest-runner",
 "roles": ["kg.writer", "workflow.executor"],
 "tenant_id": "acme",
 "authenticated": True,
 "actor_type": "automated_service"}
```

Every KG read/write in the request is scoped to that actor: ontology
permissioning rows, audit attribution, and (with `KG_BRAIN_ENFORCE` on) the
fail-closed ACL gate all see it. Under `KG_AUTH_REQUIRED=1`, any
`_actor`/`_roles`/`_tenant` tool kwargs a caller supplies are ignored entirely.

## 3. The same call with curl

```bash
TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6ImRlbW8ta2V5LTEi..."  # from your IdP  # sanitizer:ignore
curl -s http://localhost:9000/api/graph/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"cypher": "MATCH (n) RETURN count(n) AS c"}'
```

## 4. Failure modes

All captured by driving `ActorIdentityMiddleware` directly in the smoke run
(`KG_AUTH_REQUIRED=1`, JWKS configured):

| Request | Status | Body |
| --- | --- | --- |
| No `Authorization` header | `401` | `{"error": "Authentication required (KG_AUTH_REQUIRED=1): provide a valid JWT Bearer token"}` |
| Malformed/forged token | `401` | `{"error": "Invalid token signature"}` |
| Expired token | `401` | `{"error": "Token has expired"}` |
| Wrong `iss`/`aud` | `401` | `{"error": "Invalid token claim: ..."}` |
| No token, `GET /health` | `200` | (exempt) |
| No token, `GET /metrics` | `200` | (exempt) |
| Valid token | `200` | (request proceeds as the minted actor) |
| `KG_AUTH_REQUIRED=1` but `AUTH_JWT_JWKS_URI` unset, no token | `401` | `{"error": "Authentication required (KG_AUTH_REQUIRED=1): provide a valid JWT Bearer token (server misconfigured: AUTH_JWT_JWKS_URI unset)"}` |

All 401 responses carry a `WWW-Authenticate: Bearer` header. Note that this
identity layer answers in `401` terms (who are you); resource-level **denials**
for an authenticated actor surface from the ontology permissioning gate as
`PermissionError` inside the tool/endpoint result (e.g. the ORCH-1.42 workflow
permission gate — see the
[ontology-to-workflow example](ontology-to-workflow.md)), not as an HTTP 403
from this middleware.

Stdio MCP servers: with `KG_AUTH_REQUIRED=1` and no valid `KG_AUTH_TOKEN`, the
process identity falls back to a restricted read-only system actor — write
tools are gated until a validated token identity exists.

---

*Verification: smoke-run against this tree (2026-06-11). Executed:
`python3 -m pytest tests/unit/core/test_request_identity.py -q` (passed, 38
combined with the evolution-bridge suite), plus a live one-off that generated
an RSA keypair, served its JWKS from a local HTTP server, minted the exact
token above with PyJWT 2.10.1, validated it through the real
`actor_from_bearer_token` path (actor dict above captured verbatim), and drove
`ActorIdentityMiddleware` for the no-token / bad-token / health / metrics /
valid-token rows of the failure table (statuses and bodies captured verbatim).
The expired-token and wrong-claim rows are from `_decode_jwt`'s explicit
error branches in `agent_utilities/security/auth.py`, exercised by the unit
suite rather than the one-off.*
