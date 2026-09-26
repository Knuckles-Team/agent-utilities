# OAuth 2.0 / OIDC SSO Authentication Guide

This guide documents the standardized OAuth 2.0 / OIDC authentication
architecture used across the entire agent-packages ecosystem.  It replaces
personal access tokens (PATs) with enterprise SSO-based identity propagation.

## Overview

Every MCP server in the ecosystem supports three authentication patterns:

| Pattern | When to Use | Example Agents |
|---------|-------------|----------------|
| **Full Delegation** | Downstream API supports OIDC token exchange (RFC 8693) | GitLab, GitHub, ServiceNow, Atlassian*, EARs*, Ansible Tower* |
| **Hybrid** | Agent has its own auth flow (e.g. MSAL) alongside OIDC | Microsoft Agent |
| **Identity Passthrough** | Downstream API uses API keys only; SSO secures MCP layer | Langfuse |

\* = Can also use identity passthrough if the downstream service doesn't support token exchange.

## Architecture

### Authentication Flow

<div class="admonition architecture" markdown>
<p class="admonition-title">Authentication flow, in three phases</p>

**Phase 1 — client authenticates with the IdP.** The user (or AI client) logs
in via SSO, device-code, or client-credentials; the IdP returns an
access-token JWT.

**Phase 2 — client calls MCP with the IdP token.** The user calls an MCP
tool with `Authorization: Bearer [IdP-token]`. The FastMCP server's
`OIDCProxy` validates the token against the IdP's JWKS, then passes the
validated request to `UserTokenMiddleware`, which binds the verified token
and claims into request-scoped `ContextVars`.

**Phase 3 — the tool handler creates an API client.** MCP calls the agent's
`auth.py::get_client(config)`, which calls
`delegated_auth.py::get_delegated_token(audience, scopes)`. That helper
performs an RFC 8693 Token Exchange against the IdP (`subject_token` →
downstream token) and returns the delegated access token up through
`auth.py`, which calls the downstream API (Jira, GitLab, etc.) with it.
The API's response flows back through MCP as the tool result.
</div>

### Component Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Shared infrastructure feeds three agent patterns</p>

`agent-utilities`' shared infrastructure — `core/config.py` (`AgentConfig`
with OIDC fields, from XDG `config.json`) → `server_factory.py` (`OIDCProxy`
+ CLI parser) → `middlewares.py` (`UserTokenMiddleware`) →
`delegated_auth.py` (`get_delegated_token`, `get_3lo_authorization_url`,
`exchange_authorization_code`) — underlies every agent's auth. `delegated_auth.py`
supplies tokens directly to the **Full Delegation Agents** (`gitlab-api`,
`github-agent`, `servicenow-api`, `atlassian-agent`, `leanix-agent`,
`ansible-tower-mcp`, each OS-5.1 or KG-2.6) and to the **Hybrid Agent**
(`microsoft-agent`, MSAL + OIDC). `context_helpers.py` (progress,
elicitation, logging, state, and sampling) supplies tool-context utilities
to all of those agents alike. `UserTokenMiddleware` separately audits user
identity for the **Identity Passthrough** agent (`langfuse-agent`, API keys
+ audit logging), which does not receive a delegated token.
</div>

## Two-Layer Auth Architecture

Authentication is split into two independent layers:

### Layer 1: MCP Transport Security (already built into `agent-utilities`)

The MCP server itself is protected by an OIDC/OAuth proxy.  This layer
validates that the **caller** has a valid IdP-issued token before any tool
is executed.

- Configured via `--auth-type oidc-proxy` on the MCP CLI
- Uses FastMCP's built-in `OIDCProxy` / `OAuthProxy` / `JWTVerifier`
- `UserTokenMiddleware` binds the verified Bearer token and claims to
  request-scoped context variables

### Layer 2: Downstream API Delegation (this module)

Each agent's `auth.py` reads the stored user token and performs an
**RFC 8693 Token Exchange** to obtain a service-specific token for the
downstream API.

- Centralized in `agent_utilities.mcp.delegated_auth`
- Shared helper `get_delegated_token()` eliminates code duplication
- Uses an explicitly configured typed credential reference when delegation is disabled

## Quick Start

### 1. Set Environment Variables

```bash
# Required for OIDC delegation
export AUTH_TYPE=oidc-proxy
export OIDC_CONFIG_URL=https://your-idp.example.com/.well-known/openid-configuration
export OIDC_CLIENT_ID=your-client-id
export OIDC_CLIENT_SECRET_REF=vault://platform/oidc#client_secret
export ENABLE_DELEGATION=True
export AUDIENCE=https://api.downstream-service.com
export DELEGATED_SCOPES="api read write"
```

Or add them to the XDG config file at
`~/.config/agent-utilities/knuckles-team/config.json`:

```json
{
    "oidc_config_url": "https://your-idp.example.com/.well-known/openid-configuration",
    "oidc_client_id": "your-client-id",
    "oidc_client_secret_ref": "vault://platform/oidc#client_secret",
    "enable_delegation": true,
    "delegation_audience": "https://api.downstream-service.com",
    "delegated_scopes": "api read write"
}
```

### 2. Start the MCP Server

```bash
# Any agent — the auth flags are handled by create_mcp_server()
python -m gitlab_api.mcp_server \
  --transport streamable-http \
  --auth-type oidc-proxy \
  --oidc-config-url $OIDC_CONFIG_URL \
  --oidc-client-id $OIDC_CLIENT_ID \
  --oidc-client-secret-ref "$OIDC_CLIENT_SECRET_REF" \
  --enable-delegation
```

### 3. Call Tools with Bearer Token

```bash
# The client includes the IdP-issued token
curl -X POST http://localhost:8000/mcp/tools/call \
  -H "Authorization: Bearer <your-idp-token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "gitlab_projects", "arguments": {"action": "list_projects"}}'
```

## Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AUTH_TYPE` | No | `none` | Auth type: `none`, `oidc-proxy`, `oauth-proxy`, `jwt`, `remote` |
| `OIDC_CONFIG_URL` | For OIDC | — | OIDC discovery URL (`.well-known/openid-configuration`) |
| `OIDC_CLIENT_ID` | For OIDC | — | OAuth 2.0 client ID from your IdP |
| `OIDC_CLIENT_SECRET_REF` | For OIDC | — | `env://`, `vault://`, or `secret://` reference resolved only at runtime |
| `ENABLE_DELEGATION` | No | `False` | Enable RFC 8693 token exchange for downstream APIs |
| `AUDIENCE` | For delegation | — | Target audience for the delegated token |
| `DELEGATED_SCOPES` | No | `api` | Space-separated scopes for the delegated token |

### Provider Credentials When Delegation Is Not Used

Provider endpoints and non-secret client identifiers remain ordinary
configuration. Every token, password, cookie, and OAuth client secret is a
typed `SOURCE_CREDENTIALS` descriptor whose secret-bearing fields contain only
`env://`, `vault://`, or `secret://` references:

| Credential kind | Typical providers | Reference-bearing fields |
|-----------------|-------------------|--------------------------|
| API key / bearer | GitLab, GitHub, LeanIX | `secret` |
| Basic auth | ServiceNow, Ansible Tower | `username_secret`, `password_secret` |
| Cookie session | Browser-backed providers | `secret` |
| OAuth 2.0 | Atlassian, Microsoft, Langfuse | `secret`, `refresh_token_secret`, `client_secret_secret` |

Raw token/password environment credentials are not a supported path. A
connector that cannot resolve its selected credential reference fails closed.

## Auth Patterns in Detail

### Full Delegation (RFC 8693 Token Exchange)

Used when the downstream API accepts OIDC tokens or supports the
Token Exchange grant type.

```python
# In any agent's auth.py:
from agent_utilities.mcp.delegated_auth import (
    get_delegated_token,
    get_user_identity,
    is_delegation_enabled,
)

def get_client():
    if is_delegation_enabled():
        token = get_delegated_token(
            audience="https://gitlab.example.com",
            scopes="api read_repository",
        )
        return Api(url=instance, token=token)

    # Non-delegated mode uses the typed provider; its descriptor contains only
    # a runtime secret reference and materializes the header in memory.
    from agent_utilities.security.credential_provider import get_credential_provider

    material = get_credential_provider().get("gitlab").materialize()
    if not material.headers:
        raise RuntimeError("GitLab credential reference is unavailable")
    return Api(url=instance, headers=material.headers)
```

### Three-Legged OAuth (3LO)

Used for services like Atlassian Cloud that require explicit user consent
via the Authorization Code Grant flow.

```python
from agent_utilities.mcp.delegated_auth import (
    get_3lo_authorization_url,
    exchange_authorization_code,
    refresh_access_token,
)
from agent_utilities.security.secrets_client import create_secrets_client

secrets = create_secrets_client()
client_secret = secrets.resolve_ref(
    "vault://platform/atlassian#client_secret"
)
if not client_secret:
    raise RuntimeError("Atlassian client-secret reference is unavailable")

# Step 1: Build authorization URL
auth_url = get_3lo_authorization_url(
    authorization_endpoint="https://auth.atlassian.com/authorize",
    client_id="your-app-client-id",
    redirect_uri="http://localhost:8080/callback",
    scopes=["read:jira-work", "write:jira-work"],
)

# Step 2: User visits auth_url, consents, gets redirected with ?code=...
# Step 3: Exchange authorization code for tokens
tokens = exchange_authorization_code(
    token_endpoint="https://auth.atlassian.com/oauth/token",
    client_id="your-app-client-id",
    client_secret=client_secret,
    code=authorization_code,
    redirect_uri="http://localhost:8080/callback",
)

# Step 4: Use access_token, refresh when expired
new_tokens = refresh_access_token(
    token_endpoint="https://auth.atlassian.com/oauth/token",
    client_id="your-app-client-id",
    client_secret=client_secret,
    refresh_token=tokens["refresh_token"],
)
```

### Identity Passthrough

Used when the downstream API does not support OIDC at all (e.g. API key
authentication only).  SSO still secures the MCP server.

```python
from agent_utilities.security.credential_provider import get_credential_provider

def get_client():
    # The provider resolves the descriptor's secret reference in memory. Do not
    # log its value or attach user identity to the credential event.
    material = get_credential_provider().get("service").materialize()
    if not material.headers:
        raise RuntimeError("Service credential reference is unavailable")
    return ServiceClient(headers=material.headers)
```

### Hybrid (MSAL + OIDC)

Used by Microsoft Agent, which has its own MSAL device-code flow alongside
standard OIDC delegation.

```python
from agent_utilities.mcp.delegated_auth import (
    get_delegated_token,
    get_user_token,
    is_delegation_enabled,
)

async def get_client():
    # Priority 1: OIDC delegation
    if is_delegation_enabled():
        token = get_delegated_token(audience="https://graph.microsoft.com")
        auth.access_token = token
        return MicrosoftGraphApi(auth)

    # Priority 2: MSAL cached token
    token = auth.get_token()
    if token:
        return MicrosoftGraphApi(auth)

    # Priority 3: MCP user token passthrough
    user_token = get_user_token()
    if user_token:
        auth.access_token = user_token
        return MicrosoftGraphApi(auth)
```

## Troubleshooting

### "No user token available for delegation"

**Cause**: The MCP server received a request without a Bearer token, or
`UserTokenMiddleware` is not configured.

**Fix**: Ensure:
1. The MCP server is started with `--auth-type oidc-proxy`
2. The client sends `Authorization: Bearer <token>` in the request
3. `--enable-delegation` is passed at MCP startup

### "No token_endpoint configured"

**Cause**: The OIDC discovery URL wasn't resolved at startup.

**Fix**: Set `OIDC_CONFIG_URL` to your IdP's well-known endpoint:
```bash
export OIDC_CONFIG_URL=https://your-idp.example.com/.well-known/openid-configuration
```

### "Token exchange failed (HTTP 400/401)"

**Cause**: The IdP rejected the token exchange request.

**Fix**: Verify:
1. `OIDC_CLIENT_ID` is correct and `OIDC_CLIENT_SECRET_REF` resolves at runtime
2. The client is authorized for the `token-exchange` grant type in your IdP
3. The `AUDIENCE` matches the service registered in your IdP
4. The `DELEGATED_SCOPES` are valid for the target service

### "OIDC delegation failed"

**Info**: Delegation and source credentials fail closed. Correct the IdP
configuration or configure an explicitly selected `SOURCE_CREDENTIALS`
descriptor containing runtime references; the agent does not retry with raw
environment credentials.

---

## Vault & OpenBao Integration

The secrets engine (`agent_utilities.security.secrets_client`) integrates with HashiCorp Vault and OpenBao using the same OIDC identity infrastructure documented above. This means the SSO user token that protects the MCP server can also be used to **authenticate to Vault / OpenBao** without retaining a static Vault token.

### How It Works

<div class="admonition architecture" markdown>
<p class="admonition-title">SSO-derived Vault login, cached and TTL-aware</p>

**Phase 1 — already done.** The user authenticated to MCP; `UserTokenMiddleware`
bound the verified token and claims into request `ContextVars`.

**Phase 2 — the agent needs a secret.** `SecretsClient` calls
`VaultBackend.get("gitlab/token")`, which first checks whether it already
holds a valid Vault token. If not, it fetches the IdP JWT from
`UserTokenMiddleware.get_user_token()` and logs in to Vault —
`POST /auth/{auth_mount}/login` with the role and `jwt=IdP_token`. Vault
validates the JWT against the IdP's JWKS and, once valid, issues a Vault
token scoped to the user's policies, which `VaultBackend` caches
TTL-aware. Either way, `VaultBackend` then does
`GET /secret/data/{path_prefix}/gitlab/token` against Vault and returns the
resolved secret value back up to `SecretsClient`.
</div>

### Config ↔ Path Mapping

The `VaultBackend` constructs full secret paths from three components:

```
vault_mount:        secret          ← KV v2 secrets engine mount point
vault_path_prefix:  agents/mcp/     ← where in the mount to scope secrets
key:                gitlab/token    ← the key passed to get()/set()
```

The full path `secret/data/agents/mcp/gitlab/token` breaks down as
`SECRETS_VAULT_MOUNT` (`secret`) / `VAULT_PATH_PREFIX` (`agents/mcp/`) /
the key passed to `client.get()` (`gitlab/token`).

The auth method mount is **separate** from the secrets path:

```
vault_auth_mount:   oidc            ← auth method mount (custom endpoint)
vault_role:         agent-reader    ← role bound to OIDC claims

Auth endpoint: POST /auth/oidc/login
               (supports any custom mount: 'jwt', 'my-okta-auth', etc.)
```

### Authentication Strategies

The documented profiles use workload identity and do not retain a static Vault
credential:

| Method | Use Case | Required Config |
|--------|----------|-----------------|
| **OIDC/JWT** | User-scoped access via SSO token | `VAULT_ROLE`, `VAULT_AUTH_MOUNT` |
| **Kubernetes** | Workload-scoped pod access | `VAULT_ROLE`, projected service-account token |

### Non-secret Vault Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRETS_BACKEND` | No | `engine` | Set to `vault` to enable Vault |
| `SECRETS_VAULT_URL` | For vault | `http://127.0.0.1:8200` | Vault cluster URL |
| `SECRETS_VAULT_MOUNT` | No | `secret` | KV v2 mount point |
| `VAULT_AUTH_METHOD` | No | `auto` | `auto`, `oidc`, or `kubernetes` in documented profiles |
| `VAULT_AUTH_MOUNT` | No | `jwt` | Auth method mount path (custom endpoints supported) |
| `VAULT_ROLE` | For OIDC/K8s | `default` | Vault role name |
| `VAULT_PATH_PREFIX` | No | — | Path prefix within the mount |
| `VAULT_K8S_SA_TOKEN_PATH` | For K8s | `/var/run/secrets/kubernetes.io/serviceaccount/token` | SA token path |

### Usage Examples

#### OIDC/JWT Authentication (Recommended)

```bash
# Environment
export SECRETS_BACKEND=vault
export SECRETS_VAULT_URL=https://vault.example.com
export VAULT_AUTH_METHOD=oidc
export VAULT_AUTH_MOUNT=oidc               # or 'jwt', 'my-okta-auth', etc.
export VAULT_ROLE=agent-reader
export VAULT_PATH_PREFIX=agents/mcp/
```

```python
from agent_utilities.security.secrets_client import create_secrets_client

client = create_secrets_client()
# When called inside an MCP tool handler, the user's SSO token is
# automatically used to authenticate to Vault.
gitlab_token = client.get("gitlab/token")
# → reads: secret/data/agents/mcp/gitlab/token
```

#### CLI Usage

```bash
# Read a secret with OIDC auth and path prefix
secret-manager --backend vault \
  --vault-auth oidc \
  --vault-auth-mount my-okta-auth \
  --vault-role agent-reader \
  --vault-path-prefix agents/mcp/ \
  get gitlab/token
```

### Vault Admin Setup (Prerequisites)

For OIDC/JWT authentication to work, the Vault server must have the auth
method enabled and configured.  This is a one-time setup performed by
the Vault admin:

```bash
# 1. Enable the OIDC auth method (custom mount path supported)
vault auth enable -path=oidc oidc

# 2. Configure it to trust your IdP
vault write auth/oidc/config \
  oidc_discovery_url="https://your-idp.example.com" \
  oidc_client_id="vault-client-id" \
  default_role="agent-reader"

# For a confidential IdP client, an approved secret-aware provisioner resolves
# vault://platform/vault-oidc#client_secret and sends the value to the Vault API
# in memory. Never place the resolved value in this command or a committed file.

# 3. Create a role that maps OIDC claims to Vault policies
vault write auth/oidc/role/agent-reader \
  role_type="jwt" \
  bound_audiences="vault-client-id" \
  user_claim="sub" \
  groups_claim="groups" \
  policies="agent-secrets-read" \
  ttl="1h"

# 4. Create the policy granting KV v2 read access
vault policy write agent-secrets-read - <<EOF
path "secret/data/agents/mcp/*" {
  capabilities = ["read", "list"]
}
EOF
```

### XDG Configuration

All Vault settings can be persisted in the XDG config file:

```json
{
    "vault_url": "https://vault.example.com",
    "vault_mount": "secret",
    "vault_auth_method": "oidc",
    "vault_auth_mount": "oidc",
    "vault_role": "agent-reader",
    "vault_path_prefix": "agents/mcp/"
}
```

---

## 🔗 Generalized Authentication & Credentials Topology

The following diagram provides a comprehensive system-wide visualization of the unified authentication flows across the entire `agent-packages` and `agent-utilities` ecosystem, illustrating the OIDC Proxy verification layer, RFC 8693 Token Delegation, Hybrid MSAL auth, Vault/OpenBao dynamic credential extraction, and the remote loopback port-forwarding flow:

<div class="admonition architecture" markdown>
<p class="admonition-title">System-wide authentication and credentials topology</p>

**User space.** The user (an AI developer) calls a tool directly against
the remote workspace's MCP transport layer, and separately drives a local
web browser through an authorize flow that redirects to
`127.0.0.1:56121` — a local port forward into the secure remote workspace
(container or VM).

**1. MCP transport & verification.** `OIDCProxy`/`OAuthProxy`/`JWTVerifier`
(the FastMCP server) validates the incoming call and passes the validated
JWT to `UserTokenMiddleware`, which binds it to request-scoped
`ContextVars` and hands it, thread-local, to both the delegation and
secrets engines below.

**2. agent-utilities shared auth engine.** `delegated_auth.py` performs
RFC 8693 Token Exchange against the IdP (Okta, Entra ID, Keycloak) to
obtain a delegated access token. `SecretsClient`/`VaultBackend` uses the
same JWT to authenticate to the Vault/OpenBao cluster (KV v2), which
verifies it against the IdP's JWKS and issues a scoped token and secrets
back. The loopback callback server (port 56121) receives the forwarded
browser traffic, exchanges the code with the xAI OAuth provider, and seeds
credentials into the xAI-authenticated agent.

**3. Specialized agent clients**, each fed by the shared engine: Full
Delegation Agents (GitLab, GitHub, Jira, ServiceNow) get a downstream
token from `delegated_auth.py`; Passthrough Agents (Langfuse, etc.) get
fetched secrets from `SecretsClient`; the Hybrid Microsoft Agent (MSAL /
OIDC) draws on both; `x-search-agent`/`x-ingestion-team` complete an xAI
ingest-post workflow through the loopback flow above.

Every specialized agent client ultimately calls its target API or cloud
service (GitLab, Microsoft Graph, Langfuse, etc.) with the credential it
obtained.
</div>


This file lives at `~/.config/agent-utilities/knuckles-team/config.json`
(following XDG Base Directory Specification).
