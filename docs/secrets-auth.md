# Secrets & Authentication

> CONCEPT:AU-011 — Secrets & Authentication

This document covers how `agent-utilities` manages secrets, credentials,
and authentication across the agent ecosystem.

---

## Overview

The `SecretsClient` provides a unified, pluggable interface for storing and
retrieving sensitive values (API keys, tokens, SSH credentials, etc.). It
ships with three backends and supports URI-style references for maximum
flexibility.

```
┌─────────────────────────────┐
│        SecretsClient        │  ← High-level API
│  get_or_env() / resolve_ref │
└────────┬────────────────────┘
         │  (pluggable)
   ┌─────┴──────┬──────────────┐
   │ InMemory   │   SQLite     │  Vault (hvac)
   │ (default)  │ (persistent) │  (enterprise)
   └────────────┴──────────────┘
```

## Quick Start

```python
from agent_utilities.secrets_client import create_secrets_client

# Zero-config (in-memory, encrypted)
client = create_secrets_client()

# Store and retrieve
client.set("gitlab/token", "glpat-xxx")
token = client.get("gitlab/token")  # "glpat-xxx"

# Fallback to environment variable
token = client.get_or_env("gitlab/token", "GITLAB_TOKEN")

# URI resolution
token = client.resolve_ref("vault://agents/mcp/gitlab/token")
token = client.resolve_ref("env://GITLAB_TOKEN")
```

## Backends

### InMemory (Default)

- **Zero config** — works out of the box
- Values encrypted with [Fernet](https://cryptography.io/en/latest/fernet/) (AES-128-CBC)
- Lost on process restart
- Best for: development, testing, short-lived agent sessions

### SQLite (Persistent)

- Standard `sqlite3` database + Fernet field-level encryption
- Auto-generates encryption key file (`.key`) alongside the DB
- Key names visible; values encrypted at rest
- Best for: CLI/TUI usage, local development with persistence

```bash
export SECRETS_BACKEND=sqlite
export SECRETS_SQLITE_PATH=~/.agent-utilities/secrets.db
```

### HashiCorp Vault (Enterprise)

- Requires `pip install agent-utilities[vault]` (installs `hvac`)
- Uses KV v2 secrets engine
- Best for: production, multi-tenant, corporate deployments

```bash
export SECRETS_BACKEND=vault
export SECRETS_VAULT_URL=https://vault.example.com
export SECRETS_VAULT_MOUNT=secret
export VAULT_TOKEN=hvs.xxx
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SECRETS_BACKEND` | `inmemory` | Backend type: `inmemory`, `sqlite`, `vault` |
| `SECRETS_SQLITE_PATH` | `~/.agent-utilities/secrets.db` | SQLite database file path |
| `SECRETS_VAULT_URL` | `http://127.0.0.1:8200` | Vault server URL |
| `SECRETS_VAULT_MOUNT` | `secret` | Vault KV v2 mount point |
| `AGENT_SECRETS_MASTER_KEY` | *(auto-generated)* | Base64-encoded Fernet key for encryption |

## How to Load Secrets

### Current: Environment Variables (Still Supported)

The existing pattern continues to work. All `os.environ` / `.env` files are
loaded by `python-dotenv` at startup:

```bash
# .env
LLM_API_KEY=sk-xxx
GITLAB_TOKEN=glpat-xxx
```

### New: SecretsClient (Recommended for Sensitive Values)

```python
from agent_utilities.secrets_client import create_secrets_client

client = create_secrets_client()

# 1. Programmatic storage
client.set("gitlab/token", "glpat-xxx")

# 2. Retrieve with env-var fallback
token = client.get_or_env("gitlab/token", "GITLAB_TOKEN")

# 3. URI references (for config files)
token = client.resolve_ref("vault://agents/mcp/gitlab/token")
token = client.resolve_ref("env://GITLAB_TOKEN")
token = client.resolve_ref("sqlite://gitlab/token")
```

### URI Schemes

| Scheme | Example | Behavior |
|--------|---------|----------|
| `vault://` | `vault://agents/mcp/github/token` | Looks up key in backend |
| `secret://` | `secret://path/to/secret` | Alias for vault:// |
| `env://` | `env://GITLAB_TOKEN` | Reads `os.environ["GITLAB_TOKEN"]` |
| `sqlite://` | `sqlite://gitlab/token` | Looks up key in backend |
| *(plain)* | `gitlab/token` | Direct backend lookup |

### Migration Path

```
Phase 0 (current):  .env + os.environ → plaintext
Phase 1 (this PR):  SecretsClient with encrypted storage + env fallback
Phase 2 (planned):  JWT session tokens + OIDC delegation + per-user isolation
```

## Integration with GraphDeps

The `SecretsClient` is available on `GraphDeps.secrets_client` during graph
execution. Specialist nodes and MCP tools can resolve credentials from
the execution context:

```python
# In a graph step or tool
if ctx.deps.secrets_client:
    token = ctx.deps.secrets_client.get_or_env("gitlab/token", "GITLAB_TOKEN")
```

## MCP Token Delegation (Existing)

The ecosystem already supports OAuth2 token delegation for MCP servers:

1. **`UserTokenMiddleware`** captures incoming Bearer tokens into thread-local storage
2. MCP servers like `gitlab-api` read the token via `threading.local().user_token`
3. The token is exchanged via RFC 8693 (Token Exchange) for a scoped service token

See [middlewares.py](../agent_utilities/middlewares.py) and
[mcp_utilities.py](../agent_utilities/mcp_utilities.py) for the full auth stack
(`--auth-type jwt|oidc-proxy|oauth-proxy|remote-oauth`).

## Endpoint Authentication (auth.py)

The agent server supports two authentication mechanisms that can be used
independently or combined (logical OR):

### API Key (Legacy)

Static shared secret via the ``X-API-Key`` header. Enable with:

```bash
export ENABLE_API_AUTH=true
export AGENT_API_KEY=your-secret-key
```

### JWT Bearer Token (Recommended)

Validates tokens against a JWKS endpoint from any OIDC provider (Azure AD,
Okta, Keycloak, Auth0, etc.). Requires `pip install agent-utilities[auth]`.

```bash
export AUTH_JWT_JWKS_URI=https://login.microsoftonline.com/.../discovery/v2.0/keys
export AUTH_JWT_ISSUER=https://login.microsoftonline.com/.../v2.0
export AUTH_JWT_AUDIENCE=api://my-agent-api
```

The server will accept either a valid API key OR a valid JWT Bearer token.
When no auth mechanism is configured, the server operates in open mode.

| Config Variable | Description |
|----------------|-------------|
| `AUTH_JWT_JWKS_URI` | JWKS endpoint for token verification |
| `AUTH_JWT_ISSUER` | Expected `iss` claim |
| `AUTH_JWT_AUDIENCE` | Expected `aud` claim |

### OIDC Flows by Client Type

| Interface | OAuth2 Flow | Library |
|-----------|-------------|---------|
| WebUI (React/Next.js) | Authorization Code + PKCE | NextAuth.js v5 / Auth.js |
| CLI/TUI (Textual) | Device Authorization Grant | `authlib` + custom CLI flow |
| Service-to-Service | Client Credentials | `authlib` |

## CORS & Host Restriction

CORS and trusted host policies are configurable via environment variables:

```bash
# Restrict to specific origins
export ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com

# Restrict trusted hosts
export ALLOWED_HOSTS=api.example.com,*.example.com
```

Both default to `*` (allow all) when unset. **In production, always set
explicit origins.**

## MCP Token Forwarding

When the agent server invokes MCP tools via subprocess (`MCPServerStdio`),
the user's session token is automatically forwarded:

1. The server stores the user's token in `SecretsClient` or `AGENT_USER_TOKEN`
2. `mcp_utilities.py` injects `AGENT_USER_TOKEN` into the subprocess env
3. MCP tools read `os.environ["AGENT_USER_TOKEN"]` for delegated auth

```python
# In an MCP tool:
token = os.environ.get("AGENT_USER_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"
```

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use short-lived tokens** where possible (rotate every 30–90 days)
3. **Audit logging**: Every `SecretsClient.get()` / `.set()` call is logged
   at INFO level (key name only, never values)
4. **Least privilege**: Only request the scopes needed for the current graph node
5. **Encrypted at rest**: All backends encrypt values before storage
6. **Key file permissions**: SQLite backend creates `.key` files with `0o600`
7. **JWT over API keys**: Prefer JWT Bearer auth for production — tokens
   expire, carry claims, and can be revoked at the IdP
8. **Restrict CORS**: Set `ALLOWED_ORIGINS` to specific trusted origins
