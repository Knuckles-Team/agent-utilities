# Secrets & Authentication

> CONCEPT:OS-5.1 — Secrets & Authentication

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
from agent_utilities import create_secrets_client

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

## Secret Manager CLI

`agent-utilities` provides a built-in CLI to easily populate and manage your local secrets (such as the SQLite database) before running your agent.

```bash
# Set a secret
secret-manager set gitlab/token glpat-my-token

# Set a secret explicitly using the sqlite backend (overrides env var)
secret-manager --backend sqlite set my-service/api-key 12345

# Retrieve a secret
secret-manager get gitlab/token

# List all stored keys
secret-manager list

# Delete a secret
secret-manager delete gitlab/token
```

## Backends### InMemory (Default)

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

### HashiCorp Vault & OpenBao (Enterprise / Open Source)

- Requires `pip install agent-utilities[vault]` (installs `hvac`)
- Uses KV v2 secrets engine
- Best for: production, multi-tenant, corporate deployments
- **OpenBao Support**: OpenBao (an open-source fork of HashiCorp Vault initiated at Vault 1.14.7) is **fully compatible out-of-the-box**. Because OpenBao maintains complete API compatibility with HashiCorp Vault, the `hvac` Python client and all authentication methods (Static Token, AppRole, Kubernetes, OIDC/JWT) work seamlessly. No code or configuration changes are needed.

To configure your agent to use Vault or OpenBao, export:

```bash
export SECRETS_BACKEND=vault
export SECRETS_VAULT_URL=https://openbao.example.com  # Points directly to your OpenBao or Vault server
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
from agent_utilities import create_secrets_client

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

See [middlewares.py](../../agent_utilities/mcp/middlewares.py) and
[mcp_utilities.py](../../agent_utilities/mcp_utilities.py) for the full auth stack
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

When the agent server invokes MCP tools via subprocess (`MCPToolset` stdio transport),
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


## Local Secret Storage (Vault, OpenBao, & SQLite)

The ecosystem provides a unified `SecretsClient` designed to replace static `.env` files, supporting `inmemory`, `sqlite`, and `vault` (HashiCorp Vault & OpenBao) backends.

**Light Configuration Example (SQLite):**
```bash
export SECRETS_BACKEND=sqlite
export SECRETS_SQLITE_PATH=~/.agent-utilities/secrets.db
```

**Usage in Code & URI Schemes:**
Secrets can be resolved securely in Python via the context, or directly in `mcp_config.json` via URI schemes:
```python
# Direct code resolution without os.environ
token = ctx.deps.secrets_client.get_or_env("gitlab/token", "GITLAB_TOKEN")

# URI Scheme support for configuration files
"env_vars": { "GITLAB_TOKEN": "secret://gitlab/token" }
```

**Secret Manager CLI:**
Use the built-in CLI to easily populate your local database before running your agent:
```bash
secret-manager set gitlab/token glpat-xxx
secret-manager list
```

---

## Native xAI OAuth Integration

`agent-utilities` supports native xAI OAuth 2.0 PKCE authentication to access the X / xAI API and search X posts or browse individual posts without hitting static API key limitations.

### Architecture

The authentication flow utilizes the OAuth 2.0 Authorization Code Flow with Proof Key for Code Exchange (PKCE) (RFC 7636).

```
┌──────────────┐          1. Click link          ┌──────────────┐
│ Agent / CLI  ├────────────────────────────────►│ x.com Auth   │
│              │◄────────────────────────────────┤ Login Page   │
│ (Spin Server)│     2. Callback with Code       └──────┬───────┘
└──────┬───────┘    (or manual CLI input)               │
       │                                                │
       │ 3. Exchange Auth Code + Verifier               │
       ▼                                                │
┌──────────────┐                                        │
│  xAI OAuth   │◄───────────────────────────────────────┘
│  Token Endpt │
└──────┬───────┘
       │ 4. Store encrypted tokens in SecretsClient
       ▼
┌──────────────┐
│SecretsClient │
└──────────────┘
```

### Flow Options

1. **Auto-Callback Server**: Launches a temporary local web server (defaults to `http://localhost:8000`) to catch the callback and automatically parse the authorization code.
2. **Manual CLI Fallback**: If a port is occupied or a server cannot be started, prints the authorization URL to the terminal and prompts the user to paste the callback URL or code directly.

### Usage in Python

```python
from agent_utilities.security.xai_auth import XaiAuthManager
from agent_utilities.secrets_client import create_secrets_client

secrets = create_secrets_client()
manager = XaiAuthManager(secrets_client=secrets)

# Perform authentication (launches loopback server or CLI paste fallback)
tokens = manager.login()
print("Access Token:", tokens.get("access_token"))
```

### Auto Token Refresh

The `XaiAuthManager` automatically handles token expiration and token refresh using OAuth 2.0 refresh token rotation:

```python
# Get a valid, fresh token (auto-refreshes if expired; pass auto_login=True
# to trigger the interactive flow when no cached tokens exist)
valid_token = manager.resolve_credentials(auto_login=True)
```

### Loopback & Headless Authentication Support

1. **How Loopback Works Remotely**
   The OIDC callback server runs inside the workspace environment at `http://127.0.0.1:56121/callback`.
   Because `graph-os` runs as an MCP server, standard standard-input prompts (`input()`) cannot be used (since the IDE uses standard input/output for JSON-RPC communication, reading from stdin would hang the MCP server).
   Therefore, the callback server is the exclusive way to exchange tokens without crashing the MCP channel.

2. **How to Authenticate in a Headless/Remote Environment**
   To authenticate using your local browser while the MCP server runs on the remote container/VM:
   * **Forward the Callback Port**: Set up a local port forward for port `56121` in your IDE (or via SSH using `ssh -L 56121:127.0.0.1:56121`).
   * **Authorize**: Click the xAI auth link in your browser and log in.
   * **Seamless Redirect**: When the browser redirects to `http://127.0.0.1:56121/callback`, the traffic will be forwarded back to your remote workspace. The OIDC loopback server will instantly capture the authorization code, exchange it, and save the token securely—completing the setup automatically with zero manual copy-pasting!

### Configuration

The OAuth client ID, issuer, scope, and loopback redirect URI are built-in
constants in `agent_utilities/security/xai_auth.py` (`XAI_OAUTH_CLIENT_ID`,
`XAI_OAUTH_REDIRECT_PORT = 56121`, `XAI_OAUTH_REDIRECT_URI`); the public PKCE
client requires no client secret, so there are no `XAI_CLIENT_ID` /
`XAI_REDIRECT_URI` environment variables to set for the auth flow.

For the X search tools (separate from the auth flow), the following
environment variables are honored:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `XAI_BASE_URL` | xAI API base URL | Override the xAI API endpoint used by the search tool |
| `XAI_SEARCH_MODEL` | built-in default | Model used for X search/browse |

For more details on X search tools, see the [Tools Guide](tools.md).

> **Full Documentation:** HashiCorp Vault & OpenBao setup, encryption details, and API references are covered in the sections above (see [HashiCorp Vault & OpenBao](#hashicorp-vault--openbao-enterprise--open-source) and [Local Secret Storage](#local-secret-storage-vault-openbao--sqlite)).
