# Secrets & Authentication

> CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication

This document covers how `agent-utilities` manages secrets, credentials,
and authentication across the agent ecosystem.

---

## Overview

The `SecretsClient` provides a unified, pluggable interface for storing and
retrieving sensitive values (API keys, tokens, SSH credentials, etc.). It
ships with two live backends and supports URI-style references for maximum
flexibility (CONCEPT:AU-OS.identity.encrypted-secret-store).

```
┌─────────────────────────────┐
│        SecretsClient        │  ← High-level API
│  get_or_env() / resolve_ref │
└────────┬────────────────────┘
         │  (pluggable)
   ┌─────┴───────────────────┬──────────────┐
   │ InEpistemicGraphBackend │  Vault (hvac) │
   │ (engine-encrypted,      │  (enterprise) │
   │  default everywhere)    │               │
   └─────────────────────────┴──────────────┘
```

The default `InEpistemicGraphBackend` is a **durable, engine-backed** store:
secrets live as `:Secret` nodes in a dedicated `__secrets__` epistemic-graph
graph, the secret **value** held as an encrypted node property sealed by the
engine's encryption-at-rest (ChaCha20-Poly1305 over the redb value blobs, keyed
by child-only data-key material + the KMS seam — CONCEPT:EG-KG.sharding.row-level-security), while the
key **name** + metadata stay queryable plaintext. It is the store in **every**
profile, because `GraphComputeEngine` auto-starts the full engine artifact installed
by the hard-base `epistemic-graph[full]>=2.23.1,<3.0.0` dependency on demand (the
OS-5.63 resolver). AgentConfig carries
only `EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF`; production and non-tiny local modes
resolve it from an external `env://` or `vault://` source. Non-production `tiny`
mode creates one stable private XDG key. In both cases only the spawned Rust
child receives the raw value; ambient raw/ref variables are scrubbed.

GraphOS's tiny packaged-local stdio bootstrap does not read or store an identity
secret. Only `graph-os --transport stdio` with `DEPLOYMENT_PROFILE=tiny`, no
`GRAPH_SERVICE_ENDPOINTS`, and neither external process-identity source may sign
and validate a neutral short-lived JWT with an in-memory key as a one-time proof.
The key and token are destroyed before a process-lifetime session is returned.
Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point requires exactly one external identity source resolved through the
mechanisms below; an invalid source never falls back locally.

## Quick Start

```python
import os

from agent_utilities import create_secrets_client

# Zero-config encrypted engine storage
client = create_secrets_client()

# Store and retrieve
client.set("gitlab/token", os.environ["GITLAB_TOKEN"])
token = client.get("gitlab/token")

# Fallback to environment variable
token = client.get_or_env("gitlab/token", "GITLAB_TOKEN")

# URI resolution
token = client.resolve_ref("vault://agents/mcp/gitlab/token")
token = client.resolve_ref("env://GITLAB_TOKEN")
```

## Secret Manager CLI

`agent-utilities` provides a built-in CLI for the encrypted engine store or
Vault/OpenBao. Secret values are accepted only through runtime references, never
as process arguments.

```bash
# Set a runtime-injected secret without embedding it in this command
secret-manager set gitlab/token --value-ref env://GITLAB_TOKEN

# Retrieve a secret
secret-manager get gitlab/token

# List all stored keys
secret-manager list

# Delete a secret
secret-manager delete gitlab/token
```

## Backends

### Engine-encrypted `__secrets__` store (Default everywhere)

- **Zero config** — works out of the box; durable across restart
- Secrets are `:Secret` nodes in a dedicated `__secrets__` engine graph
- The secret **value** is an encrypted node property (engine encryption-at-rest,
  ChaCha20-Poly1305, keyed by `EPISTEMIC_GRAPH_ENCRYPTION_KEY`/KMS); the key
  **name** + metadata stay queryable plaintext
- Works on every profile (the OS-5.63 resolver auto-starts the mandatory full
  engine artifact); there is **no** local-disk / RAM fallback

### HashiCorp Vault & OpenBao (Enterprise / Open Source) { #vault-openbao }

- Requires `pip install agent-utilities[vault]` (installs `hvac`)
- Uses KV v2 secrets engine
- Best for: production, multi-tenant, corporate deployments
- **OpenBao Support**: OpenBao (an open-source fork of HashiCorp Vault initiated at Vault 1.14.7) is **fully compatible out-of-the-box**. Because OpenBao maintains complete API compatibility with HashiCorp Vault, the `hvac` Python client and all authentication methods (Static Token, AppRole, Kubernetes, OIDC/JWT) work seamlessly. No code or configuration changes are needed.

To configure your agent to use Vault or OpenBao, export:

```bash
export SECRETS_BACKEND=vault
export SECRETS_VAULT_URL=https://openbao.example.com  # Points directly to your OpenBao or Vault server
export SECRETS_VAULT_MOUNT=secret
```

Prefer workload identity. If token authentication is required, inject
`VAULT_TOKEN` at runtime and never persist it in a tracked file.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SECRETS_BACKEND` | `engine` | `engine` → the durable engine-encrypted `__secrets__` store (default everywhere); `vault` → enterprise OpenBao/Vault |
| `EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF` | *(tiny non-production local mode generates a stable private key; otherwise required)* | External `env://` or `vault://` bootstrap reference for the packaged local engine data key. `secret://` is rejected because it would create a circular dependency on the engine-backed store. |
| `SECRETS_VAULT_URL` | `http://127.0.0.1:8200` | Vault server URL |
| `SECRETS_VAULT_MOUNT` | `secret` | Vault KV v2 mount point |

> The Rust engine still consumes its native raw key variable internally, but
> that variable is not an AgentConfig field. The launcher resolves the reference,
> validates 32–4096 control-free UTF-8 bytes, and materializes the value only in
> the child environment.

### Two surfaces — `graph_secret` MCP tool + `/graph/secret` REST route

Secrets are reachable from the **gateway (REST)** and the **MCP server**, not just
a Python import. The `graph_secret` MCP tool and the `/graph/secret` REST twin both
dispatch into the one `SecretsClient` core. Actions: `set` / `get` / `list` /
`delete`. Mutations (`set` / `delete`) are governed by the ActionPolicy gate
(`secret.set` / `secret.delete`, `approval_required`); `get` / `list` are reads and
not gated. `list` returns key names only — never values.

## How to Load Secrets

### Runtime references

Durable `config.json` stores references, never values. `vault://` and `secret://`
resolve through the configured secret backend. `env://NAME` first uses an explicit
process value; it can also use the optional implicit `runtime-secrets.json` beside
the XDG AgentConfig.

```json
{
  "LANGFUSE_SECRET_KEY_REF": "env://LANGFUSE_SECRET_KEY",
  "GRAPH_DB_CONNECTION_PROFILE_REF": "secret://graphs/primary-profile"
}
```

Both references must resolve to real runtime material. The resolver and
`agent-utilities doctor` reject unresolved templates, redaction masks, and
obvious placeholder sentinels locally, before any network request. Validation
does not assume fixed provider key lengths and never returns or logs the
resolved values.

Raw credential aliases and header maps are process-only inputs and are rejected
when non-empty in durable XDG configuration. Nested models use `api_key_ref` and
`headers_ref`; OAuth2 uses a referenced `client_secret`. A resolved `headers_ref`
must contain one bounded JSON object and is validated against header injection and
hop-by-hop header rules before client construction.

AgentConfig never searches a repository or launch directory for a dotenv file.
The XDG runtime-secret filename is fixed and cannot be configured in `config.json`.
On POSIX it must be owned by the current user or root and have mode `0600` or
`0400`. It contains one JSON object whose keys are environment-variable names.
Only keys targeted by exact `env://` references anywhere in `config.json` are
projected; all others remain unavailable. Explicit process values win. Native
Windows uses explicit process injection because private file sources fail closed
until their ACL posture can be validated at the descriptor boundary.

The loader rejects links, special files, oversized input, malformed JSON,
duplicate or case-ambiguous keys, non-string or empty values, and a durable
configuration key that collides with a referenced secret target. Reload is staged:
invalid replacement content leaves the last valid projection active. Doctor emits
only aggregate status and counts.

### SecretsClient

```python
import os

from agent_utilities import create_secrets_client

client = create_secrets_client()

# 1. Programmatic storage
client.set("gitlab/token", os.environ["GITLAB_TOKEN"])

# 2. Retrieve with an explicit or XDG-projected runtime value
token = client.get_or_env("gitlab/token", "GITLAB_TOKEN")

# 3. URI references (for config files)
token = client.resolve_ref("vault://agents/mcp/gitlab/token")
token = client.resolve_ref("env://GITLAB_TOKEN")
```

### URI Schemes

| Scheme | Example | Behavior |
|--------|---------|----------|
| `vault://` | `vault://agents/mcp/github/token` | Looks up key in backend |
| `secret://` | `secret://path/to/secret` | Alias for vault:// |
| `env://` | `env://GITLAB_TOKEN` | Reads an explicit or referenced XDG-projected process value |

### Current storage contract

```
Durable configuration stores only runtime references; values resolve in memory
from explicit environment variables, Vault/OpenBao, or the encrypted engine
store. Local database files and sibling key files are not read.
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

The ecosystem supports OAuth2 token delegation for MCP servers:

1. **`UserTokenMiddleware`** accepts only the token and claims exposed by the
   configured FastMCP authentication provider, then binds them to
   request-scoped context variables.
2. MCP servers retrieve that verified request authority through
   `agent_utilities.mcp.delegated_auth.get_user_token()` and
   `get_user_claims()`.
3. The token is exchanged via RFC 8693 (Token Exchange) for a scoped service token

See [middlewares.py](https://github.com/Knuckles-Team/agent-utilities/blob/main/agent_utilities/mcp/middlewares.py) and
[server_factory.py](https://github.com/Knuckles-Team/agent-utilities/blob/main/agent_utilities/mcp/server_factory.py) for the full auth stack
(`--auth-type jwt|oidc-proxy|oauth-proxy|remote-oauth`).

## Endpoint Authentication (auth.py)

The agent server uses JWT bearer authentication for remote listeners.

Validates tokens against a JWKS endpoint from any OIDC provider (Azure AD,
Okta, Keycloak, Auth0, etc.). Requires `pip install agent-utilities[auth]`.

```bash
export AUTH_JWT_JWKS_URI=https://login.microsoftonline.com/.../discovery/v2.0/keys
export AUTH_JWT_ISSUER=https://login.microsoftonline.com/.../v2.0
export AUTH_JWT_AUDIENCE=api://my-agent-api
export KG_POLICY_VERSION=policy-v1
```

The server accepts a valid JWT bearer token. When no JWKS endpoint is
configured, only a loopback listener is allowed to operate without
authentication.

| Config Variable | Description |
|----------------|-------------|
| `AUTH_JWT_JWKS_URI` | JWKS endpoint for token verification |
| `AUTH_JWT_ISSUER` | Expected `iss` claim |
| `AUTH_JWT_AUDIENCE` | Expected `aud` claim |
| `KG_POLICY_VERSION` | Immutable policy revision stamped into verified graph sessions |

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

For MCP subprocesses loaded from an MCP configuration (`MCPToolset` stdio
transport), the canonical configuration loader can forward an explicitly
available process token. This startup-time path is separate from per-request
HTTP delegation:

1. `agent_utilities.core.config.load_mcp_servers_from_config()` resolves an
   existing `AGENT_USER_TOKEN`, or a `session_token` from `SecretsClient`.
2. The loader adds `AGENT_USER_TOKEN` to a child configuration only when that
   child did not already declare it.
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
6. **JWT over API keys**: Prefer JWT Bearer auth for production — tokens
   expire, carry claims, and can be revoked at the IdP
7. **Restrict CORS**: Set `ALLOWED_ORIGINS` to specific trusted origins


## Secret Storage (Encrypted Engine, Vault, and OpenBao) { #local-secret-storage }

The ecosystem provides a unified `SecretsClient` designed to replace static
`.env` files. It supports the encrypted engine store (`engine`) and Vault or
OpenBao (`vault`).

**Default encrypted engine storage:**
```bash
export SECRETS_BACKEND=engine
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
Use the built-in CLI with a runtime reference, never a value in the command line:
```bash
secret-manager set gitlab/token --value-ref env://GITLAB_TOKEN
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

> **Full Documentation:** HashiCorp Vault & OpenBao setup, encryption details, and API references are covered in the sections above (see [HashiCorp Vault & OpenBao](#vault-openbao) and [Local Secret Storage](#local-secret-storage)).
