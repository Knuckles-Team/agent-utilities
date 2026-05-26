# Implementation Plan: Universal Browser-based OAuth PKCE for the agent-packages Ecosystem

This plan outlines the design and architecture to generalize interactive, browser-based OAuth 2.0 PKCE (Proof Key for Code Exchange) authentication. We will extract our highly successful authentication mechanics from `leanix-agent` and build a generic, highly configurable `BaseBrowserAuthManager` directly inside `agent-utilities`. This enables all SaaS, cloud, and OIDC-supported agents in the `agent-packages` ecosystem to easily adopt premium browser-based SSO logins.

---

## Architectural Analysis of the Ecosystem (30+ Packages)

We have performed a comprehensive analysis of all packages inside `agent-packages/agents/*` to determine their compatibility with interactive browser-based OAuth:

| Tier | Compatibility & Capability | Target Agent Packages | Action & Roadmap |
| :--- | :--- | :--- | :--- |
| **Tier 1: SaaS/Cloud APIs** | Natively support standard Three-Legged OAuth (3LO) / PKCE. High benefit: avoids static password or manual token entry. | **LeanIX** (Completed)<br>**xAI/SuperGrok** (Supports PKCE)<br>**Microsoft Agent** (Supports MSAL interactive flow)<br>**Atlassian Agent** (Supports 3LO)<br>**ServiceNow API** (Supports Auth Code Grant)<br>**Github & Gitlab** (Supports OAuth Apps)<br>**Plane** (Supports OAuth 2.0) | **Roadmap**: Integrate our generalized `BaseBrowserAuthManager` from `agent-utilities` into these packages so they can use full interactive browser logins out of the box. |
| **Tier 2: Enterprise APIs** | Support API Keys or basic credentials against a configurable, private, or local deployment. | **Ansible Tower**<br>**Archivebox**<br>**Systems Manager**<br>**Tunnel Manager** | **No action**: Standard API tokens/PATs or SSH keys are the correct and standard integration path here. |
| **Tier 3: Local/Self-Hosted** | Do not support OAuth 2.0. Rely exclusively on direct local API Keys or basic credentials. | **Jellyfin** (Direct API Keys)<br>**arr-mcp** (Lidarr/Radarr/Sonarr/etc. - API Keys)<br>**AdGuard Home** (Basic Auth)<br>**qBittorrent** (Session Auth)<br>**Uptime Kuma** (Socket/Token)<br>**Mealie** (API Keys)<br>**Home Assistant** (PATs) | **No action**: These services do not run OAuth/OIDC providers; direct token-based authentication is the only viable path. |

---

## Proposed Architecture: Generalized `BaseBrowserAuthManager`

We will create a reusable module `agent_utilities/security/browser_auth.py` containing a generic manager class that encapsulating the entire interactive OAuth flow:

```python
class BaseBrowserAuthManager:
    def __init__(
        self,
        client_id: str,
        auth_endpoint: str,
        token_endpoint: str,
        scopes: str,
        secret_key_prefix: str,
        redirect_host: str = "127.0.0.1",
        redirect_port: int = 56122,
        redirect_path: str = "/callback",
        secrets_client: SecretsClient | None = None,
        refresh_skew_seconds: int = 120,
        extra_auth_params: dict[str, str] | None = None,
        extra_token_params: dict[str, str] | None = None,
    ):
        ...
```

### Key Capabilities Provided Out of the Box:
1. **Cryptographic PKCE**: Automates high-entropy `code_verifier` and SHA-256 `code_challenge` generation.
2. **Ephemeral Loopback Server**: Spawns a background thread running a secure callback server on localhost to capture the redirected code.
3. **Graceful Headless Fallback**: Automatically prompts the user to copy-paste the redirected authorization URL/code in headless or remote SSH environments if the local browser or loopback server fails.
4. **Token Refresh Lifecycle**: Handles silent token updates using `refresh_token` and proactive refresh checks based on expiration timestamps.
5. **Secure Encrypted Persistence**: Automatically saves and loads credentials dynamically under a scoped key structure (e.g., `{secret_key_prefix}/oauth_tokens/{domain}`) using `SecretsClient`.

---

## Proposed Changes

### 1. [NEW] [browser_auth.py](file:///home/apps/workspace/agent-packages/agent-utilities/agent_utilities/security/browser_auth.py)
Create a new file in `agent-utilities` containing the generic `BaseBrowserAuthManager` implementation:
- **`generate_pkce()`**: Shared PKCE verifier/challenge generation.
- **`BaseLoopbackCallbackServer`** & **`BaseLoopbackCallbackHandler`**: Configurable, thread-safe loopback HTTP servers.
- **`BaseBrowserAuthManager`**: The core base class containing `login()`, `refresh_tokens()`, and `resolve_credentials()`.

### 2. [MODIFY] [leanix-agent/browser_auth.py](file:///home/apps/workspace/agent-packages/agents/leanix-agent/leanix_agent/browser_auth.py)
Refactor `leanix-agent` to subclass our new `BaseBrowserAuthManager` from `agent-utilities`, showing how cleanly it reduces duplicated code:
```python
from agent_utilities.security.browser_auth import BaseBrowserAuthManager

class LeanixBrowserAuthManager(BaseBrowserAuthManager):
    def __init__(self, workspace_host: str, client_id: str, redirect_port: int, scopes: str, verify: bool = True):
        super().__init__(
            client_id=client_id,
            auth_endpoint=f"https://{workspace_host}/oauth/authorize",
            token_endpoint=f"https://{workspace_host}/oauth/token",
            scopes=scopes,
            secret_key_prefix=f"leanix/oauth_tokens/{workspace_host}",
            redirect_port=redirect_port,
        )
```

---

## Verification Plan

### Automated Tests
1. **Unit Tests in `agent-utilities`**: Create `tests/test_browser_auth.py` inside `agent-utilities` verifying:
   - PKCE generation correctness.
   - Successful browser-based login and token-exchange mock assertions.
   - Automatic silent proactive token refresh when access tokens are near expiry.
   - Fallback manual prompt when the loopback callback server cannot start.
2. **Regression Testing**: Run the existing `leanix-agent` test suites to ensure refactoring is 100% backward-compatible.
