# Plane provisioning & day-0 connector auth (generalizes to every `.arpa` MCP connector)

This captures the day-0 steps that make a freshly-deployed connector actually
**ingest into the KG** — learned provisioning Plane + the Atlassian (Jira/Confluence)
connectors end-to-end. The connector-auth section applies to **any** `*-mcp` fleet
server the `graph-os` host syncs from, not just Plane.

## 1. Plane instance provisioning (god-mode)

A fresh Plane AIO deploy registers an *instance* but has **no instance (god-mode)
admin**, so `http://plane.arpa` shows "welcome / get started" and `/god-mode/` does
not complete — there is no admin to sign in and `is_setup_done` is false.

Provision via the Django shell inside the `plane_plane-aio` container (manage.py is at
`/app/backend`):

```python
from plane.db.models import User
from plane.license.models import Instance, InstanceAdmin   # NOT plane.db.models
u = User.objects.get(email="admin@homelab.arpa")           # or User.objects.create_user(...)
u.set_password("<password>"); u.is_password_autoset = False; u.is_active = True; u.save()
inst = Instance.objects.first()
InstanceAdmin.objects.get_or_create(instance=inst, user=u, defaults={"role": 20})
Instance.objects.update(is_setup_done=True)
```

**GOTCHA — the instance config is cached in Redis.** After setting `is_setup_done`,
`GET /api/instances/` keeps returning `is_setup_done=false` (so god-mode stays in
setup mode) until you clear the cache:

```python
from django.core.cache import cache; cache.clear()
```

Then `GET /api/instances/` → `is_setup_done=true`, `/god-mode/` serves, and you can
sign in at `http://plane.arpa/god-mode/` with the admin email + password.

**Store the creds in OpenBao** at `apps/plane-mcp` (KV v2 mount `apps/`):
`PLANE_ADMIN_EMAIL`, `PLANE_ADMIN_PASSWORD`, `PLANE_API_KEY`, `PLANE_WORKSPACE_SLUG`,
`PLANE_BASE_URL`, `PLANE_GODMODE_URL`. The `*-mcp` service env mirrors these.

The Plane **API token** (workspace → Settings → API tokens, or via the workspace
user) is what `plane-mcp` uses; it is **workspace-scoped** and sent as `x-api-key`.

## 2. Connector auth — the host must mint a fleet bearer (applies to ALL connectors)

`source_sync` runs on the `graph-os` host/server and reaches each `*-mcp` server over
its **remote `.arpa` URL**. Three things must all be true or the sync **silently
returns 0 nodes** (it degrades to no-auth → the child 401s):

1. **The graph-os host AND server carry the OIDC client-credentials env** so
   `mcp/client_credentials.bearer_auth` mints a service bearer (CONCEPT:OS-5.32):
   `MCP_CLIENT_AUTH=oidc-client-credentials`, `OIDC_CLIENT_ID=mcp-multiplexer`,
   `OIDC_CLIENT_SECRET=<from OpenBao>`, `OIDC_AUDIENCE=agent-services`,
   `OIDC_TOKEN_URL=http://keycloak.arpa/realms/homelab/protocol/openid-connect/token`.
   Without these, `bearer_auth` returns `None` → unauthenticated connect → 401.

2. **Keycloak: the minting client (`mcp-multiplexer`) must inject `aud=agent-services`.**
   The fleet children validate `FASTMCP_SERVER_AUTH_JWT_AUDIENCE=agent-services`, but a
   bare client mints no custom audience → **fleet-wide 401**. Fix once in Keycloak:
   create a client-scope `agent-services` with an `oidc-audience-mapper`
   (`included.custom.audience=agent-services`, `access.token.claim=true`) and add it as
   a **default** client scope to `mcp-multiplexer` (and `claude-code`). Verify a minted
   token decodes to `aud: ["agent-services", ...]`. (See `create_mcp_clients.py`.)

3. **The connector config the host reads must route to the remote URL, not stdio.**
   The host reads `MCP_CONFIG_PATH` (e.g. `/root/.config/agent-utilities/mcp_config_source.json`,
   from the `~/.config/agent-utilities` mount) — **not** the workspace `mcp_config.json`.
   Each connector entry there must be `{"url": "http://<svc>.arpa/mcp",
   "transport": "streamable-http", "disabled": false}` — NOT a `command`/stdio entry
   (a stdio entry makes the connector try to spawn `.venv/bin/<svc>-mcp`, which does not
   exist on the host → `No such file or directory`). Mirror the working `freshrss-mcp` /
   `plane-mcp` entries.

Single-source `source_sync(source="X")` runs **inline on the graph-os server**;
`source_sync(source="all")` enqueues to the **host** worker — both need the env + config
above, so set them on `graph-os_graph-os` AND `graph-os_graph-os-host`.

## 3. Connector API-compat gotchas (per-connector)

- **Jira** — Atlassian removed the classic `/search` (now **410 Gone**); the live search
  is the **search-and-reconcile** endpoint (`/search/jql`), which **400s on an unbounded
  JQL** and **omits `key`/fields unless explicitly requested**. The `jira` preset must use
  a bounded fallback JQL (`created >= "1970-01-01" ORDER BY updated DESC`) and pass a
  `fields` list. (CONCEPT:KG-2.124, `connectors/mcp_tool.py` + `core/source_sync.py`.)
- **Plane** — the client returns a **custom `Response` model** (`.data` parsed, `.response`
  raw), not a `requests.Response`; MCP tools must (a) be annotated **`-> Any`** (a `-> dict`
  annotation makes FastMCP emit a strict outputSchema → *"outputSchema defined but no
  structured output returned"*), and (b) serialize the model to `{status_code, data}`.
  Also `BaseApiClient._validate_auth` must hit a **key-accessible** endpoint
  (`/workspaces/{slug}/projects/`, not `/workspaces/{slug}/` which 401s for API keys),
  or the client never constructs (*"Failed to resolve dependency 'client'"*). (CONCEPT:ECO-4.1.)
- **Eunomia** — `/check/bulk` hard-caps at **100 items**; servers fronting >100 tools
  (plane) must chunk the bulk authz ≤100. (CONCEPT:ECO-4.88, `mcp/eunomia_principal.py`.)

## 4. SSO procedure (Keycloak OIDC for Plane + GitLab)

Both wire to the `homelab` Keycloak realm (`http://keycloak.arpa`). General shape:
create a confidential Keycloak client per app, store the secret in OpenBao
`apps/<app>`, point the app's OIDC config at the realm.

**Plane** (god-mode → *Authentication → OIDC*, or instance config):
- Keycloak client `plane` (confidential), valid redirect
  `http://plane.arpa/auth/oidc/callback/`.
- Set the Plane instance OIDC config (`is_oidc_enabled=true`, client id/secret, the realm
  discovery URL `…/realms/homelab/.well-known/openid-configuration`). Enables the "Sign in
  with SSO" button on the workspace login.

**GitLab** (the `/users/auth/openid_connect` route 404s when OmniAuth OIDC is not wired):
- Keycloak client `gitlab` (confidential), redirect
  `http://gitlab.arpa/users/auth/openid_connect/callback`.
- In `gitlab.rb` (or `GITLAB_OMNIBUS_CONFIG`): enable `omniauth`, add an
  `openid_connect` provider (`issuer: http://keycloak.arpa/realms/homelab`,
  `client_auth_method: 'query'`, `discovery: true`, the client id/secret,
  `redirect_uri` as above), then `gitlab-ctl reconfigure`. The route then resolves and
  the login page shows the Keycloak button.

> Reconfiguring an app's auth can lock out the existing login path, so do it with
> console/root access available and verify the SSO round-trip before removing the
> password fallback.

See also: [`freshrss-and-sso.md`](freshrss-and-sso.md),
[`keycloak-realm-consolidation.md`](keycloak-realm-consolidation.md),
[`connector-catalog.md`](connector-catalog.md).
