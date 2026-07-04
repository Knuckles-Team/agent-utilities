# Plane provisioning & day-0 connector auth (generalizes to every `.arpa` MCP connector)

This captures the day-0 steps that make a freshly-deployed connector actually
**ingest into the KG** — learned provisioning Plane + the Atlassian (Jira/Confluence)
connectors end-to-end. The connector-auth section applies to **any** `*-mcp` fleet
server the `graph-os` host syncs from, not just Plane.

> **What each connector ingests + how it maps to OWL** (its `source_sync` source key, the
> entities, the OWL classes, and the creds/env) lives in the companion
> **`connector-ingestion-owl.md`**. This doc is the *auth/provisioning* half; that one is
> the *ingestion/ontology* half. Both generalize the same `.arpa` connector pattern.

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
   `mcp/client_credentials.bearer_auth` mints a service bearer (CONCEPT:AU-OS.identity.so-jwt-protected-children):
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
  `fields` list. (CONCEPT:AU-KG.compute.jira-first-class-delta, `connectors/mcp_tool.py` + `core/source_sync.py`.)
- **Plane** — the client returns a **custom `Response` model** (`.data` parsed, `.response`
  raw), not a `requests.Response`; MCP tools must (a) be annotated **`-> Any`** (a `-> dict`
  annotation makes FastMCP emit a strict outputSchema → *"outputSchema defined but no
  structured output returned"*), and (b) serialize the model to `{status_code, data}`.
  Also `BaseApiClient._validate_auth` must hit a **key-accessible** endpoint
  (`/workspaces/{slug}/projects/`, not `/workspaces/{slug}/` which 401s for API keys),
  or the client never constructs (*"Failed to resolve dependency 'client'"*). (CONCEPT:AU-ECO.mcp.fastmcp-middleware.)
- **Eunomia** — `/check/bulk` hard-caps at **100 items**; servers fronting >100 tools
  (plane) must chunk the bulk authz ≤100. (CONCEPT:AU-ECO.bus.agent-bus-awareness, `mcp/eunomia_principal.py`.)

## 4. SSO procedure (Keycloak OIDC) — and the internal-HTTP gotchas

Each app: a confidential Keycloak client in the `homelab` realm (`http://keycloak.arpa`),
secret in OpenBao `apps/<app>`, app's OIDC config pointed at the realm.

### ⚠️ The core problem: keycloak's issuer is HTTP (internal), but OIDC tooling assumes HTTPS

`keycloak.arpa` is internal HTTP. OIDC libraries that **auto-discover** force **HTTPS**
for the `.well-known` fetch (the OIDC spec mandates https) → they dial `keycloak.arpa:443`
→ TLS fails. So **`discovery: true` does not work** against this keycloak. Use
`discovery: false` and supply everything discovery would have — but watch the gaps below.

### GitLab — the VERIFIED working recipe (CONCEPT:AU-ECO.connector.plane-provisioning-auth / OS-5)

Keycloak client `gitlab` (confidential), redirect
`http://gitlab.arpa/users/auth/openid_connect/callback`. In `GITLAB_OMNIBUS_CONFIG`
(`services/gitlab/compose.yml`):

```ruby
gitlab_rails['omniauth_enabled'] = true
gitlab_rails['omniauth_allow_single_sign_on'] = ['openid_connect']
gitlab_rails['omniauth_auto_link_user'] = ['openid_connect']
gitlab_rails['omniauth_providers'] = [{
  name: "openid_connect", label: "Keycloak",
  args: {
    name: "openid_connect", scope: ["openid","profile","email"], response_type: "code",
    issuer: "http://keycloak.arpa/realms/homelab",
    discovery: false,                 # discovery:true would force https:443 → TLS fail
    client_auth_method: "query", uid_field: "preferred_username",
    client_signing_alg: :RS256,       # (1) discovery:false needs the alg declared
    client_jwk_signing_key: '<the full JWKS JSON from keycloak certs>',  # (2) SEE BELOW
    client_options: {
      identifier: "gitlab", secret: "<client secret>",
      redirect_uri: "http://gitlab.arpa/users/auth/openid_connect/callback",
      scheme: "http", host: "keycloak.arpa", port: 80,   # (3) pin back-channel to HTTP
      authorization_endpoint: "/realms/homelab/protocol/openid-connect/auth",
      token_endpoint: "/realms/homelab/protocol/openid-connect/token",
      userinfo_endpoint: "/realms/homelab/protocol/openid-connect/userinfo",
      jwks_uri: "/realms/homelab/protocol/openid-connect/certs",
      end_session_endpoint: "/realms/homelab/protocol/openid-connect/logout",
    }
  }
}]
```

The four gaps `discovery:false` opens — each surfaced as a distinct error, fix in order:
1. **`Ssl connect … :443 … tlsv1 alert`** → the back-channel defaulted to https:443; set
   `client_options` `scheme: "http", host, port: 80` (+ the explicit endpoint *paths*).
2. **`Missing parameter: code challenge method`** → the Keycloak `gitlab` client requires
   PKCE but gitlab sends none. Clear it: set the client attribute
   `pkce.code.challenge.method` to `""` (admin API / console), or add `pkce: true` to gitlab.
3. **`undefined method 'include?' for nil`** (after the sign-alg fix) → **the real one**:
   under `discovery:false` the gem builds the verification key from
   `client_jwk_signing_key`/`client_x509_signing_key` **only — it ignores `jwks_uri`**. With
   RS256 and neither set, `public_key` is nil. Fix: set `client_jwk_signing_key` to the FULL
   JWKS JSON from `http://keycloak.arpa/realms/homelab/protocol/openid-connect/certs`.
4. **CAVEAT — key rotation:** that embedded JWKS is static; when keycloak rotates its signing
   key, SSO breaks until re-embedded. Schedule a small refresh that re-fetches the JWKS into
   the gitlab config on rotation (or accept manual re-embed).

`POST /users/auth/openid_connect` → 302 means omniauth is loaded (GET is 404 — it's POST-only;
don't poll GET). Apply live with `docker service update --env-add` AND persist to the compose.

### Plane — CE has NO generic OIDC

This Plane CE build exposes only Google/GitHub/GitLab/Gitea OAuth (no `is_oidc_enabled`). For
Keycloak SSO, **chain via GitLab OAuth** (`IS_GITLAB_ENABLED` + `GITLAB_HOST=http://gitlab.arpa`,
a GitLab OAuth app for Plane) so Plane → gitlab.arpa → Keycloak; or use Plane EE.

### ⛔ Caddy-security (`authp`) startup deadlock — DO NOT restart Caddy blindly

The edge `Caddyfile` (`/home/apps/caddy/Caddyfile`) had a global `security { oauth identity
provider keycloak { metadata_url http://keycloak.arpa/... } authentication portal authp … } }`
block. caddy-security **provisions this at startup** by fetching keycloak's metadata+JWKS — via
`keycloak.arpa`, which **routes through Caddy itself**. So **any Caddy restart deadlocks**
(needs keycloak.arpa → needs Caddy) and takes down **all `.arpa`**. The discovery doc also
advertises `keycloak.arpa:80` endpoints (the Caddy frontend) that Caddy can't reach during its
own boot, so pointing metadata at the internal VIP only moves the failure to the JWKS fetch.
**Recovery used:** comment out the `security {}` block + every `authenticate with authp` /
`authorize with …` directive (only `auth.arpa` + freshrss used it) → Caddy starts → ingress
restored. **Re-enabling caddy-security safely requires a keycloak endpoint reachable at Caddy
startup that does NOT loop through Caddy** (e.g. a tiny always-up keycloak proxy on `:80`, or
serving keycloak.arpa from a non-Caddy ingress for the auth host). Treat a Caddy restart as a
high-blast-radius op; back up the Caddyfile first.

See also: [`freshrss-and-sso.md`](freshrss-and-sso.md),
[`keycloak-realm-consolidation.md`](keycloak-realm-consolidation.md),
[`connector-catalog.md`](connector-catalog.md).
