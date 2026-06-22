# FreshRSS world-model intake + Caddy/Keycloak SSO — genesis standard

How genesis stands up the **FreshRSS** world-model RSS source (the `freshrss-agent`
connector + the FreshRSS app) and the **reusable Caddy + Keycloak OIDC SSO** layer
(`caddy-security`) that protects `*.arpa` web apps. Validated live 2026-06-18.

Two independent surfaces — keep them separate:
- **Ingestion / API path** (`freshrss.arpa/api/greader.php`) — token auth (GReader
  `GoogleLogin`), driven by the `freshrss-mcp` connector. **Never** behind the SSO portal.
- **Web UI** (`freshrss.arpa/…`) — Keycloak SSO via `caddy-security` + FreshRSS `http_auth`.

---

## 1. FreshRSS app configuration (on its placement node)

FreshRSS ships uninstalled. Install + enable the API via the in-container CLI:

```bash
cid=$(docker ps --filter name=freshrss -q | head -1)
docker exec -e TZ=America/Chicago $cid php cli/do-install.php \
  --default-user admin --base-url http://freshrss.arpa --language en \
  --auth-type form --api-enabled --db-type sqlite
docker exec -e TZ=America/Chicago $cid php cli/create-user.php \
  --user admin --password "<LOGIN_PW>" --api-password "<API_PW>" --language en
# CRITICAL: the CLI writes config as root; the web user (www-data) can't read it →
# GReader ClientLogin returns "configuration cannot be found". Re-apply perms:
docker exec $cid ./cli/access-permissions.sh
```

Verify the API: `curl "http://freshrss.arpa/api/greader.php/accounts/ClientLogin?Email=admin&Passwd=<API_PW>"` → `SID=…/Auth=…` (not 503/Unauthorized).

**Secrets** → OpenBao `apps/freshrss-mcp` (openbao-mcp `open__kv kv2_put`, params
`mount_path`/`secret_path`/`data`): `FRESHRSS_URL`, `FRESHRSS_USER`,
`FRESHRSS_API_PASSWORD`, `FRESHRSS_SSL_VERIFY`. Mirror into `services/freshrss-mcp/.env`.

**Curated feeds** (the world-model categories + ScholarX arXiv unification) via the
`freshrss_subscriptions` MCP tool or `reports/freshrss/seed_feeds.py` over
`curated_feeds.json` (Markets & Finance · Tech/AI/Cyber · Macro/Geopolitics ·
Science+Breaking · **Research (ScholarX)** = arXiv cs.AI/LG/CL/MA/CR + stat.ML).

## 2. freshrss-mcp connector deploy (editable, no publish)

`services/freshrss-mcp/compose.dev.yml` — source-mount `/au`+`/src`, `PYTHONPATH=/au:/src`,
image built **locally** from `agents/freshrss-agent/docker/debug.Dockerfile` (the prod
Dockerfile pip-installs from PyPI; debug builds from local source). Registered in
`agent-utilities/deploy/mcp-fleet.registry.yml` (host_port 8254) and
`mcp_config*.json` + `services/mcp-multiplexer/mcp_config_central.json`. The KG side
(`MCP_TOOL_PRESETS["freshrss"]` + `_sync_freshrss` + `WorldModelPipelineRunner`) is in
agent-utilities; the source connector authenticates outbound to the JWT-protected
`freshrss-mcp` via the service-account bearer (`MCP_CLIENT_AUTH`, OS-5.32).

---

## 3. Caddy + Keycloak OIDC SSO standard (`caddy-security`)

The reusable pattern for putting **any** `*.arpa` web app behind Keycloak SSO.

### 3a. Build Caddy with caddy-security
`services/caddy/Dockerfile` — add to the `xcaddy build`:
```dockerfile
RUN xcaddy build \
    --with github.com/caddy-dns/cloudflare \
    --with github.com/greenpau/caddy-security
```

### 3b. ⚠️ Deploy gotcha — `registry.arpa` routes THROUGH Caddy
The private registry is served at `registry.arpa` **by Caddy itself**. During a Caddy
cutover the registry is unreachable, so a swarm task that must *pull* the new Caddy
image gets `No such image` and ingress stays down (circular dependency, observed live).
**Always pre-pull the new image onto the Caddy placement node BEFORE the swap:**
```bash
ssh <caddy-node> docker pull registry.arpa/caddy:cloudflare-authp   # while Caddy is UP
ssh <manager>   docker service update --image registry.arpa/caddy:cloudflare-authp --force caddy_caddy
```
Recovery if it goes down: the old image is still local on the node — revert with
`docker service update --image registry.arpa/caddy:cloudflare caddy_caddy` (falls back
to the cached local image).

### 3b.5 ⭐ ONE realm — `homelab` (no realm sprawl)
**All SSO/OIDC — human web apps AND the MCP/agent fleet — uses the single `homelab` realm**
(`http://keycloak.arpa/realms/homelab`); `master` is Keycloak super-admin only. Do NOT create
extra realms unless the end user explicitly asks for multi-realm isolation. See the keycloak
service `AGENTS.md` realm policy. (SSO users must have an `email` — caddy-security requires the
`email` claim, else "Unauthorized" after login.)

### 3c. ⚠️ Keycloak: remove the RSA-OAEP encryption key
caddy-security cannot parse Keycloak's default **`rsa-enc-generated`** key (algorithm
`RSA-OAEP`) in the realm JWKS → `jwks unsupported key algorithm RSA-OAEP`, provisioning
fails. Delete that **encryption** KeyProvider from the `homelab` realm (the RS256 **signing**
key all token validation uses is separate and untouched; encryption keys are only used for JWE,
which the fleet does not request):
```
keycloak-mcp keyc__agent_components delete_components_by_id {realm:homelab, id:<rsa-enc-generated id>}
```

### 3d. Keycloak client (one client, reusable for all apps)
A single confidential client `caddy-authp` in the **`homelab`** realm (standard flow) with
redirect URIs **`http://*.arpa/*`** — the wildcard makes EVERY app host a valid callback, so each
app hosts its own `/oauth2/<realm>/authorization-code-callback` (see 3e).

### 3e. ⚠️ `.arpa` is a PUBLIC SUFFIX → host the auth PER-APP (host-only cookie)
**Do NOT use a central `auth.arpa` portal with `cookie domain arpa`.** Browsers reject a
`Domain=arpa` cookie (`arpa` is on the Public Suffix List), so the session cookie never sticks —
caddy-security logs a "Successful login" but the user **loops back to the login screen**. Instead
mount the caddy-security OIDC endpoints **on each app's own host** so the cookie is **host-only**
(no `Domain`). This is also how the other homelab apps' OIDC sessions work. Use a **per-app
authorization policy** whose `set auth url` is on that app's host.

```caddyfile
{
    # …email, acme_dns…
    order authenticate before respond
    order authorize before reverse_proxy
    security {
        oauth identity provider keycloak {
            # ⚠️ caddy-security uses `realm` as the OAuth URL path segment
            # (/oauth2/<realm>) AND the `transform match realm <realm>` key — it must
            # equal the name in the policy's `set auth url …/oauth2/<realm>` and the
            # transform below (NOT the Keycloak realm, which lives in metadata_url).
            realm keycloak
            driver generic
            client_id caddy-authp
            client_secret <SECRET>
            scopes openid email profile
            metadata_url http://keycloak.arpa/realms/homelab/.well-known/openid-configuration
        }
        authentication portal authp {
            crypto key sign-verify <RANDOM_64_HEX>   # the portal SIGNS the session JWT with this
            cookie insecure on              # .arpa is plain http (no TLS); NO `cookie domain` (host-only)
            enable identity provider keycloak
            transform user {
                match realm keycloak
                action add role authp/user  # grant every keycloak user the app role
            }
        }
        authorization policy freshrss_policy {
            # ⚠️ the policy MUST carry the SAME key to VERIFY the portal's JWT — without it the
            # policy auto-generates a different (ES512) key, fails with "keystore: failed to parse
            # token" on every request, and the browser loops (ERR_TOO_MANY_REDIRECTS).
            crypto key verify <RANDOM_64_HEX>
            set auth url http://freshrss.arpa/oauth2/keycloak   # the app's OWN host
            allow roles authp/user authp/admin
            inject headers with claims
            # ⚠️ caddy-security's session user object carries ONLY sub/name/email/given_name/
            # family_name/roles — it DROPS preferred_username. {http.auth.user.id} resolves to
            # the EMAIL. Source the upstream username header from a claim that IS present AND is a
            # valid app username. given_name works once set to the app's exact login (see 3f).
            inject header "X-WebAuth-User" from given_name
        }
    }
}

http://freshrss.arpa {
    handle /api/* { reverse_proxy freshrss_freshrss:80 }   # API: token auth — BYPASS the portal
    handle /oauth2/* { authenticate with authp }           # OIDC endpoints ON this host → host-only cookie
    # ⚠️ after login caddy-security lands on /portal (its default) which a per-app host has no
    # route for → app 404. Bounce it to the app root. Use an ABSOLUTE target: `redir / 302`
    # (leading slash) is mis-parsed as a matcher and yields an empty 200.
    handle /portal* { redir http://freshrss.arpa/ 302 }
    handle {                                                # Web UI: Keycloak SSO
        authorize with freshrss_policy
        reverse_proxy freshrss_freshrss:80               # NO header_up — the policy injects X-WebAuth-User
    }
}
```
*(Each additional app repeats the `handle /oauth2/* … /portal … authorize` blocks on its own host
with its own `<app>_policy`; the one `caddy-authp` client + `*.arpa/*` redirect covers them all.)*
The **live** runtime Caddyfile is `/home/apps/caddy/<…>/Caddyfile` (NOT the
`services/caddy/Caddyfile` GitOps source). Edit live, `caddy validate` (rejects a bad
config — the running config persists), then `caddy reload`.

### 3f. App side — FreshRSS `http_auth` (TWO non-obvious requirements)
```bash
docker exec $cid php cli/reconfigure.php --auth-type http_auth   # trust X-WebAuth-User
docker exec $cid ./cli/access-permissions.sh
```
**(a) The header value must equal an existing FreshRSS username.** FreshRSS usernames are
`[0-9a-zA-Z_]` only (NO `@`/`.`) so the email can't be used. caddy drops `preferred_username`,
so we feed `given_name` (3e) — and it must match the FreshRSS account **exactly** (case-sensitive).
Set the Keycloak user's `firstName` to the lowercase login (e.g. `admin`) so `given_name=admin`.
⚠️ do NOT clear `lastName` (empty → Keycloak "Account is not fully set up" → login blocked).

**(b) FreshRSS only honors the header from a TRUSTED proxy IP.** `httpAuthUser(onlyTrusted:true)`
checks `config.php` `trusted_sources`; if the Caddy container's source IP isn't listed, FreshRSS
**ignores the header and 403s** — the error page shows the (untrusted) value, e.g.
`Remote-User=admin`, which looks like it should work. Add the Docker/overlay proxy subnet:
```php
// data/config.php → 'trusted_sources' => [ '127.0.0.0/8', '::1/128', '172.16.0.0/12' ],
```
(edit via `docker cp` out → patch → `docker cp` in → `cli/access-permissions.sh`).

The GReader API password keeps working under any web `auth_type` (the `/api/*` bypass + token
auth), so ingestion is unaffected throughout.

### 3g. End-to-end ordered checklist (every gate that must line up)
A 403/loop means ONE of these is off — they were each a separate failure mode in the live build:
1. **Realm path** — `set auth url …/oauth2/<realm>` matches the provider's `realm` label (not the
   provider name) and the `transform match realm <realm>` — else 400 "identity provider not found".
2. **One realm = `homelab`** — provider `metadata_url` → `realms/homelab`; remove that realm's
   `rsa-enc-generated` RSA-OAEP key.
3. **Host-only cookie** — auth hosted on the app's host, NO `cookie domain` (`.arpa` public-suffix
   → `Domain=arpa` cookie rejected → redirect loop).
4. **Policy verify key** — `crypto key verify` on the policy == the portal's `sign-verify` (else
   "keystore: failed to parse token" → loop).
5. **email claim** — the SSO user has an `email` (else "Unauthorized" after a successful login).
6. **/portal → /** — bounce caddy-security's post-login landing to the app root.
7. **Username header** — inject `X-WebAuth-User` from a present+valid claim (`given_name` set to
   the exact app login); NOT `{http.auth.user.id}` (=email) and NOT `preferred_username` (dropped).
8. **trusted_sources** — the app trusts the Caddy proxy subnet.

**To add another app:** add an `<app>_policy` (auth url on that app's host) to the global
`security` block, then an `http://<app>.arpa { handle /api/* … ; handle /oauth2/* {
authenticate with authp } ; handle { authorize with <app>_policy ; reverse_proxy … } }` block
(bypass any token/API subpaths; host `/oauth2/*` on the app so the cookie is host-only), and set
the app to trust the proxy header. **No** new Keycloak client/realm — the one `caddy-authp` client
(`*.arpa/*` redirect) + `homelab` realm serve all apps.
