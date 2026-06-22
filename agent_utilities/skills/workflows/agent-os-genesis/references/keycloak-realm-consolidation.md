# Keycloak realm policy + fleet realm-consolidation (master → homelab)

The homelab runs **one application realm: `homelab`**. `master` is reserved for Keycloak
super-admin only. This file documents the target architecture and the **procedure to migrate the
MCP/agent fleet off `master`** (the part that was still on `master` as of 2026-06-19). See the
keycloak/caddy service `AGENTS.md` for the policy statement and `freshrss-and-sso.md` for the
web-SSO recipe.

## Architecture (target)
- **`homelab`** — every SSO/OIDC consumer: human web apps (Grafana, Twenty, GitLab, Mattermost,
  FreshRSS-via-caddy-security) **and** the MCP/agent fleet (server-side JWT validation + the
  multiplexer's outbound minting).
- **`master`** — Keycloak super-admin console only. No app/service clients.
- **Do not create more realms** unless the end user explicitly asks for multi-realm isolation.

## How the fleet authenticates today (must understand before migrating)
- **~55 MCP services** validate inbound JWTs via three env vars (single-valued):
  `FASTMCP_SERVER_AUTH_JWT_ISSUER`, `…_JWKS_URI`, `…_AUDIENCE=agent-services`.
- **The multiplexer** (`mcp-multiplexer`) is the **outbound minter**: a Keycloak
  **service-account client** `mcp-multiplexer` (client-credentials grant, with an
  `oidc-audience-mapper` adding audience `agent-services`, token lifespan 3600) minted via
  `OIDC_TOKEN_URL`/`OIDC_CLIENT_ID`/`OIDC_CLIENT_SECRET`. It ALSO validates inbound client tokens
  (claude-code etc.) via its own `FASTMCP_SERVER_AUTH_JWT_*`.
- **graph-os** (host `.venv` child, serves the `go__` tools) validates inbound + has its own OIDC.
- **`keycloak-mcp`** is a **special case**: its service account administers Keycloak itself.
  A `master` service account can manage all realms; a `homelab` one needs explicit cross-realm
  admin roles. Plan its move last / separately.

## ⚠️ The hard constraints (why this is not a simple flip)
1. **No multi-issuer support.** `agent_utilities/security/request_identity.py` + `auth.py` validate
   against ONE issuer + ONE JWKS. A `master`-configured server REJECTS a `homelab` token
   (different RS256 signing keys), and vice-versa. There is no env-level "dual-trust".
   → A naive flip is **atomic**: the instant the minter mints `homelab` tokens, every
   not-yet-flipped server 401s.
2. **The multiplexer is the lifeline.** Clients (incl. this agent) reach the whole fleet THROUGH
   it. If its **inbound** realm is flipped while the client still presents a `master` token, the
   operator is **locked out with no tools to recover**. Recovery is `MCP_AUTH_TYPE=none` on the
   multiplexer stack (a deliberate bootstrap fallback) — but that is an operator action.
3. **Classifier gates.** In-place multiplexer env changes, `AUTH_TYPE=none`, and container
   token-scanning are auto-denied to the agent — so the multiplexer/graph-os cutover is
   operator-driven (`! docker service update …`), not agent-driven.

## Migration procedure — zero-downtime (recommended) via dual-trust
Because of constraint #1, the safe path adds dual-trust FIRST:

0. **(code) Add multi-issuer support** to `request_identity.py`/`auth.py`: accept a
   comma-separated `FASTMCP_SERVER_AUTH_JWT_ISSUER`/`…_JWKS_URI`, fetch+merge both JWKS, accept a
   token whose `iss` ∈ the list (signature verified by `kid` against the merged keys). Land on
   local main; servers pick it up on restart (editable `/au` mount).
1. **(groundwork)** In `homelab`, create the service-account clients mirroring `master` **with the
   same client_id + secret + `aud-agent-services` mapper** (`mcp-multiplexer`; others as needed),
   so the cutover is a pure issuer/token-url flip with no secret change.
2. **Dual-trust the servers** — set both vars to `master,homelab` and redeploy **in waves of 5**
   (additive; the minter still mints `master`, so nothing breaks). Verify each wave via a served
   `go__` call before the next.
3. **Dual-trust the lifeline** — multiplexer + graph-os (operator-run; keep an `MCP_AUTH_TYPE=none`
   recovery window). Verify tools still work.
4. **Flip the minter** — multiplexer/graph-os `OIDC_TOKEN_URL` → `realms/homelab`. All servers
   (dual-trust) still accept. Verify.
5. **Drop `master` trust** — set every server's vars to `homelab` only, **waves of 5**.
6. **`keycloak-mcp`** — recreate its service account in `homelab` with cross-realm admin roles
   (`realm-management` on both realms while two exist; on `master` it manages everything by
   default), flip last.
7. **Decommission** — once nothing references `realms/master` (`grep -rl realms/master services/`),
   `master` holds only the KC super-admin console; delete leftover app clients (`caddy-authp`,
   `freshrss` in `master` — superseded by the `homelab` equivalents).

## Without the code change (fallback — has an outage)
If dual-trust isn't added: flip the lifeline (multiplexer outbound + graph-os) and every server
to `homelab` as close to atomically as possible; expect each server's tools to 401 until its wave
lands. Sequence the lifeline first so `go__` core tools recover early, peripheral servers in waves.
Higher risk; only with an operator holding the `MCP_AUTH_TYPE=none` recovery lever.

## Env vars touched (per service)
`FASTMCP_SERVER_AUTH_JWT_ISSUER` + `…_JWKS_URI` (`realms/master` → `realms/homelab`); audience
stays `agent-services`. Minters also: `OIDC_TOKEN_URL` (+ any `OIDC_ISSUER`). Inventory:
`grep -rl realms/master services/*/compose*.yml`.
