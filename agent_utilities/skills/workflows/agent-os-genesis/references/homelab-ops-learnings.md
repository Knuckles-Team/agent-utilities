# Homelab operations & provisioning learnings (day-0 / day-2)

Hard-won, reproducible learnings from standing up + repairing the connector fleet, SSO,
and app provisioning. Companion to [`connector-ingestion-owl.md`](connector-ingestion-owl.md)
(what each connector ingests + its OWL mapping) and
[`plane-provisioning-and-connector-auth.md`](plane-provisioning-and-connector-auth.md)
(the GitLab-OIDC recipe + the Caddy-security deadlock). This doc is the **operations
playbook** — the patterns that recur across services.

## 1. "Deployed but unprovisioned / lost-creds" is the #1 app failure mode

Many stacks deploy (container Running, web 200) yet are unusable because their **env was
never injected or was lost** (the `.env`/secret didn't reach the swarm service, so the
container booted with empty `${VARS}`). Symptoms + the fix, by app:

- **Firefly III** → 500 on every page. Cause: `APP_KEY` + `DB_*` empty (Laravel can't boot).
  The creds existed in `services/firefly-iii/.env` but were never injected into the service.
  Fix: `docker service update --env-add` the `.env` values (`DB_DATABASE/USERNAME/PASSWORD`,
  `APP_KEY`, `APP_URL=http://firefly.arpa`, `APP_ENV=production`, `TRUSTED_PROXIES=**`) →
  the official image auto-migrates on boot → "Login to Firefly III". **GOTCHA:** if the DB
  volume already initialized with a now-lost user (`psql: role "X" does not exist`), recover
  the original `.env` creds rather than re-init (the volume keeps the first-init creds).
- **ERPNext / Frappe** → API 403 (no token). The site (`erpnext.arpa`) existed + login worked;
  it just had no API key. Fix (in the `frappe/erpnext` container, `/home/frappe/frappe-bench`):
  `bench --site <site> set-admin-password '<pw>'` then a `bench --site <site> console` snippet
  that sets `User('Administrator').api_key/api_secret = frappe.generate_hash(15)` + commits.
  The connector token is **`api_key:api_secret`** sent as `Authorization: token <key>:<secret>`.
  Verify: `GET /api/method/frappe.auth.get_logged_user` → `{"message":"Administrator"}`.
- **Plane** → "welcome/get started", god-mode won't load. Cause: **0 InstanceAdmins**. Fix
  (Django shell in `plane-aio`, `/app/backend`): create `InstanceAdmin` for the user, set its
  password, `Instance.objects.update(is_setup_done=True)`, then **`cache.clear()`** (the instance
  config is Redis-cached — without the clear the API keeps reporting `is_setup_done=false`).

**Pattern:** check `services/<app>/.env` + OpenBao `apps/<app>` for the real creds first; if the
app initialized once and lost them, reset via the app's own CLI (bench / artisan / Django shell /
`manage.py`); always verify the credential against the live API before wiring it. Store the
recovered/created creds in **OpenBao `apps/<service>`** so it survives.

## 2. Connector ingestion: the graph-os HOST is the sweeper

`source_sync` sweeps run on **`graph-os_graph-os-host`** (the worker daemon), NOT the
`graph-os_graph-os` MCP server. So connector creds + the connector package's PYTHONPATH must be
on the **host** service. The recurring "no source client for X" / `ModuleNotFoundError` was the
host missing the env (camunda/egeria/twenty) or the package path (erpnext). Fix: inject the
`<SVC>_URL`/`<SVC>_TOKEN` (copied from the working `*-mcp` service env or OpenBao) AND append
`agents/<name>-mcp` to `PYTHONPATH` on **both** `graph-os_graph-os` and `graph-os_graph-os-host`.

- **Two wiring styles:** `MATERIALIZE_SOURCES` (graph-os builds the connector's client in-process
  and talks to the app directly — camunda/twenty/egeria/erpnext) vs `MCP_TOOL_PRESETS` (declarative
  server+tool+field-map, AU-KG.ingest.mcp-tool-connector). Either way, entities must flow through `ingest_external_batch`
  as **typed OWL nodes** (`:Issue`, `:ContainerImage`, …) — the generic Document path makes
  everything `:Document`. New OWL classes go into the **canonical** ontology (`ontology_*.ttl` via
  `owl_bridge` + `PROMOTABLE_NODE_TYPES`), never a per-connector `.ttl`; keep the SHACL/valid/
  connected gate green.
- `source='all'` candidacy is gated by per-source config detection (`_MCP_TRACKER_SERVERS` /
  configured-server resolution) — an unconfigured source **skips gracefully**, never errors.

## 3. The connector-fleet MCP bug patterns (audit findings)

A fleet audit of ~36 `*-mcp` servers surfaced a dominant class: **tools list (auth works) but every
call fails** because:
- **upstream URL defaults to `localhost`** → inject `<SVC>_URL=http://<stack>_<svc>:<port>` (the
  overlay service name) — e.g. owncast `:8080/api`, mealie `:9000`, searxng `searxng_searxng:8080`,
  lgtm `lgtm_grafana:3000`+`lgtm_alertmanager:9093`, jellyfin `jellyfin_jellyfin:8096`.
- **`Failed to resolve dependency 'client'`** = the connector's `get_client` raised at FastMCP
  dependency-resolution time — usually missing creds, OR an **eager connectivity probe in
  `BaseApiClient.__init__`** that 404/401s (fix: remove the eager probe; build lazily).
- **Eunomia `/check/bulk` 400 "Maximum allowed: 100"** for servers exposing >100 tools — the
  shared `apply_bulk_check_chunking` (ECO-4.88) chunks at 100, but a container started **before**
  that commit landed in `/au` runs the old in-memory middleware → **redeploy to reload `/au`**.
  Any `*-mcp` started pre-ECO-4.88 with >100 tools needs a rolling restart.
- searxng instances 500 on `/search` = default 3s `outgoing.request_timeout` → raise to ~8s.

## 4. SSO over an internal-HTTP IdP (Keycloak `keycloak.arpa` is HTTP)

Full GitLab recipe + the four `discovery:false` gotchas live in
[`plane-provisioning-and-connector-auth.md`](plane-provisioning-and-connector-auth.md). Key
operational adds:
- **JWKS rotation** breaks `discovery:false` SSO (the embedded `client_jwk_signing_key` goes stale).
  Automated by `/home/apps/scripts/gitlab-jwks-refresh.py` + a daily cron on R820 (re-fetches the
  keycloak JWKS and `docker service update`s gitlab only when it changed).
- **Keycloak client secret divergence:** the live client secret (Keycloak admin) can differ from
  what's injected into the swarm service env / `~/.claude.json` → `401 invalid_client` mints. Always
  fetch the secret fresh from the Keycloak admin API when minting for diagnosis; for the deployed
  fleet, keep the service env's `OIDC_CLIENT_SECRET` in sync with the live client.
- **Caddy-security (`authp`) startup deadlock** (its OIDC provider provisions at boot by fetching
  keycloak metadata via `keycloak.arpa`, which loops through Caddy): documented in the auth doc —
  **never blindly restart Caddy** while that block is active; re-enabling it safely needs a
  keycloak endpoint reachable at Caddy startup that doesn't loop through Caddy.

### Multiplexer → child service-account auth (the "fleet-wide 401 that isn't the deployed mux")

The multiplexer mints a Keycloak client-credentials bearer (`mcp-multiplexer`, audience
`agent-services`) and attaches it to every jwt-protected child (CONCEPT:AU-OS.identity.so-jwt-protected-children,
`mcp/client_credentials.py` `child_auth` → `ClientCredentialsAuth`). If the mint fails it
**degrades to no auth** and the child returns **401** — so a single bad mint config looks like a
fleet-wide child outage.

There are **two multiplexers**, and the per-session one is the usual culprit:
- **Deployed swarm mux** (`mcp-multiplexer.arpa`, `mcp_config_central.json`) — fronts the fleet.
- **Per-session local mux** — each Claude session spawns its OWN: `~/.claude.json` →
  `mcpServers.mcp-multiplexer` = `python -m agent_utilities.mcp.multiplexer --config
  mcp_config_claude.json`. Its config **already points every child at the swarm `.arpa`
  services** (it's just the session's aggregator, not a second fleet), and it mints its own bearer
  from the `env` OIDC vars.

Two ways the local mux's mint silently breaks → `invalid_client` → no bearer → **every** child 401:
1. **Wrong realm in `OIDC_TOKEN_URL`.** Must be `…/realms/homelab/…` — the `mcp-multiplexer`
   client and `agent-services` audience live in **homelab**, NOT `master`. A `…/realms/master/…`
   token URL mints `invalid_client` even with the correct secret. (Observed + fixed: the local
   `~/.claude.json` had the `master` realm.)
2. **Stale `OIDC_CLIENT_SECRET`** — drifts when the Keycloak client secret rotates without
   re-syncing `~/.claude.json`.

**Diagnosis (don't chase the deployed mux):** prove the swarm mux is healthy first — inside its
container, `get_provider().get_token()` mints and a `streamablehttp_client(<child>.arpa/mcp,
auth=child_auth({}))` `initialize()` returns 200. If that works but `load_tools(<child>)` from
your session 401s, the fault is the **local** mux. Confirm by minting with `~/.claude.json`'s exact
`OIDC_TOKEN_URL` + `OIDC_CLIENT_SECRET` — an `invalid_client` pinpoints realm-or-secret.
**Fix:** set `OIDC_TOKEN_URL=http://keycloak.arpa/realms/homelab/protocol/openid-connect/token`
+ the current secret, then **reconnect** the session (the running process must respawn).

**Rotation runbook (so this can't recur):** rotating the `mcp-multiplexer` Keycloak client secret
must fan the new value — and the correct **homelab** realm — to ALL consumers in one pass: the
swarm `mcp-multiplexer` + `graph-os` (server+host) service envs, OpenBao `apps/mcp-multiplexer`,
**and every local `~/.claude.json`**. Treat it like the GitLab-JWKS cron above — a rotation that
misses one consumer causes a confusing partial outage.

## 5. Object storage / presigned URLs (browser must reach the S3 endpoint)

**Plane "failed to upload image cover"** = the backend signs presigned URLs with the **internal**
`http://minio:9000`, which the user's browser can't reach. Fix: expose the MinIO **S3 API** (`:9000`,
NOT the console `:9001`) at a browser-reachable HTTP host (`http://planeminio.arpa` via caddy →
`plane_minio:9000`) and set the app's `AWS_S3_ENDPOINT_URL` to it. The backend resolves `.arpa` too,
so one endpoint serves both signing + browser upload. Generalize to any self-hosted-S3 app: the
presigned endpoint must be resolvable+reachable from the browser, not just the backend.

## 6. Swarm deploy mechanics (the operational gotchas)

- **`RW710` is a swarm WORKER; `R820` is the manager.** Service-level commands (`docker service
  create/update`, `stack deploy`) must run on **R820**. `docker restart` of a swarm task on a
  worker spawns a **stray sibling** in the slot → reconcile from the manager with
  `docker service update --force <svc>`.
- **`docker service update --env-add` persists across `--force`** (it's in the service spec) but a
  **from-source redeploy** (Portainer GitOps / `stack deploy`) re-reads the compose → persist
  durable changes into `services/<app>/compose.yml` (and the live caddy config at
  `/home/apps/caddy/Caddyfile`, which is what the running caddy actually mounts — NOT
  `services/caddy/Caddyfile`, which has drifted).
- **graph-os reload** = `docker service update --force graph-os_graph-os{,-host}` (mounts `/au`
  canonical → reloads merged code + ontology). New ingestion handlers/ontology need this to activate.
- **Multiplexer child-resilience breaker** latches a child FAILED after ~22 restarts; a session's
  go__ tools then refuse — **reconnect the MCP session** to reset the breaker (the deployed fleet is
  unaffected). Avoid churn that racks up restarts.
- **Deploying a small new stack from a worker:** if the manager can't read the workspace compose,
  `docker service create` the single service directly on R820 (e.g. fuseki:
  `secoresearch/fuseki:latest` on the `caddy` net) rather than `stack deploy`.

## 7. Secrets

OpenBao KV v2 at `apps/<service>`; the per-service token carries the `agent-apps-rw` policy
(scoped to `apps/data/*`). Mint a **periodic, renewable** `agent-apps-rw` token from
`BAO_ROOT_TOKEN` (finite-TTL tokens silently expire — that broke connector writes once). Store
every credential you provision/recover at `apps/<service>` so the next bring-up finds it. Rotation
runbook: the OpenBao + GitLab-JWKS sections above + the rotate-credentials skill.
