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

## 8. The homelab is Kubernetes (RKE2) — Swarm AND Docker are fully sunset

As of 2026-07, the homelab runs a single **RKE2** cluster (Cilium CNI, kube-proxy-free eBPF);
Docker Swarm is dissolved and **`docker.service` is stopped + disabled on every node**. Only
RKE2's own **containerd** (`/run/k3s/containerd/containerd.sock`) runs. Sections 1-7 above are the
Swarm-era playbook; these are the k8s-native equivalents.

- **No host Docker for builds.** The GitLab runner uses the **kubernetes executor**
  (`privileged=true`, no `docker.sock`/host_path mount); image build/mirror jobs run as **in-pod
  DinD** (`registry.arpa/debian/debian-dind`) and push to the k8s-hosted `registry.arpa`. This is
  what let Docker be removed entirely — the host buildx builders were vestigial. Inspect containers
  with `crictl`, NOT docker: `sudo /var/lib/rancher/rke2/bin/crictl --runtime-endpoint
  unix:///run/k3s/containerd/containerd.sock ps` (the bare `crictl` grabs the dead `cri-dockerd.sock`).
- **Private registry trust per node.** Each node needs `/etc/rancher/rke2/registries.yaml`
  (mirror `registry.arpa` → `http://<r820>:5000` primary + `https://registry.arpa` fallback) **and**
  the CA at `/etc/rancher/rke2/registry.arpa-ca.pem`, then an `rke2-agent`/`rke2-server` restart.
  A node missing these hits `x509: certificate signed by unknown authority` on private pulls (a
  late-joined GPU node hit exactly this). To move a locally-built image onto a node without pushing:
  `sudo docker save <img> | sudo /var/lib/rancher/rke2/bin/ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images import -`.
- **Storage:** DBs / SQLite / any file-locked store → **hostPath + `nodeSelector` (data-in-place),
  NEVER NFS** (NFS locking corrupts Postgres/SQLite). Bulk read-mostly media/blobs → NFS ok.
  Migrate a stateful service by `docker stop` + `docker cp` from the STOPPED container (consistent
  snapshot) → hostPath, then a hostPath-pinned pod. `enableServiceLinks: false` for neo4j (a
  `NEO4J_`-named Service injects `NEO4J_PORT_*` env that neo4j rejects as unknown config).
- **Naming collision gotcha:** the KG mirrors already own the Deployment/Service names
  `neo4j`/`falkordb` in ns `platform`. `kubectl apply` a same-named resource CLOBBERS the mirror
  (repoints it). Check for an existing same-named resource in the target namespace before `apply`.

## 9. DNS + edge model on k8s — and the "wildcard points at the dead edge" trap ★

The homelab keeps its `*.arpa` (internal) + `*.heavenhomestead.com` (public) hostnames across the
Swarm→k8s move so nothing reconfigures. How resolution + routing works now:

- **technitium** is the `.arpa` DNS authority (the site router points DNS at it). **Every k8s
  Ingress host gets an explicit `A → 10.0.0.240`** record (the Cilium LoadBalancer VIP for
  ingress-nginx). ingress-nginx then routes by `Host` header to the right Service.
- **caddy** (hostNetwork on the control-plane node) is the **public** edge (`*.heavenhomestead.com`)
  + a few host-service routes.
- **★ THE TRAP (cost us a confusing outage):** the `.arpa` zone had a **wildcard `*.arpa → <old
  swarm edge IP>`** left over from the Swarm era. After migration, that old caddy ran a STALE
  Swarm Caddyfile whose `reverse_proxy` targets were all dead swarm services → **502**. So any
  `.arpa` name WITHOUT an explicit ingress record (a service under a slightly different name, e.g.
  `uptime.arpa` when the ingress is `uptime-kuma.arpa`) — OR any client that had **cached the old
  wildcard answer** before the explicit record existed — fell through to the dead edge and 502'd.
  Symptom: "some `.arpa` services load, some don't, seemingly at random."
  - **FIX:** repoint the wildcard to the **ingress VIP**, not the old edge:
    `POST /api/zones/records/add?zone=arpa&domain=*.arpa&type=A&ipAddress=10.0.0.240&ttl=60&overwrite=true`
    (technitium API; creds in `services/technitium-dns-mcp/.env`). Now unmatched names hit
    ingress-nginx (served if an ingress exists for that Host, else a clean 404 — never the dead 502).
  - **Name mismatches:** add the expected hostname as an extra `host` on that service's Ingress
    (e.g. add `uptime.arpa` alongside `uptime-kuma.arpa`).
  - **Client-side cache:** a device that cached `name.arpa → old-edge` keeps failing until its
    resolver TTL expires — tell the user to flush DNS. Keeping BOTH the explicit records AND the
    wildcard on the ingress VIP makes the cached-vs-fresh distinction harmless going forward.
  - **Genesis takeaway:** when standing up (or migrating) the edge, NEVER leave a `*.arpa` wildcard
    pointing at anything but the live ingress VIP. Use a low TTL (60s) on migration records.

## 10. The KG host daemon is a single-flock singleton — pin the MCP server to `client`

`graph-os` (MCP gateway) and `graph-os-host` (the consolidated host daemon: queue drain + workers +
**maintenance scheduler** that drives the loop engine, delta-sweeps, enrichment) are separate
deployments. The host daemon holds a singleton `flock` (`host_lock.py`); only ONE process may be
`host`. **Pin `graph-os` to `KG_DAEMON_ROLE=client`** — otherwise both default to `auto` and, because
each pod's lock lives on a PRIVATE emptyDir (not shared storage), BOTH self-elect as host = the
duplicate-drainer thrash that can wedge the scheduler (it silently killed ALL background maintenance
for hours once — alive + holding the lock ⇒ no failover). Defense-in-depth: a **watchdog CronJob**
that restarts `graph-os-host` if the maint-loop log goes silent (no `[maint-loop]` line in 6 min).
The loop-engine `KG_LOOP*` config (interval/topics/**breadth off**/report-only) goes on the
**host daemon** deployment, not the MCP server — the daemon runs the tick.

## 11. OpenBao write-capable service tokens — mint + store (genesis MUST automate this)

Services need an `agent-apps-rw` token to WRITE their own `apps/<service>` secrets. Genesis has to
mint it — the ESO ClusterSecretStore token and the `openbao-mcp` token are (correctly) **read-only**,
so nothing in the running cluster can create a rw token. It can ONLY be minted from the **root token
captured at `bao operator init`** (held by the operator, never stored in the cluster — a `lookup-self`
on the restricted service tokens even fails). The one-time genesis sequence, run with `BAO_ROOT_TOKEN`
exported (operator supplies it via `! bao login` or the init output):

**Where the root token lives:** genesis `operator init` captures it into **`services/openbao/.env`**
as `BAO_ROOT_TOKEN` (+ the unseal keys) — the homelab `.env` convention, NOT the cluster. That is
the ONLY copy; there is no root/unseal secret in k8s (by design). Run the mint against the openbao
pod (it has the `bao` CLI but **no `jq`/`python`** — use `-field=token`, not `-format=json | jq`):

```bash
ROOT=$(grep '^BAO_ROOT_TOKEN=' services/openbao/.env | cut -d= -f2- | tr -d '"')
POD=$(kubectl get pod -n platform -l app=openbao -o jsonpath='{.items[0].metadata.name}')
# feed the root token on STDIN (never a CLI arg); run all steps in-pod:
printf '%s\n' "$ROOT" | kubectl exec -i -n platform "$POD" -- sh -s <<'IN'
read RT; export BAO_ADDR=http://127.0.0.1:8200
# 1. policy: rw on the apps/ KV-v2 mount only (least privilege)
BAO_TOKEN="$RT" bao policy write agent-apps-rw - <<'POL'
path "apps/data/*"     { capabilities = ["create","read","update","delete"] }
path "apps/metadata/*" { capabilities = ["list","read","delete"] }
POL
# 2. PERIODIC (auto-renewing), renewable token — finite-TTL tokens silently expire (broke writes once)
NEW=$(BAO_TOKEN="$RT" bao token create -policy=agent-apps-rw -period=768h -renewable=true -field=token)
# 3. store it for the fleet + as openbao-mcp's OWN token (so it becomes write-capable)
BAO_TOKEN="$RT" bao kv put   apps/openbao     AGENT_APPS_RW_TOKEN="$NEW" >/dev/null
BAO_TOKEN="$RT" bao kv patch apps/openbao-mcp OPENBAO_TOKEN="$NEW"       >/dev/null
BAO_TOKEN="$NEW" bao kv put apps/_probe t=ok >/dev/null && echo "rw VERIFY: OK" || echo "rw VERIFY: FAILED"
IN
# 4. make ESO deliver the new token + restart the consumer
kubectl annotate externalsecret openbao-mcp-secrets -n apps force-sync=now --overwrite
kubectl rollout restart deploy/openbao-mcp -n apps
```

Each `*-mcp` / service stack gets that token as its `OPENBAO_TOKEN` (the `agent-apps-rw` policy) so it
can `bao kv put apps/<service> …` on bring-up. **Genesis automates all of this right after
`operator init` + unseal**, before deploying the fleet. Verify with a real write — a
`403 permission denied` means read-only, not rw (this is exactly how the fleet's read-only ESO +
openbao-mcp tokens present, and why they can't self-mint). Raw HTTP KV-v2 write:
`POST $BAO_ADDR/v1/apps/data/<service>` body `{"data":{…}}` (the openbao pod has no `curl` — use the
`bao` CLI in-pod, or `urllib` from a python-having pod).
