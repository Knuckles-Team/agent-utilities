# Phase-10 cutover runbook — au `638ea524` + eg `2595a24` → live RKE2

> **Status: FINALIZED RUNBOOK, NOT APPLIED.** Every fact below was gathered read-only
> (`kubectl get/describe/exec -- cat`, no `apply`/`patch`/`delete`/`rollout`/`scale`
> executed) plus source inspection of au `638ea524` and eg `2595a24`, on 2026-07-25.
> This extends `reports/phase10-redeploy-plan.md` (the investigation) with the
> **exact per-workload edit list** the live cutover executes, reconciles that plan's
> one open ambiguity (the identity contract), and records the eg build-feasibility
> result. The live cutover happens later under the user's explicit approval — nothing
> here is auto-applied by any agent or pipeline.

**Placeholder convention.** Commands below are copy-paste exact for every
`kubectl`/OpenBao verb, flag, and JSON-Patch path — only environment-specific
identifiers are written as `<placeholder>` rather than baked into a tracked
document. Resolve each once, immediately before the cutover window, with the
read-only command shown:

| Placeholder | What it is | Resolve with |
|---|---|---|
| `<hostpath-root>` | Base directory holding the `graph-os`/`graph-os-host`/engine hostPath mounts | `kubectl get deployment graph-os -n platform -o jsonpath='{.spec.template.spec.volumes[0].hostPath.path}'` (strip the trailing `kg-src-next/agent-utilities`) |
| `<engine-source-node>` | The node the engine binary + `au-src`/`eg-src` hostPaths live on | `kubectl get deployment epistemic-graph -n platform -o jsonpath='{.spec.template.spec.nodeSelector}'` |
| `<engine-tls-endpoint>` | The engine's TCP/TLS contact address | `kubectl get configmap graph-os-env -n platform -o jsonpath='{.data.ENGINE_TLS_SERVER_NAME}'` (+ port `9100`, confirmed in the engine Deployment's own `--tcp-addr`/`GRAPH_SERVICE_ENDPOINTS`) |
| `<graph-os-public-host>` | The public MCP hostname | `kubectl get ingress graph-os -n platform -o jsonpath='{.spec.rules[0].host}'` |
| `<keycloak-host>` | The Keycloak issuer hostname | `kubectl get configmap graph-os-env -n platform -o jsonpath='{.data.OIDC_ISSUER}'` |

---

## 1. Identity-contract reconciliation (definitive)

The task brief flagged a conflict between the redeploy plan's blocker #1
(`KG_AUTH_TOKEN_REF`/`KG_IDENTITY_OAUTH2`) and a messaging-bundle finding
(`AUTH_JWT_AUDIENCE`/`MCP_JWT_AUDIENCE` + `KG_POLICY_VERSION`). Both are correct —
they are two **different, sequential** gates in the same boot path, not competing
theories, and the plan under-verified the first one because it stopped at
"unverified — needs the rehearsal step" rather than actually running it.

**The two gates, read directly from `agent_utilities/security/request_identity.py`:**

1. **Process-bootstrap identity** — `acquire_process_identity_token()` (lines
   400–428). Called unconditionally for any transport other than
   `stdio`-with-local-authority (`agent_utilities/mcp/kg_server.py:3791`,
   `_mint_process_session`, the `else` branch at line 2835) — i.e. it **is** on
   `platform/graph-os`'s own boot path, since it serves `TRANSPORT=streamable-http`.
   It requires **exactly one** of `KG_AUTH_TOKEN_REF` / `KG_IDENTITY_OAUTH2`
   (`config.kg_auth_token_ref` / `config.kg_identity_oauth2`, `core/config.py:2894-2905`)
   and raises `RuntimeError` otherwise (the XOR check at line 412: both absent
   also fails the `bool(a) == bool(b)` test). Same gate for the **standalone**
   `agent-utilities-messaging` process (`messaging/daemon.py:91-115`,
   `mint_process_identity`) and for `kg-ingest-worker`/the A2A protocol server.
2. **Claims-to-session projection** — `mint_graph_session()` (lines 93-113), called
   (a) internally by gate 1 once a token is acquired, **and** (b) independently by
   `ActorIdentityMiddleware` for every served inbound HTTP/MCP request that presents
   its own Bearer JWT. It requires `config.auth_jwt_audience or config.mcp_jwt_audience`
   (aliases `AUTH_JWT_AUDIENCE` / `FASTMCP_SERVER_AUTH_JWT_AUDIENCE`,
   `core/config.py:2856` / `2812-2813`) and `config.kg_policy_version` both non-empty,
   else `PermissionError("...missing audience or policy revision")`.

So a process that mints its **own** identity (graph-os, the standalone messaging
daemon, ingest-worker) must satisfy **both** gates in order; a process that only
**validates callers' tokens** (the served HTTP path) only ever touches gate 2.

**Definitive live status, verified directly against the cluster (not inferred):**

| Key | Where it lives | Confirmed live? |
|---|---|---|
| `KG_IDENTITY_OAUTH2` | `platform/graph-os`'s **durable `config.json`** (mounted `/root/.config/agent-utilities/config.json`, sourced from the migrated `au-config` hostPath — **kubectl-invisible**, not a ConfigMap/Secret key) | **YES** — a JSON object with `audience=agent-services`, `client_id=graph-os`, `client_secret=env://OIDC_CLIENT_SECRET`, `token_url=https://<keycloak-host>/realms/homelab/protocol/openid-connect/token`. The `client_secret` ref resolves against the bare `OIDC_CLIENT_SECRET` key already present in `graph-os-secrets` (confirmed). |
| `KG_AUTH_TOKEN_REF` | (same config.json) | Not needed — mutually exclusive with the above, and the above is set |
| `AUTH_JWT_AUDIENCE` | `graph-os-env` ConfigMap | **YES** — `agent-services` (the plan's own key enumeration in §3.1 omitted this key; it is present) |
| `KG_POLICY_VERSION` | `graph-os-env` ConfigMap | **YES** — `homelab-v1` (plan already had this right) |

**Conclusion: `platform/graph-os` already satisfies BOTH gates today.** This is *why*
it has been healthy for 2 days — not because the identity requirement doesn't apply to
it (it does, unconditionally, per `kg_server.py:3791`), but because the durable
`config.json` the plan's own §6 step 1 said to check (and deferred, rather than ran)
already carries the answer. **Blocker #1 in the plan is RESOLVED, not open.** No
env/Secret/OpenBao write is needed for `graph-os`. See §2 item (d) below for the
mechanical re-verification command and how this extends to `graph-os-host`.

**Why the standalone `agent-utilities-messaging` Deployment crash-loops on this exact
chain — proven from the pod's own crash traceback, not inferred:**

```
File ".../agent_utilities/messaging/daemon.py", line 195, in main
  session = _mint_process_session()
File ".../agent_utilities/messaging/daemon.py", line 170, in _mint_process_session
  session = mint_graph_session(actor)
File ".../agent_utilities/security/request_identity.py", line 109, in mint_graph_session
  return _mint_graph_session(...)
File ".../agent_utilities/security/request_identity.py", line 161, in _mint_graph_session
  raise PermissionError(...)
PermissionError: Verified graph authority is missing audience or policy revision
```

(`kubectl logs -n apps deploy/agent-utilities-messaging --previous`.) This is the
**exact** `mint_graph_session`/gate-2 failure the messaging-bundle finding names —
confirmed live, not theoretical. Gate 1 (token acquisition) evidently **succeeds**
for this process — the exception propagates from inside `mint_graph_session`, past
frames that already returned — most likely via an identity-source entry inside its
own dedicated `config.json` (mounted read-write at `/root/.config/agent-utilities`
from its own separate `aum-config` hostPath), mirroring the same
config.json-carries-`KG_IDENTITY_OAUTH2` pattern found on `graph-os` in this
investigation. That file could not be read directly (the pod was mid-`CrashLoopBackOff`
5-minute backoff at verification time — no running container to `exec` into; restart
count 826 and climbing). What **is** directly confirmed absent from every object this
Deployment actually consumes — its Deployment-level `env:`, its own Secret
(`agent-utilities-messaging`, 9 bare plaintext keys), and `mcp-common-env` (which
has `KG_POLICY_VERSION=homelab-v1` but no audience key, and isn't even wired to this
Deployment — **its `envFrom` is only its own Secret**, confirmed via direct dump) —
is `AUTH_JWT_AUDIENCE`/`FASTMCP_SERVER_AUTH_JWT_AUDIENCE` anywhere reachable. That
gap is sufficient on its own to explain the proven failure, regardless of exactly
which of `audience`/`policy_version` its config.json leaves blank. (Function names/line
numbers in the traceback — `_mint_process_session`/`main` at 195/170 — don't match
`638ea524`'s current `daemon.py`, i.e. the crashed process was running slightly older
cached bytecode from before a recent rename to today's `mint_process_identity`
(lines 91-115); the underlying `mint_graph_session` call chain and the exact
`PermissionError` text are unaffected by that rename and remain the live, current
failure mode.) **This is a previously-undocumented independent crash cause**, layered
on the plan's already-known three (9 plaintext secrets, `tcp://` scheme, retired-key
exposure risk via its own config.json).

**Confirms the task's framing exactly:** the fix is not to patch the standalone
Deployment's identity config (a second location to keep in sync forever) — it is to
**retire it**. `graph-os`'s bundled messaging co-service
(`co_service_supervisor.start_co_services`, `kg_server.py:3840`) calls
`messaging.daemon.run_forever(engine, platforms, stop_event)` **directly** — never
`daemon.mint_process_identity()` — running the co-service thread under
`_authorized_background_thread(session, ...)` where `session` is `graph-os`'s own
already-minted `bootstrap_session` (verified: `co_service_supervisor.py:207,283-305`).
The bundled co-service **never independently mints**, so it inherits whatever identity
`graph-os` already has — verified, working. Retiring the standalone Deployment (§2
item e) resolves this crash cause completely, with no config to fix in place.

---

## 2. Per-workload edit list

Per `AGENTS.md`/`mcp-fleet-network-contract-break`: OpenBao is the only correct write
target for anything ExternalSecret-managed (`graph-os-secrets`,
`epistemic-graph-secrets`, `agent-utilities-messaging`) — a raw `kubectl patch` on any
of them is silently reverted on the next 1h reconcile (all three confirmed
`STORETYPE: ClusterSecretStore`, `STORE: openbao`, `REFRESH INTERVAL: 1h`, `path: apps`
in the ClusterSecretStore spec — so `key: graph-os` resolves to OpenBao KV-v2 path
`apps/graph-os`, `key: agent-utilities-messaging` → `apps/agent-utilities-messaging`,
and the shared `GRAPH_SERVICE_AUTH_SECRET` remoteRef → `apps/agent-utilities/deployment`
property `GRAPH_SERVICE_AUTH_SECRET`). The plain `graph-os-env` ConfigMap is **not**
ExternalSecret-managed — `kubectl patch` is the correct, direct tool for it.

### (a) Strip `GRAPH_BACKEND=fanout` from `platform/graph-os`

**Already resolved on the live object — re-verify, do not blindly re-apply.** Direct
inspection of the live Deployment's `.spec.template.spec.containers[0].env` (not the
`kubectl.kubernetes.io/last-applied-configuration` annotation) shows **no
`GRAPH_BACKEND` key at all**. The retired key only survives inside that stale
annotation (frozen at whatever full manifest was last `kubectl apply -f`'d, before
someone imperatively `kubectl set env`/patched it away) — proof: the annotation still
shows the *pre-migration* hostPaths too, while the live spec uses the migrated `-next`
paths. **This is a real landmine for the cutover mechanics, not a closed issue**: if
any future step does a full `kubectl apply -f` of a manifest matching that stale
annotation (or of the rendered `deploy/k8s/production-cell/` assets /
`deploy/swarm/graphos.stack.yml` without
hand-verifying they don't carry `GRAPH_BACKEND`), it reintroduces the retired key and
the migrated hostPaths regress in the same stroke. The cutover must use **targeted
patches only** (as this whole runbook does), never a wholesale manifest apply against
`platform/graph-os`.

- **Command (idempotent pre-flight + defensive re-assertion):**
  ```bash
  # Verify (read-only) — expect empty output:
  kubectl get deployment graph-os -n platform \
    -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="GRAPH_BACKEND")]}'
  # If it ever reappears, remove it atomically (no-op today; safe to run regardless):
  kubectl set env deployment/graph-os -n platform GRAPH_BACKEND-
  ```
- **Workload:** `platform/graph-os` (Deployment env only; `graph-os-host` was never
  confirmed to carry this key in its live spec either).
- **Rollback:** N/A — there is nothing to roll back to; reintroducing `GRAPH_BACKEND`
  is the regression this step guards against, not a valid prior state.
- **Status: READY** (already correct live; the action here is a guard-rail check, not
  a change).

### (b) `graph-os-host` drifted twin — ALIGN (recommended) or RETIRE

Confirmed live via direct hostPath/config.json inspection (not inference):

| | `graph-os` (migrated) | `graph-os-host` (drifted) |
|---|---|---|
| `au-src` | `<hostpath-root>/kg-src-next/agent-utilities` | `<hostpath-root>/kg-src/agent-utilities` |
| `eg-src` | `<hostpath-root>/kg-src-next/epistemic-graph` | `<hostpath-root>/kg-src/epistemic-graph` |
| `au-config` | `<hostpath-root>/.config/agent-utilities-next` | `<hostpath-root>/.config/agent-utilities` |
| config.json `secrets_backend` | `engine` (valid) | **`inmemory`** (invalid literal — `pydantic.ValidationError`) |
| config.json retired keys present | none (swept all 19: none found) | **`ENGINE_MODE=remote`, `ENGINE_ENDPOINT=tcp://<engine-tls-endpoint>`, `GRAPH_SERVICE_TCP_ADDR=<engine-tls-endpoint>`** — all three are in `_RETIRED_CONFIGURATION_KEYS` (`core/config.py:165-186`, confirmed by direct source read) |
| `KG_DAEMON_ROLE` | `client` | `auto` (self-elects host lock) |

This directly, first-hand confirms plan §3.3/§9 blocker #3: this pod is one
unplanned restart away from the full retired-key + invalid-`secrets_backend` crash
chain, **and** it would also fail identity gate 1 (its config.json has neither
`KG_IDENTITY_OAUTH2` nor `KG_AUTH_TOKEN_REF` — confirmed absent).

**Option 1 — ALIGN (matches the plan and the existing "twin daemon" architecture
doc; recommended default).** JSON-Patch with `test` guards so the patch fails loud
(atomically, no partial application) if the array order ever differs from what was
just verified, rather than silently patching the wrong volume. Resolve
`<hostpath-root>` (see the conventions table) before running:

```bash
# Save a full rollback artifact first:
kubectl get deployment graph-os-host -n platform -o yaml > graph-os-host-pre-cutover-$(date +%s).yaml

kubectl patch deployment graph-os-host -n platform --type=json -p '[
  {"op":"test","path":"/spec/template/spec/volumes/0/name","value":"au-src"},
  {"op":"replace","path":"/spec/template/spec/volumes/0/hostPath/path","value":"<hostpath-root>/kg-src-next/agent-utilities"},
  {"op":"test","path":"/spec/template/spec/volumes/1/name","value":"eg-src"},
  {"op":"replace","path":"/spec/template/spec/volumes/1/hostPath/path","value":"<hostpath-root>/kg-src-next/epistemic-graph"},
  {"op":"test","path":"/spec/template/spec/volumes/2/name","value":"au-config"},
  {"op":"replace","path":"/spec/template/spec/volumes/2/hostPath/path","value":"<hostpath-root>/.config/agent-utilities-next"}
]'
# Deployment strategy is `Recreate` (confirmed) — the patch alone triggers the
# recreate; no separate rollout restart is needed. Watch it:
kubectl rollout status deployment/graph-os-host -n platform
```

- **Workload:** `platform/graph-os-host`.
- **Rollback:** the same `--type=json` pattern with the three `hostPath.path` values
  swapped back to the pre-migration paths (`<hostpath-root>/kg-src/agent-utilities`,
  `<hostpath-root>/kg-src/epistemic-graph`, `<hostpath-root>/.config/agent-utilities`
  — the originals, confirmed still present on disk, not deleted by this step) — or
  `kubectl apply -f graph-os-host-pre-cutover-<ts>.yaml` using the saved backup.

**Option 2 — RETIRE (simpler; a genuine, code-verified alternative worth the
operator's consideration, not just a fallback).** `graph-os` runs `KG_DAEMON_ROLE=client`;
`co_service_supervisor.host_daemon_needed()` (`agent_utilities/mcp/co_service_supervisor.py:113-123`)
is `True` exactly when the deployment is `client`-configured **and no host currently
holds the lock** — so if `graph-os-host` (the only `auto`-role pod) is scaled to 0,
`graph-os`'s own process would self-elect and bring up the KG host daemon **in-process**
(`bring_up_host_daemon_if_needed()`, called before `_start_engine_bootstrap` in
`kg_server.py:3833`) with zero coverage gap — this is the exact designed fallback, not
an improvised one. This drops one pod from the fleet at the cost of collapsing the
documented "twin daemon" separation (host-daemon CPU work would now share process/GIL
with `graph-os`'s own MCP serving — the same isolation concern
`messaging/daemon.py`'s own module docstring cites as the reason messaging got its
*own* process). **Recommendation: Option 1 (align)** for Phase-10 — it's a pure
hostPath fix with no architecture change, and the isolation trade-off in Option 2
deserves its own deliberate decision, not a side effect of fixing drift.

- **Status: BLOCKED — needs execution at cutover** (real work item, not yet applied).

### (c) Add `MCP_PUBLIC_BASE_URL=https://<graph-os-public-host>`

Confirmed **absent** from the live `graph-os-env` ConfigMap (full key enumeration).
Not ExternalSecret-managed — direct `kubectl patch` is correct. Because **both**
`graph-os` and `graph-os-host` consume this same ConfigMap via `envFrom`, one patch
covers both workloads once each is restarted. (`<graph-os-public-host>` is the same
value already set, unmodified, in the already-corrected
`deploy/k8s/graphos-homelab-live-config-fix.yaml` reference file — resolve it from
there or the Ingress, per the conventions table.)

```bash
kubectl patch configmap graph-os-env -n platform --type merge \
  -p "{\"data\":{\"MCP_PUBLIC_BASE_URL\":\"https://<graph-os-public-host>\"}}"
kubectl rollout restart deployment/graph-os -n platform
kubectl rollout restart deployment/graph-os-host -n platform   # picks it up on its own restart in (b) too
```

- **Workload:** `graph-os-env` ConfigMap → both `graph-os` and `graph-os-host`.
- **Rollback:** `kubectl patch configmap graph-os-env -n platform --type json -p '[{"op":"remove","path":"/data/MCP_PUBLIC_BASE_URL"}]'` then re-roll both Deployments (or `kubectl rollout undo`).
- **Status: READY** (value known, mechanical, zero ambiguity — matches the already-corrected reference file in item 1 of this cutover).

### (d) Identity keys

**Already present live for `graph-os` — nothing to add.** See §1 above for the full
reconciliation. The only action here is a **pre-flight re-confirmation** immediately
before the cutover window (belt-and-suspenders on the single highest-value fact, per
the plan's own emphasis), plus carrying the same config.json forward unmodified when
`graph-os-host` is aligned in (b):

```bash
kubectl exec -n platform deploy/graph-os -- python3 -c \
  "import json; d = json.load(open('/root/.config/agent-utilities/config.json')); \
   oauth2 = d.get('KG_IDENTITY_OAUTH2'); \
   print('KG_IDENTITY_OAUTH2 present:', bool(oauth2)); \
   print('client_secret ref set:', bool((oauth2 or {}).get('client_secret')))"
# Expect: True / True. If EITHER prints False, STOP — this is the one real
# boot-blocker in the whole migration and must be resolved before any hostPath flip.
```

- **Workload:** `graph-os` (and, after (b), `graph-os-host` — same config.json).
- **Rollback:** N/A (verification-only step, no mutation).
- **Status: READY** (confirmed present; re-verify, don't skip, immediately pre-cutover).

### (e) Retire `apps/agent-utilities-messaging`

Uses the already-prepared `deploy/k8s/messaging-bundle-retirement.yaml` almost
verbatim — its STEP 1–3 sequencing is correct and independently confirmed against the
current live objects. **The one easy-to-miss prerequisite: retiring the standalone
Deployment BEFORE moving its channel token(s) into `graph-os`'s own secret would
silently drop messaging entirely** (no crash, just `configured_platforms()` returning
empty, so the co-service never starts) rather than fixing it — sequencing matters.
Because the retired Deployment is being **deleted**, none of the original plan §4.4
in-place fixes (rewrite its 9 plaintext keys, fix its `tcp://` scheme, add it its own
`GRAPH_SERVICE_AUTH_SECRET`) are needed at all — retirement obviates them.

```bash
# STEP 1 (read-only) — confirm the bundle is live on graph-os already:
kubectl logs -n platform deploy/graph-os --since=10m | grep -i "co-service"
# "co-service messaging started." only appears once a channel token is configured —
# absence here is expected/correct until STEP 2 lands.

# STEP 2 — move the channel token(s) into graph-os's OWN OpenBao path (apps/graph-os),
# NOT a kubectl patch on graph-os-secrets (external-secrets reverts that within 1h).
# The live agent-utilities-messaging Secret carries TELEGRAM_BOT_TOKEN and
# MATTERMOST_TOKEN among its 9 keys — these are the two that matter for
# `configured_platforms()`; read them from the CURRENT plaintext OpenBao entry at
# apps/agent-utilities-messaging and merge into apps/graph-os (KV-v2 patch = merge,
# preserves the existing OIDC_*/other keys already there):
vault kv patch apps/graph-os \
  TELEGRAM_BOT_TOKEN="<value read from apps/agent-utilities-messaging>" \
  MATTERMOST_TOKEN="<value read from apps/agent-utilities-messaging>"
# (or the graph-os `write`/`open__write_secret` MCP tool against the same path —
# either is fine; OpenBao is the write target either way, never kubectl patch.)
kubectl annotate externalsecret graph-os-secrets -n platform force-sync=$(date +%s) --overwrite
kubectl rollout restart deployment/graph-os -n platform
kubectl logs -n platform deploy/graph-os --since=10m | grep -i "co-service"
# Now expect: "co-service messaging started."

# STEP 3 — only after STEP 2 is confirmed serving: retire the standalone Deployment.
kubectl scale deployment/agent-utilities-messaging -n apps --replicas=0
# park; confirm no regression for one full operational cycle, THEN:
kubectl delete deployment/agent-utilities-messaging -n apps
kubectl delete secret agent-utilities-messaging -n apps
# (no dedicated ConfigMap exists for this Deployment — confirmed; nothing to delete there)
```

- **Workload:** `apps/agent-utilities-messaging` (retired) + `platform/graph-os-secrets`
  (OpenBao `apps/graph-os` path gains 2 keys) + `platform/graph-os` (restart to pick
  them up).
- **Rollback:** `kubectl scale deployment/agent-utilities-messaging -n apps --replicas=1`
  (fully reversible up until the `delete` calls in STEP 3's second half — the
  scale-to-0-first-then-delete sequencing is deliberately the rollback window). Once
  deleted, restoring it means re-applying the pre-existing (broken) Deployment/Secret
  manifests — no worse than today's state, since it starts crash-looping either way.
- **Status: BLOCKED — needs execution at cutover**, specifically **needs the two token
  values copied from OpenBao `apps/agent-utilities-messaging` into `apps/graph-os`**
  (a value the operator/OpenBao holds, not something inferable from cluster metadata).

---

## 3. eg engine binary — build feasibility (2595a24)

See §6 below for the executed build result. Cutover mechanism (plan §5.1, confirmed
against the live Deployment's exact hostPath/args). Run the build on
`<engine-source-node>` (the node the `engine-bin` hostPath and `au-src`/`eg-src`
sources already live on):

```bash
# On <engine-source-node>, from the eg checkout at 2595a24:
cd <eg checkout at 2595a24>
cargo build --release --features full --target-dir <scratch>   # NOT the shared default
                                                                  # target dir — see
                                                                  # eg-shared-cargo-target-
                                                                  # corruption memory
mkdir -p <hostpath-root>/eg-bin-2595a24
cp <scratch>/release/epistemic-graph-server <hostpath-root>/eg-bin-2595a24/

kubectl get deployment epistemic-graph -n platform -o yaml > epistemic-graph-pre-cutover-$(date +%s).yaml
kubectl patch deployment epistemic-graph -n platform --type=json -p '[
  {"op":"test","path":"/spec/template/spec/volumes/0/name","value":"engine-bin"},
  {"op":"replace","path":"/spec/template/spec/volumes/0/hostPath/path","value":"<hostpath-root>/eg-bin-2595a24"}
]'
# Strategy is Recreate (confirmed) — expect ~60s downtime (plan's own historical
# estimate for this store size; re-measure for real during the rehearsal window).
kubectl rollout status deployment/epistemic-graph -n platform
```

- **Workload:** `platform/epistemic-graph`. `epistemic-graph-secrets` is unchanged —
  `GRAPH_SERVICE_AUTH_SECRET`/signer keys must be **reused verbatim, never
  regenerated** (rotating them breaks every connected client fleet-wide).
- **Rollback:** same `--type=json` pattern, `hostPath.path` back to
  `<hostpath-root>/eg-bin` (kept on disk, not deleted) — or
  `kubectl apply -f epistemic-graph-pre-cutover-<ts>.yaml`.
- **Ordering constraint (plan §5.1, unchanged):** the engine rebuild and the
  `graph-os`/`graph-os-host` source sync must land in the **same window, engine
  first** — a client built against one commit failed `create tenant graph` auth
  against a server built from a different commit in a prior incident
  (`golden-loop-embed-hang-and-engine-redeploy` memory).

---

## 4. `au-src`/`eg-src` sync for `graph-os`/`graph-os-host`

Not one of the 6 gating items on its own (the plan treats it as part of the ordered
cutover, §7b/7c) but required for either workload to actually run `638ea524`/`2595a24`
source rather than config alone:

```bash
rsync -a --delete <au checkout @ 638ea524>/ <hostpath-root>/kg-src-2/agent-utilities/
rsync -a --delete <eg checkout @ 2595a24>/  <hostpath-root>/kg-src-2/epistemic-graph/
# Stage at a NEW parallel path (kg-src-2), do not overwrite kg-src-next in place —
# keeps the proven "stage, then flip" rollback lever intact.
kubectl patch deployment graph-os -n platform --type=json -p '[
  {"op":"test","path":"/spec/template/spec/volumes/0/name","value":"au-src"},
  {"op":"replace","path":"/spec/template/spec/volumes/0/hostPath/path","value":"<hostpath-root>/kg-src-2/agent-utilities"},
  {"op":"test","path":"/spec/template/spec/volumes/1/name","value":"eg-src"},
  {"op":"replace","path":"/spec/template/spec/volumes/1/hostPath/path","value":"<hostpath-root>/kg-src-2/epistemic-graph"}
]'
kubectl rollout status deployment/graph-os -n platform
```

`au-config`/`au-src`/`eg-src` indices for `graph-os` were confirmed in the same order
as `graph-os-host`'s (au-src=0, eg-src=1, au-config=2) — re-verify with
`kubectl get deployment graph-os -n platform -o jsonpath='{.spec.template.spec.volumes[*].name}'`
immediately before applying, same defensive posture as §2(b).

- **Rollback:** patch the two `hostPath.path` values back to
  `<hostpath-root>/kg-src-next/{agent-utilities,epistemic-graph}` (kept intact).

---

## 5. Image dependency vintage — confirmed gap, not just "plausible" (plan §5.2/§9 blocker #6)

Directly checked, not assumed:

```bash
kubectl exec -n platform deploy/graph-os -- pip show pydantic-ai-slim
# Version: 2.7.0   (au's locked/required version, uv.lock: 2.16.0)
kubectl exec -n platform deploy/graph-os -- python -c "import pydantic_acp"
# ModuleNotFoundError: No module named 'pydantic_acp'   (locked: pydantic-acp==1.5.1)
```

**Worse than the plan's own hedge** ("plausible, not directly confirmed") —
`pydantic_acp` isn't merely old, it is **entirely absent** from the running image.
Mitigating factor: every `pydantic_acp` import in au is lazy (inside function bodies
in `protocols/acp_adapter.py`/`acp_providers.py`, never at module load time), and
`graph-os`'s own `kg_server.mcp_server()` boot path never reads `enable_acp` (that
flag belongs to the separate generic `server/app.py` agent-server builder) — so this
specific absence is **unlikely to crash `graph-os` on boot**. The `pydantic-ai-slim`
version gap is broader (core orchestration plumbing PA-R0/R1 was written against,
not an optional adapter) and was not exhaustively traced against `638ea524`'s actual
call sites in this pass — the plan's own risk framing stands, now **confirmed rather
than assumed**: silent misbehavior or an `AttributeError`/`TypeError` class of
failure, not necessarily a clean crash-loop. **The hostPath source-sync in §4 cannot
fix this** — `PYTHONPATH=/au:/eg` only overlays the first-party `agent_utilities`/
`epistemic_graph` packages; third-party site-packages come from whatever the live
serving image (`graph-os`'s Deployment `.spec.template.spec.containers[0].image`)
was last built with. Closing this gap requires an image rebuild + republish from
`638ea524`'s lockfile — out of this (non-live)
task's scope (no image push), and the single highest-priority remaining item for
the live cutover decision (see the go/no-go in `reports/phase10-cutover-manifest.md`).

---

## 6. eg `2595a24` build result

`cargo build --features full --target-dir ./target-isolated -j 4` at eg `main`
(= `2595a24`), run in the canonical checkout for this feasibility check: **succeeded —
0 errors, 0 warnings, `Finished \`dev\` profile [unoptimized + debuginfo] target(s)
in 10m 04s`.** Binary confirmed present and functional
(`target-isolated/debug/epistemic-graph-server`, runs, `--help` output matches the
live Deployment's exact CLI contract — `--socket-path`, `--tcp-addr`,
`--tcp-tls-cert`/`-key`, `--auth-secret`/`GRAPH_SERVICE_AUTH_SECRET`,
`--persist-dir`/`GRAPH_SERVICE_PERSIST_DIR` — confirming no new required flag and that
`--allow-insecure`/`--checkpoint-interval` are indeed gone, as the plan stated). **This
proves the redeploy artifact is buildable at the named commit.** Production deployment
uses `--release` (this feasibility check ran the plain, unoptimized `cargo build`, per
the task's exact instruction); the reach-pods path is the hostPath flip in §3 — a real
rebuild + stage-at-a-new-path + flip, never an in-place overwrite (the running binary
holds the file open).

---

## 7. Cross-references

- Input investigation: `reports/phase10-redeploy-plan.md` (workspace-level, this
  runbook's companion — not committed to any repo; the workspace root is not a git
  repository).
- Reference-file fix: `deploy/k8s/graphos-homelab-live-config-fix.yaml` (issuer scheme
  corrected on this same branch).
- Messaging retirement source: `deploy/k8s/messaging-bundle-retirement.yaml`
  (unchanged — already accurate).
- Identity/session architecture: `agent_utilities/security/request_identity.py`,
  `agent_utilities/mcp/co_service_supervisor.py`, `agent_utilities/mcp/kg_server.py`
  (`_mint_process_session`, `mcp_server()`).
