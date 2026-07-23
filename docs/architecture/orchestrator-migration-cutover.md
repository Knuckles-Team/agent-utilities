# Orchestrator Migration & Cutover — hard-won hardening (provider-neutral)

> How to migrate a running fleet from one orchestrator to another (**e.g.** Docker
> Swarm/Compose → Kubernetes) **without downtime, credential loss, or a fleet-wide
> auth outage** — and without asking clients to reconfigure anything. Every rule here
> was paid for in a live cutover; each is written generically so ANY future migration
> avoids the same trap. The step-by-step operator runbook that executes these rules is
> the genesis skill reference
> [`agent-os-genesis/references/orchestrator-migration-cutover.md`](../../agent_utilities/skills/workflows/agent-os-genesis/references/orchestrator-migration-cutover.md);
> this page is the *why* and the diagnosis catalog.

The recurring theme: **preserve every externally observable contract** — the service
hostnames clients dial, the credentials they present, the data behind each service —
and move the *substrate* underneath them one coupled unit at a time, with the old
backend kept warm as rollback until each unit is verified. Names, tokens, and data are
the contract; the orchestrator is an implementation detail.

Roles are named by function so the guidance is portable:

- **the IdP** — the OIDC/OAuth2 identity provider that issues + validates JWTs (**e.g.** Keycloak).
- **the secret store** — the KV secret backend (**e.g.** OpenBao/Vault) with an operator-gated write path.
- **the DNS authority** — the resolver that owns the internal service zone (**e.g.** the `.arpa` zone).
- **the edge proxy** — the reverse proxy / ingress terminating client traffic (**e.g.** Caddy, ingress-nginx).
- **the gateway** — the MCP fleet gateway that fronts JWT-protected child MCP servers (**e.g.** graph-os / the multiplexer).
- **the secret sync operator** — the controller that projects secret-store values into the target orchestrator (**e.g.** External Secrets Operator → a k8s `Secret`).

---

## 1. The gateway's outbound service token must be minted against the *in-cluster* IdP

**The single most expensive finding of the whole migration: a fleet-wide 401.**

When a gateway fronts JWT-protected child MCP servers, it mints an OAuth2
**client-credentials service token** and attaches it to every outbound child call.
Each child returns **401 without it**. The mint is enabled with
`MCP_CLIENT_AUTH=oidc-client-credentials` plus `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET`
/ `OIDC_AUDIENCE` / `OIDC_ISSUER`.

On a single flat network the token endpoint is auto-discovered from the issuer and
everything works. **Inside the new orchestrator that assumption breaks.**

- **Symptom.** Every lazily-loaded child MCP 401s through the gateway — surfacing as
  `Session terminated`, `no such host`, or a generic fleet outage — even though the
  gateway's inbound JWT validation is fine and the client config looks correct
  (right client id, secret, audience, issuer).
- **Diagnosis.** The gateway discovers the token endpoint from the *public* issuer
  hostname, which now routes through the **edge proxy** — and during a progressive
  migration the edge can **502 the token endpoint**. A failed mint **degrades to no
  header** rather than erroring, so `Authorization` is silently absent and *every*
  child 401s. Confirm from inside the gateway pod: mint a token and assert it is
  **non-empty** and carries the **right audience**. An empty/absent bearer pinpoints
  the mint; a present bearer that still 401s is a routing/dual-instance issue (§4).

  ```bash
  # inside the gateway pod — a healthy mint returns True; an empty/None bearer is the bug
  python3 -c 'from agent_utilities.mcp.client_credentials import bearer_header; \
              h=bearer_header({}); print(bool(h.get("Authorization")))'
  ```

- **Fix.** **Pin the token endpoint to the in-cluster IdP Service**, bypassing the
  edge, exactly as you pin JWKS to the in-cluster IdP for *inbound* validation:

  ```
  OIDC_TOKEN_URL=http://<idp-service>.<idp-namespace>.svc:8080/realms/<realm>/protocol/openid-connect/token
  ```

  This is safe because the IdP stamps the **external issuer claim regardless of which
  endpoint minted the token** — so the child's issuer-validation still matches. The
  in-cluster path removes the edge (and its 502s) from the critical auth path. Make it
  durable in the gateway manifest, not just a live patch.

> **General rule.** For *any* service that both **validates** inbound JWTs and
> **mints** outbound tokens, pin **both** the JWKS URI and the token URL to the
> in-cluster IdP Service. Public hostnames are for clients outside the cluster; the
> mesh should never leave the mesh to talk to its own IdP.

---

## 2. The cluster-driving MCP needs a dedicated ServiceAccount + a scoped ClusterRole

A deployed MCP that operates the cluster (**e.g.** a container-manager / ops MCP)
auto-detects in-cluster config and uses the pod's ServiceAccount whenever
`KUBERNETES_SERVICE_HOST` is set. Two failure modes bracket the correct setup:

- **Symptom (too little).** Every cluster call 403s — the pod is on the namespace
  **default ServiceAccount**, which has **zero permissions**.
- **Symptom (too much).** The MCP is bound to `cluster-admin` (or `nonResourceURLs: *`),
  so a prompt-injected or buggy call can do anything, including granting itself more.
- **Fix.** Give it its **own** ServiceAccount plus a **least-privilege, explicit**
  ClusterRole covering exactly the verbs it needs across the resource groups it drives
  — workloads, config, networking, storage, and any CRDs (**e.g.** the secret-sync
  CRDs) — with **no RBAC-write verbs** (`roles`/`clusterroles`/`*bindings`), so it
  **cannot self-escalate**, and no wildcard `nonResourceURLs`. Bind SA→ClusterRole and
  set the deployment's `serviceAccountName`. The MCP's own credentials/roles must be
  honored on the product side too (its identity is a first-class principal, not the
  namespace default).

> Also bake the tooling the MCP shells out to (**e.g.** `kubectl` and cloud
> exec-plugins) into its image — a client library alone is not enough if the code
> calls out to a binary for some operations.

---

## 3. App + DB secrets come from the secret store via ExternalSecrets — never hand-created

Do **not** hand-create target-orchestrator Secrets during a migration; they drift and
are lost on redeploy. Instead:

1. **Seed the secret store** at a per-app path (`<store>/<app>`) from each stack's
   existing env (the values the old orchestrator injected).
2. **An ExternalSecret** (owned by the **secret sync operator**) syncs `<store>/<app>`
   into a target `Secret`, which the app consumes.

- **Operator-gated writes.** Secret **writes** to the store are intentionally gated
  (a human/operator approves them), so ship a **seed script** the operator runs once
  per stack (reads the root/rw token + each stack's env, writes `<store>/<app>`), then
  the sync is automatic and read-only thereafter.
- **KV-v2 path nuance.** A KV-v2 store addresses data at `<mount>/data/<app>` on the
  write API but the sync manifest references the logical `<mount>/<app>` — mismatching
  these yields empty secrets. Match the store's real mount + API version (**e.g.** the
  sync CRD served version can be `v1` even when examples show `v1beta1`; use what the
  cluster actually serves, or the manifest is silently not reconciled).
- **Stale-sync tip.**
    - **Symptom.** The ExternalSecret shows `SecretSyncError` / a stale value after you
      (re-)seed the store.
    - **Fix.** Force a refresh — annotate the ExternalSecret (**e.g.** bump a
      `force-sync` annotation) to trigger immediate reconciliation instead of waiting
      for the refresh interval.
- **Blanket `envFrom` is a trap.** A per-app secret may carry keys the service must
  **not** inherit (**e.g.** a policy-engine `*_TYPE`/`*_POLICY_FILE` pair that
  activates middleware with no policy file → boot crash-loop). Prefer an **explicit
  env allow-list** over `envFrom`-the-whole-secret, or pin the sensitive key to a
  safe value in the pod env (explicit env beats `envFrom`).

---

## 4. Preserve the canonical service hostnames (zero client reconfiguration)

The migration **must keep the existing `<svc>` hostnames working**. Do **not** tell
clients to switch to in-cluster `.svc` names — every client, every stored config, and
every cross-service call would need editing, and rollback becomes impossible.

Achieve zero-reconfig continuity with two moving parts per service:

1. A target-orchestrator **Ingress carrying the SAME hostname** the client already dials.
2. The **DNS authority** pointing that hostname at the cluster **ingress VIP**.

**Automate step 2 with an ingress-watching DNS controller** (**e.g.** external-dns):
it watches Ingress `host` values and writes the corresponding A-records into the DNS
authority, so newly-migrated services **self-publish** without a manual DNS edit.

- **CRITICAL SAFETY — selective, never global.** Only flip DNS for hostnames that
  **already have a target-orchestrator Ingress** (i.e. are actually migrated). A
  **global/wildcard flip** (pointing the whole zone at the ingress VIP) **breaks every
  still-unmigrated service**, which the cluster has no Ingress for. Scope the DNS
  controller to the migrated set (a domain filter / annotation allow-list), or drive
  it per-service.
- **The failure mode we hit.**
    - **Symptom.** After a cutover, clients **401** (or connect to the wrong data) even
      though tokens and manifests are correct.
    - **Diagnosis.** The hostname still resolves to the **OLD, pre-migration backend**
      (a different instance with different auth/state). It is not a token problem — it
      is a *name-resolution* problem. Prove it by dialing the in-cluster name directly:
      if `<svc>.svc` returns 200 with the same bearer that 401s on `<svc>` (the public
      name), the public name is still pointing at the old backend.
    - **Fix.** Complete the selective DNS flip for that hostname (or repoint the
      caller at the in-cluster name as an interim), then re-verify.

---

## 5. Edge / reverse-proxy cutover caveat

During a progressive migration the **edge proxy still points every hostname at the old
backend**. When you cut a service over, **prefer flipping the DNS authority** (§4) over
editing the edge per-service — it is atomic, automatable, and has one source of truth.

If you *must* edit the edge:

- **The edit must reach the ACTUAL serving instance.** Multi-replica / global edge
  services and per-node config files are a trap: a host-file edit may not be the file
  the replica that serves a given request actually reads. Identify the running
  instance and its mounted config (**e.g.** the live mounted `Caddyfile`, not a
  drifted copy in the source tree) and edit *that*.
- **`validate` before reload.** Run the edge's config check before reloading so a typo
  doesn't take the whole edge down.
- **Curl THROUGH the edge to confirm** the new backend answers before you stop the old
  one — verify the *path clients actually take*, not just the pod directly.
- **Keep the old backend as rollback** until the new path is verified end-to-end; only
  then stop it.

> Do not enable a competing edge on the same host ports while the old edge is live
> (**e.g.** an ingress controller's host-port `:80/:443` can shadow the existing edge
> and serve the wrong certificate to every vhost on that node). Use a **LoadBalancer
> VIP** for the new ingress — which becomes the eventual edge at the DNS flip — so the
> two coexist without shadowing.

---

## 6. Stateful data-in-place patterns

Moving state is where migrations silently corrupt or lose data. The patterns that held
up:

- **`enableServiceLinks: false`** for any app that reads `<NAME>_*` env as its own
  configuration. The orchestrator auto-injects `<SVC>_PORT_*` service-link env vars
  that **collide** with an app's `<NAME>_*` config convention (**e.g.** a graph
  database that treats `NEO4J_*` as settings crash-loops on the injected
  `NEO4J_PORT_7687_TCP_PORT`). Turn service links off for those apps.
- **Pin data-in-place to the node holding the volume.** A hostPath (or equivalent
  node-local) volume must be pinned with a `nodeSelector` to the node where the data
  physically lives. If the data node is **not yet a cluster node**, copy the data to a
  cluster node **first** — you cannot node-pin a pod to a host the scheduler doesn't
  manage.
- **Postgres via `pg_dump`/`restore`, not file-copy.** A stopped named volume can read
  **empty at the host level** (the engine's on-disk files are only consistent while
  mounted), so a raw file copy of a stopped DB yields an empty or corrupt datadir. Use
  a logical `pg_dump | psql` between the running old DB and the new one (also
  version-safe). For a host-path datadir owned by a non-root user, run the DB pod as
  root so its entrypoint can `chown` the datadir.
- **Hold the app at 0 replicas until the DB is restored.** If the app starts against
  an empty DB it will **initialize a fresh schema**, clobbering the restore. Keep
  `replicas: 0` until the data is in and verified, *then* scale to 1.
- **Verify parity before cutover.** Confirm row counts / table counts / object counts
  **match the source** before flipping any traffic. "The pod is Running" is not
  "the data is correct."

---

## 7. The coupled-unit migration runbook (migrate-mode sequence)

Migrate an app **and its dependency (DB, cache, media) as one coupled unit**, in this
order. This is the reusable sequence the genesis migrate-mode executes per stack:

1. **Deploy the DB** (data-in-place or `pg_dump`/restore per §6), pinned to its data node.
2. **Hold the app at 0 replicas** (so it can't initialize a fresh schema — §6).
3. **Restore the DB** and **verify parity** (row/table counts match the source — §6).
4. **Copy media / blob data** to the app's node-local volume (from the *stopped* source
   container/volume for a consistent snapshot — §6).
5. **Scale the app to 1**; wait for a healthy readiness probe (use a `tcpSocket` or the
   app's real health path — some apps 403 the probe by trusted-host ACL yet serve fine).
6. **Verify parity + health** end-to-end against the new instance (data correct, app
   answers).
7. **Flip DNS, not the edge** (§4/§5) — selectively, for this hostname only; keep the
   old backend warm.
8. **Stop the old backend** (scale the old orchestrator's service to 0 — keep the
   definition for rollback).
9. **Re-verify with the old backend down** — this is what proves the cutover is real
   and nothing was still quietly served by the old instance (§4).

Rollback at any step: revert the DNS/edge change and scale the old service back to 1 —
the data copy was a point-in-time snapshot, and the old datadir is untouched.

---

## Quick reference — symptom → cause → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| Every child MCP 401s through the gateway | Outbound token mint routes through the edge, which 502s → empty bearer | Pin `OIDC_TOKEN_URL` to the in-cluster IdP Service (§1) |
| Cluster-driving MCP 403s every call | Pod on the namespace default ServiceAccount | Dedicated SA + scoped ClusterRole, no RBAC-write (§2) |
| App boots with empty secrets | Hand-created Secret / wrong KV path / wrong CRD version | Seed `<store>/<app>` + ExternalSecret; match mount + served version (§3) |
| ExternalSecret stale after reseed | Waiting on the refresh interval | Force-refresh annotation (§3) |
| Client 401s after cutover, tokens correct | Hostname still resolves to the old backend | Complete the **selective** DNS flip (§4) |
| Global DNS flip broke unmigrated services | Wildcard flip pointed the whole zone at the ingress VIP | Only flip hostnames that already have an Ingress (§4) |
| Edge edit didn't take effect | Edited a config the serving replica doesn't read | Edit the live/mounted config of the serving instance; `validate`; curl through (§5) |
| DB app crash-loops on `<NAME>_PORT_*` | Service-link env collides with app config convention | `enableServiceLinks: false` (§6) |
| Restored DB comes up empty | File-copied a stopped named volume, or app initialized a fresh schema | `pg_dump`/restore; hold app at 0 until restored (§6) |

## Related

- Operator runbook (the *how*):
  [`agent-os-genesis/references/orchestrator-migration-cutover.md`](../../agent_utilities/skills/workflows/agent-os-genesis/references/orchestrator-migration-cutover.md)
- Gateway auth wiring (inbound + outbound, in-cluster pinning):
  [`agent-os-genesis/references/graph-os-fleet-gateway-auth.md`](../../agent_utilities/skills/workflows/agent-os-genesis/references/graph-os-fleet-gateway-auth.md)
- [Containerized Deployment](containerized-deployment.md) — the platform-as-microservices target shape.
- [Troubleshooting](troubleshooting.md) — cross-layer diagnosis.
</content>
