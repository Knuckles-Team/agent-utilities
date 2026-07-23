# Orchestrator migration & cutover — migrate-mode operator runbook

> The step-by-step runbook for genesis **`mode=migrate`**: moving a running fleet from
> one orchestrator to another (**e.g.** Docker Swarm/Compose → Kubernetes) with **zero
> client reconfiguration, no credential loss, and no fleet-wide auth outage**. This is
> the *how*; the *why* + full symptom→diagnosis→fix catalog is the architecture page
> `docs/architecture/orchestrator-migration-cutover.md`. Provider-neutral: **the IdP**
> (**e.g.** Keycloak), **the secret store** (**e.g.** OpenBao/Vault), **the DNS
> authority** (**e.g.** the internal zone resolver), **the edge proxy** (**e.g.** Caddy
> / ingress), **the gateway** (**e.g.** graph-os), **the secret sync operator**
> (**e.g.** External Secrets Operator).

## The invariant: preserve the contract, move the substrate

Clients see three contracts — the **hostname** they dial, the **credentials** they
present, and the **data** behind each service. Migrate the orchestrator *underneath*
those contracts one coupled unit at a time; keep the old backend warm as rollback until
each unit is verified. Never make a client change a name or a token to follow a service
to its new home.

## Pre-flight (do once, before any stack moves)

1. **Registry/images self-sufficient on the new orchestrator** — the cluster must be
   able to pull every image (mirror the in-use images to a cluster-reachable registry
   and place its **CA on every node** when you write the registry trust config; a
   missing CA blocks all private-registry pulls).
2. **In-cluster IdP reachable** — the new orchestrator's DNS resolves the IdP Service;
   inbound JWKS **and** outbound token minting will pin to it (Step 2, §1).
3. **Secret sync wired** — the **secret sync operator** is installed and a
   ClusterSecretStore points at the secret store on its real mount + served API version.
4. **Data placement mapped** — for every stateful stack, know which node holds its
   volume. If that node is **not yet a cluster node**, plan to copy its data to a
   cluster node first (you cannot node-pin a pod to an unmanaged host).

## Gateway auth — the fleet-wide-401 preventer (do this before moving MCP children)

The gateway mints an OAuth2 client-credentials **service token** and attaches it to
every JWT-protected child (`MCP_CLIENT_AUTH=oidc-client-credentials` +
`OIDC_CLIENT_ID/SECRET/AUDIENCE/ISSUER`). A failed mint **degrades to no header** →
**every child 401s** (surfacing as `Session terminated` / `no such host`).

- **On the new orchestrator, PIN the mint to the in-cluster IdP Service:**
  `OIDC_TOKEN_URL=http://<idp-service>.<idp-ns>.svc:8080/realms/<realm>/protocol/openid-connect/token`.
  Auto-discovery routes through the edge, which can **502** mid-migration → empty
  bearer. The IdP stamps the external issuer claim regardless of which endpoint minted
  the token, so child issuer-validation still matches. Same resilience trick as pinning
  JWKS to the in-cluster IdP for inbound validation.
- **One-line diagnosis** (inside the gateway pod): mint a token and assert it is
  non-empty with the right audience —
  `python3 -c 'from agent_utilities.mcp.client_credentials import bearer_header; print(bool(bearer_header({}).get("Authorization")))'`.
  Empty/None ⇒ the mint is the bug (fix the token URL). Non-empty but child still 401s
  ⇒ the child hostname still resolves to the OLD backend (see DNS, below).

## Cluster-driving MCP — its own identity

A deployed MCP that operates the cluster auto-uses in-cluster config when
`KUBERNETES_SERVICE_HOST` is set. It must run on **its own ServiceAccount** with a
**scoped ClusterRole** (workload/config/network/storage/CRD verbs it needs), with **no
RBAC-write** (can't self-escalate) and **not** `cluster-admin`. The namespace default
SA has zero perms → every call 403s. Bake the CLIs it shells out to into its image.

## App + DB secrets — from the secret store, never hand-created

Seed `<store>/<app>` from each stack's existing env, then an ExternalSecret syncs it to
a target Secret. Secret **writes** are operator-gated → ship a **seed script** the
operator runs once per stack; the sync is read-only thereafter. Match the store's KV-v2
data path + the CRD's **served** version. If a synced Secret shows a stale
`SecretSyncError`, **force a refresh annotation**. Prefer an explicit env allow-list
over `envFrom`-the-whole-secret (a per-app secret may carry keys the service must not
inherit — **e.g.** a policy-engine `*_TYPE` pair that crash-loops the boot).

## Hostnames + DNS — zero client reconfiguration

**Keep the existing `<svc>` hostnames working.** Do **not** tell clients to switch to
in-cluster `.svc` names. Per migrated service: a cluster **Ingress carrying the SAME
hostname** + the **DNS authority** pointing that hostname at the ingress VIP. Automate
with an ingress-watching DNS controller (**e.g.** external-dns) so new services
self-publish.

> **SAFETY: selective flip only.** Flip DNS **only** for hostnames that already have a
> cluster Ingress (are migrated). A **global/wildcard flip breaks every unmigrated
> service.** Scope the controller with a domain filter / annotation allow-list.

**Failure mode:** clients 401 *not because tokens are wrong* but because the hostname
still resolves to the **old, pre-migration backend**. Prove it: if `<svc>.svc` returns
200 with the same bearer that 401s on `<svc>`, the public name is still on the old
backend — finish its DNS flip.

## Edge cutover caveat

During progressive migration the edge still points hostnames at the old backend.
**Prefer flipping the DNS authority over editing the edge per-service.** If you must
edit the edge: the edit must reach the **actual serving instance** (multi-replica /
global services + per-node config files are a trap — a host-file edit may not be what
the serving replica reads; edit the live *mounted* config), run the edge's `validate`
before reload, and **curl THROUGH the edge** to confirm before stopping the old
backend. Keep the old backend as rollback until verified. Do not enable a competing
ingress on the same host ports while the old edge is live — use a LoadBalancer VIP
(which becomes the eventual edge at the DNS flip).

## Stateful data-in-place patterns

- `enableServiceLinks: false` for apps that read `<NAME>_*` env as config (injected
  `<SVC>_PORT_*` links collide — **e.g.** a graph DB crash-loops on `NEO4J_PORT_*`).
- Pin hostPath data to the node holding the volume (`nodeSelector`); copy data to a
  cluster node first if its current node isn't in the cluster.
- **Postgres via `pg_dump`/restore, not file-copy** — a stopped named volume can read
  empty at the host level. Run the DB pod as root for a non-root-owned host-path datadir.
- **Hold the app at 0 replicas until the DB is restored** so it can't initialize a
  fresh schema over the restore; then scale to 1.
- **Verify row/table counts match the source** before cutover — "Running" ≠ "correct".

## The coupled-unit sequence (per stack)

Migrate the app **and its DB/cache/media as one unit**, in order:

1. Deploy DB (data-in-place / `pg_dump`-restore), pinned to its data node.
2. Hold app at **0 replicas**.
3. Restore DB → **verify parity** (counts match source).
4. Copy media/blob (from the *stopped* source for a consistent snapshot).
5. App → **1 replica**; wait for a real readiness probe.
6. Verify parity + health end-to-end against the new instance.
7. **Flip DNS (not the edge)** — selectively, this hostname only; old backend stays warm.
8. Stop the old backend (scale old service to 0; keep the definition for rollback).
9. **Re-verify with the old backend down** (proves nothing is still served by the old
   instance).

Rollback at any step: revert the DNS/edge change + scale the old service back to 1 (the
copy was a snapshot; the old datadir is untouched).

## Related

- Full symptom→diagnosis→fix catalog + rationale:
  `docs/architecture/orchestrator-migration-cutover.md`.
- Gateway auth env (inbound JWKS + outbound mint, in-cluster pinning):
  [`graph-os-fleet-gateway-auth.md`](graph-os-fleet-gateway-auth.md).
- Recurring day-2 ops patterns (lost-creds, connector auth, secrets):
  [`homelab-ops-learnings.md`](homelab-ops-learnings.md).
</content>
