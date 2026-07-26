# graph-os Horizontal Scaling — the HPA Blocker, Precisely

> Scope: the **`out-of-process-shared`** engine topology only — a single (optionally
> Raft-clustered) `epistemic-graph` engine served to **N independent graph-os client
> pods**, scaled with a k8s `HorizontalPodAutoscaler` (the "hyperscaling" shape in
> `reports/unified-binary-program.md` and the genesis skill reference
> [`agent-os-genesis/references/engine-topology-and-hyperscaling.md`](../../agent_utilities/skills/workflows/agent-os-genesis/references/engine-topology-and-hyperscaling.md)
> §"Shape 2"). It does **not** apply to the self-contained `unified-in-process` default
> (nothing to horizontally scale there — see
> [`graphos-self-hosting-cutover.md`](graphos-self-hosting-cutover.md), which is about
> *that* shape and is orthogonal to this one). Read those two first for how the shapes
> differ; this page is the current, precise state of the ONE remaining blocker on
> Shape 2's graph-os replica count, grounded against the live cluster
> (`kubectl get`/`-o yaml`, `platform` namespace, 2026-07-25) — not the older staged
> manifests, which this page corrects in one place below.

## TL;DR

Engine→client **TLS transport itself is no longer the blocker — it is live in
production today.** What still caps `graph-os` at `replicas: 1` is three *separate,
smaller* things, none of which is "add a TLS profile":

1. **A host-local config mount** (§2) — mechanical, ~an hour of work.
2. **A static-IP TLS identity pin, not a routable/discoverable one** (§3) — the
   "TLS follow-on" this doc's title refers to; not urgent today but a correctness gap.
3. **No shared MCP session store** (§4) — an accepted, documented limitation with a
   partial mitigation already live (cookie affinity), not something this program
   needs to solve to ship HPA.

None of the three requires re-deriving PKI from scratch — the engine has shipped a
native TLS listener with cert/key/client-CA flags for a while, and it is now wired,
issued, and proven end-to-end. What is missing is the *last mile*: making graph-os
schedulable anywhere, and making the engine's address something other than a
hand-typed IP literal. §5 is the concrete sequence.

## 1. What is live today (verified against the running cluster)

| Component | Live fact | Source |
|---|---|---|
| `epistemic-graph` (engine) | `--tcp-addr 0.0.0.0:9100 --tcp-tls-cert /etc/eg-tls/eng-tls.crt --tcp-tls-key /etc/eg-tls/eng-tls.key`, `hostNetwork: true`, `nodeSelector: r510`, `replicas: 1`, `strategy: Recreate` | `kubectl get deploy epistemic-graph -n platform -o yaml` |
| Engine auth | `GRAPH_SERVICE_AUTH_SECRET` (HMAC) + `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` from Secret `epistemic-graph-secrets`; `EPISTEMIC_GRAPH_REQUIRE_OIDC=false` | same |
| Engine cert | k8s `Secret epistemic-graph-tls` (`kubernetes.io/tls`), issued off the `homelab-arpa-ca` `ClusterIssuer`, SAN for the engine's node IP | `kubectl get secret epistemic-graph-tls -n platform` (type/keys only — no values read) |
| Engine Service | `ClusterIP`, ports `9100` (engine RPC) + `9130` (kvcache) — routable in-cluster from any node despite `hostNetwork` (kube-proxy targets the node IP) | `kubectl get svc epistemic-graph -n platform` |
| `graph-os` (client) | image `knucklessg1/graph-os-unified@sha256:…` (the **unified image**, W-C), command `python3 -m agent_utilities.mcp.kg_server`, **not** `hostNetwork`, but still `nodeSelector: r510`, `replicas: 1`, `strategy: RollingUpdate` (`maxSurge: 1, maxUnavailable: 0`) | `kubectl get deploy graph-os -n platform -o yaml` |
| graph-os → engine TLS config | ConfigMap `graph-os-env`: `ENGINE_CA_BUNDLE=/etc/ssl/homelab/ca-bundle.pem`, `ENGINE_TLS_SERVER_NAME=10.0.0.10` (the engine's literal node IP) | `kubectl get configmap graph-os-env -n platform -o yaml` |
| graph-os auth | Secret `graph-os-secrets` carries the **same** `GRAPH_SERVICE_AUTH_SECRET` key as the engine's secret (shared HMAC, as `engine_sharding.md` requires fleet-wide) | key list only, no values read |
| Ingress | `graph-os.arpa`, **`nginx.ingress.kubernetes.io/affinity: cookie` already set**, TLS via `graph-os-tls` | `kubectl get ingress graph-os -n platform -o yaml` |
| Autoscaling objects | **none** — `kubectl get hpa,pdb -n platform` returns nothing | same |

`ENGINE_CA_BUNDLE`/`ENGINE_TLS_SERVER_NAME` are exactly the inputs
`core/transport_security.py`'s `resolve_configured_tls_profile("ENGINE", …)` needs to
build a working TLS trust profile without a named `ENGINE_TLS_PROFILE` — configuring
them is pointless unless `GRAPH_SERVICE_ENDPOINTS` is actually a `tls://` value (a bare
`unix://`/loopback `tcp://` endpoint never reaches that code path at all —
`engine_transport.py`'s `engine_client_transport_kwargs`). `GRAPH_SERVICE_ENDPOINTS`
itself is not in the ConfigMap or either Secret's key list; it lives in the durable
`config.json` mounted from the host (see §2) — this page did not read that file, so the
literal endpoint scheme is inferred from the above, not directly observed. That
inference is corroborated by `reports/HANDOFF-2026-07-22.md` §2e, which proves
(against a copy of the real 9.9G store) `tls://` connect-OK from the merged client to
this exact engine binary/cert, and by the `graph-os` Deployment's own
`Progressing=True`/`readyReplicas: 1` status today — it is not crash-looping against an
engine it cannot legally reach.

**What this corrects:** `services/epistemic-graph/k8s/production-separate.yaml`'s
header (and the genesis skill reference cited above) both still describe a design that
predates this — "without adding a TLS profile, the only zero-new-work way to reach the
engine over TCP is loopback" / "which this staged manifest does not yet add". That was
true when those files were last substantively edited; it stopped being true on
2026-07-22 (see `reports/HANDOFF-2026-07-22.md` §1, §2e, §7). Neither file has been
updated since — treat their TLS-availability claims as historical, not current;
this page is now the current source for that fact. (Their *other* content —
the 4-independent-Deployments shape, the Raft/HPA architecture — is unaffected and
still accurate.)

## 2. Blocker #1 — graph-os is still node-pinned, but no longer for a TLS reason

`graph-os`'s `nodeSelector: kubernetes.io/hostname: r510` is not there because of
`hostNetwork` (it doesn't set that field at all — a real change from the staged
`production-separate.yaml` design). It is there because two of its volumes are plain
**host-local** `hostPath` mounts on r510, not the NFS mounts the rest of the pod
already uses:

```yaml
volumes:
  - hostPath: {path: /home/genius/.config/agent-utilities-next, type: Directory}
    name: au-config          # durable AgentConfig dir (config.json) — LOCAL DISK
  - hostPath: {path: /run/epistemic-graph, type: DirectoryOrCreate}
    name: uds                # UDS socket dir — LOCAL DISK, role unconfirmed (see below)
  - name: au-src              # NFS 10.0.0.12:/home/apps/workspace/.../agent-utilities
  - name: eg-wheel             # NFS 10.0.0.12:/home/apps/eg-wheel
```

`au-src` and `eg-wheel` were already migrated to NFS (any node can mount them); the
config directory was not. A second replica scheduled on `r710`/`rw710`/`gb10`/`r820`
would find no `/home/genius/.config/agent-utilities-next` on that host and boot with an
empty (or absent) durable config — this is a **config-distribution** gap, unrelated to
whether the engine speaks TLS. It is also the more mundane of the two Blockers and
should be fixed first, independent of §3.

Fix options (either is sufficient — pick one, don't do both):

- **(a) Publish `config.json` onto the same NFS export** `au-src`/`eg-wheel` already
  use, and mount it read-only from there instead of the r510-local hostPath. Minimal
  diff from today's shape.
- **(b) Drop the hostPath entirely.** Per `AGENTS.md` → *Configuration discipline*,
  deployment-varying values belong in `config.json`-via-OpenBao-reference or the typed
  `AgentConfig`/Secret path already used for almost everything else in this Deployment
  (`graph-os-env` ConfigMap + `graph-os-secrets`/`epistemic-graph-secrets` Secrets). If
  everything currently living only in the host-local `config.json` can be expressed as
  ConfigMap data + `secret://`/`vault://` references, this removes the last node-local
  state from the pod spec and is the more idiomatic fix given the existing pattern
  every *other* setting in this Deployment already follows.

The `/run/epistemic-graph` UDS hostPath mount's current role is **not confirmed by this
page** — it may be a live fallback/secondary path, or a leftover from an earlier
bundled-pod-style iteration now superseded by the TLS path in §1. Either way it is
**also** host-local, so it does not change the fix above; if it turns out to be
load-bearing for something other than "reach the co-located engine," that needs its
own network-reachable substitute before removing the nodeSelector.

## 3. Blocker #2 — the TLS follow-on this doc is named for: identity is a static IP pin, not routable/discovered

This is the gap the unified-binary-program task named "routable mTLS/SNI instead of
loopback." Loopback is already gone (§1) — what remains is that the TLS identity model
in place today is **hand-wired to one literal address**, not the general mechanism the
engine already ships for exactly this purpose:

- **Today:** the engine's cert has a SAN for its node's literal IP (`10.0.0.10`), and
  graph-os is configured with `ENGINE_TLS_SERVER_NAME=10.0.0.10` to match it verbatim.
  This works — but only because the engine is (and, being a single-writer redb store,
  will structurally remain — see §6) permanently pinned to that one node. Every
  graph-os replica, wherever k8s schedules it, needs the *identical* hardcoded pin;
  there is no discovery, so a future engine move (a raft failover to a different
  voter, a re-platform to a different host) requires a coordinated manual edit of
  every client's config, with no live signal that it happened.
- **No client mTLS.** The engine binary supports `--tcp-tls-client-ca` for verified
  client certificates (`src/main.rs` `Args::tcp_tls_client_ca`), but it is not
  configured on the live Deployment (only `--tcp-tls-cert`/`--tcp-tls-key`). Today the
  ONLY thing distinguishing an authorized caller from anyone who can route to
  `10.0.0.10:9100` is the app-layer HMAC bearer secret
  (`GRAPH_SERVICE_AUTH_SECRET`/`eg2.` envelope signing) — TLS here authenticates the
  *server* to the client, not the reverse.
- **The mechanism that already exists for this, unused so far:** eg `AGENTS.md`'s
  cluster-topology discovery — `EPISTEMIC_GRAPH_ADVERTISED_CLIENT_ADDR` +
  `EPISTEMIC_GRAPH_ADVERTISED_TLS_SERVER_NAME`, self-reported by each engine node at
  startup (`Method::NodeInfoUpsert`) and handed back to a discovering client
  (`Method::ClusterMembers`, `epistemic_graph/pool.py`'s `resolve_cluster_endpoints`).
  This is explicitly designed to replace "a static hand-maintained client map" — which
  is exactly what `ENGINE_TLS_SERVER_NAME=10.0.0.10` is, today, by hand.

**Why this is not an urgent blocker today, and is a correctness gap tomorrow:** with
one static engine node, the IP pin is functionally sufficient — the cluster's flat
host network (`MCP_TRUSTED_PROXY_CIDRS` already trusts `10.0.0.0/24`) means any
graph-os replica on any of the 5 nodes can already reach `10.0.0.10:9100`. Nothing in
§2's fix is blocked on this section. It becomes load-bearing the moment either (a) the
engine adopts the `cluster`/Raft build (the genesis reference's Shape 2 subsections
"Build the `cluster` engine" / "Stand up the Raft members" — a *different*, larger,
separately-scoped program) so its address can legitimately change on failover,
or (b) an operator wants real client-identity enforcement (mTLS) rather than a shared
bearer secret. Track it as a named follow-on, not a step in this sequence's critical
path — see §5 step 4.

## 4. Blocker #3 — MCP session affinity (documented limitation, already mitigated, not solved)

The MCP Python SDK's `StreamableHTTPSessionManager` keeps `_server_instances` as a
plain in-process dict — no shared/external session store exists (or is planned by any
current program; this is a genuine SDK-level constraint, not an au gap). Once
`replicas > 1`:

- A **new** session can land on any replica and works fine.
- A request **continuing** an existing session must land back on the *same* replica.
  The Ingress already carries `nginx.ingress.kubernetes.io/affinity: cookie` (§1 table)
  — live today, a no-op at `replicas: 1`, and becomes load-bearing the moment a second
  replica exists.
- Cookie affinity does **not** survive a **pod restart** — a scaled-down or rescheduled
  pod's dict is empty regardless of the cookie, so its in-flight sessions are lost and
  the client must reconnect. This is the direct implication for HPA specifically: a
  scale-down event (not just a crash) drops whatever sessions were pinned to the
  terminated replica.

**This is not something this program needs to fix to ship HPA** — it is an accepted,
already-mitigated limitation (same conclusion as the genesis skill reference and the
older `production-separate.yaml`). Document it in the HPA rollout (a conservative
`scaleDown` policy/stabilization window reduces how often it bites) rather than
blocking on solving it; solving it for real means an externalized session store, which
is a distinct, larger piece of work (parallel to the `STATE_DB_URI` externalization
track already used for checkpoints/queues — see
[`state_externalization.md`](state_externalization.md)) that nothing here depends on.

## 5. The sequence to unblock — in order

1. **Fix the config-locality pin (§2).** Either move `config.json` onto the existing
   NFS export or fold its contents into `graph-os-env`/OpenBao references. Verify by
   scheduling a throwaway pod with the SAME volumes on a non-r510 node and confirming
   `agent_utilities.core.config` builds cleanly.
2. **Drop `graph-os`'s `nodeSelector`.** With §2's fix in place, nothing left in the
   pod spec is node-local. Confirm a rolling restart lands the single replica on any
   node (not just r510) before touching replica count.
3. **Add the `HorizontalPodAutoscaler`.** Target the `graph-os` Deployment, CPU- or
   request-rate-driven (the gateway already emits
   `agent_utilities_gateway_in_flight_requests` /
   `agent_utilities_gateway_requests_total`, per `gateway_scaling.md` — a custom-metric
   HPA can use these instead of bare CPU). Add a `PodDisruptionBudget` once `replicas`
   can legitimately be `>1` (a PDB at `replicas: 1` only blocks eviction — see the
   engine's own comment on why it deliberately has none).
4. **(Follow-on, not gating) Replace the static TLS pin with discovery (§3).** Wire
   graph-os's engine-endpoint resolution through
   `epistemic_graph.pool.resolve_cluster_endpoints`/`ClusterMembers` instead of a fixed
   `ENGINE_TLS_SERVER_NAME`, and evaluate turning on `--tcp-tls-client-ca` for real
   mTLS. Natural to bundle with any future move to the `cluster`/Raft engine build,
   since that is the point at which a fixed IP pin actually breaks.
5. **Validate (§7 checklist) at `replicas: 2`, then let the HPA drive further.**

## 6. What this document does NOT change

The engine (`epistemic-graph` Deployment) stays at `replicas: 1`, `strategy: Recreate`,
pinned to r510, in every step above — that is a **structural** property of a
single-writer redb store, not a topology choice HPA on graph-os can or should affect.
Horizontal engine capacity is a separate axis (the `cluster`/Raft build,
`EPISTEMIC_GRAPH_RAFT_GROUPS` write-sharding — see `docs/architecture/engine_sharding.md`
and the genesis reference's §"3. Shard for write-scaling") that this document
deliberately does not attempt. This document is scoped to graph-os the **client**
scaling independently of the engine, which is the entire premise of the
`out-of-process-shared` shape.

This document also does not apply to — and should not be read as blocking —
`graphos-self-hosting-cutover.md`'s design, which runs the OPPOSITE shape (engine
embedded per-pod, `unified-in-process`) and is explicitly single-replica by design
(nothing to horizontally scale in a self-contained unit — see that shape's own "nothing
to hyperscale here by design" note in the genesis reference).

## 7. Validation checklist (for whoever executes §5)

- [ ] A graph-os pod with §2's fix applied, scheduled on a node **other than r510**,
      reaches `Ready` and its `/health` reports the engine reachable.
- [ ] Two graph-os replicas (`kubectl get deploy graph-os -o jsonpath='{.status.readyReplicas}'`
      → `2`) on **different** nodes, both healthy.
- [ ] A single MCP streamable-http session, opened against the Ingress, survives
      several requests (cookie affinity keeps it on one replica) — confirm via the
      `Set-Cookie`/repeat-request pattern, not just pod logs.
- [ ] Kill the replica an active session is pinned to; confirm the client reconnects
      (new session on a surviving replica) rather than hanging — i.e. the documented
      limitation in §4 fails the way this page says it fails, not some other way.
- [ ] `agent_utilities_gateway_requests_total`/`in_flight_requests` are visible
      per-replica in Prometheus (each replica's own registry, per `gateway_scaling.md`)
      before wiring the HPA to a custom metric.
- [ ] HPA scale-up and scale-down both observed at least once in a soak window; no
      `CrashLoopBackOff` attributable to the removed nodeSelector.
- [ ] `epistemic-graph` (the engine) untouched throughout — still `replicas: 1`,
      same pod, no restart caused by graph-os scaling events.
