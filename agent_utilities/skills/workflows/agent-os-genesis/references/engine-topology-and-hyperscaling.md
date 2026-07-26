# Engine topology: unified self-contained (default) vs. out-of-process hyperscaling

> The two-shapes edict (au `AGENTS.md` "Engine transport", eg `AGENTS.md` two-shapes,
> `reports/unified-binary-program.md`) made explicit as a genesis run-plan axis —
> `run_plan.engine_topology` / per-profile `engine_topology` in `genesis.yaml`. This is
> the full depth Step 0a′ and Step A2 point to; read those first for the summary.
>
> **Status (read before treating this as fully live):** `unified-in-process` is the
> **declared default** for `tiny`/`single-node-prod` — the target the platform is
> converging on. The PyO3 in-process binding that makes it literally one process
> (`reports/unified-binary-program.md` workstream W-A) is **landing, not yet shipped**.
> Until it ships, genesis realizes the SAME externally-visible contract (one thing to
> install, no separate engine service to operate/version/drift) via the existing
> bundled/autostarted engine sharing the host with graph-os. Nothing about the
> profile's *steps* changes when W-A lands — only the internal transport. Similarly,
> the k8s manifests this doc cites under `services/epistemic-graph/k8s/` and
> `services/epistemic-graph/soak/` are **staged references** (their own file headers say
> "PROPOSED — NOT LIVE" / "isolated soak"), not applied production state — adapt them,
> don't apply verbatim.

## The two shapes, at a glance

| | `unified-in-process` (default) | `out-of-process-shared` (hyperscaling) |
|---|---|---|
| Default profiles | `tiny`, `single-node-prod` | `enterprise` |
| Process boundary | Engine embedded **in-process via PyO3**, same binary as graph-os (`epistemic_graph.engine`) | Engine is a **separate, independently-scaled** process; graph-os talks over UDS/TCP (`epistemic_graph.client`, GIL-free) |
| `genesis.yaml` `engine` shape | `autostart` (tiny) / `container` (single-node-prod) | `remote` |
| Engine build | `default`/`full` cargo features (single-node, no raft) | `cluster` cargo feature (`full` + `raft` + more — multi-Raft sharding, HA) |
| Scaling axis | None — one self-contained unit | Engine (Raft group count / voters) and graph-os (pod replicas) scale **independently** |
| What you deploy | ONE image/pod: engine + KG + numeric kernel + messaging + gateway | The engine cluster (its own Deployments/StatefulSet) **and** N graph-os pods behind the gateway |
| Non-negotiable rule (both shapes) | Engine calls stay **batched** (one call = one batch op over graph-resident data, never a per-element Python loop); the engine's compute internals stay pure Rust | (same) |

Orthogonal axis, not to be confused with the above: `genesis.yaml`'s existing `engine`
field (`autostart` / `container` / `remote`) is **where/how the engine process is
obtained** on a host. `engine_topology` is **which side of a process boundary** the
engine sits on relative to graph-os. In practice they move together per profile (see
the table above) but they are independent knobs in the manifest.

## Shape 1 — `unified-in-process` (the self-contained default)

```
┌─────────────────────────────────────────────┐
│ ONE process / ONE pod / ONE image            │
│                                               │
│   graph-os (kg_server, MCP + REST gateway)   │
│        │  in-process call (PyO3), no socket  │
│        ▼                                     │
│   epistemic_graph.engine  (Rust, embedded)   │
│        │                                     │
│        ├─ numeric kernel (xp / ABI3 .so)     │
│        ├─ redb durable store                 │
│        └─ OWL/SPARQL, vector/ANN, …          │
│                                               │
│   messaging daemon (Telegram/Mattermost/…)    │
│   host daemon (queue drain, KG_LOOP, …)      │
└─────────────────────────────────────────────┘
```

- **No separate engine service to deploy, version, or let drift** out of sync with
  graph-os — one image tag, one restart, one set of logs. This is what "consolidated"
  means for `tiny`/`single-node-prod`: the profile table's "one self-contained
  graph-os" is this diagram.
- **tiny** realizes it as `engine: autostart` — the `EngineResolver`
  (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision) constructs the engine on first use,
  refcounted (self-stops ~60s idle) or `persistent`.
- **single-node-prod** realizes it as `engine: container` — the SAME consolidation,
  containerized: the `graph-os` image IS the engine + gateway + messaging, not a
  sidecar arrangement. (An intermediate step some deployments may still run — 3-4
  explicit containers in ONE pod sharing a socket/`emptyDir`, e.g.
  `services/epistemic-graph/k8s/bundled-core-pod.yaml` Variant A — is a valid
  stepping stone toward the tighter PyO3 embedding, not a competing target.)
- **Nothing to hyperscale here by design.** If load outgrows one host, that is the
  signal to move the profile/topology to `out-of-process-shared`, not to add
  replicas of a self-contained unit (its engine has no shared state to coordinate
  across copies).

## Shape 2 — `out-of-process-shared` (the hyperscaling shape)

```
                    k8s HPA (independent scaling)
                    ┌─────────────┐
        ┌──────────▶│ graph-os-1  │──┐
        │           └─────────────┘  │
gateway │           ┌─────────────┐  │   GRAPH_SERVICE_ENDPOINTS
ingress ├──────────▶│ graph-os-2  │──┼──▶ (UDS/TCP, GIL-free client)
        │           └─────────────┘  │
        │                 ⋮          │
        │           ┌─────────────┐  │
        └──────────▶│ graph-os-N  │──┘
                    └─────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │  epistemic-graph engine — `cluster`│
                    │  cargo-feature build (raft)         │
                    │                                     │
                    │  MultiRaft: N groups, each a         │
                    │  quorum-committed redb shard          │
                    │  (group g owns shard g)               │
                    │  node1 (seed) · node2 · node3 (voters)│
                    └───────────────────────────────────┘
```

### 1. Build the `cluster` engine

The `raft` cargo feature is the **opt-in `cluster` build layer only** — `cluster =
["full", "raft", …]`, NOT in `default`/`full`/`all`, so the plain build links no
`openraft` (eg `AGENTS.md` "Opt-in: in-engine Raft replication"). Build/pull it
explicitly (`EG_FEATURES=cluster` in the image build, or `cargo build --release
--features cluster,ast-extended`). This is a **different build of the same engine**,
not a separate product line — genesis's `check_genesis_manifest.py` gate forbids
reintroducing multiple engine *artifact* tiers; the `cluster` feature is a build-time
capability toggle on the ONE engine, selected at deploy time like any other runtime
configuration.

### 2. Stand up the Raft members (HA)

Each voting node runs its own engine process:

- `EPISTEMIC_GRAPH_RAFT_NODE_ID` (this node's id — absent ⇒ single-node), `_PEERS`
  (`id@host:port,…`, identical on every node), `_BIND_ADDR`, and **one shared**
  `EPISTEMIC_GRAPH_RAFT_AUTH_SECRET`. Reference env file:
  `services/epistemic-graph/flavors/cluster.env`
  (4-node fleet example — peers list, port model, per-node overrides, reversibility).
- **Raft peering MUST use a headless Service (`clusterIP: None`) with pod DNS, never a
  plain ClusterIP.** A ClusterIP's DNAT does not carry the engine's long-lived pooled
  Raft transport reliably — a live soak proved followers cannot elect a new leader
  through one (`Vote … timeout` despite open L4 TCP). The fix is the standard
  etcd/CockroachDB pattern: a headless Service + per-pod DNS peering. See
  `services/epistemic-graph/soak/README.md` ("★ Failover finding") for the evidence,
  and `services/epistemic-graph/soak/21-engine-statefulset.yaml` for the working
  manifest shape.
- **k8s reference manifests (staged, adapt don't apply verbatim):**
  `services/epistemic-graph/k8s/raft-cluster/` — a 3-voter production design (one
  Deployment per node, deterministic host pinning, headless peer Services +
  ONE ClusterIP fronting all voters' client port for remote dialers).
  3 voters tolerate 1 node failure (the HA sweet spot); node 1 must be the host
  already holding the authoritative data (it becomes the Raft SEED, never wiped).
- **Validate**: `services/epistemic-graph/soak/README.md`'s procedure — formation +
  leader election in logs, a write on the leader read back on a follower, and
  killing the leader re-electing (~5s) while the survivors keep serving on quorum.

### 3. Shard for write-scaling (optional, orthogonal to HA)

HA (above) tolerates node loss; it does not by itself scale writes — one Raft group
is one write lock. For write-scaling, `EPISTEMIC_GRAPH_RAFT_GROUPS` stands up **N
independent Raft groups** (`MultiRaft`), each with its own redb shard and its own
apply loop/durable writer — group *g* owns shard *g*, so HA and write-scaling
coexist (eg `AGENTS.md` "Multi-Raft groups"). Client-side this is the identical
tenant-partitioned routing already documented for the non-raft case —
[`docs/architecture/engine_sharding.md`](../../../../../docs/architecture/engine_sharding.md):
`tenant → named graph → HRW (rendezvous hash) → shard endpoint`, configured via a
`GRAPH_SERVICE_ENDPOINTS` list (one entry per group/leader) shared verbatim by every
client. The engine side just now replicates each shard instead of running it
single-node — the routing math on the Python side is unchanged.

### 4. N graph-os client pods behind the gateway, k8s HPA

Every graph-os pod is a **plain, stateless-w.r.t.-the-engine client**: point
`GRAPH_SERVICE_ENDPOINTS` at the engine (single member or the sharded list), and
authenticate with the fleet's ONE `GRAPH_SERVICE_AUTH_SECRET`. This is the piece
that actually hyperscales graph-os itself:

- **Reference (staged, current honest limitation):**
  `services/epistemic-graph/k8s/production-separate.yaml` already decouples graph-os
  into its own Deployment (independent restart/roll from the engine), reached over
  **loopback TCP** — because a non-loopback `tcp://` engine endpoint requires TLS
  (`engine_transport.py` enforces this), which this staged manifest does not yet add.
  Loopback-only means graph-os is still capped at `replicas: 1` today (hostNetwork
  pods can't share a port on one node).
- **The follow-on that unlocks real horizontal scale:** a TLS profile for the engine
  endpoint so a graph-os replica on a **different** node can dial the engine over a
  real (non-loopback) address. Once that lands, put an `autoscaling/v2
  HorizontalPodAutoscaler` on the `graph-os` Deployment (CPU or request-rate driven)
  — it scales graph-os pods **independently** of the engine's own Raft-group/voter
  count, which is exactly the point of this shape.
- **Session affinity is mandatory once graph-os has >1 replica.** The MCP Python
  SDK's `StreamableHTTPSessionManager` keeps sessions in an in-process dict — a
  request continuing a session that lands on a *different* replica finds nothing.
  Pin clients to their originating pod with cookie affinity on the Ingress
  (`nginx.ingress.kubernetes.io/affinity: cookie`) — this does not survive a pod
  *restart* (the dict is empty regardless of the cookie), only spreads live traffic
  correctly across replicas.
- **Gateway worker model (single-host multi-worker, orthogonal to k8s HPA):** if you
  are not yet on k8s, `GATEWAY_WORKERS=N` pre-forks N workers behind one listen
  socket on a single host — see
  [`docs/architecture/gateway_scaling.md`](../../../../../docs/architecture/gateway_scaling.md)
  for exactly what is/isn't per-process state (metrics, rate limits, the circuit
  breaker) under that model. It is the compose/single-node equivalent of "more than
  one graph-os" before you have a cluster to HPA against.

### 5. What this buys you vs. what it costs

- **Buys:** the engine's HA/write-throughput scales via Raft groups/voters; graph-os's
  request-serving capacity scales via pod replicas — the two scale **on different
  axes, independently**, which is the entire reason `out-of-process-shared` exists as
  a distinct shape from the self-contained default.
- **Costs:** more moving parts to operate (a Raft cluster to monitor, a headless
  Service + StatefulSet/per-node-Deployment topology, the TLS follow-on for real
  graph-os scale-out) — which is exactly why it is NOT the default and is reserved
  for the profile that actually needs it (`enterprise`).

## Choosing / changing the topology

- Genesis seeds the **default** from the profile (`genesis.yaml` per-profile
  `engine_topology`); an operator can still override it at Step 0 like any other
  run-plan axis (e.g. a beefy single host that wants HA without a full k8s
  enterprise rollout can opt `single-node-prod` into `out-of-process-shared` early;
  an enterprise that is not yet at scale can start `unified-in-process` and graduate
  later).
- Moving from `unified-in-process` to `out-of-process-shared` is a **data migration**,
  not a config flip: stand up the target engine (cluster build), point
  `GRAPH_SERVICE_ENDPOINTS` at it, migrate the existing durable store onto it (the
  same `.mp`/redb snapshot tooling `docs/architecture/engine_sharding.md`'s
  rebalancing section uses), verify, then cut graph-os over. Never run both against
  the same persist directory at once (single-writer-per-shard).

## Pointers (single sources of truth — do not duplicate their detail here)

| Concern | Source |
|---|---|
| **Skill-embedded k8s recipes (genesis DEFAULT, template-shaped, apply directly)** | [`k8s/README.md`](k8s/README.md) — `graphos-unified.yaml` (unified-in-process), `hyperscale-engine-and-graphos.yaml` (out-of-process-shared), `mcp-server-editable.yaml` (fleet) |
| The edict itself (why two shapes, the batching rule) | au `AGENTS.md` → "Engine transport (two shapes)"; eg `AGENTS.md` → PyO3/in-process section |
| Program status (what's shipped vs. landing) | `reports/unified-binary-program.md` |
| In-engine Raft replication, multi-Raft groups, sharded writer | eg `AGENTS.md` → "Sharded K-way durable writer", "Opt-in: in-engine Raft replication", "Multi-Raft groups" |
| Tenant-partitioned HRW sharding (client-side routing, non-raft or raft) | `docs/architecture/engine_sharding.md` |
| Gateway multi-worker model, per-process state | `docs/architecture/gateway_scaling.md` |
| Raft HA cluster env (4-node fleet example) | `services/epistemic-graph/flavors/cluster.env` |
| Fast-storage/compute split (a related but distinct scale pattern) | `services/epistemic-graph/flavors/split-storage.env`, `docs/recipes/split-storage-engine.md` |
| Staged k8s: single-pod bundle (dev/uvx/compose fit) | `services/epistemic-graph/k8s/bundled-core-pod.yaml` |
| Staged k8s: 4 independent Deployments (current k8s production target) | `services/epistemic-graph/k8s/production-separate.yaml` |
| Staged k8s: 3-voter Raft cluster | `services/epistemic-graph/k8s/raft-cluster/` |
| Raft soak evidence (headless-Service fix, validation procedure) | `services/epistemic-graph/soak/README.md` |
