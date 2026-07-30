# graph-os Self-Hosting Cutover — Design (EXECUTED 2026-07-26)

> **✅ EXECUTED 2026-07-26 against `platform/graph-os` + `platform/epistemic-graph`.**
> Result: one self-contained `graph-os` pod on `<ENGINE_NODE>` (engine native-sidecar + graph-os,
> `2/2 Ready`), engine loaded 87 graphs (matches pre-cutover baseline), graph-os connected
> `resolved_mode=remote endpoint_count=1 reachable_count=1 breaker=0 latency=0.4ms` over the
> in-pod unix socket, messaging up (mattermost+telegram). Old `epistemic-graph` Deployment
> kept scaled to 0 (not deleted) for the rollback/soak window. **One deviation from the spec
> below:** the sidecar binds `--tcp-addr 0.0.0.0:9100` (not `127.0.0.1:9100`) because a
> `tcpSocket` startupProbe dials the POD IP, not loopback — a loopback-only bind makes the
> probe "connection refused" forever. Same TLS+HMAC posture as the pre-cutover engine (which
> also bound 0.0.0.0), so no security regression. FOLLOW-UP to restore true loopback-only:
> switch the three probes to an `exec` python `socket.create_connection(('127.0.0.1',9100))`
> check, then revert the bind to `127.0.0.1`. Live manifest persisted to
> `services/graph-os/k8s/graph-os.deployment.yaml`.


> Scope: the **`unified-in-process`** engine topology (the self-contained default —
> `reports/unified-binary-program.md`, genesis skill reference
> [`agent-os-genesis/references/engine-topology-and-hyperscaling.md`](../../agent_utilities/skills/workflows/agent-os-genesis/references/engine-topology-and-hyperscaling.md)
> §"Shape 1"). This is the OPPOSITE shape from
> [`graphos-horizontal-scaling.md`](graphos-horizontal-scaling.md) (`out-of-process-shared`,
> N graph-os pods against one external engine) — do not run both cutovers on the same
> Deployment. This document is **design only**; the orchestrator executes it live, on
> its own schedule, against the real `platform/epistemic-graph` + `platform/graph-os`.
> **Nothing in this document has been applied.**
>
> **Placeholders used below** (this is a public page — the literal values are
> homelab-specific and deliberately not repeated here; resolve them with the cited
> `kubectl get` commands, exactly as in
> [`graphos-horizontal-scaling.md`](graphos-horizontal-scaling.md)):
> `<ENGINE_NODE>` / `<ENGINE_NODE_IP>` — the node the redb hostPath pins the engine to,
> and its cluster IP. `<ENGINE_HOST_HOME>` — the operator home directory backing the
> redb/TLS/config hostPath mounts. `<WORKSPACE_NFS_SERVER>` / `<WORKSPACE_NFS_EXPORT>`
> — the existing NFS export `au-src`/`eg-wheel` already use. `<REGISTRY>` — wherever
> the unified image is published.

## Why this is possible now, and why it wasn't before

`docker/graphos-unified.Dockerfile` (W-C) now bakes the engine binary
(`epistemic-graph-server`, at `/usr/local/bin/`) **into the same image** as
`graph-os`/`kg_server` — confirmed live: `platform/graph-os` already runs
`<REGISTRY>/graph-os-unified@sha256:…` today. Before this image existed, co-locating
engine + graph-os in one pod required two *different* container images (see
`services/epistemic-graph/k8s/bundled-core-pod.yaml` Variant A: an `ubuntu:26.04` engine
sidecar with a hostPath-mounted binary, alongside a separate `<REGISTRY>/agent-utilities`
graph-os container) — the genesis reference already names that shape "a valid stepping
stone toward the tighter PyO3 embedding, not a competing target." This document is that
stepping stone made concrete against the current unified image and the real 87-graph
production store, superseding the need for two images: **both containers below use the
SAME unified image**, differing only in `command`.

**Two mechanisms, only one designed here:**

- **(a) INTERIM (this document).** The engine runs as a second process **in the same
  pod** as graph-os, using the engine's OWN existing production config — a k8s **native
  sidecar** (`initContainers` entry with `restartPolicy: Always`, k8s ≥1.29; this cluster
  is 1.35), not a single shell script that backgrounds a process. graph-os **connects**
  to it over a local socket exactly as it connects to today's separate Deployment —
  nothing in `kg_server` changes.
- **(b) FINAL — W-A's in-process PyO3 engine.** `epistemic_graph.engine` embedded via
  PyO3 in the SAME process as `kg_server` (no second process, no socket at all). Per
  `reports/unified-binary-program.md`, this is **NOT STARTED** (the largest workstream).
  Nothing below depends on it landing; (a) is a complete, shippable end state on its own
  and is not "temporary scaffolding" for (b) — it removes the separate Deployment today,
  and (b) can replace *this pod's* sidecar with an in-process call later without another
  data migration (same store, same identity, only the transport inside the pod changes).

## The identity trap this design exists to avoid

**Do not** delete the separate `epistemic-graph` Deployment and simply let graph-os's
existing `EngineResolver` **autostart** a local engine against the same `--persist-dir`.
It will start, but it will not be able to serve the 87-graph store, and it will fail in
one of two ways depending on whether graph-os's own process already happens to carry an
`EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` (it does not, today):

- Read `agent_utilities/knowledge_graph/core/graph_compute.py`
  (`_autostart_engine`, ~L1656–1711): when it spawns a child engine, it takes the
  **current request's own verified actor** (`bootstrap_session.engine_verified_context()`
  — a per-caller Keycloak identity, not a static service account) and either (i) if the
  parent process already carries an `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`, **requires that
  exact actor to already be a key in it** — `raise RuntimeError("verified process
  identity is absent from the engine signer registry")` if not — or (ii) if no registry
  is configured at all, **mints a brand-new throwaway signer key for just that one
  actor** and hands it to the fresh child.
- Neither case is what the 87-graph store's `rbac.redb` was actually provisioned with.
  Per `reports/HANDOFF-2026-07-22.md` §7, the store's RBAC was bootstrapped for
  `homelab-system` and graph-os's own service-account subject — a per-request end-user
  identity (case i) or a brand-new ad hoc one (case ii) matches neither, and the engine
  answers `ACCESS_DENIED: a provisioned identity/RBAC policy is required` (or, against a
  genuinely empty store, would instead treat it as a *fresh* bootstrap — also wrong,
  since this store is not fresh).
- **Autostart is designed for a zero-config local/tiny engine, not for attaching to an
  already-governed production store.** `resolve_engine`
  (`knowledge_graph/core/engine_resolver.py`) only takes the autostart leg when
  `GRAPH_SERVICE_ENDPOINTS` is **absent**. The fix is simply to never let that happen:
  **always set `GRAPH_SERVICE_ENDPOINTS` explicitly**, pointing at the sidecar engine.
  With it set, the resolver's precedence is `remote` — "every configured topology is a
  hard contract: connect to it, never auto-spawn a local stand-in" — the exact same code
  path graph-os already uses to reach today's separate engine, just now over a socket
  inside its own pod instead of TLS to a sibling pod. The sidecar engine is launched
  directly (its own binary, its own CLI flags, its own env) — **never** through
  `_autostart_engine`.

This is the whole of "sidesteps the autostart identity wall": not a workaround, but
*not exercising* the autostart code path at all, by construction.

## The redb single-writer-lock handoff

The engine enforces an OS-level single-writer lock on `--persist-dir`
(`src/main.rs`, `persist_lock::acquire`, held for the whole process lifetime; a second
engine pointed at the same directory refuses to start while the lock is held). There is
**no hot handoff** — the old engine must fully release the lock before the new one can
take it:

```
1. Freeze writes / note the current graph count for a post-cutover sanity check.
2. Snapshot the redb store (see "Pre-flight" below) — the ~2-3 min window in step 3
   is exactly the kind of change the workspace's own migration discipline requires
   rehearsing against a COPY first (proven practice: reports/HANDOFF-2026-07-22.md
   caught two production-breaking regressions this way before they hit the real
   9.9G store).
3. Scale platform/epistemic-graph to 0 (NOT delete — see Rollback).
     -> lock released. KG is now DOWN for every client (graph-os has no other engine).
4. Roll platform/graph-os to the new pod spec (below). The engine sidecar acquires
   the lock, loads the redb catalog (4 shards, ~45s-2min historically —
   startupProbe budgets to it), and the sidecar's startupProbe passes.
     -> kubelet was blocking graph-os's own container start until this passed;
        it now starts, connects (mode="remote", see above), and serves.
   Total KG-down window: stop (near-instant) + new-engine-load — the same ballpark
   as today's engine `Recreate` restarts (an eg cutover on 2026-07-22 measured
   ~2.5 min end-to-end for a comparable stop/reload).
5. Verify (checklist below), THEN delete the old epistemic-graph/Service, or leave
   them scaled to 0 for a rollback window before deleting.
```

**Out of scope / do not touch:** `platform/epistemic-kvcache` (image
`registry.arpa/epistemic-graph:kvcache`, runs on the cluster's GPU node) shares the
engine codebase but is an **unrelated** workload (vLLM KV-cache layering, not KG
storage) on a different node — this cutover does not affect it.

**Verify-before-cutover item (do not assume):** the live `epistemic-graph` container
declares `containerPort: 9130 name: kvcache` alongside `9100`, and
`graph-os-env`'s `EPISTEMIC_GRAPH_KVCACHE_URL=http://<ENGINE_NODE_IP>:9130` points at it — but
the container's actual CLI args (`kubectl get deploy epistemic-graph -o yaml`) show no
corresponding `--kvcache-addr`-style flag today. Confirm whether anything is really
listening on 9130 on the CURRENT engine before cutover (e.g. `kubectl exec` a connect
test against it, or check the engine's own startup log for a "kvcache" listener line —
both read-only). If something is, add the equivalent flag to the sidecar's `args` below;
if nothing is, the port declaration is vestigial and needs no equivalent.

## Pre-flight (mandatory — do not skip)

Mirrors the two techniques that already prevented outages in this exact system
(`reports/HANDOFF-2026-07-22.md` §"pre-flight techniques"):

1. **Copy the redb store** (`<ENGINE_HOST_HOME>/epistemic-graph/graph_snapshots`, hostPath on
   <ENGINE_NODE>) to a scratch path. Run the new pod spec's engine sidecar command **standalone**
   (a throwaway pod, `--persist-dir` pointed at the COPY, same TLS/signer/auth env)
   against the copy first. Confirm: catalog loads, graph count matches the live store,
   `EPISTEMIC_GRAPH_REQUIRE_OIDC=false` + the shared HMAC secret authenticate a test
   request.
2. **Import-test inside the running pod without restarting it** — not applicable here
   the same way (this is a topology change, not a code drop), but the equivalent is:
   deploy the new pod spec to a **scratch Deployment name** (e.g. `graph-os-selfhost-rehearsal`)
   pointed at the COPY from step 1, verify it reaches `Ready`, THEN do the real cutover
   against production. This costs one extra pod but converts an irreversible-feeling
   cutover into a rehearsed one.
3. Confirm the kvcache item above.
4. Confirm nothing besides `graph-os` currently dials the `epistemic-graph` Service
   directly (§"redb single-writer-lock handoff" already established `graph-os-host` and
   `agent-utilities-messaging` are both scaled to `0/0` fleet-wide — `kubectl get deploy
   --all-namespaces` — i.e. graph-os is already the sole live consumer; messaging/
   host-daemon are bundled INTO the unified graph-os image per
   `reports/unified-binary-program.md`'s status ledger, not separate processes anymore).
   Re-verify this is still true immediately before cutover, since it is the basis for
   binding the sidecar's TCP listener to loopback-only in the spec below.

## The pod spec

Both containers reference the **same** unified image
(`<REGISTRY>/graph-os-unified:<tag>` — the validated `latest`/`w-c-validation`, not the
`:langfuse` follow-up validation tag from `graphos-unified-kaniko-job.yaml` unless that
work has separately been promoted). Deltas from today's live `graph-os` Deployment are
called out inline; everything not mentioned (au-config/au-src/eg-wheel mounts,
`graph-os-env`/`graph-os-secrets`, probes, resources, the existing `graph-os`
Service/Ingress) is **unchanged**.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-os
  namespace: platform
  labels: {app: graph-os, tier: kg-mcp, topology: unified-in-process}
spec:
  replicas: 1
  strategy:
    type: Recreate   # HARD requirement, not style: a second engine can never hold the
                      # redb lock while the first is still shutting down — a rolling
                      # update (maxSurge>=1) would start a NEW pod's sidecar against the
                      # SAME --persist-dir while the OLD pod's sidecar still holds the
                      # lock, guaranteeing the new one fails to start.
  selector:
    matchLabels: {app: graph-os}
  template:
    metadata:
      labels: {app: graph-os, tier: kg-mcp, topology: unified-in-process}
    spec:
      # No hostNetwork. Both containers share ONE pod network namespace by
      # construction — the reason today's SEPARATE Deployments need hostNetwork
      # (reaching a sibling POD's loopback) does not apply to sibling CONTAINERS in the
      # same pod. Only the redb hostPath still forces node pinning.
      nodeSelector: {kubernetes.io/hostname: <ENGINE_NODE>}
      terminationGracePeriodSeconds: 60   # covers the engine's clean shutdown + graph-os preStop

      # ── NATIVE SIDECAR (k8s >=1.29; this cluster is 1.35) ─────────────────────────
      # kubelet blocks the `containers:` below from starting until this startupProbe
      # passes — no polling wait-init script, no sleep, no crash-loop-and-retry on
      # graph-os. Same mechanism bundled-core-pod.yaml's Variant A already uses and
      # already documents as the reason to prefer this over "a manual wait-for-socket
      # loop" — reused here verbatim, just on the ONE unified image instead of two.
      initContainers:
      - name: engine
        restartPolicy: Always
        image: <REGISTRY>/graph-os-unified:<tag>   # SAME image as graph-os below
        imagePullPolicy: Always
        command: ["sh", "-c"]
        args:
        - ulimit -Sn 524288; exec /usr/local/bin/epistemic-graph-server
          --socket-path /run/epistemic-graph/epistemic-graph.sock
          --tcp-addr 127.0.0.1:9100
          --tcp-tls-cert /etc/eg-tls/tls.crt --tcp-tls-key /etc/eg-tls/tls.key
          --persist-dir /data/graph_snapshots
          # loopback-only TCP (was 0.0.0.0): nothing outside this pod needs to reach
          # the engine directly anymore (see pre-flight #4) — the TCP listener exists
          # ONLY so this container's own probes below can use a cheap tcpSocket check;
          # the real traffic from graph-os uses the unix socket. Kept on TLS anyway
          # (harmless, and the cert material is already provisioned) rather than
          # dropping it for loopback, so nothing about the engine's own security
          # posture regresses relative to today.
        envFrom:
        - secretRef: {name: epistemic-graph-secrets}   # UNCHANGED — same 6 keys
        env:
        - {name: EPISTEMIC_GRAPH_REQUIRE_OIDC, value: "false"}   # matches today
        - {name: RUST_LOG, value: info}
        ports:
        - {containerPort: 9100, name: engine, protocol: TCP}
        startupProbe:
          tcpSocket: {port: 9100}
          periodSeconds: 3
          failureThreshold: 40      # ~120s ceiling, matches today's engine Deployment
          timeoutSeconds: 1
        readinessProbe:
          tcpSocket: {port: 9100}
          initialDelaySeconds: 5
          periodSeconds: 15
          failureThreshold: 3
          timeoutSeconds: 1
        livenessProbe:
          tcpSocket: {port: 9100}
          initialDelaySeconds: 5
          periodSeconds: 20
          failureThreshold: 3
          timeoutSeconds: 1
        resources:
          requests: {cpu: "1", memory: 4Gi}
          limits: {cpu: "6", memory: 24Gi}     # unchanged from today's engine Deployment
        volumeMounts:
        - {mountPath: /data/graph_snapshots, name: redb}
        - {mountPath: /run/epistemic-graph, name: engine-socket}
        - {mountPath: /etc/eg-tls, name: eg-tls, readOnly: true}

      containers:
      - name: graph-os
        image: <REGISTRY>/graph-os-unified:<tag>   # SAME image as the sidecar above
        imagePullPolicy: Always
        command: ["python3", "-m", "agent_utilities.mcp.kg_server"]   # unchanged
        env:
        # THE line that makes this connect, not autostart (see identity-trap section
        # above) — overrides whatever the durable config.json otherwise resolves.
        - {name: GRAPH_SERVICE_ENDPOINTS, value: "unix:///run/epistemic-graph/epistemic-graph.sock"}
        - {name: EUNOMIA_TYPE, value: none}
        - {name: KG_DAEMON_ROLE, value: client}
        - {name: KG_LOOP, value: "0"}  # client serves; an explicit host owns loops
        - {name: KG_FUSEKI_ENDPOINT, value: "http://fuseki.apps.svc.cluster.local:80"}
        - {name: PYTHONPATH, value: /au}
        envFrom:
        - configMapRef: {name: graph-os-env}   # ENGINE_CA_BUNDLE/ENGINE_TLS_SERVER_NAME
                                                 # become IRRELEVANT once GRAPH_SERVICE_ENDPOINTS
                                                 # above is unix:// (engine_transport.py only
                                                 # consults them for a tls:// endpoint) — left
                                                 # in place rather than pruned, so falling back
                                                 # to the separate-engine shape (rollback) needs
                                                 # no ConfigMap edit, only the env override above removed.
        - secretRef: {name: graph-os-secrets}
        ports:
        - {containerPort: 8000, name: http, protocol: TCP}
        lifecycle:
          preStop:
            exec: {command: ["/bin/sh", "-c", "sleep 15"]}
        readinessProbe:      # unchanged from today's live graph-os Deployment
          httpGet: {path: /health/ready, port: http, httpHeaders: [{name: Host, value: graph-os.arpa}]}
          initialDelaySeconds: 20
          periodSeconds: 10
          failureThreshold: 12
          timeoutSeconds: 5
        livenessProbe:
          httpGet: {path: /health, port: http, httpHeaders: [{name: Host, value: graph-os.arpa}]}
          initialDelaySeconds: 10
          periodSeconds: 20
          timeoutSeconds: 5
        startupProbe:
          httpGet: {path: /health, port: http, httpHeaders: [{name: Host, value: graph-os.arpa}]}
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 60
        resources:
          requests: {cpu: 500m, memory: 1Gi}
          limits: {cpu: "4", memory: 6Gi}
        volumeMounts:
        # UNCHANGED from today: au-config, local-data, homelab-ca, eg-wheel, au-src.
        - {mountPath: /root/.config/agent-utilities, name: au-config, readOnly: true}
        - {mountPath: /root/.local/share/agent-utilities, name: local-data}
        - {mountPath: /etc/ssl/homelab, name: homelab-ca, readOnly: true}
        - {mountPath: /eg-wheel, name: eg-wheel, readOnly: true}
        - {mountPath: /au, name: au-src, readOnly: true}
        # NEW: the socket the sidecar writes to.
        - {mountPath: /run/epistemic-graph, name: engine-socket}

      volumes:
      # NEW — was a hostPath (`/run/epistemic-graph` on <ENGINE_NODE>) shared via co-scheduling;
      # now an in-pod emptyDir, the SAME simplification bundled-core-pod.yaml already
      # made for the identical reason (makes the coupling explicit: these two
      # containers share a socket because they are IN one pod, not because of an
      # incidental host mount).
      - {name: engine-socket, emptyDir: {}}
      # UNCHANGED — must stay hostPath, never NFS (redb needs real POSIX advisory
      # locks; see MEMORY.md k8s-swarm-cutover "DBs never NFS").
      - name: redb
        hostPath: {path: <ENGINE_HOST_HOME>/epistemic-graph/graph_snapshots, type: Directory}
      # RECOMMENDED small deviation from today: mount the k8s Secret directly instead
      # of the hostPath copy (`<ENGINE_HOST_HOME>/eg-tls`) the live Deployment currently
      # uses — one less manually-synced copy of the same material. Revert to the
      # hostPath (unchanged from today) if the operator prefers zero deviation here.
      - name: eg-tls
        secret: {secretName: epistemic-graph-tls}
      # UNCHANGED from today's graph-os Deployment:
      - {name: au-config, hostPath: {path: <ENGINE_HOST_HOME>/.config/agent-utilities-next, type: Directory}}
      - {name: local-data, emptyDir: {}}
      - {name: homelab-ca, configMap: {name: homelab-ca-bundle, defaultMode: 420}}
      - {name: eg-wheel, nfs: {server: <WORKSPACE_NFS_SERVER>, path: <WORKSPACE_NFS_EXPORT>/eg-wheel, readOnly: true}}
      - {name: au-src, nfs: {server: <WORKSPACE_NFS_SERVER>, path: <WORKSPACE_NFS_EXPORT>/agent-packages/agent-utilities, readOnly: true}}
```

**Not shown/unchanged:** the `graph-os` Service (port 80 → container `http`) and
Ingress (`graph-os.arpa`, cookie affinity, TLS) need **no edits at all** — they already
target only the `graph-os` container's port 8000, which is unaffected by this cutover.
Delete (or scale to 0) the `epistemic-graph` Deployment and Service once the new pod is
verified — see the checklist.

## Rollback

The redb single-writer lock makes rollback a mirror of cutover, not a special case:

1. Scale the new `graph-os` (pod with the sidecar) to 0 — releases the lock.
2. Scale the OLD `epistemic-graph` Deployment back to 1 (this is why cutover step 3
   scales to 0 rather than deleting — keep it, its ConfigMap/Secret refs, and its
   Service around for the whole verification window before deleting anything).
3. `kubectl rollout undo deployment/graph-os -n platform` (or reapply the pre-cutover
   manifest) to restore the old pod spec (`GRAPH_SERVICE_ENDPOINTS=tls://<ENGINE_NODE_IP>:9100`
   via the durable config, no sidecar).
4. Confirm `graph-os` reaches `Ready` against the restored separate engine before
   declaring rollback complete.

Total rollback window is the same ballpark as cutover step 4 (one redb reload) — there
is no faster path given the single-writer lock; this is inherent to redb, not a gap in
this design.

## Validation checklist

- [ ] Pre-flight steps 1-4 all passed (copy-based rehearsal, kvcache confirmed,
      sole-consumer confirmed).
- [ ] New pod: `kubectl get pod -n platform -l app=graph-os` shows the sidecar
      `engine` container `Ready` before the `graph-os` container even starts
      (confirm via `kubectl describe pod` event ordering — this is the native-sidecar
      contract, not just a hopeful timing).
- [ ] `graph-os`'s `/health` (and `/health/ready`) report the engine reachable, mode
      consistent with `unix://` (not `tls://`/`remote-network`).
- [ ] Graph count in the catalog matches the pre-cutover count noted in pre-flight step 1
      (85-87 graphs per the most recent recorded count — confirm the CURRENT count
      immediately before cutover, don't reuse a stale figure).
- [ ] A representative KG query (e.g. `graph_query`/`graph_search` against `__commons__`
      or a known `code_*` graph) returns the same answer it did before cutover.
- [ ] The kvcache verify-before-cutover item resolved one way or the other, and the
      resolution reflected in the sidecar's final `args`.
- [ ] `platform/epistemic-graph` scaled to 0, not deleted, until this checklist is fully
      green AND a rollback rehearsal (§Rollback) has been dry-run at least once.
- [ ] Only after a soak period: delete the old `epistemic-graph` Deployment + Service,
      and the now-unused `ENGINE_CA_BUNDLE`/`ENGINE_TLS_SERVER_NAME` keys from
      `graph-os-env` (left in place until then, per the inline comment above).

## Pointers

| Concern | Source |
|---|---|
| The two-shapes edict + program status | `reports/unified-binary-program.md`; au `AGENTS.md` → "Engine transport"; eg `AGENTS.md` → PyO3/in-process section |
| Why this is possible now (the unified image) | `docker/graphos-unified.Dockerfile` |
| The multi-container-pod precedent this design reuses | `services/epistemic-graph/k8s/bundled-core-pod.yaml` (Variant A) |
| The identity/RBAC chain this design must not disturb | `reports/HANDOFF-2026-07-22.md` §1, §2e, §7 |
| The opposite (hyperscaling) shape — do not conflate | `graphos-horizontal-scaling.md` |
| Engine autostart precedence (why an explicit endpoint avoids it) | `agent_utilities/knowledge_graph/core/engine_resolver.py` |
| The redb single-writer lock | `epistemic-graph` `src/main.rs` (`persist_lock::acquire`) |
