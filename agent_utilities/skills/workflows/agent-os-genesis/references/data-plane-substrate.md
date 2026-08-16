# Optional data-plane substrate

A lakehouse/streaming/triple-store data plane — object storage, a table catalog,
a SQL query engine, batch compute, event streaming, a triple store — is
**optional infrastructure a Kubernetes deployment may add**, never a required
component. Default every service below to `skip`. A laptop, single-host, or
minimal-profile run must resolve to zero selected data-plane services and must
never be blocked by their absence.

Never copy host names, addresses, storage paths, registries, credentials,
cluster names, or DNS suffixes from this file into an operator's plan (same
top-level rule as `SKILL.md`). Every placeholder below (`<namespace>`,
`<node>`, `<name>.<dns-suffix>`, `<ingress-vip>`, `<hostpath-root>`, `<realm>`,
`<idp-host>`, …) is a value **discovery must resolve for the target cluster**,
not a default to assume. The gotchas recorded here were each learned once,
against one real deployment of this catalog of services; they are reported
generically because the specific cluster they came from is exactly the kind of
detail this skill must never hardcode.

## Why these are a distinct capability, not `connectors`/`components`

The Helm chart's `connectors`/`components` value lists are `agent-utilities`
**application** pods rendered by the chart. These services are independently
versioned **platform** services with their own release cadence, storage, and
failure domains — each ships as its own checked-in manifest
(`services/<name>/k8s/manifests.yaml` in a config-as-code workspace) applied
directly with `kubectl`, the same way the chart itself is rendered and
applied, but as a sibling artifact, never a chart template. Genesis owns
provisioning/attaching them (Day-0); wiring the resolved endpoint/auth
references into `agent-utilities`/`epistemic-graph` runtime config is
`agent-utilities-deployment`'s job, per the responsibility boundary in
`SKILL.md`.

## Discovery — check before assuming missing

Before planning a `deploy` action for any service in the catalog, classify its
current state (Phase 1's `ready | degraded | missing | incompatible`):

1. Resolve the in-cluster DNS name from inside the cluster
   (`<service>.<namespace>.svc.cluster.local`); a resolvable name with a
   listening port is at least `degraded`, not `missing`.
2. `kubectl -n <namespace> get deploy,sts,svc <name>` — check replica
   readiness, restart counts, and the image tag actually running (not just the
   tag in the checked-in manifest — they can drift).
3. Probe the LAN ingress host for the service and confirm a real application
   response, not the ingress controller's default-backend 404 — a
   default-backend response means DNS/ingress exists but the service behind it
   does not.
4. Diff the live object set against `services/<name>/k8s/manifests.yaml`
   (`kubectl diff -f`) to detect drift before treating "already deployed" as
   "matches the checked-in contract."

Never mutate merely to discover (Safety invariant 2).

## Deploy — idempotent apply of the checked-in manifest

Each service already has a config-as-code manifest in a workspace built with
this pattern. Deploying means applying it, not hand-authoring resources:

```bash
kubectl apply --dry-run=server -n <namespace> -f services/<name>/k8s/manifests.yaml
kubectl apply -n <namespace> -f services/<name>/k8s/manifests.yaml
kubectl rollout status -n <namespace> deploy/<name> --timeout=<bounded>
```

Re-applying an unchanged manifest must be a no-op (idempotency invariant). Do
not `kubectl edit`/`kubectl patch` a running data-plane workload directly —
the manifest is the source of truth; change it and re-apply.

### Cluster conventions a data-plane manifest must discover and follow

Discover these from the target cluster; do not assume any operator's values:

- **Namespace**: whichever namespace the plan selected for this substrate
  (commonly a shared `apps`-style namespace, distinct from the
  `agent-utilities` release namespace).
- **Placement**: `nodeSelector: kubernetes.io/hostname: <node>` — pin
  explicitly rather than leaving scheduling to chance whenever the cluster
  has heterogeneous nodes (different CPU generations, a cordoned
  control-plane node, or a tainted accelerator/edge node — see the Trino
  gotcha below for why heterogeneous CPU baselines matter).
- **Storage**: if no StorageClass exists, persistent workloads fall back to a
  raw `hostPath` under a fixed, discovered root (e.g. `<hostpath-root>/<service>`)
  instead of a PVC/StorageClass claim. Confirm which mode the cluster actually
  supports before writing the manifest.
- Every pod spec **must** set `enableServiceLinks: false` on any cluster
  whose convention requires it. Omitting it, on a cluster with that
  convention, injects Kubernetes service-link env vars that can make
  `AgentConfig()` unbuildable fleet-wide — check the cluster's own
  standing convention for every workload, not only the data plane.
- **Ingress**: an ingress controller fronting a stable VIP (`<ingress-vip>`),
  with a per-service host of the form `<name>.<dns-suffix>` — discover the
  controller class and DNS suffix from the cluster rather than assuming one.
- **Secrets**: route through the cluster's declared secret-store integration
  (e.g. an External-Secrets `ClusterSecretStore`). Never `kubectl patch` a
  managed Secret directly on a cluster with that convention — an
  external-secrets-style controller reverts the patch on its next reconcile.
- **Node roles**: discover which nodes are cordoned/control-plane (do not
  schedule application workloads there), which are tainted for a different
  architecture or accelerator (do not schedule a mismatched image there
  without an explicit toleration and a compatible image), and which nodes'
  CPU generation constrains which image tags can run (see Trino below).

## Service catalog

The catalog below names the *kind* of service and its role; discover the
concrete endpoints, image references, and manifest paths for the target
cluster rather than assuming these.

| Service | Purpose | In-cluster endpoint pattern | LAN ingress pattern | Manifest | Default action |
|---|---|---|---|---|---|
| SeaweedFS | S3-compatible object store | `<name>.<namespace>.svc.cluster.local` — S3/filer, master, volume, and filer-UI ports | `<name>.<dns-suffix>` | `services/<seaweedfs-service>` | `skip` |
| Lakekeeper (or equivalent) | Iceberg REST catalog | `<name>.<namespace>.svc.cluster.local:<port>` | `<name>.<dns-suffix>` | `services/<catalog-service>` | `skip` |
| Catalog's backing Postgres | The catalog service's own store only | `<name>.<namespace>.svc.cluster.local:5432` | none (ClusterIP) | `services/<catalog-db-service>` | `skip` |
| Trino (or equivalent) | SQL query engine over the catalog | `<name>.<namespace>.svc.cluster.local:<port>` | `<name>.<dns-suffix>` | `services/<query-engine-service>` | `skip` |
| Spark runner (or equivalent) | Batch compute | `<name>.<namespace>.svc.cluster.local:<port>` | none — reach via `kubectl exec` | `services/<compute-service>` | `skip` |
| Kafka *(often pre-existing)* | Event streaming | `<name>.<namespace>.svc.cluster.local:9092` | none | — | `use-existing` where already wired |
| Apache Jena/Fuseki *(often pre-existing)* | Triple store | `<name>.<namespace>.svc.cluster.local:80` | `<name>.<dns-suffix>` | — | `use-existing` where already wired |

**Kafka and Fuseki already have a first-class connect surface in
`agent-utilities` core** — `KAFKA_BOOTSTRAP_SERVERS` and
`KG_FUSEKI_ENDPOINT`/`GRAPH_FUSEKI_DATASET`/`GRAPH_FUSEKI_USER`/
`GRAPH_FUSEKI_PASSWORD_REF` are existing `AgentConfig` fields
(`agent_utilities/core/config.py`). For those two, Genesis's job is discovery
+ recording the resolved reference in the handoff; `agent-utilities-deployment`
sets the field. **The lakehouse services (object store/catalog/query
engine/compute) have no existing config surface in `agent-utilities` core as
of this writing** — record the resolved endpoints/auth in the handoff's
`data_plane_ref` regardless, and treat wiring an actual config field for them
as `agent-utilities-deployment`'s open work, not something Genesis can
complete on its own.

**MinIO is not an approved choice for a new object-store deploy on a stack
that has adopted this catalog**, when the operator's own MinIO edition has
been discontinued for the deployment's support window (verify current
support/release status at plan time — this is a fact to re-check, not to copy
forward as permanently true). Reaching that conclusion during one prior
deployment is why SeaweedFS is listed as the default object-store choice
above; re-verify rather than trusting either choice indefinitely.

## Connect — endpoints, auth, and per-service detail

### Object store (e.g. SeaweedFS)
Some S3-compatible object stores have no S3 STS support. If the selected
store doesn't support STS, every Iceberg warehouse whose storage backend
points at it **must** explicitly disable STS in its storage profile —
leaving the (often-default) STS-enabled setting on fails silently at first
data access, not at warehouse creation, so the failure surfaces far from its
cause. Confirm the store's STS support before wiring a warehouse, not after.

### Catalog service (e.g. Lakekeeper)
- **The catalog URI is not necessarily the bare host.** Some REST catalog
  implementations root their API under a path prefix (e.g. `/catalog/v1`
  rather than `/v1`). Point any Iceberg-speaking engine at the documented
  root, and treat a `404` on an unauthenticated config probe as "wrong path,"
  not "service down" — a correctly-configured catalog returns `401` on that
  same probe, which is the fail-closed, *expected* response.
- **Auth**: expect an OIDC identity provider (issuer + realm) and a
  confidential machine client using client-credentials for service callers.
  Confirm both the issuer URL and the client's actual granted scopes before
  wiring an engine.
- **Pin the OAuth2 client's scope explicitly; do not trust the library
  default.** A generic Iceberg REST/OAuth2 client commonly defaults to a
  generic scope name that a scoped-down service client may not hold — that
  mismatch produces a 403 that looks like a broken client, not a scope
  mismatch. Read the service account's actual granted scope and pin it.
- **If the identity provider is reachable both via an in-cluster plain-HTTP
  Service and an HTTPS ingress, route OAuth2 clients through the HTTPS
  route.** A token minted via the plain-HTTP path can carry an `http://`
  issuer claim that a catalog configured to trust only `https://` will
  correctly 401 — that looks like a broken credential, not an
  issuer-routing mismatch.
- **Translate an upstream Compose `command: [...]` override into k8s `args:`,
  never `command:`.** Several services ship compose examples that override
  the container's default command (e.g. to run a one-shot migration). A
  literal k8s `command:` replaces the container's ENTRYPOINT and breaks argv
  dispatch, so the intended override silently never runs and the container
  just serves against whatever state it started with (e.g. an unmigrated
  schema).

### Query engine (e.g. Trino)
A query-engine image can assume a CPU microarchitecture baseline (e.g.
x86-64-v3/AVX2) that an older node in the cluster does not meet. On such a
node, images past a certain release began crashing instantly with
`Fatal glibc error`; the newest tag confirmed to still run on that hardware,
as last verified, was several releases behind the then-current one. Probe the
target node's CPU flags (`/proc/cpuinfo`) before pinning an image tag, keep a
record of the newest confirmed-working tag per node generation, and re-probe
before bumping rather than assuming a later release fixed it.

### Compute runner (e.g. Spark)
No LAN ingress is expected for a batch-compute runner; reach it via
`kubectl exec` into the pod. Do not expose it for direct external job
submission unless the plan explicitly adds that surface.

### Internal CA trust (cross-cutting — every JVM-based data-plane client)
A cluster's internal CA bundle Secret can be a multi-certificate bundle with
the operator's actual root CA appended anywhere in the file, not necessarily
first. A naive single-certificate import (e.g. one `keytool -importcert -file
<bundle>.pem` call) silently imports only the *first* certificate in the
file. Split the bundle and import every certificate individually into any
JVM truststore (a query engine, a compute runner, any Java client) that must
trust the cluster's internal TLS names — then verify the trust chain by
listing the resulting truststore's entry count, not by assuming the import
succeeded.

### Object-store filer/gateway startup timing
Some object-store server processes (e.g. a combined filer+S3 gateway) take
significantly longer to bind their service port than a typical container's
default probe delay assumes — on the order of 90–120 seconds, not the more
common ~30-second default. A liveness-probe delay shorter than the process's
real cold-start time crash-loops the pod permanently, because each restart
resets the clock before the port ever binds. Measure the process's actual
cold bind time for the selected image and set both readiness and liveness
`initialDelaySeconds` comfortably past it.

## Exit gates (data-plane substrate)

In addition to the standard Kubernetes production gates
(`kubernetes-and-helm.md`), before declaring a data-plane service `verified`:

1. An unauthenticated probe of the catalog's config endpoint (at its
   documented root path) returns `401`, and the same probe with a
   client-credentials token scoped correctly for the calling service
   returns `200`.
2. A warehouse created against an STS-incapable object-store backend has STS
   explicitly disabled in its storage profile, and a first read **and**
   write against it succeeds — not merely warehouse creation.
3. The query engine reaches ready with no CPU-baseline crash-loop on the
   deployed image tag, on every node it is scheduled to.
4. The object-store filer/gateway pod passes readiness without
   restart-looping through its real cold-start window.
5. Any migration Job's logs/exit code show the migration step actually ran
   (not just "container started").
6. Every JVM-based client that must reach an internal TLS name trusts the
   full internal CA bundle, not just one certificate from it.

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for the symptom-first lookup
table covering these same failure modes.
