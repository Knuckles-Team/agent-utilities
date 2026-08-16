# Troubleshooting

Start from the failed exit gate and preserve the trace/evidence bundle.

| Symptom | Check first | Safe response |
|---|---|---|
| Render fails | values schema, unresolved references, API versions | correct the plan; do not bypass validation |
| Cluster apply is forbidden | authority mode and exact RBAC denial | emit an administrator requirement or remove the cluster-scoped capability |
| Workload is pending | requests, quotas, taints, architecture, PVC binding | adjust the declared placement/resources or provide capacity |
| Readiness fails | application health, engine reachability, identity/secrets | inspect correlated logs/traces; do not restart-loop a dependency failure |
| Engine will not start | data format, lock owner, disk space/IO, encryption reference | stop extra writers, restore compatible state, or run the supported migration |
| Connector returns 401/403 | issuer/audience, service token, target policy, clock | repair identity routing or authorization; never disable auth as a test |
| UI/API authenticates fine but every KG-backed read is empty, missing tools, or 503, while the underlying data provably exists | which client/service-account minted the token, and whether its granted roles are graph-capability scopes or only UI-level roles | never substitute a UI-level admin role for a graph-capability scope; grant the correct scope to the correct identity (browser client vs. backend service account) — see `agent-utilities-deployment`'s *Identity & authorization* section — then re-authenticate, since a role grant never applies to an already-minted token |
| External route fails | DNS, certificate names, ingress/Gateway backend, network policy | test each hop and keep the prior route available |
| Background work starves users | foreground priority, IO/model/GPU limits, checkpoints | pause/checkpoint lower lanes and verify foreground latency |
| Second apply changes resources | nondeterministic render/defaults or manual drift | identify the owner; reconcile through the declared GitOps/runtime path |
| Local model answers without a tool | tool allow-list, exact server/tool binding, prompt, model capability | inspect the run trace; do not count it as delegated execution |
| Iceberg REST catalog probe returns `404` instead of an auth error | the catalog URI is missing the `/catalog` prefix — the surface is rooted at `/catalog/v1`, not `/v1` | point the client at `<catalog-host>/catalog`, then re-probe `/catalog/v1/config` and expect `401` |
| Catalog request 401s despite a token that looks valid | which Keycloak route issued it — the in-cluster plain-HTTP Service issues `iss=http://…`, but the catalog is configured for `https://` | mint the token through the HTTPS Keycloak ingress, not the in-cluster Service |
| Catalog request 403s despite a valid, correctly-issued token | the OAuth2 client's default `scope` — most Iceberg REST clients default to `scope=catalog`, which a scoped-down service client may not hold | pin the client's OAuth2 scope to the value the service account actually has (e.g. `lakekeeper`), not the library default |
| A lakehouse query fails only on first data access, not at warehouse/table creation | whether the object-store backend supports S3 STS | for a store with no STS (e.g. SeaweedFS), the warehouse's storage profile must disable STS explicitly — credential vending otherwise fails silently until the first real read/write |
| A JVM-based query engine crash-loops with a `Fatal glibc error` | the node's CPU microarchitecture vs. the image's build baseline | pin to the newest image tag confirmed to run on that node's CPU; re-probe newer tags before bumping rather than assuming a later release fixed it |
| A JVM client trusts one internal CA but not another sharing the same bundle file | whether the truststore import grabbed every certificate in a multi-cert bundle or only the first | split a multi-certificate CA bundle and import each certificate individually — a naive single-cert import silently drops every cert after the first |
| A migration/init container never appears to run its migration step | whether a Compose `command:` was translated to a k8s `command:` (replaces ENTRYPOINT) instead of `args:` | translate an upstream Compose `command:` override into k8s `args:`, never `command:`, unless the image truly has no ENTRYPOINT |
| An object-store/filer pod crash-loops forever immediately after deploy, never stabilizing | the probe's `initialDelaySeconds` against the process's real bind-time, not just "is it eventually up" | measure how long the process takes to bind its port cold and set both readiness and liveness `initialDelaySeconds` comfortably past that, not a generic default |
| A new object-store deploy is being planned around MinIO | whether MinIO community edition is still receiving images | MinIO CE is archived (no images since ~Oct 2025 as of this writing) — do not select it for a new deploy; use the operator-approved replacement |

For runtime-only Graph-OS incidents after a healthy deployment, use the
`graph-runtime-and-governance` or `epistemic-graph-troubleshooting` skill. Use Genesis
only when the fix changes the substrate contract.

Always retain: plan digest, source revision, rendered artifacts, runtime events,
redacted logs/traces, exact failing gate, remediation, and rollback result.
