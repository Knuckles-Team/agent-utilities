# Production cell runbook

Status vocabulary: the checked-in cell is a **renderable production candidate**;
it is not deployed, live, or one-million-user certification. A separately retained
measured topology input and a signed release are required before any apply.

The production reference is a global GraphOS control plane plus one or more cell
data planes. A cell is the failure and recovery boundary; a tenant placement maps to
one cell and one of 20 MultiRaft groups. Every group has three replicas spread across
zones. A replacement release is prepared in isolation and activated by one fenced,
atomic route cutover; two application/protocol versions never serve concurrently.

## Platform preflight

Before rendering a release, verify all of these conditions:

1. Capture a measured ResourcePool input with a timestamp and non-secret evidence
   reference. The renderer is the sole source of replica/resource bounds; the
   checked-in example and older W3/reference manifests are not inventory.
2. Three zones are schedulable and the engine anti-affinity/PVC requirements fit.
3. Istio injection and native sidecars are available. Both GraphOS namespaces must
   show `PeerAuthentication` in `STRICT` mode before application traffic is admitted.
   Label the mesh control-plane namespace
   `graphos.network/role=service-mesh-control` so default-deny egress still permits
   certificate issuance and rotation.
4. `graphos-retained-rwo` retains 512 Gi engine claims.
   `graphos-cross-cell-object-rwx` is backed by versioned, cross-cell replicated
   object storage, not node-local disk.
5. Prometheus Operator, Prometheus Adapter and the metrics pipeline are healthy;
   the adapter evidence binds the exact `graphos_workload` selectors and
   `agent_utilities_gateway_in_flight_requests`,
   `agent_utilities_dispatch_queue_depth`, and
   `agent_utilities_kg_ingest_queue_depth` names. Mining stays fixed until a
   bounded signal is published.
   The mesh-injected observability namespace is named `graphos-observability` and
   carries `graphos.network/role=observability`.
6. The secret synchronizer has created `graphos-runtime-secrets` and
   `graphos-trust-bundle` in both namespaces, plus `graphos-engine-tls` in the
   data-plane namespace. No secret value belongs in Git.
7. OIDC, the event backbone, PostgreSQL state/usage storage, OTLP, Langfuse, ingress,
   observability and controlled egress are reachable through the permitted namespaces.

The runtime Secret supplies the engine authentication secret, the
`PERSISTENCE_IDENTITY_HMAC_KEY` value, workload principals and
tenants, audience, policy version, OIDC/JWKS settings, database/broker locations,
Langfuse settings and `OTEL_EXPORTER_OTLP_ENDPOINT`. The analytics worker additionally
requires `GRAPH_OS_ANALYTICS_PRINCIPAL` and `GRAPH_OS_ANALYTICS_TENANT`. All values are
runtime references; documentation, manifests, logs and evidence must not retain their
contents. ConfigMaps contain only the neutral
`PERSISTENCE_IDENTITY_HMAC_KEY_REF=env://PERSISTENCE_IDENTITY_HMAC_KEY` reference.

The trust Secret supplies a PEM bundle containing every required intermediate and root
CA. `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, and runtime TLS profiles are the supported
contract. For a uv-managed Langfuse MCP process, also use `UV_NATIVE_TLS=true`; never
replace the PEM bundle with a binary certificate or disable verification.

The engine TLS Secret supplies runtime-mounted `tls.crt` and `tls.key`; the certificate
must cover the coordinator service name. The committed contract uses only
`tls://` native-engine endpoints. Engine readiness performs a real TLS handshake with
the mounted trust bundle and expected server name, so a missing intermediate, wrong
identity, expired certificate or plaintext listener never becomes ready. Signed,
tenant-bound request envelopes remain mandatory inside TLS. Certificate/key material
and custom trust profiles stay external to the repository and rotate through the
secret synchronizer.

## Render and deploy

The source Kustomization contains non-resolving image sentinels. Render only from a
signed exact release:

```sh
check-graphos-compatibility --manifest RELEASE_MANIFEST
python scripts/release/render_production_cell.py \
  --manifest RELEASE_MANIFEST \
  --topology-input /secure/evidence/graphos/production-input.json \
  --output RENDERED_DIRECTORY
python scripts/deployment/check_production_assets.py \
  --directory RENDERED_DIRECTORY --rendered
# after the write fence, signed snapshot, and one-time migration gate:
kubectl apply -k RENDERED_DIRECTORY
```

The topology input must also bind exact current and rollback image digests,
engine Service/TLS/discovery identities, OIDC discovery, retained ConfigMap,
Secret, external session-store and action-audit authorities, and separate
rollout/rollback evidence references. The renderer emits an immutable
`graphos-topology-contract` alongside `graphos-release-pins`; inspect both
before apply. It refuses to render on missing capacity, wrong identity or port,
unverified TLS/discovery/OIDC, digest drift, non-retained authority, or an
unbounded metric. Run the rollback renderer with `--rollback` and review it as a
separate artifact; it never changes a live cluster by itself.

The compatibility gate verifies exact Epistemic Operations schemas, Epistemic Graph,
Agent Utilities, connector catalog, the ten consolidated skills, ontology lock and
index migration catalog. It invokes external signature verifiers and refuses tags,
`latest`, sentinel digests, unknown keys or unsigned artifacts.

Assemble and verify the release in this order: protocol schemas, Epistemic Graph,
Agent Utilities, connector bundles, skills, ontology lock, then the migration catalog.
Fence admissions and writes, capture the signed pre-cutover snapshot, execute the
one-time persisted-state migration, start only the exact new release, verify its
watermarks, then atomically move traffic. The prior release remains fenced and cannot
read or write migrated state.

## Authority and worker checks

- `epistemic-graph-raft` must have exactly three ready members and three distinct
  nodes/zones. Scaling it with an HPA is forbidden because quorum membership is an
  operator action.
- `epistemic-graph-coordinator` must select all three ready members. Replicated job,
  reasoning, placement, and transaction authorities fence requests and redirect them
  to the current group leader; losing any one contact must not remove authority.
- Dispatch, ingestion and analytics workers scale from queue depth/consumer lag/job
  backlog. CPU is a secondary front-door signal only.
- The analytics Deployment must run `graph-os-analytics-worker`, while the engine has
  `EG_ANALYTICS_WORKERS=0`. The worker uses a verified v2 context with `kg:write` and
  `analytics:worker`, renewable leases, fencing, cooperative cancellation and governed
  KnowledgeBatch results.
- Projection, indexing, truth maintenance and reasoning remain inside the engine
  authority process. Their cursor is acknowledged only after authoritative state and
  the projection/index transition are durable.

## Observability and SLO response

Production startup requires PostgreSQL usage accounting, self-ingest WAL-before-send,
OTLP export and runtime-injected OTLP location. The committed ConfigMaps set
`USAGE_DB_BACKEND=postgres`, `USAGE_TRACKING_ENABLED=true`,
`USAGE_CONTENT_RETENTION=metadata`, `LANGFUSE_CAPTURE_CONTENT=false`,
`ENABLE_OTEL=true` and omit the exporter location. Traces retain aggregate counts,
timings, status and opaque HMAC references only; prompts, completions, graph content,
endpoints, host names, raw identities and filesystem locations are prohibited.

The recording rules and dashboard cover front/engine p99, availability error-budget
burn, dispatch backlog, ingestion lag, analytics backlog, quorum risk, WAL loss,
checkpoint age and restore validation. On an alert:

1. stop rollouts and resharding;
2. preserve raw aggregate telemetry and the exact release/configuration digests;
3. protect quorum and interactive serving before background work;
4. scale only the worker tier associated with the measured queue;
5. if the RPO or coordinator integrity alert fires, execute the disaster-recovery
   procedure rather than deleting or recreating retained claims.

Langfuse reachability is not equivalent to trace delivery. Validate both the MCP trace
list operation and a new opaque local-model trace correlated by a non-identifying trace
reference. A successful empty list proves only authentication/TLS reachability.
The `identity-tls-policy-trace` certification hook must additionally prove an
unverified request is rejected, the peer workload identity is mTLS-authenticated, a
stale policy is rejected, the new opaque trace is visible, and no trace content was
captured.

## Hot-swap and rollback

Hot-swap is blue/green at the release boundary, not a mixed-version rolling update.
Prepare the exact replacement while isolated; fence the active release, migrate its
signed snapshot once, start the replacement, and atomically switch routing only after
all acceptance probes pass. StatefulSet pods use `OnDelete` so Kubernetes cannot mix
engine versions implicitly.

Rollback restores the signed pre-cutover snapshot under the prior exact manifest,
ontology lock and connector catalog. It never asks prior code to read new state and
never uses dual readers or compatibility shims. The certification campaign must prove
both cutover and snapshot restore.
