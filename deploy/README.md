# GraphOS deployment assets

`k8s/production-cell/` is the production reference topology. It separates a
stateless global control plane from one cell data plane, keeps authoritative graph
state in a three-member MultiRaft StatefulSet, and assigns independent resource
partitions to dispatch, ingestion and analytics workers. The previous single-owner,
unmounted-volume manifest was removed.

Do not apply the template directory directly. Its image names intentionally point
to a non-resolving registry. A release is deployable only after the compatibility
gate verifies every component signature and the renderer substitutes exact OCI
digests:

```sh
check-graphos-compatibility \
  --manifest RELEASE_MANIFEST
python scripts/release/render_production_cell.py \
  --manifest RELEASE_MANIFEST \
  --output RENDERED_DIRECTORY
python scripts/deployment/check_production_assets.py \
  --directory RENDERED_DIRECTORY --rendered
kubectl apply -k RENDERED_DIRECTORY
```

The uppercase operands above are runtime/operator inputs, not committed local
paths. Generated release and certification evidence must remain outside the source
tree.

## Production cell contract

- `graphos-control`: three or more `graphos-front` replicas, OIDC-authenticated
  streamable HTTP, usage accounting backed by PostgreSQL, and OTLP/Langfuse export.
- `graphos-cell`: three engine members spread across nodes and zones. Every member
  mounts a retained 512 Gi PVC; all members host the configured 20 Raft placement
  groups. Correctness-critical projection, indexing, truth maintenance and reasoning
  stay colocated with graph authority.
- `epistemic-graph-coordinator`: a stable Service selecting StatefulSet pod index
  zero. Non-graph coordinator ledgers, including analytics jobs and admin mutation
  receipts, always use this one durable authority and are never load-balanced across
  independent stores. Pod loss is recovered by StatefulSet recreation and PVC
  reattachment; MultiRaft continues to replicate authoritative graph state.
- Worker Deployments: queue-driven dispatch, broker-lag-driven ingestion, and the
  authenticated `graph-os-analytics-worker`. The engine has
  `EG_ANALYTICS_WORKERS=0`, so remote analytics compute cannot silently fall back to
  a colocated pool.
- Recovery: minute-level online bundles on a cross-cell replicated RWX/object-CSI
  archive and daily offline restore validation. Bundle format v3 includes
  `admin-mutations.redb`, encrypted prepared plans, parent/child receipts, retained
  cross-shard decisions, and exact portable-file digests. The mounted hot window keeps two bundles; bucket versioning
  and lifecycle policy provide longer retention without unbounded PVC growth.

## Required platform services

The manifests fail closed unless the platform provides:

- Kubernetes with stable StatefulSet pod-index labels and native sidecars;
- three schedulable failure zones and a retained RWO class named
  `graphos-retained-rwo`;
- an RWX class named `graphos-cross-cell-object-rwx` whose versioned backing object
  store is replicated outside the cell;
- Istio CRDs/control plane with namespace injection; `PeerAuthentication` is `STRICT`
  and `AuthorizationPolicy` binds engine ports to workload service accounts;
- Prometheus Operator and a Prometheus Adapter configured with the generated external
  metric rules;
- an OIDC issuer, event backbone, PostgreSQL state/usage store, OTLP collector,
  Langfuse service, secret synchronizer, and external evidence signer/verifier;
- an ingress namespace labeled `graphos.network/role=ingress`, an observability
  namespace named `graphos-observability` and labeled
  `graphos.network/role=observability`, and event/state/egress namespaces carrying the labels
  referenced by the NetworkPolicies. The mesh control-plane namespace must carry
  `graphos.network/role=service-mesh-control` so injected sidecars can obtain and
  rotate workload certificates.

## Runtime Secret contract

The deployment contains no Kubernetes `Secret` objects. A secret controller must
materialize `graphos-runtime-secrets` in both namespaces and `graphos-trust-bundle`
in each namespace, plus `graphos-engine-tls` in `graphos-cell`. The engine TLS Secret
contains `tls.crt`/`tls.key` for the coordinator service identity; certificate
material is never committed. Runtime settings include OIDC/JWKS authority, engine HMAC secret,
database/broker locations, workload principals/tenants, policy version, Langfuse
credentials/base location, and `OTEL_EXPORTER_OTLP_ENDPOINT`.

TLS verification is mandatory. Trust is configured by the mounted PEM bundle through
`REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE`; no boolean downgrade control exists.
All committed native-engine client endpoints use `tls://`, and engine readiness
validates the trust chain and expected service name rather than probing only for an
open TCP port.
`OTEL_EXPORTER_OTLP_ENDPOINT` is deliberately absent from ConfigMaps and must be
runtime-injected. `PERSISTENCE_IDENTITY_HMAC_KEY_REF` points only to the neutral
runtime environment reference `env://PERSISTENCE_IDENTITY_HMAC_KEY`; the referenced
key is supplied by the Secret controller. The production observability guard also
requires `USAGE_DB_BACKEND=postgres`, `USAGE_TRACKING_ENABLED=true`,
`USAGE_CONTENT_RETENTION=metadata`, `LANGFUSE_CAPTURE_CONTENT=false`, and
`ENABLE_OTEL=true`.

See [the production runbook](../docs/operations/production-cell-runbook.md),
[disaster recovery](../docs/operations/disaster-recovery.md), and
[release certification](../docs/release/compatibility-and-certification.md).

## Other profiles

`swarm/graphos.stack.yml` is the hardened, downscaled non-cell Swarm profile. It
requires an immutable `GRAPHOS_IMAGE_DIGEST`, an encrypted non-attachable
overlay, external Docker secrets, direct GraphOS TLS, and mutually authenticated
native-engine TLS. The source contract rejects inline endpoints, credentials,
mutable images, plaintext engine exposure, and weakened container settings:

```bash
python scripts/deployment/check_swarm_assets.py --self-check \
  --runtime-image "$GRAPHOS_IMAGE_DIGEST"
docker stack deploy -c deploy/swarm/graphos.stack.yml graphos
```

Create the external secrets and `graphos-engine-data` volume before deployment;
the stack deliberately contains no environment-specific values. The Swarm
profile and `postgres/tenant_rls.sql` are not substitutes for the signed,
HA production-cell release gate.
