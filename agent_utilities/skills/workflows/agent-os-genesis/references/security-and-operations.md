# Security and operations

Apply these controls consistently across every runtime.

## Identity and tenancy

Use an existing OIDC provider when compatible or deploy the selected provider as a
separate capability. Validate issuer, audience, JWKS rotation, clock skew, service
identity, user identity, and logout/revocation behavior. Map tenants and roles
explicitly. Never share conversation, retrieval, or semantic/prompt caches across
tenants without a tenant-bound key.

## Secrets and PKI

Prefer workload identity to static credentials. Store secret values in a
Vault-compatible store, Kubernetes/Swarm secret, system credential facility, or the
engine’s encrypted secret graph according to the profile. Configuration carries only
references. Prove rotation and redact tool arguments, traces, and exception text.

Discover or create a trust hierarchy according to the plan. Distribute only the
required trust bundle. Test internal and external TLS names, expiry monitoring, and
rotation before cutover.

## Network and permissions

Default-deny inbound and egress where the runtime supports it. Allow only declared
service and provider flows. Treat tool descriptions as untrusted metadata. Enforce
tool allow-lists, argument validation, user consent for side effects, idempotency
keys, and tenant/permission context at the execution layer.

## Observability

Correlate ingress request, agent run, Pydantic node/step, model request, retrieval,
tool call, connector request, engine transaction, and background work under one
trace. Capture tokens, cost, latency, queue delay, retries, schema repairs, cache
hit/miss/tenant, termination reason, and redacted error details.

Export through OpenTelemetry to operator-selected backends. An observability vendor
is optional; trace semantics are not.

## Resource arbitration

Classify foreground user execution above background ingestion, indexing, research,
evaluation, and evolution. Background work must checkpoint and yield CPU, memory,
GPU, disk IO, model concurrency, and connector quotas when foreground demand rises.
Resume from durable checkpoints after the quiet period. Record preemption and lost
work.

## Supply chain and operations

Pin source revisions, images, charts, and dependencies; verify provenance, signatures,
SBOM, and vulnerability policy. Attribute compute/model/storage/connector cost by
tenant, user journey, workflow, and feature.

Define SLOs, alerts, runbooks, backup retention, restore drills, upgrade order, data
migration compatibility, and rollback. A successful snapshot without a restore test
is not a recovery gate.
