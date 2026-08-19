# Security and operations

Apply these controls consistently across every runtime.

## Identity and tenancy

Use an existing OIDC provider when compatible or deploy the selected provider as a
separate capability. Validate issuer, audience, JWKS rotation, clock skew, service
identity, user identity, and logout/revocation behavior. Map tenants and roles
explicitly. Never share conversation, retrieval, or semantic/prompt caches across
tenants without a tenant-bound key.

IdP wiring is not the whole identity job: the deployed applications still need
their own role/scope contract provisioned on top of it, and a browser-facing
client's identity must never be conflated with a backend service's own
service-account identity — they are separate credentials with separate role
assignments, even when both belong to the same application. A UI-level admin
role and a data/graph-capability admin role are two different things; granting
one is never a substitute for the other, and a deployment is not verified
until both the human and the service identities hold the scopes their calls
actually require. For agent-utilities specifically, this contract (which
roles, which two OIDC clients, and how to diagnose it when wrong) is
provisioned and diagnosed in `agent-utilities-deployment`'s *Identity &
authorization* section, not duplicated here to keep this skill
environment-neutral.

## Secrets and PKI

Prefer workload identity to static credentials. Store secret values in a
Vault-compatible store, Kubernetes/Swarm secret, system credential facility, or the
engine’s encrypted secret graph according to the profile. Configuration carries only
references. Prove rotation and redact tool arguments, traces, and exception text.

Discover or create a trust hierarchy according to the plan. Distribute only the
required trust bundle. Test internal and external TLS names, expiry monitoring, and
rotation before cutover.

### Workload identity must be provisioned, not assumed

"Prefer workload identity" above is a day-0 provisioning step, not a property a
Vault-compatible store has by default. A store can hold every secret correctly and
still have **no way for a workload to prove who it is** — in which case every
consumer falls back to a static token, and any design that requires workload
identity is silently unrunnable.

Verify, do not assume: list the store's enabled auth methods. If only a token
method is mounted, workload identity does not exist yet regardless of what any
runbook claims. A store fronted by an external-secrets operator is *not* evidence —
that operator commonly authenticates with a static token reference of its own.

Provisioning it is four ordered steps, and each must be verified by reading it back:

1. **Enable the Kubernetes auth method** on the store.
2. **Configure it** with the cluster API host and CA. Sourcing both from the store
   pod's own projected ServiceAccount avoids hand-copied values drifting.
3. **Write a policy scoped to exactly the one secret path** the workload needs —
   never the whole mount. A signing workload needs one secret, not `apps/data/*`.
4. **Bind a role to exactly one ServiceAccount in one namespace**, and set the
   **audience** to match the audience on the workload's projected token. An
   unset audience silently accepts any token that satisfies the other bindings;
   a mismatched one fails at run time with an opaque auth error.

Only then can a controlled job read a credential live at use time instead of having
it materialized into a cluster Secret. That distinction is the point: a materialized
Secret persists the value at rest for its whole lifetime and adds a second custody
surface, and in this workspace a hand-patched externally-managed Secret is silently
reverted by the operator that owns it.

**Signing keys specifically:** a key that signs release artifacts belongs in a
controlled job, never in an interactive session. Expect a correctly-built signer to
*refuse* an environment-variable reference and accept only a versioned store
reference — that refusal is the control working. Expect it to fail closed rather
than fall back to an ephemeral key, and expect a keyless "review" run to be
impossible if the generator resolves its signer at build time. Budget for a **built
artifact at the frozen commit** as a prerequisite; a source-tree install is
deliberately rejected, because the thing being signed must be the thing that was
reviewed.

The rule that makes all of this worth the effort: **a signed-but-stale artifact is
strictly worse than an unsigned one, because it manufactures trust nothing
reviewed.** Never regenerate signatures over drifted content to make a gate pass.

**A different "signer," easy to conflate with the one above:** au's own background
daemons authenticate to the engine using a *separate* shared-HMAC "signer" concept —
an engine-side trusted-signer registry that authorizes registering identities and
granting RBAC roles, not signing build artifacts. It is not scoped by the engine to
any particular role a signer may grant, it is a symmetric secret with no rotation
overlap window by default, and on this platform's default configuration it can be
stored in the very store it authorizes writes to. Provisioning it is a Phase 5 step
in its own right, not a variant of artifact signing above. Read
[engine-identity-admission.md](engine-identity-admission.md) before provisioning,
diagnosing, or rotating it.

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
