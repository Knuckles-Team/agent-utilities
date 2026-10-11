# Delivery plan — AU-SEC-01

## Decisions and order

1. Inventory current authority seams and failing tests. Pin exact scope names/classes and public contract with engine/gateway owners before changing AU projection.
2. Repair local process and control-view behavior at existing entry points; prove negative tests first.
3. Generate scope projection and contract tests from the engine-owned registry. Remove obsolete hand-maintained allowlist values after parity is proven.
4. Implement elevation request orchestration against a typed engine lease API. Connect the actual graph-os/A2A/chat execution paths and tool guard to the final engine decision. Coordinate browser presentation in its owner repository.
5. Implement guardrail proposal and activation with current decision/approval plumbing, immutable evidence, revision check, and audit. Connect cache invalidation and drift quarantine to real event/connector paths.
6. Run focused, contract, full repository and hosted gates on the exact candidate commit. Record evidence and update delivery state per requirement.

## Integration rule

Each task must replace or extend a named live call site. A helper with passing unit tests but no request, tool execution, connector, or cache path is incomplete. Keep one identity decoder, one scope registry, one policy decision per layer, and one durable lease authority.

## Design

### Existing wiring to reuse

`agent_utilities/security/request_identity.py` provides `ActorIdentityMiddleware`, `actor_from_claims`, `mint_graph_session`, `local_process_authority_enabled`, `mint_local_process_session`, and `system_write_session`. It reuses `security/auth.py` JWKS validation. `security/permissions_kernel.py` resolves tool policy; `security/tool_guard.py` connects that policy to pydantic-ai approval, preserving hard denial. `knowledge_graph/core/engine.py` binds the dedicated control view. `core/config.py` exposes registry invalidation; `caching/semantic_cache.py` has explicit invalidation. Reuse these entry points and existing typed `ActorContext`/`GraphSession` rather than adding parallel token or policy stores.

### Trust and data flow

```mermaid
flowchart LR
  P[Principal] --> G[graph-os verified public boundary]
  G --> A[AU actor and session projection]
  A --> T[AU tool policy and decision]
  T --> E[epistemic-graph final authorization]
  E --> L[Durable lease and audit]
  S[SDK connector sync] --> E
  E --> I[Invalidation event]
  I --> C[AU cache]
```

1. The boundary validates signature, issuer, audience, expiry and claims before AU receives a verified actor. AU projects only classified scopes and tenant from trusted claims. An unknown registry revision fails closed or enters a documented compatibility window without granting new permissions.
2. `tiny` local authority stays tied to a private packaged stdio process, ephemeral signing key, 120-second proof, and bounded session renewal. No network transport may invoke it. Local setup may use deterministic in-process fixtures, but a test fixture cannot silently become production authority.
3. A request for elevation carries `{request_id, actor_id, tenant_id, action, resource, reason, expires_at, trace_id}`. AU submits a typed intent to graph-os/engine. The engine persists a lease and audits transitions; `check_access` consumes only an active, scoped lease. The approver identity must differ from requester; model output has proposal authority only. Idempotency key prevents duplicate grants. Revocation is effective before the next protected action; expiry uses server time.
4. Guardrail profile proposals carry current profile revision, candidate digest, bounded delta, evidence references, and decision ID. AU submits through its existing orchestration/decision path. A tightening operation is monotone within configured bounds; a loosening operation must reference a live explicit approval. Commit uses compare-and-swap on revision; failure leaves the old profile active and emits a typed audit event.
5. Cache invalidation is keyed by tenant and versioned object. Lost event transport must not produce indefinite stale reads: configured TTL remains a hard cap. Rate learning cannot lengthen it. Security decisions do not rely on a stale permissive cache; revalidation occurs at the engine chokepoint.
6. Connector drift reports are generated at the SDK/engine boundary. AU receives typed `SchemaDriftReport` metadata, suspends orchestration when classification or policy fails, and resumes only after explicit, versioned disposition. The SDK owns checkpoint advancement.

### Compatibility and migration

Add the classed scope projection behind the existing `GraphSession` contract. Regenerate from one registry source, reject stale checked-in projections, and migrate callers of string literals without inventing aliases. Keep the existing health path exemption narrow. Public API changes require graph-os contract compatibility tests and matching browser behavior. Durable record schema evolution belongs to epistemic-graph and must be versioned there. No change may alter tenant or scope claims while fixing placement or transport.

### Failure and observability

Return 401 for absent/invalid identity, 403 for valid but forbidden identity, and a typed unavailable/retryable response for control or policy storage failure; never treat unavailable as allowed. Audit decisions and lifecycle transitions with redacted actor/trace metadata. Emit counters for deny, approval, expiry, revoke, drift quarantine, invalidation lag, and failed policy reload without logging tokens or sensitive payloads.

### Developer setup

Fresh clone: install Python and `uv` as declared by this repository, run `python3 scripts/uv_workspace.py doctor`, then `python3 scripts/uv_workspace.py run --all-extras pytest tests/unit/security -q`. Unit and contract suites must use local fixtures and provision only their declared dependencies. A served integration test that needs an engine must start or provision its own disposable fixture and fail with a clear missing-dependency reason; ordinary PR checks must not depend on a private network or a running deployment.

## Input and tool-output screening (AU-SEC-R011)

Data flow: `execute_agent (api/agent_control_plane.py:361) -> _prepare_agent_task (:348) -> screen_and_redact_agent_task (orchestration/task_guard.py:32) -> capability resolution`; tool results `-> output screen (capabilities/composition.py:147) -> next model step`.

Types: `ScreenVerdict` with source, closed reason and rule identifier. The injection scanner, the redaction guard and `TaskInjectionRejected` are reused. The tool-output screen and the approval-required refusal are new.

Failure: a failed screen stops the call before any model or tool runs. A scanner error counts as a failed screen and never as a pass.

Decision: one screen policy serves both sources, so the rule sets cannot drift apart. The rejected alternative is a separate tool-output policy with its own patterns.

## Approval after untrusted content (AU-SEC-R012)

Data flow: `execute_agent (api/agent_control_plane.py:361) -> tool result enters run -> taint mark -> tool call -> _requires_approval (security/tool_guard.py:383) -> ApprovalRequiredToolset (:464) -> tool-call record (capabilities/kg_audit_sink.py:113)`.

Types: `ContentInfluence` is new. `ActionClass` uses the producer values `egress`, `privileged_call` and `delegation` without translation. A contract test asserts exact equality and refuses `privileged`. `ApprovalRequiredToolset`, the policy decision at `tool_guard.py:383` and `ToolCallRecord` are reused. This rollup extends the screening of `AU-SEC-R011`: that rollup refuses bad text, this one governs actions after untrusted text.

Failure: an unknown action class returns `invalid_action_class` before any effect. A missing taint mark counts as tainted. A denied approval stops the action and records the outcome.

Decision: the run carries the received taint mark and pauses on the existing approval hook. The rejected alternative is a second approval engine for tainted runs.

Search terms used: untrusted, taint, influence, ApprovalRequiredToolset, requires_approval.
Existing code: `agent_utilities/security/tool_guard.py:383`, `:464`; `agent_utilities/agent/factory.py:1054`; `agent_utilities/capabilities/kg_audit_sink.py:113`, `:540`. Reachable today: approval yes, taint input no.
Moves to GRAPHOS-A2A: the approval decision in `_requires_approval` (`agent_utilities/security/tool_guard.py:383`).
Cross-repository rows: EG-SRC-AUTH-R011 — the taint mark on content; GRAPHOS-A2A-R005 — the approval admission for a gated action.
