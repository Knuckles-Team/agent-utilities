# Verified Identity Carrier Contract (GOC-15)

CONCEPT:AU-OS.identity.verified-carrier-contract

> **Status:** contract freeze (GOC-15-W02) for the carrier that already ships
> in production. This document is the canonical reference GOC-17–23 (and any
> other lane minting or consuming caller identity — GOC-85's remote MCP
> broker in particular) must build against. It **corrects two false premises**
> that circulated in planning before this lane verified `main`; see
> [Premise corrections](#premise-corrections-read-first) before designing
> anything against this contract.

## Premise corrections (read first)

1. **There is no `CommitDescriptorV1` on `main`.** A prior planning note
   claimed a sibling lane had landed `crates/eg-types/src/commit_descriptor.rs`
   with a `tenant_ref`/`principal_ref`/`authority_ref`/`authority_epoch` shape.
   That file and that type do not exist anywhere in `epistemic-graph` as of
   this lane's verification (2026-08-16, `epistemic-graph@02594f7`). Do not
   build against it. What DOES exist, and IS the real identity carrier, is
   documented below.
2. **The JSON-Schema `RequestContext` model is dead code, not the carrier.**
   `agent_utilities/protocols/epistemic_operations/schemas/v1/request-context.schema.json`
   (`$id: urn:epistemic-operations:v2:request-context`, `schema_version` const
   `"2"`) generates a pydantic `RequestContext` class
   (`agent_utilities/protocols/epistemic_operations/_generated.py`). It has
   **zero live callers** anywhere in `agent-utilities` outside its own
   `tests/unit/protocols/test_epistemic_operations.py` — grep confirms no
   production code ever constructs or `model_validate`s a `RequestContext`.
   Promoting this schema to "v3" (as an earlier lane plan proposed) would be
   formalizing a projection nothing uses, while the *actual* wire carrier
   (below) stays undocumented. This lane does not wire that dead model into a
   live path — see [What this lane did NOT do](#what-this-lane-did-not-do).

## The carrier that is actually live today

Two verified, already-fail-closed, already-tested primitives, one per
language, connected by one wire dict. **This lane does not invent a third.**

```
AU:  ActorContext  →  GraphSession  →  GraphSession.engine_verified_context()
                                              │  (Python dict, MessagePack over eg2.)
                                              ▼
EG:  RequestContextClaims  →  server::auth::VerifiedRequestContext (post-verify)
                                              │
                                              ▼
                                    server::access::CarrierAuthority
                                    (tenant_scope / actor_scope / owner_scope / admin)
```

* **AU identity primitive:** `agent_utilities.security.brain_context.ActorContext`
  (`security/brain_context.py`) — minted only from validated credentials
  (`actor_from_claims`, `security/request_identity.py`).
* **AU session/authority primitive:** `agent_utilities.knowledge_graph.core.session.GraphSession`
  (`knowledge_graph/core/session.py`) — the one explicit currency (actor,
  tenant, scopes, graph, policy_version, audience, trace_context). Its
  `__post_init__` already fails closed on an unauthenticated actor, an empty
  `actor_id`, an empty tenant, or `tenant != actor.tenant_id`
  (`SessionRequiredError`) — proven by
  `tests/unit/test_graph_session.py::test_session_construction_rejects_unverified_or_mismatched_authority`,
  parametrized over unauthenticated / empty-actor / empty-tenant /
  **cross-tenant** / empty-session-tenant.
* **The wire projection (the actual carrier bytes):**
  `GraphSession.engine_verified_context()` returns exactly the dict below,
  which the native `epistemic_graph` client packs into the `eg2.` MsgPack
  envelope. This dict shape — not the JSON Schema — is the operative carrier.
* **EG identity primitive:** `crates/eg-types/src/acl.rs::RequestContextClaims`
  — the wire *representation*, untrusted until verified.
* **EG verified/authoritative primitive:** `server::auth::VerifiedRequestContext`
  (non-constructible outside the module; only produced after MAC + audience +
  tenant + policy-version + replay-nonce verification) →
  `server::access::CarrierAuthority::from_verified()`, which derives
  `tenant_scope` / `actor_scope` / **`owner_scope()`** (the actual per-agent
  isolation key — `tenant_scope + actor_scope`, opaque-hashed) / `admin`.

## Canonical field table (identity/claims layer)

These are the fields `GraphSession.engine_verified_context()` emits and
`RequestContextClaims` deserializes (`#[serde(deny_unknown_fields)]` — an
unrecognized field on either side is a hard reject, not a soft ignore). This
is the field-level contract every consumer of this lane must match.

| Field | Type | Required | Notes |
|---|---|---|---|
| `principal` | string | always | Authenticated subject. AU sets it to `actor.actor_id` (the verified JWT `sub`/`client_id`/`azp`). Hashed by the engine before persisting `ChangeEnvelope` provenance — never logged raw. |
| `tenant` | string | always | Tenant security boundary. Must equal `GraphSession.actor.tenant_id`; EG separately compares it against the deployment's own configured tenant on auxiliary surfaces (Iceberg — see below). |
| `audience` | string | always | Server-validated intended audience (`AUTH_JWT_AUDIENCE`/`MCP_JWT_AUDIENCE`), never accepted from a request payload. |
| `agent_id` | string | always | Effective ACL/RBAC subject. Equals `principal` unless a spawn is running under an ENFORCED delegation (see Delegation below), in which case it is the per-run agent-instance id. |
| `roles` | string[] | always (may be empty) | IdP-agnostic capability set (`ActorContext.roles`), sorted, deduped. |
| `scopes` | string[] | always (may be empty) | `GraphSession.scopes`, restricted to the hierarchical `kg:read`/`kg:write`/`kg:admin` set with `kg:admin ⊇ kg:write ⊇ kg:read` expansion applied once at mint time (`_mint_graph_session`). |
| `delegation` | string[] | always (may be `[]`) | Ordered chain from `principal` to `agent_id` inclusive. Empty ⇒ no delegation, requires `principal == agent_id`. Only emitted non-empty when `ENABLE_DELEGATED_IDENTITY=on` AND the ambient `SpawnDelegation.principal` matches this session's own principal — a spawn can never forge a chain for a principal it does not run under (`security/delegation.py`, `session.py::_apply_spawn_delegation`). |
| `policy_version` | string | always | Exact policy bundle/version this authority was minted under (`KG_POLICY_VERSION`). |
| `node` | string \| absent | optional | ADR-3/W1.9 node-bound envelope target. `#[serde(default)]`; omitted claim ≠ error on either side. |
| `priority` | string \| absent | optional | Advisory QoS class (W2.4), one of `interactive`/`orchestration`/`hydration`/`background_ingestion`; MAC-covered so it cannot be forged to jump the admission queue. Omitted entirely for an untagged caller (byte-identical to a pre-W2.4 envelope). |
| `oidc_token` | string \| absent | optional | RFC 8693 exchanged token, present only under an enforced `SpawnDelegation` that carries one. **Not MAC-covered** — its own signature is its trust anchor; the engine independently RSA/JWKS-verifies it and cross-checks subject/tenant against the SAME context (`server::auth::bind_verified_identity`). |

**Explicitly NOT part of this layer** (a deliberate separation, not a gap):
`request_id`, `trace_id`, `issued_at`/`expires_at`, `schema_version`. Replay/
idempotency identity lives on `MutationBatch`/`ChangeEnvelope`
(`envelope_id`, `idempotency_key`, `submitted_at_ms`) — a **different**
currency with its own dedup/replay semantics — and trace correlation stays in
`GraphSession.trace_context`, which is attached to observability spans, not
sent to the engine as an identity claim. A consumer needing correlated
request/trace ids for a new surface (SPARQL, federation, observability
exporters) should propagate `GraphSession.trace_context` alongside — not
inside — the claims dict above; do not add these fields to
`RequestContextClaims` without a corresponding EG wire-compat sweep (adding a
field to a `deny_unknown_fields` struct is a breaking deploy-order change,
exactly like W2.4's `priority` rollout note already documents in
`session.py::engine_verified_context`'s docstring).

## Per-surface carrier status (verified against `main`)

| Surface | Carrier today | Per-caller principal? | Reject-bad proof |
|---|---|---|---|
| `eg2.` primary protocol (REST/WS/SSE/MCP → engine) | `RequestContextClaims` above, MAC + audience + tenant + policy + replay-nonce verified (`server/auth.rs`) | Yes | `tests/unit/test_graph_session.py` (AU side, cross-tenant/unauthenticated); EG `server/auth.rs` unit suite (MAC/replay/audience) |
| Iceberg REST catalog | `authenticated_iceberg_bearer` — projects an **already OIDC-verified** bearer's own `subject`/`tenant` claim; **tenant claim required (not merely compared-if-present)** and rejected on mismatch against `EPISTEMIC_GRAPH_TENANT` (`src/server/auth.rs:334`) | Yes (per-subject, since GOC-222) | **Already a full negative matrix**: `src/server/auth.rs` `mod iceberg_bearer_carrier` — `different_tenant_mints_no_carrier_and_is_denied`, `missing_tenant_claim_mints_no_carrier`, `no_verified_claims_mints_no_carrier`, `empty_subject_mints_no_carrier`, `distinct_subjects_bind_distinct_non_admin_principals`. **No further work needed here** — do not re-implement. |
| S3 SigV4 / KV-cache bearer / `/sparql` SELECT-CONSTRUCT-ASK bearer | `mint_fixed_service_carrier` — ONE fixed `service:<name>` principal for every caller who passes that surface's own protocol-native check | **No** — deliberately: "none of these protocols carries a distinguishable per-caller principal" (doc comment, `server/auth.rs`) | N/A by design — this is a real architectural decision already made, not an oversight. **If GOC-18/19/20 (SPARQL/federation) need per-caller isolation, that requires adding an OIDC-bearer leg to those surfaces mirroring `authenticated_iceberg_bearer`** — same shape, new call site. That is future work this lane scopes but does not implement (see below). |
| Native SQL wire (pgwire/mysql-wire/mssql-wire) | `authenticated_sql_wire_actor` — HMAC-derived opaque principal per `(protocol, agent_id)` after SCRAM/HMAC password proof | Yes | Covered by that module's own SCRAM/HMAC test suite (not re-audited by this lane; out of the listed W01 surface set) |
| WebUI browser boundary | See below | Yes (the signed-in human's own token) | See below |
| Observability exports | **Not yet instrumented with a carrier at all** (W01 gap, confirmed) | No | Out of scope for this lane's implementation pass; flagged for GOC-15-W06 follow-on |

## WebUI browser credential boundary (verified)

`agent-webui`'s `oidc_session.py` (`agent/agent_webui/oidc_session.py`) is a
pure-ASGI middleware mounted **outside**
`agent_utilities.security.request_identity.ActorIdentityMiddleware`. Verified
mechanism, reading the live code (not the module docstring alone):

1. The authorization-code exchange runs **server-side** with a confidential
   client (`WEBUI_OIDC_CLIENT_ID`/`WEBUI_OIDC_CLIENT_SECRET`) — the client
   secret never reaches the browser.
2. The resulting `access_token` is stored in a **Fernet-sealed** session
   cookie (`WEBUI_SESSION_KEY`, 32-byte urlsafe-base64 key) — chunked across
   `_chunk_name(index)` cookies, `secure`, cleared on logout. The browser
   holds only the encrypted blob; it cannot read or replay the token content
   itself.
3. Per request, the middleware **decrypts server-side** and re-attaches
   `Authorization: Bearer {token}` (`oidc_session.py:704`) before handing the
   request down to the **unmodified** `ActorIdentityMiddleware`, which
   validates it via the same JWKS path as any other bearer caller — no
   parallel identity primitive, no widened authority.
4. The carried token's `sub`/`email`/`realm_access.roles` are the **signed-in
   human's own** — the WebUI backend never elevates or acts "on behalf of"
   with a shared privileged credential.

**This satisfies the lane's "no data-plane surface may infer identity from an
untrusted... browser credential" invariant already**: the browser's own
credential is opaque (sealed) and the identity actually used downstream is
independently re-verified server-side on every request, not trusted from the
cookie's mere presence. No code change was required here; this section is the
recorded proof.

**One doc-drift correction while verifying this surface:**
`docs/architecture/identity-inheritance.md`'s "Deferred / roadmap" section
(unchanged since commit `3e13feeec`) still lists **"graph-os on-behalf-of
token exchange in `execute_agent`"** — RFC 8693 delegation carrying the
original caller's identity to downstream calls — as **not yet implemented**.
It has been implemented since commit `df3b69b90`
(`feat(identity): per-agent on-behalf-of delegation — connect the three
primitives (W2.1)`): `security/delegation.py` (`SpawnDelegation`,
`ENABLE_DELEGATED_IDENTITY`) plus `session.py::_apply_spawn_delegation`
forward exactly this — the `delegation` chain and, since W2.1-1, the
`oidc_token` claim documented in the field table above. This lane corrects
that stale entry (see the diff to `identity-inheritance.md` in this change).

## Bounds already enforced (vs. the lane's proposed bounds)

The lane brief proposed carrier ≤16 KiB / delegation depth ≤8 / scopes ≤128 /
TTL ≤15 min / clock skew ≤60 s as new bounds to add. Verified against `main`:

* **TTL/clock skew**: already enforced — `RequestContextPolicy` +
  `server/auth.rs`'s envelope timestamp/nonce checks (durable replay-nonce
  acceptance, per this file's module docstring) and AU's
  `GraphSession.ensure_authority_current()` (fails closed on expired bearer).
* **Delegation depth / scope count / total size**: **not currently bounded**
  as an explicit numeric limit on either side — this is a real, verified gap.
  `_apply_spawn_delegation` only validates principal-first/agent-last/`len≥2`
  shape, not a maximum chain length; `RequestContextClaims.scopes` has no
  cardinality cap; there is no explicit total-envelope-size cap distinct from
  the transport's own frame limits. **This lane records the gap and defers
  the fix** rather than adding an unreviewed bound to a shared, already-live,
  `deny_unknown_fields` wire struct without the security-owner sign-off this
  lane's own acceptance gates require (Acceptance gate 1). Recommended next
  step for GOC-15-W06: add depth/count checks in
  `VerifiedRequestContext::from_verified_claims` (AU mint side) and a
  matching cap in `_mint_graph_session`/`_apply_spawn_delegation`, sized from
  real observed delegation chains (there are currently none deeper than 2 in
  the codebase's own call sites), not from an arbitrary constant.

## What this lane did NOT do (and why)

* **Did not promote the dead JSON Schema to "v3."** Formalizing an unused
  projection would create a second, competing "canonical schema" next to the
  one actually on the wire — exactly the duplicate-implementation failure
  this program has hit three times. If a future lane wants a
  language-neutral schema artifact for the carrier (e.g. for a non-Rust,
  non-Python consumer), it should regenerate that schema **from** the field
  table above, then wire a real caller to it in the same change — not before.
* **Did not add a competing carrier type in AU or EG.** `ActorContext`/
  `GraphSession` (AU) and `RequestContextClaims`/`VerifiedRequestContext`/
  `CarrierAuthority` (EG) are the carrier. GOC-85 and every other consumer
  should construct/consume through these, not a new dataclass.
* **Did not re-implement the Iceberg tenant-mismatch negative matrix** — it
  already exists and is complete (see table above).
* **Did add** (this change): `agent_utilities.security.request_identity.CARRIER_CLAIM_FIELDS`
  / `OPTIONAL_CARRIER_CLAIM_FIELDS` (the exact field-name sets from the table
  above, exported as the one place to import them from) and
  `validate_carrier_claims()`, a fail-closed shape-check helper for any new
  adapter (SPARQL/federation/observability) that needs to verify a claims
  dict has the right shape before propagating it — so those lanes import a
  shared check instead of hand-rolling a second key list. See
  `agent_utilities/security/request_identity.py` and its test,
  `tests/unit/security/test_carrier_claim_contract.py`.

## Handoff to GOC-85 (remote browser-OAuth MCP broker)

GOC-85's per-principal remote MCP session isolation in
`agent_utilities/mcp/multiplexer.py` should:

1. **Mint one `GraphSession` per verified remote principal** through the
   existing `mint_graph_session(actor)` /
   `_mint_graph_session(actor, audience=..., policy_version=...)` path — the
   same function every other served transport already uses. Do not invent a
   parallel per-session identity object.
2. **Key session isolation on `(tenant, actor_id)`**, mirroring EG's own
   `CarrierAuthority.owner_scope()` (`tenant_scope + actor_scope`, opaque-
   hashed) — this is the field-level isolation key this lane's contract
   defines. Two remote principals must never share a `GraphSession`/engine
   authority even if they share a tenant.
3. **Reject reconnect that changes actor/tenant mid-session** — the same
   invariant `ActorIdentityMiddleware` already enforces per-request; a
   long-lived remote MCP session must re-verify on reconnect, not persist the
   first principal across a changed credential.
4. **Narrow via `GraphSession.with_actor`/scopes intersection only** — never
   widen. If GOC-85 needs a scoped child session (e.g. one OAuth-broker
   session fanning out to multiple downstream tool calls with narrower
   scopes), that is a scope-intersecting copy of the SAME session type, not a
   new carrier shape.
5. Import `CARRIER_CLAIM_FIELDS`/`validate_carrier_claims` from
   `agent_utilities.security.request_identity` if it needs to validate a
   claims dict shape before forwarding it (e.g. logging/audit redaction of an
   outbound carrier) — do not redefine the field list.

## Unverified / open

* Observability export carrier (per this lane's own W01 surface list) is
  **not instrumented at all** — confirmed absent, not merely unproven.
* SPARQL/federation per-caller isolation (as opposed to today's fixed-service
  identity) is **not implemented**; the Iceberg-bearer pattern is the
  template but was not extended to these surfaces in this pass.
* Delegation-depth/scope-count/size bounds are **unenforced** (see above).
* Key rotation / overlap window for the OIDC verifiers EG already runs
  (`server::oidc`) was not re-audited by this lane — assume unchanged from
  whatever GOC-01 verified.
* Mixed-version carrier behavior (an older client's envelope missing
  `priority`/`oidc_token` against a newer engine, and vice versa) is
  documented as a **deploy-ordering constraint** in
  `session.py::engine_verified_context`'s docstring
  (`deny_unknown_fields` rejects the WHOLE request on one unrecognized
  field) but has no live mixed-version test proving it — recorded, not
  fixed, by this lane.
