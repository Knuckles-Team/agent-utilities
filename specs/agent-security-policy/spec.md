# Agent security and governed policy — AU-SEC-01

**Owner:** agent-utilities. **Delivery:** partial source implementation; **acceptance:** pending. This status describes the repository as inspected, not a release certification. Source obligations: EH-097, EH-379, EH-401, EH-402, EH-403, EH-405, EH-407, EH-541.

## Purpose and actors

An operator, service principal, approver, and agent need predictable identity and policy decisions across interactive, background, and local development execution. An untrusted tool request or model proposal must never acquire authority from a caller supplied field, a broad local default, a stale cache, or a failed control graph lookup. AU owns agent execution policy and authenticated session projection. `epistemic-graph` owns durable capability checks, leases, schema and drift facts; `graph-os` owns public HTTP/MCP/A2A authentication and request endpoints; `agent-connector-sdk` owns connector sync; `agent-webui` owns browser interaction. An adjacent implementation must follow these contracts but can be built independently.

## User stories and acceptance

1. **P0 — bounded authority.** As a non-admin caller, I can perform permitted read/write operations without an admin grant; forbidden operations return a typed denial, including from background jobs. A forged, expired, cross-tenant, or unknown scope is denied. A local `tiny` stdio session is process private, short lived, and unavailable to network listeners or any configured external identity.
2. **P0 — dependable control.** As an operator, I receive a typed unavailable result when the control view cannot open; work does not silently target a content graph. A drift classification or failed schema contract check quarantines connector work and does not advance its checkpoint.
3. **P0 — human governed elevation.** As a requester, I can ask for a specific temporary capability. An independent authorized approver can approve or revoke it; expiry and revocation are observed at the authorization chokepoint before every protected action. The requesting agent and a learned policy cannot issue their own elevation. Denial and audit records contain subject, tenant, action, decision, lease ID, and trace ID without credentials.
4. **P1 — safe adaptation.** As an operator, I can inspect a proposed guardrail change and its evidence. Tightening within declared bounds may auto apply through the governed decision path; loosening requires explicit approval and remains denied when policy state is unavailable. Caches observe authoritative invalidation, and a learned volatility rate can only shorten a configured TTL.
5. **P1 — consistent registry.** As a contributor, I can add one classified scope in the durable registry and regenerate the AU allowlist. Unknown or service-only scopes cannot be smuggled through a human JWT. Public surfaces and the engine agree on exact scope spelling and class.

## Functional requirements

| ID | Requirement | Source | Evidence required |
|---|---|---|---|
| SEC-01 | Mint a server-owned actor/session only from validated claims, retaining audience, tenant, expiry, scope and policy revision; prohibit request supplied session authority. | EH-097, EH-541 | unit + gateway contract tests |
| SEC-02 | Restrict local process authority to packaged `tiny` stdio, no external endpoint/identity, and expiry bounded renewal; deny network reuse. | EH-097 | positive/negative profile tests |
| SEC-03 | Fail closed with typed error on missing control view; never substitute a content graph. | EH-379 | injected backend failure test |
| SEC-04 | Consume the canonical classed scope registry; regenerate and verify AU projection with exact values. | EH-541 | generator diff + scope parity |
| SEC-05 | Request, approve, revoke, and expire elevation through typed, tenant scoped operations; enforce two distinct actors and audit every transition. AU orchestrates only, with durable lease and final authorization in the engine. | EH-403, EH-405 | contract + end to end policy tests |
| SEC-06 | A tool guard denial stays a denial; approval can satisfy only a declared approval requirement and cannot override RBAC or engine policy. | EH-405 | tool execution negative tests |
| SEC-07 | Evolve guardrail profiles through a governed decision: bounded tightening may auto apply; loosening requires a live approval; reject missing bounds or stale evidence. | EH-407 | decision and replay tests |
| SEC-08 | Subscribe AU caches to invalidation, bound TTL by declared volatility class, and allow learned change rates to shorten TTL only. | EH-401 | event, loss, and clock tests |
| SEC-09 | Treat schema drift as a typed, fail-closed result at the SDK/engine seam, quarantine without checkpoint advance, and never load AU-owned SHACL as authority. | EH-402 | connector contract tests |

EH-405 and EH-403 concern public request surfaces owned by graph-os and agent-webui as well as AU's orchestration. EH-402's classifier and durable report belong to SDK and epistemic-graph; this spec covers AU's control decision and failure behavior. These IDs indicate cross-repository obligations, not duplicate implementations.

## Edge cases and success criteria

Concurrent approve/revoke, renewal across expiry, replayed requests, mixed human/service claims, unknown registry version, control backend outage, dropped invalidation, malformed drift report, and malformed model proposal must have deterministic denials or safe retry semantics. No grant survives its expiry; no denied call executes; no quarantined batch advances its source checkpoint; a failed control graph never redirects writes. Every requirement has a runnable test and a linkable exact-commit result before **ACCEPTED**.

Out of scope: implementing JWT validation twice, new graph authorization storage in AU, connector transport, browser UI, or a second public gateway.
