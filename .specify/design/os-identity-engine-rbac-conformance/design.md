# Design Document: GOC-62 two-tier auth standard — engine RBAC admission (D3a) and the conformance suite (D2)

CONCEPT:AU-OS.identity.engine-rbac-admission
CONCEPT:AU-OS.identity.stack-wide-auth-conformance

> `agent_utilities/security/engine_rbac_admission.py` (D3a);
> `agent_utilities/security/conformance/` (D2).
> Full record: `plans/graph-os-completion-program/decisions/GOC-62-keycloak-auth-standard.md`

## Decision 1 — `engine_rbac_admission.py`: bridge Keycloak Tier-1 grants into the engine's independent Tier-2 RBAC store (D3a)

The engine (epistemic-graph) gates 5 of its 86 `authz_action` strings —
spanning 20 wire `Method` variants (`admin:cluster-read` incl.
`PlacementRoute`, `admin:cluster`, `admin:backup`, `admin:sqlite-file`,
`security:admin`) — behind a SECOND, independent check:
`IsolationLayer::has_admin_capability` (`crates/eg-core/src/isolation.rs`),
which consults ONLY a durable, engine-local `agents`/RBAC store. No Keycloak
scope, however broad, ever satisfies it — that store is written only by an
explicit `RegisterIdentity`/`RbacAdmin` engine RPC, and nothing in
Keycloak-side provisioning (`agent-webui/scripts/provision_identity.py`) calls
it. A fresh engine store therefore has NO admin identity at all, and every
Tier-2 action fails `ACCESS_DENIED` even for a service Keycloak granted the
matching scope to (BUG-030's live-cluster gap; BUG-038's "fresh-store deploy
would fail" proof).

`engine_rbac_admission.py` is the missing bridge: an idempotent,
source-controlled admission pass run by deployment tooling immediately after
Keycloak-side (Tier-1) provisioning. It performs the SAME two-step admission a
human operator did by hand on the live cluster (DEC-015 §4's verdict on commit
`ee35179`), but reproducibly, from source: (1) bootstrap — once, only on a
pristine store, the provisioner's own identity self-registers via
`ConsensusClient.bootstrap_system_identity`, the engine's one-time,
`security:bootstrap`-scoped gate, deliberately NOT IdP-derived because the
first admin cannot be granted by an RBAC store with no admin in it yet; (2)
admission — every deploy, idempotent, using an already-admitted identity's
signer credentials to register/grant every service identity the Tier-1
Keycloak manifest says needs a Tier-2 action.

**The rejected alternative** was granting Tier-2 capability by hand per
deployment (the status quo the bug describes) — unreproducible, undocumented
in source, and exactly what left the live cluster in an unverified state
(commit `ee35179`, unpushed).

## Decision 2 — `security/conformance/`: live introspection, not a hardcoded surface list (D2)

The conformance suite's goal is to prove, per surface, across the entire
GOC-15 28-surface inventory: authenticated-allowed, unauthenticated-denied,
wrong-audience-denied, wrong-tenant-denied, insufficient-scope-denied, and —
the check that matters most — no fail-open. DEC-015's own §1 table is already
a hardcoded list; the moment a surface #29 is added, nothing re-derives it. So
the 28-surface inventory is this standard's SEED DATA, not its enumeration
mechanism — the mechanism must be a live introspection pass over the actual
registries surfaces are registered in, cross-checked against a reviewed
disposition manifest (`surface_manifest.py`) via a drift test that FAILS when
introspection finds a surface the manifest does not know about.

`security/conformance/` is the first, working instance: a live introspection
pass over the `graph_query` dialect dispatcher
(`surface_inventory.py`'s `enumerate_query_dialect_surfaces` parses the ACTUAL
source of `mcp/tools/query_tools.py`, never a hand-maintained list, to find
every `if scope == "<literal>":` branch). Deliberately scoped to one surface
family rather than the full 28-surface inventory: live introspection for
FastAPI routes requires constructing a full agent app (too heavy for a fast
unit-test harness), and the EG wire-`Method` family already has its own
complete, generated enumeration on the Rust side
(`eg_capabilities::ALL_METHODS`) needing no Python mirror. The query-dialect
family is real, live, mechanically enumerable, and is exactly where BUG-036
(the AU federation fail-open,
`CONCEPT:AU-KG.identity.verified-carrier-required-federation`) lives — so it
doubles as the proof-of-concept AND the harness for that fail-open.
Deliberately does NOT auto-classify a surface's authorization posture via
static analysis (an early design considered scanning for a literal
`resolve_session(`/`current_session(` call, but `sql`/`sparql` are
session-required only at the WIRE layer, defeating a shallow heuristic) — so
disposition stays a reviewed, human-authored manifest field, matching this
codebase's "gates report more coverage than they have" lesson.

## Risk Assessment

- **Blast Radius**: `agent_utilities/security/engine_rbac_admission.py`,
  `agent_utilities/security/conformance/*.py`.
- **Backward Compatible**: Yes — the admission bridge is additive deployment
  tooling; the conformance suite is test/audit-only, no production code path
  changes.
- **Known weak point**: the conformance suite covers ONE surface family
  (`graph_query` dialects) of the 28-surface GOC-15 inventory today —
  extending coverage to FastAPI routes and the EG wire-`Method` family is
  explicitly out of scope for this pass.
