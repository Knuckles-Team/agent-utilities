# Design Document: The ontology gets first-class governed ACTIONS — closing the one real gap versus Palantir AIP's unified data+logic+action+security model

CONCEPT:AU-KG.ontology.ontology-action-system ·
CONCEPT:AU-KG.ontology.batch-actions-executor ·
CONCEPT:AU-KG.ontology.submission-criteria-gate ·
CONCEPT:AU-KG.ontology.vendor-writeback-action

> `agent_utilities/knowledge_graph/actions/` (`__init__.py`, `dispatch.py`,
> `effects.py`, `executor.py`, `fleet_writeback.py`, `models.py`,
> `registry.py`).

## Decision — a parameterized, permission-gated, audited verb layer over ontology objects, reusing the existing permissions/audit/edit-ledger fabric rather than reinventing any of it

`actions/__init__.py:4-43` states the gap and the design directly: the
ontology already modeled nouns (typed nodes) and capability properties
(`providesCapability`/`requiresCapability`/`swappableWith`) but not Palantir
AIP's fourth pillar, **Actions**. `OntologyAction` (a verb definition: params,
`acts_on` object types, `required_capability`, `produces_effect`),
`ActionRegistry` (binds definitions to handlers), `ActionInvocation` (the
audited per-call KG record), and `ActionExecutor` (authorize → validate →
run → audit → persist) close that gap. Every governance concern is explicitly
reused, not reinvented: authorization goes through the existing
`PermissionsKernel`, audit through the existing `AuditLogger`, and OWL
reasoning gets a real dividend — the `mayBeInvokedBy` property chain
(`requiresCapability` ∘ `providedBy`) means an agent that provides an action's
required capability is *reasoned* eligible to invoke it, not manually
enumerated. `DEFAULT_REGISTRY` is import-populated with real built-ins
(`kg.search`, `finance.forensic_screen`) rather than shipped empty; an executor
requires an explicitly injected permissions kernel — there is no default/open
executor.

**The rejected alternative is bespoke, per-caller mutation code** — the
pattern this whole subsystem replaces. Without a governed action layer, every
"do a thing to an ontology object" caller would need its own authorization
check, its own audit-log call, and its own decision about whether the mutation
is revertible — with nothing guaranteeing those three are done consistently.
Centralizing them in `ActionExecutor` means a new action type gets all three
for free by construction.

### Pointer — `CONCEPT:AU-KG.ontology.batch-actions-executor`

`actions/dispatch.py:4-22`. The Notifications/Webhooks half of Palantir's
Action Type contract ("after an Action Type's edits are submitted it may
notify recipients and call external systems"), made real rather than a silent
no-op: `RecordingNotifier` is the default sink — with no live channel wired, a
notification is still durably journaled (`delivered=False`,
`transport="recorded"`) rather than dropped, and `send_webhook` degrades the
same way when `httpx` isn't importable. **The rejected alternative is a no-op
dispatch when no channel is configured** — silently swallowing a notification
that was supposed to fire. Recording it instead means "did the system attempt
to notify" stays observable even when nothing was actually wired to receive
it, closing what the module calls the "Wire-First loop (a dispatch that
happened is observable)."

### Pointer — `CONCEPT:AU-KG.ontology.submission-criteria-gate`

`actions/effects.py:4-15`. The other half of Palantir's Action Type contract:
validate parameters against submission criteria, then apply typed edits
(create/modify/delete object, add/remove link) atomically on submission. The
real decision is WHERE the edit leg is wired: through the existing C1 Edit
Ledger (`ontology/edits`), so "every applied side-effect is a durable,
revertible `object_edit`" — the SAME ledger `CONCEPT:AU-KG.ontology.edit-ledger-writeback`
covers (see `.specify/design/kgo-cat-edit-ledger/design.md`) — rather than a
second, action-specific mutation-recording mechanism. The module states its
own scope discipline: "pure orchestration over existing fabric — it neither
reinvents permissions/audit (the executor owns those) nor the edit journal
(C1 owns that)."

### Pointer — `CONCEPT:AU-KG.ontology.vendor-writeback-action`

`actions/fleet_writeback.py:1-23`. The symmetric write path to KG-2.59's read
ingestion: one governed `fleet.write_record` action pushes a mutation back to
an external system of record through the SAME fleet MCP tools
(`call_tool_once`) the read-side source connector uses. **The rejected
alternative is a bespoke write path per vendor** (a `servicenow_writeback.py`,
a `jira_writeback.py`, …) — instead, because the write runs through the
governed action executor, every external write is "authorization-gated,
approval-gateable, and audited as an `ActionInvocation`" for free, turning the
KG "from a read model into a *system of action* over the enterprise without a
single bespoke per-vendor write path." The caller supplies exactly
`server`/`tool`/`action`/`params` — the same shape a source preset already
uses for reads, so writing a new vendor's write-back requires no new code,
only a call.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/actions/*.py`,
  `knowledge_graph/ontology/edits.py`, `shapes/governance.shapes.ttl`
  (`OntologyActionShape`), `ontology_action.ttl`.
- **Backward Compatible**: Yes — an additive verb layer; no existing node/edge
  type is changed.
- **Known weak point**: `fleet_writeback`'s generality is also its risk —
  because the caller supplies raw `server`/`tool`/`action`/`params`, the
  action-layer authorization gate can approve or deny the WRITE ATTEMPT, but
  cannot itself validate that the specific `params` are semantically safe for
  the target vendor tool; that validation is delegated entirely to the called
  MCP tool.
