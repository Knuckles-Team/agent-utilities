# Epistemic Audit & Compliance (CONCEPT:AU-KG.enrichment.compliance-posture-rollup)

**Persona:** internal audit, SOC2/compliance officer, or anyone building
regulated decision support on top of the knowledge graph — "what did we
believe, on what evidence, as of what date, and can I prove the record
hasn't been tampered with."

This guide is a walkthrough for a specific narrow question a plain
`graph_query` Cypher result can't answer on its own: not just *what is in the
graph*, but *why do we believe it, who could see it, and is the trail
provably intact*. Everything below wraps existing primitives — no new
compliance or redaction logic was added to write this guide; the tools it
documents (`graph_audit`, `graph_compliance`, `graph_epistemic`) are thin
aggregation/naming layers over the engine's own audit chain and belief
machinery (`agent_utilities/mcp/tools/audit_tools.py`,
`agent_utilities/mcp/tools/compliance_tools.py`,
`agent_utilities/mcp/tools/epistemic_tools.py`).

## What you get

| Question | Tool / action | Backing primitive |
|---|---|---|
| "Has the audit log been tampered with?" | `graph_audit(action="verify")` | Rust SHA-256 hash-chained mutation log (`epistemic-graph/src/audit.rs`, redb `AUDIT` table) |
| "What happened to this entity, in order?" | `graph_audit(action="for_target", target_id=...)` | KG's own `:ToolCall` provenance, reverse-indexed |
| "What's our overall compliance posture right now?" | `graph_compliance(action="posture")` | audit-chain `verify()` + node-count/status rollup of the governance labels the CISO Assistant extractor + TRM engine already ingest |
| "Export this subgraph for an auditor, with redaction applied" | `graph_compliance(action="export")` | bulk `explain_belief(node_id, disclosure_level)` over an id list or Cypher selection |
| "Why do we believe this claim?" | `graph_epistemic(action="why", node_id=...)` | `explain_belief` justification tree (`Asserted` / `DerivedSupport` / `DerivedContradiction` / `BayesianUpdate`) |
| "Do we still believe it, since when, and what would flip it?" | `graph_epistemic(action="status", node_id=...)` | `epistemic_status` acceptance capstone (opt-in `epistemic-tms` engine feature) |
| "What changed between two audit periods?" | `graph_epistemic(action="what_changed", tx_from=..., tx_to=...)` | whole-graph bitemporal diff (opt-in `epistemic-tms`) |
| "As of last quarter-end, what did the graph say?" | `graph_query(cypher=..., as_of="2026-03-31T00:00:00Z")` | bitemporal `as_of` cutoff on the read path |

All of these are also plain MCP tools you can call from any client that
speaks MCP, and each has a REST twin (`POST /audit`, `POST /compliance`,
`POST /epistemic` on the graph-os gateway — the generic REST-twin factory in
`kg_server._build_server` mounts one for every entry in `ACTION_TOOL_ROUTES`
without a bespoke handler).

---

## 1. Verify the audit ledger hasn't been tampered with

Every durable mutation to the graph already chains into a per-graph SHA-256
hash chain (`epistemic-graph/src/audit.rs`): each entry binds the previous
entry's hash, so altering, reordering, deleting, or inserting any entry
breaks the chain at that exact position. `graph_audit(action="verify")`
walks the chain and reports the first break, if any.

```jsonc
// Call
graph_audit(action="verify")

// Response shape (AuditReport, eg-types/src/protocol.rs):
{
  "surface": "audit",
  "action": "verify",
  "available": true,
  "graph": "__commons__",
  "ok": true,
  "entries": 4218,
  "first_broken_seq": null,
  "detail": "chain verified: 4218 entries, no breaks"
}
```

If the chain has been altered, `ok` is `false` and `first_broken_seq` names
the exact sequence number where verification first failed — the entry at
that `seq` (and everything after it) is suspect.

**Degrade contract.** `AuditVerify` requires the engine to be built with the
`security` cargo feature (part of the default `full` build) **and** a durable
redb persist dir configured (`GRAPH_SERVICE_PERSIST_DIR` /
`--persist-dir`). An in-memory-only engine, or a slim build without
`security`, returns a clean error instead of raising:

```jsonc
{
  "surface": "audit",
  "action": "verify",
  "available": false,
  "error": "audit ledger not exposed by this engine build/config (...). Requires the epistemic-graph `security` cargo feature (part of the default `full` build) AND a durable redb persist dir configured — otherwise this is a corresponding epistemic-graph-side gap, not an agent-utilities one."
}
```

If you're building toward an audit-defensible deployment, confirm this
returns `available: true` before relying on the chain for anything — the
[Enterprise](../recipes/enterprise.md) and [Single-node prod](../recipes/single-node-prod.md)
recipes both configure a durable persist dir; the zero-infra
[Tiny](../recipes/tiny.md) recipe may not, depending on your `.env`.

## 2. Reconstruct "what happened to entity X"

`graph_audit(action="for_target", target_id="...")` is the entity-anchored
half of the same primitive: every `:ToolCall` that acted on a given entity
id, in call order, reverse-indexed off the KG's own tool-call provenance
(`Orchestrator.get_tool_calls_for_target`), plus a best-effort chain-verify
snapshot alongside it — so you get both "what touched this record" and "is
the ledger recording it still intact" in one call.

```jsonc
graph_audit(action="for_target", target_id="claim:mine:abc123")
```

## 3. One-view compliance posture

`graph_compliance(action="posture")` is a rollup, not new logic: it joins the
same `verify()` report from step 1 with a node-count / status-breakdown of
the governance labels already ingested into the graph by the CISO Assistant
GRC extractor and the TRM portfolio-intelligence engine —
`Control`, `Policy`, `Risk`, `ComplianceRequirement`, `ComplianceGate`,
`Regulation`, `ComplianceAssessment`, `Assessment`, `Incident`,
`RemediationProposal`, `Finding`, `SecurityException`.

```jsonc
graph_compliance(action="posture")

// ->
{
  "surface": "compliance",
  "action": "posture",
  "audit_ledger": { "ok": true, "entries": 4218, "first_broken_seq": null, ... },
  "node_counts": {
    "Control": 214, "Policy": 38, "Risk": 91, "ComplianceRequirement": 156,
    "ComplianceGate": 12, "Regulation": 9, "ComplianceAssessment": 47,
    "Assessment": 47, "Incident": 6, "RemediationProposal": 3,
    "Finding": 22, "SecurityException": 1
  },
  "status_breakdown": {
    "Control": {"implemented": 190, "planned": 24},
    "Incident": {"open": 2, "resolved": 4}
  }
}
```

A read failure on any one label degrades that label's count to `0` rather
than failing the whole rollup — a partial governance mirror still gives you
a usable posture view.

## 4. Redaction-compliant bulk export, as-of a date

Per-node redaction already existed at the engine (`Method::ExplainBelief`'s
`disclosure_level` — `Full` / `Skeleton` / `ExistenceOnly` — *masks*, never
silently drops, an evidence node the caller's row-level security can't see).
What was missing was a bulk "export this subgraph, redacted, for an
auditor" primitive — that's `graph_compliance(action="export")`: it takes
either an explicit id list or a read-only Cypher query selecting an `id`
column, and calls the engine's own `explain_belief(node_id, disclosure_level)`
per id — the exact same per-node redaction primitive `graph_epistemic`'s
`why` action uses, just batched.

```jsonc
// Select by Cypher (any read-only query returning an `id` column) and
// export at Skeleton disclosure, as of a fixed instant, capped at 500 rows:
graph_compliance(
  action="export",
  cypher="MATCH (c:ComplianceAssessment) WHERE c.framework = 'SOC2' RETURN c.id AS id",
  disclosure_level="Skeleton",
  as_of="2026-06-30T23:59:59Z",
  limit=500
)

// ->
{
  "surface": "compliance",
  "action": "export",
  "disclosure_level": "Skeleton",
  "as_of": "2026-06-30T23:59:59Z",
  "requested": 47,
  "exported": 47,
  "truncated": false,
  "entries": [
    {"node_id": "compliance:soc2:cc6.1", "belief": { "root": { "claim": "compliance:soc2:cc6.1", "rule": "DerivedSupport", "confidence": 0.94, "premises": [ ... ] } } },
    ...
  ]
}
```

`disclosure_level` controls what an out-of-scope evidence node in the
justification tree looks like to the recipient of this export:

- **`Full`** (default) — every node in the proof tree, as the caller's own
  RLS view sees it.
- **`Skeleton`** — structure (claim ids, rule kind, confidence) preserved,
  but the *content* of nodes the recipient's policy can't see is masked.
- **`ExistenceOnly`** — the recipient learns a supporting/contradicting node
  exists at all, nothing about its content.

This is the mechanism you want when handing a subgraph to an external
auditor who should see that a control claim is supported, without seeing
the (possibly sensitive) evidence text itself.

## 5. Explainable decision logs — why do we believe this

For a single claim rather than a bulk export, `graph_epistemic` is the
purpose-named wrapper (SKILL: `kg-epistemic-answer`) over the same
justification machinery:

```jsonc
// The justification tree — what supports/contradicts this claim, and how
graph_epistemic(action="why", node_id="claim:mine:abc123")
// -> { "engine_method": "explain_belief",
//      "result": { "root": {
//        "claim": "claim:mine:abc123", "rule": "DerivedSupport",
//        "confidence": 0.82,
//        "premises": [
//          {"claim": "evidence:doc:44", "rule": "Asserted", "confidence": 1.0, "premises": []}
//        ]
//      } } }
```

`rule` is one of `Asserted` / `DerivedSupport` / `DerivedContradiction` /
`BayesianUpdate` — the actual proof-tree vocabulary the engine uses
(`eg-epistemic::model::JustRule`), not a paraphrase.

The **acceptance capstone** goes one step further — "do we still believe
this, since when, on what evidence, and what would flip it" — via
`epistemic_status` (opt-in engine feature `epistemic-tms`, **not** in the
default `full` build; check for a clean `{"error": ...}` degrade before
assuming it ran):

```jsonc
graph_epistemic(action="status", node_id="claim:mine:abc123")
// -> { "engine_method": "epistemic_status",
//      "result": {
//        "claim": "claim:mine:abc123",
//        "believed": true,
//        "confidence": 0.82,
//        "uncertainty": 0.05,
//        "proof": { "root": { "rule": "DerivedSupport", ... } },
//        "why_not": null,
//        "evidence": ["evidence:doc:44"],
//        "valid_time": [1751328000, null],
//        "tx_time": [1751328042, null],
//        "what_would_invalidate": { ... }
//      } }
```

And for "what changed between two audit periods" rather than one claim's
history, `what_changed` (same opt-in feature) gives a whole-graph bitemporal
diff between two transaction-time bounds:

```jsonc
graph_epistemic(action="what_changed", tx_from=1748736000, tx_to=1751328000)
```

## 6. As-of reads for point-in-time audit snapshots

Every read that goes through `graph_query`/`graph_ask` accepts an `as_of`
ISO-8601 instant — a bitemporal cutoff (`valid_from <= as_of < valid_to`) —
so you can reconstruct exactly what the graph asserted on a given date,
independent of what's been written since:

```jsonc
graph_query(
  cypher="MATCH (c:Control {framework:'SOC2'}) RETURN c.id AS id, c.status AS status",
  as_of="2026-03-31T00:00:00Z"
)
```

Add `include_epistemic=true` to get each row back as a full `EpistemicRow`
(confidence, bitemporal window, evidence refs, policy labels) in the same
call, instead of a bare row plus a second `explain_belief` round-trip.

---

## Putting it together: a SOC2 evidence-collection pass

1. **Confirm the ledger is trustworthy first.** `graph_audit(action="verify")`
   → `ok: true`. If this returns `available: false`, your engine build/config
   doesn't carry a durable audit trail yet — treat that as a finding, not a
   tool bug.
2. **Pull the posture rollup.** `graph_compliance(action="posture")` for the
   control-coverage/status snapshot to attach to the audit narrative.
3. **Export the in-scope control evidence, redacted, as of period-end.**
   `graph_compliance(action="export", cypher=..., disclosure_level="Skeleton", as_of=<period-end>)`.
4. **Deep-dive any control an auditor flags.** `graph_epistemic(action="why", node_id=...)`
   for the justification tree; `action="status"` for the acceptance capstone
   if your engine build has `epistemic-tms`.
5. **Answer "what changed since last audit."** `graph_epistemic(action="what_changed", tx_from=<last_audit_tx>, tx_to=<now>)`.

## Honest limitations

- `epistemic_status`/`what_changed` need the opt-in engine feature
  `epistemic-tms` — not folded into the default `full` build. A clean
  `{"error": ...}` means the connected engine doesn't have it, not that the
  claim is unbelieved.
- `disclosure_level` redaction (`Full`/`Skeleton`/`ExistenceOnly`) needs the
  opt-in `epistemic-redaction` feature for the masking behavior; without it,
  `explain_belief` still runs but does not mask.
- `graph_audit(action="verify")` needs the `security` cargo feature
  (default `full` build carries it) **and** a durable redb persist dir —
  an in-memory-only engine has no chain to verify.
- These are read-only diagnostics over what was already written. They
  explain and export existing belief/audit structure; they do not create
  compliance controls or remediate findings themselves — see
  `graph_ops_causal`'s `control_evidence` action and the CISO Assistant
  writeback sink for the write side of GRC.

## See also

- Skill: `kg-epistemic-answer` — the full four-layer epistemic-answer
  surface this guide's §5 draws from, including `explain_provenance_by_ids`
  (currency-upgrading any id list) and `explain_policy` (visible vs
  policy-denied ids for a plan).
- [Configuration Reference & Flag Audit](../architecture/configuration.md) —
  `KG_BRAIN_ENFORCE` (fail-closed node ACLs), `KG_AUTH_REQUIRED` (identity on
  the KG surface) — the enforcement layer this guide's redaction sits on top
  of.
- [Supported Deployment Configurations](deployment-configurations.md) — rung
  (b)/(c) for how to get a durable, identity-gated deployment this guide's
  audit chain can actually attest to.
