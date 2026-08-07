# Design Document: Dispatch writes through the typed engine API, never raw Cypher, so the native backend's narrower write grammar can't silently fail a call

CONCEPT:AU-KG.query.register-each-user-table

> `scripts/security/check_cypher_write_subset.py:1-40` (module docstring —
> the gate that keeps this decision enforced).

## Decision — every write goes through `IntelligenceGraphEngine.link_nodes`/`add_node`/`add_edge`, never a raw multi-MATCH/edge-MERGE `.execute()`/`.execute_write()` call

The epistemic-graph engine's NATIVE Cypher implementation
(`eg-query/src/cypher/parser.rs`) supports only a narrow WRITE subset: at
most one leading `MATCH`, `MERGE` limited to a single bare node (never an
edge pattern `MERGE (a)-[:REL]->(b)`, regardless of how many `MATCH` clauses
precede it), no `WITH` between a `MATCH` and a write clause, and `SET`
values restricted to literals (no `SET n += $x` map-merge). A query using
any of these shapes sent to a NATIVE `EpistemicGraphBackend`
(`typed_mutation_support == "native"`) does not silently do less than
intended — it raises at the wire boundary
(`EpistemicGraphBackend.execute_write` -> `CypherEngineError`).

~24 call sites across the codebase were found using shapes outside this
subset and fixed by hand: dispatch through the typed engine API
(`IntelligenceGraphEngine.link_nodes`/`add_node`/`add_edge`) instead of a
raw `.execute()`/`.execute_write()` call with hand-written Cypher — the same
`_upsert_edge`/`_upsert_node` convention 4 earlier sites already used. The
typed API is expressed entirely in terms the native subset supports by
construction, so a call written against it cannot regress into the
unsupported grammar the way hand-written Cypher can.

**The rejected alternative is what was actually in place before the fix**:
hand-written Cypher strings that happened to work against whichever backend
was tested at the time, with no mechanism stopping a new site from being
added in the unsupported shape. `scripts/security/check_cypher_write_subset.py`
is the gate that keeps the ~24-site fix from silently regressing — it is
explicitly a follow-on to the dispatch decision, not a substitute for it: it
catches new violations, it does not fix existing ones.

**Not every rich multi-MATCH/edge-MERGE query is treated as a violation** —
a backend targeting a genuinely different store (e.g. `LadybugBackend`,
which hands the query to Kuzu's full openCypher engine) may legitimately
emit the fuller grammar; such sites carry an inline
`# cypher-write-subset-allow: <reason>` pragma rather than being rewritten
to the typed API, since the constraint they are exempt from is native-engine-
specific, not universal.

## Risk Assessment

- **Blast Radius**: every call site that performs a graph write — the ~24
  originally-fixed sites plus any future write path; enforced by
  `scripts/security/check_cypher_write_subset.py` at merge-queue gate time
  (a `scripts/security/check_*.py`, discovered automatically, per
  `AU-OS.governance.tiered-merge-gate`).
- **Backward Compatible**: Yes for callers already using the typed API;
  breaking (by design, via the gate) for any new raw-Cypher write outside
  the native subset.
- **Known weak point**: the pragma escape hatch
  (`# cypher-write-subset-allow: <reason>`) is a textual, per-call-site
  opt-out — it trusts the annotation is accurate rather than verifying the
  backend at that call site is genuinely non-native.
