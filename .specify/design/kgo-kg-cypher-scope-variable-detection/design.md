# Design Document: Detect the query's own bound variable for a tenant/visibility predicate, don't assume the caller wrote `n`

CONCEPT:AU-KG.query.cypher-scope-variable-detection

> `agent_utilities/knowledge_graph/core/cypher_scope_vars.py:1-30`. Fixes
> the divergence D-SH-4 found between `TenancyManager.scope_cypher_query`
> and `tenant_sharing.apply_visibility` (both previously hardcoded `n`) and
> `engine.backend.execute` on the identical, un-scoped query text.

## Decision — a small, dependency-free regex detector for the first node-pattern variable bound by the query's own MATCH clause, replacing two independent hardcoded `"n"` defaults

Two call sites — `TenancyManager.scope_cypher_query` and
`tenant_sharing.apply_visibility` — injected a tenant/visibility predicate
written against a hardcoded `n` variable (`n.tenant_id = '...'`), on the
assumption every read query aliases its primary node `n`, matching the
module docstrings' own examples (`MATCH (n:Entity) RETURN n`). Real callers
do not follow that convention — `MATCH (s:Skill) ... RETURN s` or
`MATCH (w:WorkItem) RETURN count(w) AS c` are both real, observed shapes —
so the injected predicate referenced a variable that does not exist
anywhere in the query. Cypher engines evaluate a reference to an unbound
variable as never matching (not as an error), so the query silently
returned zero rows, or for an aggregate projection, a single zero-count row
— exactly the divergence D-SH-4 found between `engine.query_cypher` (goes
through this injection) and `engine.backend.execute` with the identical
query text run unscoped.

The fix is a shared detector: the first node-pattern variable bound by the
query's own (first) `MATCH`/`OPTIONAL MATCH` clause — exactly the variable a
caller who *did* follow the `n` convention would have used, so callers that
already worked keep working, and callers that used a different alias now
also work. It replaces both independent hardcoded defaults with one
detector, so a third future call site cannot reintroduce the same
divergence by hand.

**The rejected alternative is named directly in the docstring**: a general
Cypher parser. Deliberately not built — the detector only looks at the
node pattern of the query's first `MATCH`/`OPTIONAL MATCH` clause, which is
sufficient for "which variable should the injected predicate reference" and
avoids the cost/risk of parsing arbitrary Cypher for a narrow, specific
question. It is also kept dependency-free (no other project imports) on
purpose, to avoid any import-cycle risk between `company_brain.py` and
`tenant_sharing.py`, the two modules that consume it.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/cypher_scope_vars.py`,
  consumed by `company_brain.TenancyManager.scope_cypher_query` and
  `tenant_sharing.apply_visibility`.
- **Backward Compatible**: Yes — callers using the `n` convention see
  identical behavior; callers using a different alias are now correctly
  scoped instead of silently returning zero rows.
- **Known weak point**: regex-based, not a real parser, and correct by
  construction only for the shapes it recognizes — a fully anonymous first
  pattern (`MATCH () ...`, `MATCH (:Label) ...`) returns `None` rather than
  guessing, and the docstring places the burden on the caller: "fail-closed
  callers should skip scoping rather than reference a fabricated name." A
  caller that does not honor that contract (injects a predicate anyway on
  `None`) would reintroduce the original silent-zero-row failure mode this
  module exists to fix.
