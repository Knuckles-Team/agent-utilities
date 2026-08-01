# Design Document: The vector/hybrid retrieval path enforces the same session ACL as raw Cypher, opt-in and fail-closed once supplied

CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval

> `agent_utilities/knowledge_graph/orchestration/engine_query.py:711-800`
> (`QueryMixin.search` / `search_hybrid`), surfaced through
> `agent_utilities/mcp/tools/analysis_tools.py:1382`,
> `agent_utilities/mcp/tools/query_tools.py:1224`, and measured by
> `agent_utilities/observability/gateway_metrics.py:610`.

## Decision — `session` is a no-op unless supplied, but once supplied it is fail-closed, not fail-open

`engine_query.py:711` states the gap directly: `query_cypher` already applies a
per-node ACL + owner/scope + audit boundary via an explicit
`GraphSession`, but the vector/hybrid path (`search`/`search_hybrid`) "previously
returned every ranked node completely unfiltered even when the caller held a
verified session, unlike the guarded Cypher path." Two properties define the fix:

1. **Backward compatible by construction.** `session=None` (the default) is an
   exact no-op — the prior, unfiltered behavior — so the many existing
   internal/test callers with no backing ACL infrastructure are unaffected.
2. **Fail-closed once opted in.** When a session IS supplied, enforcement never
   silently no-ops: "an infrastructure failure raises `PermissionError` rather
   than falling back to unfiltered results." The served `graph_search` MCP tool
   passes its ambient session explicitly, so the externally-reachable surface is
   governed by default with no operator configuration.

**The rejected alternative** is symmetric with every other ACL gap in this
codebase and is named by omission rather than argued for: silently degrading to
unfiltered results when the session/ACL check itself fails (a missing index, a
backend hiccup) is exactly the failure mode this decision refuses. A retrieval
path that "fails open" on infrastructure trouble is indistinguishable, from a
security standpoint, from having no ACL at all — it only holds under the
happy path, which is when it matters least. The explicit `PermissionError` on
infrastructure failure trades availability for correctness at the one moment
availability would otherwise mask a security regression.

A second, narrower rejected alternative: applying the ACL filter as a
post-hoc row-drop after ranking (filter the top-k) rather than scoping the
retrieval itself. The chosen approach reuses the *same* per-node ACL + owner/
scope boundary `query_cypher` already applies, so the vector path and the
Cypher path share one enforcement primitive instead of growing two ACL
implementations that could drift.

## Risk Assessment

- **Blast Radius**: `engine_query.py` (`search`/`search_hybrid`), the served
  `graph_search` MCP tool, `analysis_tools.py`, `query_tools.py`,
  `gateway_metrics.py` (enforcement is now a measured metric, not just a code
  path).
- **Backward Compatible**: Yes — `session=None` is a byte-for-byte no-op.
- **Known weak point**: enforcement is opt-in per call site. A future retrieval
  entrypoint that forgets to thread `session` through silently reproduces the
  original unfiltered-vector-path gap; nothing mechanically forces every new
  caller to pass a session, only convention (the served MCP tool does it, but a
  new internal caller could omit it without a governance gate catching it, per
  `tests/unit/knowledge_graph/orchestration/test_engine_query_acl_wiring.py`
  existing to catch regressions on the ones that DO wire it).
