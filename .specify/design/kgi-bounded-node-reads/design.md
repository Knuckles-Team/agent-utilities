# Design Document: One shared helper — bounded per-type/per-id engine reads — replaces whole-graph node scans, and never falls back to a full scan on a legitimately empty result

CONCEPT:AU-KG.ingest.never-scan-whole-graph

> `agent_utilities/knowledge_graph/core/bounded_read.py` (primary — the
> shared `iter_nodes_by_types`/`get_node_data` helpers), `agent_utilities/capabilities/teams.py:314-317`,
> `agent_utilities/knowledge_graph/kb/ingestion.py:349-353,417-421`,
> `agent_utilities/knowledge_graph/research/fleet_relevance.py:157-160`,
> `agent_utilities/knowledge_graph/security/policy_ingestor.py:635-638`,
> `agent_utilities/knowledge_graph/security/rule_ingestor.py:654-657`.

## Decision — a whole-graph `graph.nodes(data=True)` pull is replaced everywhere by a bounded, engine-side per-label/per-id fetch, and a bounded empty result is trusted rather than triggering a full-scan fallback

`bounded_read.py:2-9` states the failure mode this fixes concretely: **"A
whole-graph `graph.nodes(data=True)` on the live multi-tenant engine
materializes EVERY node (166K+ on `__commons__`, with 1024-dim embeddings)
into one MessagePack frame — a gigabyte-scale payload that overloads and
resets the connection."** The fix is to iterate by TYPE through the
engine-side bounded label fetch (`get_nodes_by_label`), which scopes the
wire payload per label instead of dumping the whole graph.

`iter_nodes_by_types` is documented as "the one helper every type-filtered
reader should use" (`bounded_read.py:10`), and it makes a specific,
deliberately non-obvious choice: it uses `get_nodes_by_label` when available
and **"TRUSTS that bounded result — it does NOT fall back to a full scan on
an empty result, because falling back would re-introduce the very 166K-node
pull we are avoiding for a legitimately-empty type"** (`bounded_read.py:16-19`).
It only degrades to a plain `graph.nodes(data=True)` scan when the graph
exposes NO bounded fetch at all — a small in-memory/test/pipeline graph,
where a full pass is cheap and correct (`bounded_read.py:19-20`).

**The rejected alternative is the naive, more defensive-looking design**:
treat an empty bounded-fetch result as ambiguous (maybe the label casing
didn't match, maybe there really are zero nodes) and fall back to a full
scan "just in case." It is rejected explicitly and by name because it would
silently reintroduce the exact gigabyte-payload failure the bounded fetch
exists to avoid, every time a legitimately-empty type is queried — which,
for many of the six call sites (teams, knowledge bases, policies,
engineering rules), is the common case on a graph that hasn't populated that
type yet. `get_node_data` applies the identical rule at single-node
granularity — "a single engine round-trip via the facade's per-id properties
fetch, NEVER a whole-graph scan to find one node" (`bounded_read.py:41-44`).

`bounded_read.py:22-23`'s `_label_casings` helper is a second, smaller
decision bundled in: live labels are inconsistently cased (`article` vs.
`Concept`), so the bounded fetch probes the common casings rather than
requiring every caller to already know the exact stored casing.

Six independent call sites (`teams.py`, `kb/ingestion.py` twice,
`fleet_relevance.py`, `policy_ingestor.py`, `rule_ingestor.py`) converge on
this one helper rather than each reimplementing bounded iteration, each
citing the concept marker at the call site as the reason.

## Risk Assessment

- **Blast Radius**: `core/bounded_read.py` and every reader listed above;
  any future type-filtered reader is expected to adopt the same helper.
- **Backward Compatible**: Yes — the degrade-to-full-scan path preserves
  correctness on small/test graphs.
- **Breaking Changes**: None.
- **Known weak point**: correctness of "trust the bounded empty result"
  depends entirely on the engine-side `get_nodes_by_label` actually being
  reliable for every probed casing — if the engine silently returns empty
  for a label it doesn't recognize (rather than erroring), a genuinely
  populated type under an unprobed casing variant would be read as
  empty with no full-scan safety net, by design.
