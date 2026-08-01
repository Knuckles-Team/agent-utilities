# Design Document: A cohort/target-scoped pass fetches only the ids it already knows, and skips whole-graph sub-stages entirely

CONCEPT:AU-KG.ingest.fetch-only-requested-ids

> `agent_utilities/knowledge_graph/assimilation/gap_analysis.py:118-135`
> (the `_node_data_by_id` per-id fetch), `agent_utilities/knowledge_graph/assimilation/synergy.py:75-89`
> (`restrict_to`-scoped feature-node fetch), `agent_utilities/knowledge_graph/assimilation/concept_matcher.py:212-219`
> (HNSW recall instead of a full node scan), `agent_utilities/knowledge_graph/research/cohort.py:257-270`
> (`cohort_source_ids` — recovering a cohort's node set from task provenance,
> never from a graph scan), `agent_utilities/knowledge_graph/research/loop_controller.py:729-765`
> (skipping WHOLE-GRAPH sub-stages — dedup, embed-backfill — for a scoped
> pass), `agent_utilities/automation/research_pipeline.py:770-777`
> (per-id membership probe over `has_node`, not `in graph.nodes`).

## Decision — when the assimilation/research pipeline already knows which node ids it needs (a cohort's members, an explicit membership check), it fetches/iterates only those ids, and a scoped pass skips whole-graph sub-operations rather than running them narrowed

Six sites across the assimilation and research-cohort pipeline apply the
same rule, stated most fully in `gap_analysis.py:118-127`: fetch **"ONE
node's full data by id without a whole-graph pull... The live
`GraphComputeEngine` facade exposes a per-id properties fetch
(`get_node_properties`/`_get_node_properties`) that does a single engine
round-trip — NOT a `GetNodes` whole-graph list, which on a large multi-tenant
engine returns a huge payload and resets the socket."**

**The rejected alternative, named explicitly at every site, is the
whole-graph pull**: `graph.nodes(data=True)` or an unfiltered node scan to
find or check specific ids. `research_pipeline.py:770-777` names it most
concretely for the membership case: `article_id in graph.nodes` "materializes
the whole node list (a gigabyte-scale payload on the live multi-tenant
engine that resets the connection)" — so `has_node()` (a per-id round-trip)
is used instead.

**The decision goes beyond single-node fetch to whole PIPELINE STAGES.**
`synergy.py:80-89` scopes feature-node fetch to a `restrict_to` id set when
one is supplied, calling this out as what "avoids the whole-graph node pull
that makes per-cohort synthesis O(graph) not O(cohort)." `cohort.py:264-270`
goes a step further: a cohort's source-node set is recovered from **durable
task provenance** (each member task records the node it created) rather than
by scanning the graph at all — "this is exactly what scopes the matrix to
the cohort... instead of the whole 15k-feature graph."
`loop_controller.py:729-765` then applies the same principle at the stage
level: `dedup_features` (a whole-graph SUPERSEDES-clustering op) and the
embed-backfill pass (iterates every node to find vectorless concepts) are
both **skipped entirely** — not narrowed, not run scoped — for a `restrict_to`
cohort pass, because "the registry is embedded ecosystem-wide once... a
cohort finalize must not re-scan the whole graph (which resets the socket at
scale)." `concept_matcher.py:212-219` applies the same idea to vector
search: candidate recall goes through the engine's HNSW index via
`semantic_search` rather than an in-memory scan when the index is empty.

**Why this needed its own family of call sites rather than reusing the
general `AU-KG.ingest.never-scan-whole-graph` helper** (`bounded_read.py`):
the assimilation/research-cohort call sites are scoped by an explicit,
already-known *id set* (`restrict_to`, task provenance, a membership check)
rather than by *node type* (which is what `bounded_read.iter_nodes_by_types`
scopes by). Both decisions attack the same root failure mode — a
gigabyte-scale whole-graph payload resetting the connection on the live
multi-tenant engine — but this one applies specifically to id-scoped
cohort/target reads and to skipping whole-graph pipeline stages outright,
which `bounded_read.py`'s type-iteration helper does not address.

## Risk Assessment

- **Blast Radius**: `assimilation/gap_analysis.py`, `assimilation/synergy.py`,
  `assimilation/concept_matcher.py`, `research/cohort.py`,
  `research/loop_controller.py`, `automation/research_pipeline.py`.
- **Backward Compatible**: Yes — the ecosystem-wide (non-scoped) pass is
  unaffected; scoping only activates when a `restrict_to`/cohort context is
  supplied.
- **Breaking Changes**: None.
- **Known weak point**: the per-id fetch and the local `_node_data_by_id`
  helper in `gap_analysis.py` are NOT the same function as
  `bounded_read.get_node_data` (`AU-KG.ingest.never-scan-whole-graph`) even
  though their docstrings describe the identical rationale nearly verbatim —
  this is a duplicated implementation of the same principle in two modules
  rather than one shared helper, so a future fix to one (e.g. a new label
  casing, a new facade method name) is not guaranteed to reach the other.
