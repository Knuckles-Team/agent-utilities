# Design Document: How a codebase ingest TASK is scoped before it runs — split by SIZE, or scope by DIRTY-CHECKOUT identity

CONCEPT:AU-KG.ingest.big-repo-structural-split · CONCEPT:AU-KG.ingest.subtask-routing-key · CONCEPT:AU-KG.ingest.code-source-partition · CONCEPT:AU-KG.ingest.agent-utilities-checkout

> `agent_utilities/knowledge_graph/ingestion/repo_split.py` (the split planner),
> `agent_utilities/knowledge_graph/core/engine_tasks.py`
> (`_maybe_fanout_codebase`, the admission-time decision), `agent_utilities/
> knowledge_graph/ingestion/engine.py` (routing-key application + the
> per-repo source id), `agent_utilities/knowledge_graph/assimilation/
> breadth_ingest.py` (`_is_self_repo` / `_git_modified_source_files` /
> `_default_codebase_ingest`, the dirty-checkout scoping), pinned by
> `tests/unit/knowledge_graph/test_ingest_tail_optimization.py` and
> `tests/unit/knowledge_graph/test_breadth_preskip.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.ingest.over-same-tree-fan` / `AU-KG.ingest.deterministic-classifier` | classifies a repo's files BY TYPE (code/skill/doc/…) — orthogonal axis to this document's by-SIZE / by-DIRTY-STATE scoping | 0.35 | KG |
| `AU-KG.ingest.self-tool-surface` | graph-os's own MCP tool-surface ingestion — shares `engine.py` module locality only | 0.20 | KG |

### Extension Analysis

- **Primary Extension Point**: `engine_tasks._maybe_fanout_codebase`'s guard
  clause (`engine_tasks.py:3137-3139`) is the single admission gate both
  decisions below flow through — it is where "is this task already scoped"
  gets decided before either the split or the self-checkout path can apply.
- **Extension Strategy**: augment — a third scoping trigger would be a new
  guard clause here, not a new task type.
- **New Concept Required?**: No — this document is the home for the four
  markers above.

## Decision 1 — a repo above the file-count threshold is split into K shard-routed sub-tasks that commit in PARALLEL, not one task on one shard writer

`CONCEPT:AU-KG.ingest.big-repo-structural-split` · `CONCEPT:AU-KG.ingest.subtask-routing-key` · `CONCEPT:AU-KG.ingest.code-source-partition`

The tail problem, named with live numbers in `repo_split.py:4-10`: one huge
repo (agent-utilities, epistemic-graph — thousands of files) is one
`codebase` WorkItem → one per-repo graph → one redb shard writer, so its
structural write *serializes* on one worker for minutes (p50=36s but
p95=650s/max=797s) while the other K-1 shard writers sit idle. `engine_tasks.
_maybe_fanout_codebase` (`engine_tasks.py:3116-3167`) is the admission-time
fix: for a repo over `SPLIT_MIN_FILES` (1200, `repo_split.py:44`),
`plan_repo_split` (`repo_split.py:79-127`) partitions its files into `k`
balanced buckets — grouped by a coarse path prefix, deepened only until
there are enough groups to bin-pack evenly (`_choose_depth`,
`repo_split.py:69-76`), then LPT-scheduled largest-group-first into the
emptiest bucket for a deterministic, reproducible assignment
(`repo_split.py:116-124`). Each bucket becomes its own sub-task carrying
`only_files` + a `route_repo` of `<repo>__s<i>` (`engine_tasks.py:3170-3176`,
`engine.py:1805-1811`), so the K buckets hash to K *different* shard writers
and commit concurrently (`subtask-routing-key`, `engine_tasks.py:3121-3126`).

**The rejected alternative is splitting by an arbitrary/fine-grained key (per
file, or per directory regardless of size)**: the docstring is explicit that
grouping stays COARSE, "deepened only until there are enough groups to
balance `k` buckets, capped at `_MAX_SPLIT_DEPTH`" (`repo_split.py:17-19`,
46-49), because keeping each sub-package's files together in one bucket
preserves *intra*-package cross-file CALL/INHERIT resolution — only calls
that cross a bucket boundary go unresolved, "a bounded, coarse-grained
tradeoff" the module accepts deliberately (`repo_split.py:20-23`). A
per-file split would have maximized fan-out at the cost of resolving almost
no cross-file edges at all.

**A second rejected alternative is leaving the K physical shards as K
separate logical sources.** `code-source-partition` closes that gap: every
code node any shard writes is stamped `source_system = code:<repo>` —
**keyed on the repo name, not the per-shard routing key** — so all shards of
one repo share ONE `urn:source:code:<repo>` source partition
(`engine.py:1856-1861`, `2083-2084`; the buffering layer that actually stamps
it, `enrichment/pipeline.py:174-182`, 362-364). Splitting the WRITE path
(for parallel commit throughput) without also anchoring a shared source id
would have made a big repo look like K unrelated repos to any source-scoped
query — the split is invisible to everything downstream except the shard
writers themselves.

Guards keep the median and the recursion safe: a sub-task
(`route_repo`/`split_child` set) never re-splits, an explicitly-scoped task
(`only_files` already set — see Decision 2) is left as-is, and small/medium
repos fall straight through to the unchanged inline path
(`engine_tasks.py:3129-3139`).

## Decision 2 — the agent-utilities self-checkout is scoped to its git-status-modified files, not forced through a full re-walk every tick

`CONCEPT:AU-KG.ingest.agent-utilities-checkout`

The always-on breadth-ingest loop re-submits every tracked repo every few
minutes. For a clean repo the engine's `codebase_git` watermark makes an
unchanged HEAD a no-op. But `agent-utilities` is "the one box that is
routinely DIRTY (active development)" (`breadth_ingest.py:315-320`) — active
development on the very repo doing the ingesting — which disables the
engine's clean-tree git-diff delta and forces a FULL tree re-walk every tick
(`breadth_ingest.py:381-384`). `_is_self_repo` detects this one case (the
presence of `agent_utilities/__init__.py` at the root,
`breadth_ingest.py:314-322`) and `_git_modified_source_files` parses `git
status --porcelain` to collect just the added/modified/untracked source
files (`breadth_ingest.py:325-366`); `_default_codebase_ingest` then submits
the task with `only_files` set to exactly that list
(`breadth_ingest.py:377-387`), which the engine honors verbatim ahead of its
own git-diff path (`engine.py:1900-1909`) and — critically — does NOT use to
advance the whole-repo git-sha watermark afterward, since a scoped partial
ingest didn't cover the whole tree at that HEAD
(`engine.py:1947-1951`); the per-file content-hash skip remains the
correctness backstop instead.

**The rejected alternative is forcing the routinely-dirty self-repo through
the same full re-walk every other repo gets on a dirty tree.** That is the
default behavior for every OTHER repo, and it is accepted there because most
repos are dirty only rarely; for the one repo that is dirty essentially
always, the same default would mean the breadth loop re-stats and re-hashes
the entire agent-utilities tree on every tick indefinitely. Scoping to
`only_files` is what makes the breadth tick cheap specifically for the
box that can never rely on the clean-tree fast path.

**A second rejected alternative — generalizing dirty-tree scoping to every
repo** — was not taken. The mechanism is deliberately special-cased to the
self-repo (`_is_self_repo`) rather than built as a general "any dirty repo
gets `only_files`" rule; the comment frames it as "the case worth scoping"
(`breadth_ingest.py:317`), not the general case. This keeps the git-diff /
full-walk semantics that every other tracked repo relies on unchanged.

`test_breadth_preskip.py`'s `test_dirty_self_repo_scopes_to_modified_files`
pins the behavior end to end: a synthetic checkout with an
`agent_utilities/__init__.py` marker submits a task scoped to only its
modified files.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/repo_split.py`,
  `agent_utilities/knowledge_graph/core/engine_tasks.py`
  (`_maybe_fanout_codebase`), `agent_utilities/knowledge_graph/ingestion/
  engine.py` (`_run_codebase_structural`, `_route_classified_artifacts`),
  `agent_utilities/knowledge_graph/assimilation/breadth_ingest.py`.
- **Backward Compatible**: Yes — both paths are opt-in gates (file-count
  threshold; `_is_self_repo` detection) that fall through to the prior
  whole-repo behavior when they don't fire.
- **Known weak point**: Decision 1 and Decision 2 share the same `only_files`
  field and are mutually exclusive by construction
  (`_maybe_fanout_codebase` bails out if `only_files` is already set,
  `engine_tasks.py:3138`) — a future third caller of `only_files` on a
  `codebase` task would silently also suppress the big-repo split for that
  task, with no assertion or log line calling that interaction out.
