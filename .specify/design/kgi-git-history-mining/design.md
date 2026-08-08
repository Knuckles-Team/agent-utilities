# Design Document: A normal codebase ingest ALSO mines commit history as first-class graph data, not just current-state structure

CONCEPT:AU-KG.ingest.mine-git-history-files · CONCEPT:AU-KG.ingest.normal-codebase-ingest-also

> `agent_utilities/knowledge_graph/enrichment/git_history.py` (commit/author/
> file mining, the primary module), `agent_utilities/knowledge_graph/
> enrichment/git_coupling.py` (change-coupling, folded in as one of
> `git_history`'s edge types), `agent_utilities/knowledge_graph/ingestion/
> engine.py:1966-2011` (the call site inside `_run_codebase_structural`),
> pinned by `tests/unit/knowledge_graph/enrichment/test_git_history.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.ingest.over-same-tree-fan` / `AU-KG.ingest.deterministic-classifier` | classifies the CURRENT file tree by type — this document mines the repo's TEMPORAL dimension instead (commits, not files-as-they-are) | 0.30 | KG |
| `AU-KG.ingest.code-source-partition` | the `file:<path>` id namespace this document's `TOUCHED`/`FILE_CHANGES_WITH` edges attach to, so history and structure are one graph | 0.30 | KG |

### Extension Analysis

- **Primary Extension Point**: `ingest_commit_history`'s return dict
  (`git_history.py`) and the `TOUCHED`/`AUTHORED`/`PARENT`/
  `FILE_CHANGES_WITH` edge types it emits.
- **Extension Strategy**: augment — a new evolution signal (e.g. per-file
  age-at-last-change) is a new field on the existing commit/file walk, not a
  new ingestion phase.
- **New Concept Required?**: No — this document is the home for both markers.

## Decision — commit history is mined as first-class `:Commit`/`:Author`/`:File` graph data on every codebase ingest, not left to external visualization tools

`CONCEPT:AU-KG.ingest.mine-git-history-files` · `CONCEPT:AU-KG.ingest.normal-codebase-ingest-also`

`git_history.py:1-7` states the thesis directly: "A repo's commit history *is*
a graph: commits → authors → files, evolving over time — which fits the KG
natively. Tools like Gource / SourceTree only *render* that evolution; we
INGEST it as first-class graph data" so who-owns-what, change-coupling,
churn hotspots, per-file timelines and blast-radius become free native
queries instead of something only a separate visualization tool can show.
`_run_codebase_structural` runs this immediately after the structural code
pass and the git-delta/`only_files` decision, gated only by
`commit_history` metadata (default on) and a real `head_sha`
(`engine.py:1975-1979`): "a normal codebase ingest ALSO ingests the repo's
evolution... so codebase evolution is a free native KG query... exceeding
Gource/SourceTree, which only render it" (`engine.py:1966-1973`). It is
best-effort and never breaks the structural ingest that already succeeded
(`engine.py:2010-2011`).

**The rejected alternative is a separate, opt-in "history import" tool**
(what Gource/SourceTree effectively are) that a user runs standalone and
that renders rather than ingests. That alternative would leave
change-coupling and ownership permanently unreachable to the KG's own
query/reasoning layer — exactly the gap `git_history.py`'s module docstring
calls out. Instead, mining rides the SAME codebase ingest every repo already
gets, linked to the SAME `file:<path>` ids the structural pass and
`code-source-partition` use, "so history and structure are ONE graph"
(`git_history.py:19-21`).

**A second, narrower rejected alternative — a subprocess-per-commit walk —
is called out explicitly in the design notes** (`git_history.py:9-13`): "One
`git log --numstat` pass..., NOT a subprocess per commit, the Gource-slow
way." A per-commit subprocess is the naive translation of "mine every
commit" and would not scale to a repo with thousands of commits; the chosen
design streams and parses ONE `git log` invocation's machine-parseable
output in memory and batch-writes through the engine's bulk path.
Auto-bounding (`DEFAULT_MAX_COMMITS = 5000`, `--since` support,
`git_history.py:46-48`) keeps even a busy repo's ingest sub-second, and
delta/idempotency (commits already in the KG by sha are skipped,
`git_history.py:16-17`) makes a no-change re-ingest a no-op.

Change-coupling — `git_coupling.py`'s `FILE_CHANGES_WITH` edges — is folded
into this same mining pass rather than being a separate concept: two files
that keep changing in the same commits are coupled "even when nothing in the
AST connects them — a hidden dependency the call graph can't see"
(`git_coupling.py:3-4`). It reuses `git_history`'s own commit walk
(`git_history.py:42`, `parse_change_coupling`) instead of a second `git log`
pass, with noise filtered by a minimum co-change support count
(`DEFAULT_MIN_SUPPORT = 3`) and bulk-reformat/vendoring commits excluded via
a per-commit file-count cap (`_MAX_FILES_PER_COMMIT = 50`,
`git_coupling.py:17-21`).

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/enrichment/git_history.py`,
  `agent_utilities/knowledge_graph/enrichment/git_coupling.py`,
  `agent_utilities/knowledge_graph/ingestion/engine.py`
  (`_run_codebase_structural`).
- **Backward Compatible**: Yes — `metadata["commit_history"]=False` opts a
  caller out; any failure is caught and logged without failing the
  structural ingest (`engine.py:2010-2011`).
- **Known weak point**: `DEFAULT_MAX_COMMITS = 5000` silently truncates
  history mining for a repo with a longer log unless a caller raises it or
  sets `--since`; the truncation is logged but nothing surfaces it as a gap
  to a KG consumer querying "full" history on such a repo.
