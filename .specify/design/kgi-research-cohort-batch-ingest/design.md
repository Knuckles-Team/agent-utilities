# Design Document: A research cohort is one self-polling barrier over N papers + M repos, not N+M independent ingests

> `agent_utilities/mcp/tools/write_ingest_tools.py:786-800` (the `cohort_create`
> action), `agent_utilities/knowledge_graph/research/cohort.py` (`create_cohort`).

CONCEPT:AU-KG.ingest.batch-research-cohort

## Decision — batch-ingest a cohort behind one barrier that synthesizes a comparative matrix on completion

`write_ingest_tools.py:786-796`.

**The problem**: comparative research evolution (KG-2.173's feature/innovation
matrix) needs to compare N papers against M repos as a SET — e.g. "how do
these 3 papers' claimed innovations map onto these 2 competing
implementations" — which requires every member to have finished draining
before the comparison is meaningful. Ingesting each paper/repo independently
gives no natural signal for "the whole set is now ready to compare."

**The rejected alternative**: the caller polls each of the N+M independent
ingest jobs itself and manually triggers the comparative-matrix synthesis
once it observes all of them complete — pushing barrier/completion-detection
logic out to every caller instead of owning it once.

**The design chosen**: `cohort_create` (`base_path` = JSON list of paper
URLs/ids, `target_path` = JSON list of repo paths, `description` = the
comparison goal) creates ONE cohort whose members are all batch-ingested,
with a SELF-POLLING BARRIER that automatically synthesizes the comparative
feature/innovation matrix (KG-2.173) the moment every member has drained —
the caller doesn't poll each member; it polls the cohort via `cohort_status`
(`job_id=cohort_id`), which returns per-member progress AND the matrix counts
once ready.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/write_ingest_tools.py`
  (`cohort_create`/`cohort_status` actions),
  `agent_utilities/knowledge_graph/research/cohort.py` (`create_cohort`).
- **Backward Compatible**: Yes — cohorts are an additive grouping over the
  same underlying per-paper/per-repo ingest paths; ingesting a paper or repo
  outside a cohort is unaffected.
- **Breaking Changes**: None.
- **Known weak point**: the barrier is all-or-nothing on "every member
  drains" — a single permanently-stuck or failing member (a paper URL that
  404s, a repo path that doesn't exist) means the comparative matrix never
  synthesizes for the whole cohort, with no partial-matrix fallback for the
  members that DID complete.
