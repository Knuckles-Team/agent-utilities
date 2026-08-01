# Design Document: ~300 skill-workflow SKILL.md files become dispatchable WorkflowDefinition DAGs, in the exact shape the orchestrator already reads

> `agent_utilities/knowledge_graph/ingestion/skill_workflow_ingest.py` (the
> parser/ingest module); `agent_utilities/mcp/tools/write_ingest_tools.py:964-985`
> (the `skill_workflows` MCP action that fires it as a background job).

CONCEPT:AU-KG.ingest.skill-workflow-corpus ·
CONCEPT:AU-KG.ingest.skill-workflow-ingestion

## Decision — parse the on-disk skill-workflow corpus into the SAME `WorkflowDefinition`/`WorkflowStep` shape `execute_workflow` already reads, as a background job

`skill_workflow_ingest.py:1-15`.

**The problem, quantified in the module docstring**: `universal_skills/<domain>-workflows/<name>/SKILL.md`
files are dual-mode artefacts — YAML frontmatter plus a body `## Steps`
section whose `### Step N: <component> [depends_on: ...]` headings encode a
machine DAG — but "until now those workflows lived only on disk": a live
query showed ~2 `WorkflowDefinition` nodes versus ~300 workflows on disk, so
`graph_orchestrate execute_workflow` had almost nothing to dispatch even
though the corpus existed.

**The rejected alternative**: a bespoke, workflow-specific dispatch
mechanism that reads `SKILL.md` files directly at execution time rather than
materializing them into the graph first. That would mean `execute_workflow`
carries two code paths — one for graph-resident `WorkflowDefinition`
nodes, one for on-disk `SKILL.md` — doubling the discovery/execution
surface.

**The design chosen**: `skill_workflow_ingest` parses each `SKILL.md` and
lands a `WorkflowDefinition` (+ `WorkflowStep` DAG with `depends_on` edges +
`Skill`/`USES_SKILL` links) in the EXACT shape `execute_workflow` already
reads — so `graph_orchestrate execute_workflow` can discover and fire any of
the ~300 workflows with zero new discovery/execution code. Idempotent:
content-addressed re-ingest of an unchanged `SKILL.md` is a no-op.

### Pointer — `CONCEPT:AU-KG.ingest.skill-workflow-corpus`

`write_ingest_tools.py:964-976`. This is the MCP tool-surface entrypoint:
the `skill_workflows` action on `graph_write`/ingest tool. **The rejected
alternative here specifically**: running the full-corpus ingest synchronously
on the MCP call path. The docstring is explicit about why that fails:
"durable per-node writes for the full corpus (~315 workflows) take ~150s —
over the MCP call ceiling — and the backend can't bulk-write durably here."
Instead the action enqueues a BACKGROUND job (run by the task worker, off
the request path) and returns a `job_id` immediately; the caller polls with
`action=job_status job_id=<id>`. `target_path` optionally overrides the
corpus root; default is the installed `universal_skills` package.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/skill_workflow_ingest.py`,
  `agent_utilities/mcp/tools/write_ingest_tools.py` (`skill_workflows` action).
- **Backward Compatible**: Yes — re-ingesting an unchanged corpus is a
  content-addressed no-op; a workflow removed from disk is not automatically
  retracted from the graph (additive-only by default).
- **Breaking Changes**: None.
- **Known weak point**: the DAG parser depends on the `### Step N:
  <component> [depends_on: ...]` heading convention holding exactly — a
  `SKILL.md` author who deviates from that heading shape (a typo in
  `depends_on:`, a renumbered step) fails to parse into a correct DAG, likely
  silently producing a partial or disconnected `WorkflowStep` chain rather
  than a loud validation error at ingest time.
