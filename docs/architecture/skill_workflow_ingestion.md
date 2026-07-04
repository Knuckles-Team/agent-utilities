# Skill-Workflow → Knowledge-Graph Ingestion

**Concept:** KG-2.97 (skill-workflow ingestion — "Claude drives, graph-os runs").

The `universal-skills` package ships ~300 **skill-workflows** under
`universal_skills/workflows/<domain>/<name>/SKILL.md` — dual-mode artefacts whose
frontmatter + `## Steps` DAG compose *atomic* skills into an ordered/parallel
pipeline. Until KG-2.97 those workflows lived only on disk: a live query showed
~2 `WorkflowDefinition` nodes in the KG vs ~300 workflows on disk, so the
graph-os orchestrator had **nothing to dispatch**. This pipeline closes that gap.

## Why it exists

The "Claude drives, graph-os runs" offload (`kg-delegate` /
`graph_orchestrate action=execute_workflow`) reads `WorkflowDefinition` nodes
from the KG and dispatches them. For that to work, the on-disk workflow corpus
must first be *in* the KG, in the exact node/edge shape the orchestrator reads.
KG-2.97 parses each `SKILL.md` and upserts it as a dispatchable definition.

## What it does

`agent_utilities/knowledge_graph/ingestion/skill_workflow_ingest.py`:

- **Discovers** every `workflows/**/SKILL.md` via the installed `universal_skills`
  package (with a package-path fallback for editable installs that yield an empty
  enable-flag list), or an explicit `root` argument for tests / out-of-tree
  corpora.
- **Parses** frontmatter (`name`, `description`, `domain`, `tags`,
  `team_config.specialist_ids`/`tool_assignments`, `concept`) and the step DAG.
  The step heading dialects are both handled:
  - `### Step N: kebab-skill-name [depends_on: Step M]` — the component *is* the
    atomic skill;
  - `### Step N: Title Case [depends_on: title_slug]` with a `**Agent**` /
    `**Tools**` body — the atomic skill comes from `**Agent**`.
  `depends_on` resolves both numeric (`Step 2` / `2`) and name-based (slug of the
  component/title) references.
- **Upserts** the workflow into the KG (the same shape ORCH-1.22 `WorkflowStore`
  writes and ORCH-1.41 `ProcessPlanCompiler` mirrors).
- Is **idempotent**: deterministic ids upsert in place, and a `content_hash` on
  the definition makes an unchanged re-ingest a no-op (counted under `skipped`).

## Node / edge shape

```
(:WorkflowDefinition {id: "skill_workflow:<name>", name, description, domain,
                      source: "universal-skills", tags_json, specialist_ids_json,
                      nl_spec, step_count, content_hash, source_path,
                      use_count, version})
  -[:HAS_STEP {step_order}]->
(:WorkflowStep {node_id, step_order, component, skill_name, is_parallel,
                timeout, depends_on_json, tools_json, refined_subtask})
  -[:TRANSITION_TO {condition: "on_success"}]->  (:WorkflowStep)   # depends_on
  -[:USES_SKILL]->                               (:Skill {id: "skill:<slug>",
                                                          name})
```

`WorkflowDefinition.name` is the lookup key the orchestrator and
`WorkflowStore.load_workflow` use; `source = "universal-skills"` lets callers
filter the ingested corpus. The `USES_SKILL` edges connect each step to the
atomic `Skill` node it composes (created if absent), making the
workflow↔atomic-skill graph queryable.

## Surfaces

Two surfaces, one action core (the shared `_execute_tool` dispatch):

- **MCP:** `graph_ingest action=skill_workflows` (`target_path` optionally
  overrides the corpus root) — returns
  `{workflows, steps, skill_links, skipped, errors, scanned}`.
- **REST:** `POST /graph/ingest` with `{"action": "skill_workflows"}` — the same
  endpoint/core.

## Execution seam (out of scope)

KG-2.97 covers **ingestion + discoverability + linking + a dispatchable
definition** only. The step-by-step *execution* dispatch
(`execute_workflow` firing each atomic skill — a known-flaky area with 300s
hang risk) is deliberately untouched. The ingested `WorkflowDefinition` lands in
the exact store/shape `execute_workflow` reads, so it **can** be dispatched, but
KG-2.97 does not attempt to fix or drive execution.
