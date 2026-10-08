# AU-BASELINE-001 — Design and implementation plan

Status: BUILDING. Governing spec: [spec.md](spec.md).

## Existing system and reuse

- Graph OS bootstrap calls `engine.start_background_daemons()` on a non-client role after materialization.
  That method already starts the graph writer, the maintenance scheduler and the embedding backfill thread.
- `submit_task` writes durable WorkItems with target deduplication, priority buckets and lane routing.
- `_bg_skill_workflows` ingests a skill corpus root through `ingest_skill_workflows` and `ingest_atomic_skills`.
  Both upsert content-addressed nodes.
- `ingest_prompts_to_graph` ingests the base, fleet and overlay prompts behind a content-hash checkpoint.
- `_bg_codebase` runs the structural enrichment pipeline. It skips unchanged files by content hash.
- `iter_provider_dirs` and the provider generation resolver locate verified skill and ontology roots.
- The connector SDK `agent_connector_sdk.mcp.content` module already builds `ontology://` and `shapes://` resources.

The design adds no ingestion path. It schedules the existing paths and adds one resource registration call.

## Architecture

```text
start_background_daemons (daemon role)
  └─ start_baseline_ingest ── thread KG-Baseline-Ingest
       └─ plan_baseline ── prompts, skills per provider, codebase per repository
            └─ enqueue_baseline ── submit_task (fast first, then medium)
                 ├─ scheduled_job  -> _tick_baseline_prompts -> ingest_prompts_to_graph
                 ├─ skill_workflows -> resolve_skill_corpus_root -> skill ingest
                 └─ codebase       -> structural enrichment pipeline
KG-Embedding-Backfill thread (existing) ── slow_heavy vector work
```

Failure modes:

- A missing workspace manifest yields no codebase items. Prompt and skill items still enqueue.
- A target outside the configured workspace is rejected by `submit_task` and reported per item.
- An absent skill provider fails only its own WorkItem with `LookupError`.
- The prompt tick runs its coroutine on a helper thread. The copied context carries the verified session.

Security boundary: the thread runs under the captured daemon session through `_authorized_background_thread`.
Durable targets hold provider names or workspace-relative paths, never machine roots.

## Interfaces and data model

- Skill target reference: `skill-provider:<name>`. `resolve_skill_corpus_root` maps it at run time.
  The `universal-skills` sentinel keeps its meaning.
- Baseline job ID: `baseline-<boot>-<leg>-<slug>`. The boot token is random per process.
- WorkItem metadata key `baseline`: `{leg, name, queue_class, stage}`.
- Stage table (`semantic_tiers.STAGE_QUEUE_CLASS`): S1 `fast`; S2, S3 `medium`; S4, S5, S6 `slow_heavy`.
- Class priority buckets: `fast` 1, `medium` 2, `slow_heavy` 3.
- Task entry stages: skill, prompt and connector metadata enter at S1. Code and documents enter at S2.
  Card enrichment enters at S4.
- MCP resources: `ontology://<provider>/<file>.ttl` and `shapes://<provider>/<file>.ttl`, MIME type `text/turtle`.

Configuration lives in `AgentConfig`. No module reads the environment directly.

## Live integration path

1. Graph OS bootstrap completes materialization and calls `start_background_daemons` on the daemon role.
2. `start_background_daemons` starts its threads, then calls `start_baseline_ingest`.
3. Task workers claim the WorkItems by lane. The existing handlers write the content.
4. The embedding backfill thread embeds new nodes. The sparse-index ratio then rises above zero.
5. `create_mcp_server` calls `register_ontology_providers` after the skill and prompt providers.

A client-role process never reaches step 1. The serving-role deployment therefore needs a daemon-role process.

## Quality and release gates

- `uvx ruff check` and `uvx ruff format` on changed files.
- `scripts/check_complexity_staged.py`: cyclomatic 10 or less and cognitive 15 or less per function.
- `scripts/check_no_env_sprawl.py`, `scripts/check_swallowed_errors.py`, `scripts/check_event_loop_blocking.py`.
- `scripts/check_dupehound.py --base-ref` and `scripts/check_duplication.py enforce --base-ref`.
- `scripts/docs_contract.py --write --runtime-config-only` regenerates the configuration catalog.
- Focused pytest files listed in [test-spec.md](test-spec.md).

## Risks, alternatives, and decisions

- Decision: reuse the durable queue instead of a new daemon. The task surface then reports progress for free.
- Decision: keep the EG `SemanticIndex` queue out of scope. It binds SQL sources, not graph content.
- Alternative rejected: `source_sync source=all` at boot. It copies external systems into the graph.
- Risk: a large workspace scope saturates the ingestion lane. The cap and the background priority bound it.
- Risk: the SDK helper ships in a later SDK release. Until then registration logs one warning and serves none.
