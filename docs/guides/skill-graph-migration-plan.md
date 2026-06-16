# Skill-Graph Migration Plan — updating all existing graphs to the KG-driven format

The unified pipeline (CONCEPT:KG-2.7) builds every skill-graph the same way:
standardized `SKILL.md` + `index.json` + `sources.json`, content-optimized
`reference/`, optional distilled `OVERVIEW.md`, and **document-grade KG ingestion**
(chunk → embed → Concept/Fact extraction). This plan migrates the existing library
onto that contract, in waves, smallest-first.

> How-to + crawl4ai/Chrome setup + the refresh/cron runbook live in
> `skill-graph-migration.md`. This file is the **prioritized rollout** for the current
> library.

## Snapshot (74 graphs)

| Mode | Count | Action |
|------|-------|--------|
| `managed` | 5 | ✅ done (framer, minio, pytorch, reactrouter, redis) |
| `reacquire` | 56 | re-crawl `source_url` → standardized + KG-ingested |
| `wrap` | 1 | `quant-career-docs` — re-package existing corpus (offline) |
| `native` | 12 | hand-authored (`agent-utilities/*`, `trading-systems/*`, …) — **leave alone** |

## Waves (use `migrate --root … --apply --only …` per batch; commit per batch)

**Wave 1 — smallest reacquire (≤10 files) + the wrap graph (~13).** Fast, validates the
pipeline live. Includes the 1 wrap graph.
`redis*`, `radix-ui-docs`, `redux-docs`, `temporal-docs`, `testing-library-docs`,
`uptime-kuma-docs`, `vitejs-docs`, … + `quant-career-docs` (wrap).

**Wave 2 — medium (11–100 files), ~24, in batches of ~8.** Review diffs + `status`
between batches. e.g. `react-docs`(71), `scipy-docs`(70), `shadcn-docs`(86),
`svelte-docs`(21), `terraform-docs`(19), `qdrant-docs`(128)…

**Wave 3 — large (>100 files), 20, run off-peak / overnight with KG ingest on.** Ordered
by size: `vuejs`(103) → `docker`(105) → `fastapi`(243) → `nodejs`(305) →
`pydantic-ai`(407) → `python`(744) → `linux`(2038) → `nextjs`(3406) →
`vercel`(9836). The big ones dominate KG storage — schedule, don't run interactively.

**Special — 3 zero-file graphs** (`django-docs`, `neo4j-docs`, `tanstack-docs`): their
*original* crawls produced 0 files. **Check the `source_url` first** — these sites
likely restructured; the shrink-guard will flag a sparse re-crawl as `stale_url`. Fix
the URL, then reacquire.

## Per-graph command

```bash
# one graph (auto picks reacquire/wrap; shrink-guard on)
python -m agent_utilities.knowledge_graph.distillation.skill_graph_pipeline \
    migrate --dir <skill_graphs>/<cat>/<name>-docs

# a wave (dry-run, then apply a batch)
python -m ...skill_graph_pipeline migrate --root <skill_graphs>            # preview
python -m ...skill_graph_pipeline migrate --root <skill_graphs> --apply \
    --only react-docs,scipy-docs,shadcn-docs --limit 8
```

Set the crawler env first (`SKILL_GRAPH_CRAWLER_PYTHON`, `SKILL_GRAPH_CRAWLER`,
`SKILL_GRAPH_CRAWL_TIMEOUT`, `SKILL_GRAPH_MAX_PAGES`) — see the runbook.

## After migration: cheap upkeep via **delta refresh**

Once a graph is `managed`, keep it current without rebuilding it:

```bash
python -m ...skill_graph_pipeline refresh --root <skill_graphs>     # delta only
```

`refresh` now **writes + re-ingests only the changed files** — it diffs the re-crawl
against the live `reference/` (path + sha256), writes only added/changed files, deletes
removed ones, leaves unchanged files (and their embeddings) untouched, and re-ingests
**only** the changed files into the KG. A graph where one page moved costs one file
write + one re-embed, not a full rebuild. Schedule it nightly (cron) — see the runbook.

## Verification per wave

```bash
python <skill-graphs repo>/scripts/validate_skill_graphs.py   # managed count should rise
python -m ...skill_graph_pipeline status --dir <graph> --quick
```
