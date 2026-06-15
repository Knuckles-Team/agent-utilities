# Skill-Graph Migration Runbook (legacy → unified KG-2.7 contract)

The unified pipeline (`agent_utilities.knowledge_graph.distillation.skill_graph_pipeline`)
builds every skill-graph one way: a standardized `SKILL.md` + a `sources.json`
provenance/freshness manifest, with hybrid-auto Knowledge-Graph ingestion. The ~74
existing graphs in the `skill-graphs` repo were built the old way (per-source-type,
inconsistent frontmatter, no manifest). This runbook migrates them in place.

Each graph converted hardens the system: it exercises a real acquisition path end to
end and adds a real graph to the contract gate's "managed" set.

## 1. Classify (dry-run plan)

```bash
python -m agent_utilities.knowledge_graph.distillation.skill_graph_pipeline \
    plan --root /home/apps/workspace/agent-packages/skills/skill-graphs/skill_graphs
```

Every graph is classified into one of four migration modes:

| Mode | Meaning | Action |
|------|---------|--------|
| `reacquire` | Legacy `source_url` in `SKILL.md` (a crawled doc-site) | Re-crawl those URLs → freshest content + full provenance |
| `wrap` | Has a `reference/` corpus but no `source_url` (e.g. `quant-career-docs`) | Re-package the existing markdown as a `dir` source (offline, content-preserving) |
| `managed` | Already carries `sources.json` | Nothing to do |
| `native` | Hand-authored / nested, no corpus (e.g. `agent-utilities/*`, `trading-systems/*`) | Left alone — these are authored, not distilled |

Current snapshot: **61 reacquire, 1 wrap, 12 native.**

## 2. Migrate

A single graph:

```bash
# auto picks reacquire/wrap from classification; --no-kg skips KG ingest
python -m ...skill_graph_pipeline migrate --dir <skill_graphs>/python/fastapi-docs
```

Batch (dry-run first, then apply):

```bash
# preview what would migrate
python -m ...skill_graph_pipeline migrate --root <skill_graphs>

# apply, a few at a time to validate quality before doing the whole set
python -m ...skill_graph_pipeline migrate --root <skill_graphs> --apply --limit 5
python -m ...skill_graph_pipeline migrate --root <skill_graphs> --apply \
    --only fastapi-docs,flask-docs
```

The first standardized build of each graph is versioned `1.0.0`. `reacquire`
re-crawls; `wrap` re-packages the existing `reference/`. Hybrid-auto KG ingestion runs
when the graph daemon is reachable and degrades cleanly to offline-only otherwise — so
migration is safe to run even while the KG engine is down.

### Crawl quality

The core CLI's `reacquire` uses the in-tree recursive web connector (self-contained).
For JS-heavy sites that need the richer crawl4ai renderer, drive the migration through
the `skill-graph-builder` skill instead (it injects crawl4ai as the pipeline's
`crawler_fn`):

```bash
python <universal-skills>/.../skill-graph-builder/scripts/generate_skill.py \
    "" fastapi --from-kg ""   # or pass the source_url(s) directly as the source arg
```

## 3. Recommended rollout order

1. **Smallest reacquire graphs first** (1–20 files) — cheap, fast feedback, surface
   crawler edge-cases early (e.g. `radix-ui-docs`, `redux-docs`, `temporal-docs`).
2. **The one `wrap` graph** (`quant-career-docs`) — offline, deterministic, proves the
   content-preserving path.
3. **Mid-size doc-sites** in batches of ~5, reviewing diffs and `status` between batches.
4. **The large graphs last** (`vercel-docs` ~9.8k files, `python-docs`, `rust-docs`) —
   long crawls; run with KG ingest on so the corpus also lands in the KG.
5. **Leave `native` graphs alone** unless you want to stamp them — they have no
   re-acquirable source.

## 4. Verify each migration

```bash
# contract gate (run from the skill-graphs repo root)
python scripts/validate_skill_graphs.py        # managed count should rise each round

# freshness of a migrated graph
python -m ...skill_graph_pipeline status --dir <graph> --quick
```

A migrated graph has: standardized `SKILL.md` frontmatter (`skill_graph_version`,
`source_types`, `built_at`, `file_count`, `kg_ingested`, …), a `sources.json` manifest,
and — when the daemon was up — `kg_ingested: true` with the corpus linked to the
`SkillGraph` ontology interface. Commit the migrated graphs to the `skill-graphs` repo
in batches.
