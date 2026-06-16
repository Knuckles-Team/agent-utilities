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

### Crawl engine (crawl4ai) setup

`reacquire`/`refresh` re-crawl with the real JS-rendering **crawl4ai** web-crawler
when it is available, falling back to the in-tree basic web connector otherwise. The
pipeline runs the crawler in a **separate interpreter** so crawl4ai/Playwright can
live in a dedicated venv, pointed at via two env vars:

```bash
# one-time: dedicated venv with crawl4ai (Playwright needs a browser)
uv venv --python 3.12 /home/apps/.venvs/skillgraph-crawler
uv pip install --python /home/apps/.venvs/skillgraph-crawler/bin/python crawl4ai

# Playwright has no bundled Chromium for some new OSes (e.g. Ubuntu 26.04); use system
# Google Chrome by symlinking it into the path Playwright expects:
sudo apt-get install -y ./google-chrome-stable_current_amd64.deb   # from dl.google.com
PWDIR=~/.cache/ms-playwright/chromium-1223/chrome-linux64
mkdir -p "$PWDIR" && ln -sf /usr/bin/google-chrome "$PWDIR/chrome"

# then tell the pipeline which interpreter + crawler script to use:
export SKILL_GRAPH_CRAWLER_PYTHON=/home/apps/.venvs/skillgraph-crawler/bin/python
export SKILL_GRAPH_CRAWLER=<universal-skills>/research/web-crawler/scripts/crawl.py
export SKILL_GRAPH_CRAWL_TIMEOUT=900   # per-site wall-clock bound (default 900s)
```

With those set, `migrate --mode reacquire` and `refresh` re-crawl with crawl4ai. A
crawl that fails or times out raises, and because `build()` acquires *before* wiping
`reference/`, the existing content is preserved (the graph is reported `failed`, not
emptied).

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

## 4. Keep updated — periodic re-download + delta re-ingest

Once graphs are on the contract, `refresh` re-downloads them and **re-ingests only the
deltas**: it re-crawls each managed graph's recorded sources, compares each source's
content hash to `sources.json`, and skips the rewrite + KG re-ingest for anything
unchanged (and the KG ingest itself is content-hash delta-skipped, KG-2.8). Only
genuinely-changed corpora bump their `skill_graph_version` and re-embed.

```bash
# refresh one graph / all managed graphs under a root
python -m ...skill_graph_pipeline refresh --dir <graph>
python -m ...skill_graph_pipeline refresh --root <skill_graphs> --limit 10
python -m ...skill_graph_pipeline refresh --root <skill_graphs> --only fastapi-docs,redis-docs
python -m ...skill_graph_pipeline refresh --root <skill_graphs> --force   # rebuild even if unchanged
```

Schedule it (daily, off-peak) with cron or a systemd timer — set the crawler env vars
in the job's environment:

```cron
# /etc/cron.d/skill-graph-refresh — re-download + delta re-ingest nightly at 03:30
30 3 * * *  apps  SKILL_GRAPH_CRAWLER_PYTHON=/home/apps/.venvs/skillgraph-crawler/bin/python \
  SKILL_GRAPH_CRAWLER=/path/to/web-crawler/scripts/crawl.py SKILL_GRAPH_CRAWL_TIMEOUT=900 \
  /usr/bin/python3 -m agent_utilities.knowledge_graph.distillation.skill_graph_pipeline \
  refresh --root /path/to/skill-graphs/skill_graphs >> /var/log/skill-graph-refresh.log 2>&1
```

Each run reports per-graph `fresh` / `refreshed` / `failed`; commit the changed graphs
(SKILL.md + sources.json + reference/) after a refresh pass.

## 5. Verify each migration

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
