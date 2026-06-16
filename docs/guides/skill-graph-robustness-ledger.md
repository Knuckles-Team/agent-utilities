# Skill-Graph Acquisition — Robustness Ledger

A running catalogue of per-site discrepancies and failure modes found while
migrating legacy skill-graphs, and the robustness mechanism each one drove. The
goal is a build pipeline that picks the *right* acquisition strategy per site
automatically and never silently degrades.

The detector is `skill_graph_pipeline.detect_scrape_strategy(url)` (the
**SiteProfiler**); every build records what it detected in `sources.json`
under `scrape_profile` so a graph carries its own provenance of *how* it was built.

## Strategy ladder (what the SiteProfiler picks, in order)

1. **`llms-full.txt`** — the whole docs corpus in one fetch (no JS, no bot-block,
   no shrink). Best case. Split by H1/H2; coalesced to ~48 KB files past 400
   sections so huge corpora stay navigable.
2. **`llms.txt` index** — a curated `[name](url)` link list; each page fetched
   (HTML stripped to markdown if the link doesn't serve `.md`).
3. **`sitemap.xml`** (or a `Sitemap:` line in robots.txt) → bounded web crawl.
4. **recursive render** (crawl4ai) — last resort for JS-rendered sites with no
   machine-readable index.

Scope matters: each of (1) and (2) is probed at the **docs-path scope first**
(`/docs/llms-full.txt`), then the domain root — because a root `/llms.txt` is
often a marketing-site index, not docs.

## Ledger

| # | Site / case | Discrepancy observed | Root cause | Robustness mechanism |
|---|---|---|---|---|
| R1 | langchain (`docs.langchain.com`) | Legacy graph had only **2 files**; recursive crawl yielded almost nothing | Client-side-rendered docs site; blind HTML crawl gets shell pages | SiteProfiler detects `llms-full.txt` → **301 files / 11.8 MB** full corpus |
| R2 | langchain | A naive H1 split of `llms-full.txt` makes **2,548 tiny per-API-method files** | Each API method is its own `#` heading | `_split_llms_full` coalesces adjacent sections into ~48 KB files past `_MAX_LLMS_SECTIONS=400` → 301 navigable files |
| R3 | mariadb (`mariadb.com`) | Root `llms.txt` returned **marketing pages** (pricing/about-us/industry), not docs | Domain-root `/llms.txt` indexes the *marketing site*; real docs corpus is at `/docs/llms-full.txt` | `_llms_scopes` probes the **docs-path scope before root**; mariadb now resolves to `/docs` → 107 real-doc files |
| R4 | mariadb | Fetched pages stored as **raw `<!doctype html>`** in the corpus | `llms.txt` links served HTML, not markdown; index path stored the body verbatim | `_fetch_llms_index` detects HTML (`_looks_like_html`) and strips to markdown (`_html_to_markdown`: trafilatura → regex fallback). 1.65 MB raw HTML → 10.8 KB clean text; **0 HTML leaks** |
| R5 | uptime-kuma, gcp | No `llms.txt`; recursive crawl is slow/hang-prone | Sites publish a sitemap but no LLM index | SiteProfiler falls back to **`sitemap.xml`-driven** bounded crawl before recursive render |
| R6 | chakra-ui, pandas (Wave-1 KG pass) | KG-processing returned **0 nodes** after vitejs succeeded in the same batch | Embedder/engine dropped mid-batch (recurring GB10 power fault → 502) — *not* a content problem | Bounded embed + per-cycle embedder + health-gate already abandon-on-timeout instead of hanging; **retry these two once the engine is healthy** (open) |
| R7 | All `llms-full.txt` | A tiny stub or 404 body could be mistaken for a real corpus | Some sites return a 200 with a near-empty `llms-full.txt` | `_fetch_llms_docs` requires `len(full) > 2000` before treating it as the corpus; else falls through to the `llms.txt` index / sitemap |
| R8 | Batch migration | One slow web crawl (gcp first in the batch) **blocked all 24 following graphs** for up to the 900 s crawl bound — zero visible results | A serial batch runs in list order; a heavy crawl at the front hides every fast result behind it | Batch runner pre-detects strategy (HTTP-only) and runs **llms graphs first, web crawls last**; web crawls get a tight `SKILL_GRAPH_CRAWL_TIMEOUT=240` + `SKILL_GRAPH_MAX_PAGES=400` so a slow site fails fast and keeps existing content |
| R9 | gcp (`cloud.google.com`) | A generic depth-3 / 1000-page crawl from `/docs/overview` is unbounded-feeling and low-value (pulls the whole GCP marketing+docs tree) | `cloud.google.com` is enormous and has no `llms.txt`; "overview" is not a scoped corpus | **Deferred**: a meaningful gcp graph needs a *scoped product path* (e.g. `/run/docs`, `/compute/docs`) as the seed, not the docs root. Open — pick target product(s) (open) |

## Open items

- **R6**: re-run KG-processing for `chakra-ui-docs` and `pandas-docs` once the KG
  engine/embedder is confirmed healthy (GB10 power fault is hardware — see the
  `gb10-power-fault-and-vllm-topology` note). The reference/ trees are already
  migrated; only the KG ingest needs a retry.
- **gcp** (`cloud.google.com`): very large; sitemap crawl must stay bounded
  (`max_pages`) — confirm the cap is logged, not silently truncating.

## Invariants the pipeline now guarantees

- Never store raw HTML in a reference corpus (R4).
- Prefer a curated docs corpus over a marketing index (R3).
- Keep file counts navigable regardless of corpus size (R2).
- Degrade strategy gracefully (llms → sitemap → render), never hang (bounded
  fetch + process-group kill), and record the chosen strategy in `sources.json`.
