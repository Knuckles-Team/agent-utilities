# Design Document: One web-fetch front door with a fixed backend precedence, shared by ingestion and skill-graph distillation

CONCEPT:AU-KG.ingest.web-fetch-front-door

> `agent_utilities/knowledge_graph/ingestion/web_fetch.py:1-16`.

## Decision — a single resolver picks ArchiveBox → crawl4ai → requests+markitdown, in that fixed order, so acquisition backend selection happens once instead of per-caller

`web_fetch.py:1-16` states the decision as both a mechanism and a rationale:
**"One front door that both the ingestion engine (the `DOCUMENT` URL path)
and the skill-graph distillation pipeline call, so the acquisition backend
is chosen once, consistently"**, in this precedence:

1. **ArchiveBox** (when `config.archivebox_url` is set) — serve the
   *preserved* snapshot through the `archivebox-api` MCP server,
   archive-on-miss. "Fast, no live crawl, immune to a site going down or
   rate-limiting us."
2. **crawl4ai** (when installed) — render JS, recursive-capable.
3. **requests + markitdown** — the zero-dependency floor.

**The rejected alternative is per-caller backend selection** — the ingestion
engine and the skill-graph distillation pipeline each independently deciding
which fetch backend to use, and each re-implementing the ArchiveBox/crawl4ai/
requests fallback chain. The docstring names the outcome directly:
"ArchiveBox + crawl4ai are thus first-class without each caller
re-implementing backend selection" (`web_fetch.py:14-16`). Two callers
independently choosing backends would risk drift (one caller picks up
ArchiveBox support, the other doesn't) and duplicate the precedence logic
twice.

**The precedence order itself is a decision, not an arbitrary list**:
ArchiveBox is tried first specifically because it is a *preserved snapshot*
— faster and immune to live-site failure modes (down, rate-limited) that
would affect a live crawl. crawl4ai is second because it can render JS and
recurse, capabilities the zero-dependency floor (requests+markitdown)
lacks. requests+markitdown is last, kept as the backend that always works
with no external dependency, so the front door degrades gracefully rather
than failing outright when neither ArchiveBox nor crawl4ai is available.

**Scope boundary named explicitly**: "A single page is the unit here;
bulk/recursive crawling stays in the skill-graph pipeline" (`web_fetch.py:13`)
— this resolver is deliberately not trying to also own multi-page crawl
orchestration; it answers "how do I fetch THIS one URL" and leaves "what
URLs to fetch" to its callers.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ingestion/web_fetch.py`; both the
  ingestion engine's `DOCUMENT` URL path and the skill-graph distillation
  pipeline depend on this resolver's precedence.
- **Backward Compatible**: Yes — each backend degrades to the next when
  unavailable/unconfigured.
- **Breaking Changes**: None.
- **Known weak point**: an ArchiveBox snapshot can be stale relative to the
  live page; because ArchiveBox is tried first whenever configured, a caller
  that needs the CURRENT page content (not the last-archived one) has no
  override documented here short of unsetting `archivebox_url` entirely.
