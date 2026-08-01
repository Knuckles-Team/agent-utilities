# Design Document: Every research/news feed path converges on the SAME canonical record

> `agent_utilities/protocols/source_connectors/connectors/arxiv.py`,
> `agent_utilities/protocols/source_connectors/connectors/rss.py`, and
> `agent_utilities/automation/feed_sources.py` (ScholarX preset bridging).

CONCEPT:AU-KG.ingest.arxiv-feed-connector ·
CONCEPT:AU-KG.ingest.research-connector-presets ·
CONCEPT:AU-KG.ingest.rss-feed-connector

## Decision — three independent feed paths, one canonical envelope, so a paper never becomes three nodes

`CONCEPT:AU-KG.ingest.arxiv-feed-connector` — `arxiv.py:1-25`.

There are deliberately THREE parallel ways an arXiv paper can enter the KG:
FreshRSS-curated world-model RSS (gated by `WorldModelPipelineRunner`),
ScholarX (a paper search/dedup service reached via `scholarx-mcp`), and this
native, zero-infra arXiv connector — added as "a THIRD, narrower research
feed... the raw per-category arXiv listing, useful when neither of those is
deployed" (module docstring). **The rejected alternative was collapsing to
one path** (e.g. always requiring FreshRSS or ScholarX to be deployed first).
That loses for operators who want arXiv coverage with zero extra
infrastructure. Instead, every path emits the SAME `metadata["record"]`
envelope shape (`canonical` + `origin.streamId`) that
`feed_sources.scholarx_feed_documents` and the native `rss` connector also
produce, so the moment `WorldModelPipelineRunner` routes an item to the
research path, all three converge on the identical `arxiv:<id>` node — one
paper arriving via three feeders collapses to one KG node, never three.

**Budget-bounded by construction** (the docstring calls this the "★ critical
constraint"): `categories` has NO default — an operator must opt in
explicitly via `KG_ARXIV_CATEGORIES` — and `max_results` caps each category's
page size. The connector does no relevance scoring itself; every entry still
passes the SAME downstream gate as every other research feed
(`grade_and_enqueue_paper` — keyword score + novelty dedup + KG-stored
watchlists). The rejected alternative here is a connector that widens BOTH
the funnel's mouth and its throat (i.e., also decides relevance) — rejected
because that would let this narrow, zero-infra connector silently admit more
than the shared quality gate allows.

### Pointer — `CONCEPT:AU-KG.ingest.research-connector-presets`

`feed_sources.py:63-73`. When the local `scholarx` Python package is not
installed, `_scholarx_mcp_documents` is the fallback. **The rejected
alternative, named directly in the docstring**: writing a new bespoke HTTP
client for ScholarX. Instead the fallback drives the fleet's `scholarx-mcp`
server through the generic `mcp_tool` connector — "prefer driving the fleet
server over a new HTTP client." Because the generic `mcp_tool` connector
returns the tool's raw record shape (no `origin`/`canonical` envelope), this
function re-shapes each drained document into the EXACT SAME record envelope
the direct-import branch produces, so `WorldModelPipelineRunner._is_research`/
`_arxiv_id` route and dedup it identically regardless of which of the two
paths fetched it — the same convergence discipline as the arXiv connector
above, applied to ScholarX's two acquisition paths instead of three feed
sources.

### Pointer — `CONCEPT:AU-KG.ingest.rss-feed-connector`

Two related but distinct realizations share this id:

1. **The generic native RSS/Atom connector** (`rss.py:1-19`) — a zero-infra
   feed extractor requiring no external service (unlike FreshRSS, which
   aggregates many feeds behind its own server), the RSS analog of the
   zero-infra `filesystem`/`web` connectors. It emits the same
   `metadata["record"]` envelope so entries flow through the SAME
   `WorldModelPipelineRunner` gate — arXiv/research entries route to the
   research path, news entries to the world-model gate (see
   `AU-KG.ingest.world-model-gate`). A real, load-bearing sub-decision here:
   the sweep bounds each feed fetch to `_FEED_FETCH_TIMEOUT_S = 20.0` seconds
   and caps concurrency at `_FEED_FETCH_CONCURRENCY = 12` (`rss.py:38-41`) —
   the rejected alternative is an unbounded serial sweep, where one slow feed
   stalls the whole pass; instead a slow feed is skipped and the sweep's
   wall-clock is bounded by the timeout, not by N×timeout.
2. **ScholarX's own RSS items** (`feed_sources.py:139-150`,
   `scholarx_feed_documents`) — unified into the SAME `SourceDocument` shape,
   preferring the local `scholarx` package's specialized arXiv parser and
   falling back to the MCP preset above (research-connector-presets) — a
   no-op only when neither path is available.

Both realizations exist so that "a feed is durable" holds true regardless of
which of the three feed technologies (native RSS, ScholarX, arXiv) produced
it; see also `.specify/design/kg-first-class-rss-atom/design.md`
(`AU-KG.compute.first-class-rss-atom`) for the companion decision that a
registered feed source is materialized as a durable `:FeedSource`/`:RssFeed`
KG node (via `feed_sources.upsert_feed_source`), not left as pure
configuration — the same "collapse to one canonical representation" instinct
applied to the feed's OWN identity rather than to the documents it produces.

## Risk Assessment

- **Blast Radius**: `agent_utilities/protocols/source_connectors/connectors/arxiv.py`,
  `agent_utilities/protocols/source_connectors/connectors/rss.py`,
  `agent_utilities/automation/feed_sources.py`,
  `agent_utilities/automation/worldmodel_pipeline.py`.
- **Backward Compatible**: Yes — each connector is additive; disabling one
  (e.g. unsetting `KG_ARXIV_CATEGORIES`) leaves the other two paths unaffected.
- **Breaking Changes**: None.
- **Known weak point**: convergence depends on every current AND future feed
  path independently reproducing the exact `metadata["record"]` envelope
  shape (`canonical` + `origin.streamId`) by convention — nothing mechanically
  enforces it. A new feed connector that emits a slightly different shape
  would silently create a duplicate node instead of converging, and nothing
  would flag it short of a KG dedup audit.
