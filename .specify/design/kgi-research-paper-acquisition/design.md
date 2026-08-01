# Design Document: Paper acquisition goes through scholarx-mcp first, with a direct-download fallback

CONCEPT:AU-KG.ingest.content-acquisition

> `agent_utilities/knowledge_graph/ingestion/research_acquisition.py:1-11`.

## Decision — prefer the deployed `scholarx-mcp` server; fall back to a direct arXiv/PDF download only when it's unreachable

Given `PaperRef`s extracted from a research roundup, this module downloads
the referenced PDFs into the shared scholarx store
(`paths.research_dir()/papers`) so the ingestion engine can later ingest them
as documents/papers (`research_acquisition.py:1-6`).

The module docstring states the decision directly: **"Prefers the deployed
`scholarx-mcp` server (dedup, metadata, queue) and falls back to a direct
arXiv/PDF download so acquisition still works when the MCP is unreachable —
both write to the same directory."** (`research_acquisition.py:8-10`).

**The rejected alternative is a direct-download-only implementation.** It is
simpler (no MCP dependency, no fallback branch to maintain) and it loses on
exactly the three things the docstring calls out as scholarx-mcp's value:
dedup (an already-fetched paper isn't re-downloaded), metadata (title/author/
abstract extraction beyond a bare PDF blob), and queue (rate-limited,
orderly fetch rather than a burst of direct HTTP calls against
arXiv/publisher hosts). A pure direct-download path would have to
reimplement all three or ship without them.

**The rejected alternative on the other side — MCP-only, no fallback — is
also named implicitly**: acquisition must "still work when the MCP is
unreachable" (`research_acquisition.py:9`), which is a normal operating
condition for any fleet MCP server (down for maintenance, not yet deployed on
this host). An MCP-only implementation would make research ingestion hard-
fail on infrastructure state rather than degrade gracefully. Writing both
paths to the same directory (`paths.research_dir()/papers`) is what makes the
fallback transparent to the ingestion engine downstream — it doesn't need to
know which acquisition path produced a given file.

## Risk Assessment

- **Blast Radius**: `research_acquisition.py` only; downstream consumers
  (the document ingestion adaptor) are agnostic to which path fetched a file.
- **Backward Compatible**: Yes.
- **Breaking Changes**: None.
- **Known weak point**: the fallback path forgoes dedup/metadata/queue, so a
  sustained scholarx-mcp outage silently degrades acquisition quality (more
  duplicate downloads, thinner metadata) without a distinct alerting signal
  distinguishing "fell back" from "used the primary path."
