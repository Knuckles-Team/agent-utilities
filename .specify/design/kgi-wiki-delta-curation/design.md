# Design Document: A self-curating wiki directory ingests by content-hash delta, not a full re-walk every time

> `agent_utilities/knowledge_graph/ingestion/wiki_curator.py`;
> `agent_utilities/mcp/tools/write_ingest_tools.py:1000-1013` (the
> `curate_wiki` MCP action).

CONCEPT:AU-KG.ingest.wiki-delta-ingest

## Decision — `curate_wiki` skips unchanged pages by content hash, continuously

`wiki_curator.py:1-30`, `write_ingest_tools.py:1000-1010`.

**The problem**: a wiki-style directory of curated markdown is expected to be
re-ingested repeatedly (a "continuous" curation loop, not a one-time import)
as pages are edited. Re-ingesting the ENTIRE directory in full on every pass
— re-chunking, re-embedding, re-enriching every page whether or not it
changed — wastes work proportional to corpus size on every single pass, most
of which touches pages that haven't moved since the last pass.

**The rejected alternative**: a full re-ingest of the wiki directory on every
curation pass (the simplest implementation, and what a naive "just re-run
the ingest pipeline on a schedule" approach would do).

**The design chosen**: `curate_wiki(engine, target_path)` is explicitly a
"delta-skip continuous ingest of a self-curating wiki dir" — pages are
compared by content hash against what the KG already holds, and unchanged
pages are skipped entirely rather than re-processed. This lets curation run
as a continuous/frequent loop over a growing wiki directory without the cost
scaling with total corpus size on every pass — only genuinely CHANGED pages
pay the chunk/embed/enrich cost. The MCP action requires `target_path` (the
wiki directory) and surfaces the resulting summary as JSON.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/wiki_curator.py`,
  `agent_utilities/mcp/tools/write_ingest_tools.py` (`curate_wiki` action).
- **Backward Compatible**: Yes — the first pass over a directory still
  ingests every page (nothing yet exists in the KG to hash-match against);
  the delta-skip only activates on subsequent passes.
- **Breaking Changes**: None.
- **Known weak point**: content-hash delta detects CHANGED content but not a
  page that was DELETED from the wiki directory — a removed page's prior KG
  node is not automatically retracted by a curation pass that simply never
  encounters it again, unless a separate reconcile/tombstone step is run.
