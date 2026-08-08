# Design Document: One `ingest_agent_toolkit` pipeline auto-detects and ingests MCP configs, agent skill directories, and A2A agent cards from a mixed source list, instead of separate ingestion calls per source type

CONCEPT:AU-ECO.mcp.unified-mcp-skill-a2a-ingest

> `agent_utilities/knowledge_graph/core/engine_ingestion.py:774-855`
> (`ingest_agent_toolkit`, `_detect_toolkit_source_type`).

## Decision — `ingest_agent_toolkit(sources, agent_card_path)` accepts a single mixed list of file paths, directory paths, and URLs, classifies each source's type automatically (`a2a_url` / `mcp_config` / `skill_directory` / `remote_json`), and routes it to the matching ingestor internally, rather than exposing four separate typed ingestion entry points

An operator populating the KG with "everything this agent ecosystem can reach"
today has three structurally different kinds of sources to describe: MCP server
configs (JSON with an `mcpServers` key), agent skill directories (containing
`SKILL.md`), and A2A agent cards (fetched from a URL's well-known path). Rather than
requiring the caller to sort sources into three separate calls, `ingest_agent_toolkit`
takes one flat list and auto-detects each entry's type
(`_detect_toolkit_source_type`, `engine_ingestion.py:822`): an `http(s)://` string
is an A2A URL fetch; a JSON file containing `mcpServers` is an MCP config (ingested
per-server with **live tool discovery**, `AU-ECO.mcp.live-server-metadata-cache`,
falling back to tool-flag parsing if the live connect fails); a directory
containing `SKILL.md` is a skill; a remote `.json` URL is checked for `mcpServers`
to disambiguate from a plain A2A card fetch. Each source is processed
independently and failures are collected per-source (`errors`/`skipped` in the
summary) rather than aborting the whole batch.

## Rejected alternative — four separate typed ingestion functions/tools, one per source kind

The alternative that avoids auto-detection entirely is the more conventional API
shape: `ingest_mcp_config(path)`, `ingest_skill_directory(path)`,
`ingest_a2a_card(url)`, `ingest_remote_json(url)` as four distinct entry points,
leaving the caller to classify each source itself. That is rejected implicitly by
the chosen signature (`sources: list[str]`, one heterogeneous list) and explicitly
by the KG freshness check built into the SAME pipeline — "check KG freshness via
config hash before re-ingesting" (`engine_ingestion.py:803`) — which only makes
sense as ONE dedupe/freshness pass over a mixed batch; four separate entry points
would need that freshness logic either duplicated four times or factored out
separately anyway, at which point the unified entry point is strictly less code to
maintain for the same guarantee. Auto-detection also matches how an operator
actually thinks about "onboard this whole toolkit" — as one list of things to add,
not four categorized ones.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/engine_ingestion.py`.
- **Backward Compatible**: Yes — additive ingestion entry point.
- **Known weak point**: type detection is heuristic (JSON-key sniffing, directory
  contents, URL shape) — a source that is ambiguous under these heuristics (e.g.
  a `.json` URL that happens to contain an `mcpServers`-shaped key but is not
  actually an MCP config) can be misclassified with no override the caller can
  pass to force a specific ingestor.
