---
name: kg-feeds
skill_type: skill
description: >-
  Manages the unified RSS/Atom feed registry — list, bulk add/remove feed sources, and run
  the world-model-gated feed sweep. Use for feed ingestion — "add these RSS feeds", "list
  feeds", "sync feeds now".
license: MIT
tags: [graph-os, feeds]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-feeds

`graph_feeds` (CONCEPT:AU-KG.ingest.rss-feed-connector/2.122) manages `:FeedSource` nodes (native RSS, FreshRSS, ScholarX arXiv) ingested through one world-model gate. Actions: `list`, `add` (one `url=` or bulk `urls=` a JSON array / comma-or-newline list), `remove` (by url/urls), `sync` (run the feed sweep now — native RSS + ScholarX through the gate, concurrent; `mode=delta|full`).

## Invoke
- **MCP:** `load_tools(tools=["graph_feeds"])`, then `graph_feeds(action="sync", mode="delta")`.
- **REST twin:** `POST /graph/feeds` with `{"action": "add", "urls": "[\"https://a/feed\",\"https://b/rss\"]"}`.

## Example
```
graph_feeds(action="add", url="https://hnrss.org/frontpage")
```
