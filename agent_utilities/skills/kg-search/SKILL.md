---
name: kg-search
skill_type: skill
description: >-
  Unified semantic/keyword/concept search over the KG (hybrid, HyDE, concept, analogy,
  memory, discover, ADORE…) plus multi-hop search-task synthesis and federated cross-graph
  search. Use to find things by meaning — "search the KG for…", "find concepts like…",
  "look up CONCEPT:ID", "search across external graphs".
license: MIT
tags: [graph-os, search]
tier: core
wraps: [graph_search, graph_search_synthesis, graph_federated_search]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-search

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_search`, `graph_search_synthesis`, `graph_federated_search` are held back from the default tool list (nothing removed — REST + `_execute_tool` still reach them exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_search"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_search` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


This skill fronts three search verbs:
- **`graph_search`** — the main search (`mode`: `hybrid` default, `hyde`, `deep`, `concept` (look up a `CONCEPT:ID`), `analogy`, `memory`, `discover`, `latent`, `rerank`, `adore`, `hard_negatives`, `chrono_ids`; `top_k`, `self_correct`, `as_of`, `target` for named/`all` connections).
- **`graph_search_synthesis`** — `synthesize` an evidence subgraph + a multi-hop question around an `answer_id`, or `diagnose` solver trajectories (FORT signatures).
- **`graph_federated_search`** — fan a `query` across registered external graph `references` (cap `top_k`).

## Invoke
- **MCP:** `load_tools(tools=["graph_search"])` (or `graph_search_synthesis` / `graph_federated_search`), then call it.
- **REST twin:** `POST /graph/search` · `POST /graph/search-synthesis` · `POST /graph/federated-search` with a JSON body.

## Example
```
graph_search(query="warm-fork sandboxes", mode="hybrid", top_k=10)
graph_search(query="KG-2.7", mode="concept")
```
