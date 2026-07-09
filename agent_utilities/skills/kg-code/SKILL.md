---
name: kg-code
skill_type: skill
description: >-
  Code intelligence over the ingested code graph — cited how/usage/impact answers, cross-
  repo usages, call graphs, similar code, blast radius, metrics, arch reports — plus
  templated symbol navigation. Use to understand code via the KG instead of grepping —
  "how does X work", "who calls this", "impact of changing this", "find the definition".
license: MIT
tags: [graph-os, code]
tier: core
wraps: [graph_code, graph_code_nav]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-code

This skill fronts two code verbs:
- **`graph_code`** (`action`): `code_context` (cited how|usage|impact via `target`), `cross_repo_usages`, `call_graph` (callees|callers|inherits), `similar_code`, `routes`, `change_coupling`, `code_evolution`, `blast_radius`, `code_metrics`, `arch_report`, `adr`.
- **`graph_code_nav`** (`action`): `find_definition`, `find_references`, `trace_call_graph`, `impact_of_change`, `connects` (shortest path between two symbols). Start from a `symbol` or exact `node_id`; optionally scope by `source_system`.

Per CLAUDE.md, **query the KG before grepping**: this is the free native path for "how does this code work?".

## Invoke
- **MCP:** `load_tools(tools=["graph_code"])` (or `graph_code_nav`), then call it.
- **REST twin:** `POST /graph/code` · `POST /graph/code-nav` with a JSON body.

## Example
```
graph_code(action="code_context", query="how does the ingestion lane claim tasks", target="how")
graph_code_nav(action="find_references", symbol="reach_user")
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
