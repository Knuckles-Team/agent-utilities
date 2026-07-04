---
name: kg-etl
description: >-
  Runs a unified source→(ontological transform)→sink ETL flow over the KG hub — pull any
  connector in and/or load a write-back SoR, graph store, or engine SQL table out, with
  lineage. Use for cross-system data movement — "ETL servicenow into stardog", "list ETL
  sources/sinks", "show ETL lineage".
license: MIT
tags: [graph-os, ingestion, etl]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-etl

`graph_etl` (CONCEPT:AU-KG.ontology.one-source) composes ingestion + write-back + graph-store machinery. `action='run'`: pull `source` into the KG (any registered ingestion source; `mode` delta|full|reconcile) and/or load `sink` from the KG — sink is a write-back SoR (dry-run + approval, pass `ops_json`), a graph store (`stardog/neo4j/age/jena_fuseki` or a registered connection), or the native engine SQL table (`sink='table'`). `action='list'` shows sources/sinks/backends; `action='lineage'` shows recorded runs (AU-KG.ontology.kg-3).

## Invoke
- **MCP:** `load_tools(tools=["graph_etl"])`, then `graph_etl(action="list")`.
- **REST twin:** `POST /graph/etl` with `{"action": "run", "source": "servicenow", "sink": "stardog", "mode": "delta"}`.

## Example
```
graph_etl(action="run", source="leanix", sink="table", ops_json='{"table":"apps","replace":true}')
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
