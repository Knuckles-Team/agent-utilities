---
name: kg-table
skill_type: skill
description: >-
  Manages engine-native SQL tables — mirror a connector into a table, insert rows,
  create/list/drop tables, and run read-only SELECTs. Use for tabular/relational data in
  the KG — "ingest this source into a table", "query the SQL table", "list tables".
license: MIT
tags: [graph-os, query, sql, table]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-table

`graph_table` is the SQL-table surface of the engine. Actions: `ingest` (mirror a registered connector's data into a table), `rows` (insert row dicts), `create` (declare columns), `list`, `drop`, and `query` (run a read-only `SELECT`).

## Invoke
- **MCP:** `load_tools(tools=["graph_table"])`, then `graph_table(action="query", sql="SELECT * FROM incidents LIMIT 10")`.
- **REST twin:** `POST /graph/table` with `{"action": "query", "sql": "SELECT ..."}`.

## Example
```
graph_table(action="ingest", source="servicenow", table="incidents", limit=1000, replace=true)
graph_table(action="query", sql="SELECT priority, count(*) FROM incidents GROUP BY priority")
```
