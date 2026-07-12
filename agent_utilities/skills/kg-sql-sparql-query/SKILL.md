---
name: kg-sql-sparql-query
skill_type: skill
description: >-
  Runs read-only SQL (DataFusion, over the KG + user tables — the same path the pg-wire
  listener uses) or SPARQL 1.1 SELECT/ASK/CONSTRUCT/DESCRIBE (over the engine's RDF
  projection of the graph) via `graph_query`'s `scope` parameter — the SQL/relational
  and RDF/OWL faces of the SAME graph Cypher already reads. Also covers `scope="federated"`
  (query one registered external graph by reference id). Use when you want to read the
  KG as a relational table or an RDF graph instead of writing Cypher — "run SQL over the
  KG", "SELECT FROM nodes", "run a SPARQL query", "query this federated reference".
license: MIT
tags: [graph-os, query, sql, sparql, rdf, federated, datafusion]
tier: core
wraps: [graph_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-sql-sparql-query

`graph_query` is not only Cypher (see `kg-query` for that path) — its `scope` parameter
selects the dialect entirely, over the SAME underlying graph:

- **`scope="sql"`** — read-only SQL over the KG + user tables via the engine's
  DataFusion surface (`AU-KG.query.read-only-sql-over`) — e.g. `SELECT * FROM nodes
  WHERE type = 'Service'`. The exact same path the engine's pg-wire listener serves to a
  real Postgres driver/ORM. The `cypher` argument carries the SQL string (yes, the field
  is still literally named `cypher` — pass your SQL there). Non-`SELECT` statements are
  refused (read-only, mirrors the engine's own `sql()` refusing non-SELECT).
- **`scope="sparql"`** — SPARQL 1.1 `SELECT`/`ASK`/`CONSTRUCT`/`DESCRIBE` over the
  engine's RDF projection of the live graph (`AU-KG.ingest.mirror-inbound`). RLS-governed
  (visibility-filters rows exactly like every other read path). The `cypher` argument
  carries the SPARQL string.
- **`scope="federated"`** — query one registered `ExternalGraphReference` node by
  `reference_id`, running `cypher`'s query text against that external engine.
- Both `sql` and `sparql` honor `target` fan-out (a named connection, `"all"`, or a
  comma-separated list) exactly like the default Cypher path — per-target results come
  back labeled, with per-target timeouts so one slow backend can't stall the set.

## Invoke

- **MCP:** `load_tools(tools=["graph_query"])`.
- **REST twin:** `POST /graph/query` with `{"cypher": "<sql or sparql text>", "scope": "sql" | "sparql" | "federated", ...}`.

## Example — SQL

```jsonc
graph_query(cypher="SELECT type, count(*) AS n FROM nodes GROUP BY type ORDER BY n DESC LIMIT 10",
            scope="sql")
```

## Example — SPARQL

```jsonc
graph_query(
  cypher="SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(CONTAINS(STR(?s), 'svc:payments')) } LIMIT 20",
  scope="sparql"
)
```

## Example — federated

```jsonc
graph_query(cypher="MATCH (n:Service) RETURN n LIMIT 10",
            scope="federated", reference_id="extgraph:prod-neo4j")
```

## Honest limitations

- `scope="sql"` is read-only SELECT-only, same discipline as the default Cypher path
  (write keywords are refused). It is NOT a general admin SQL console.
- SPARQL visibility filtering mirrors RLS on the read path — a row your identity cannot
  see under RLS is filtered from SPARQL results the same way it would be from Cypher, not
  bypassed.
- `scope="federated"` needs a REAL `ExternalGraphReference` node already registered
  (`reference_id`) — there is no discovery/registration action on `graph_query` itself.
  To fan a query across SEVERAL registered external graphs at once with ranking rather
  than one reference at a time, use `graph_federated_search` (see `kg-search`) instead —
  the two are complementary: this skill targets ONE specific reference by id, that tool
  fans out to several by content.
- For SQL/Cypher/GraphQL/UQL at the raw ENGINE level (bypassing AU's `graph_query`
  wrapper entirely — e.g. `unified`/`uql`/`explain_plan`), the `engine_query` domain
  (see its `kg-modality-sql` skill, shipped from the `epistemic-graph` package) is the
  lower-level surface this skill's `graph_query` wrapper sits on top of.
