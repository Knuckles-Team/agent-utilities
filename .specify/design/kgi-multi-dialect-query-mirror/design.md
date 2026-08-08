# Design Document: The KG is queryable through multiple mirrored surfaces — native SQL tables, a SPARQL RDF projection, and NL→query — not just Cypher

CONCEPT:AU-KG.ingest.mirror-inbound

> `agent_utilities/knowledge_graph/core/table_ingest.py:1-11` (connector/ETL
> → native SQL tables), `agent_utilities/knowledge_graph/core/nl_query.py:1-9`
> (NL → query dialect selection), `agent_utilities/knowledge_graph/etl/pipeline.py:110-129`
> (the ETL `sink="table"` wiring), `agent_utilities/mcp/tools/ontology_tools.py:1358`
> (the `graph_etl` tool description), `agent_utilities/mcp/tools/query_tools.py:324,395-400,845-865`
> (`graph_query scope='sparql'`, `graph_ask`, `graph_table` tool registrations).

## Decision — mirror external/connector data and the graph itself into additional native query surfaces (SQL tables, SPARQL projection), and let an LLM pick the right dialect, instead of forcing every caller through Cypher

Three sub-surfaces share this one marker because they are the same design
move applied to three query dialects: **make the KG reachable through
whichever query language/shape a caller or dataset is already native to,
rather than requiring translation into Cypher first.**

1. **Connector/ETL data mirrored into native engine SQL tables.**
   `table_ingest.py:4-9`: "The engine gained arbitrary user tables (DataFusion
   + pg-wire). This module makes that reachable from the platform: mirror any
   registered source connector's documents — or arbitrary ETL output rows —
   into a native engine SQL table via `CREATE TABLE` + bulk `INSERT`, so
   'ingest tables from any connector / mirror data into our DB' works
   end-to-end." The `graph_table` MCP tool (`query_tools.py:849-865`) exposes
   this as `ingest`/`rows`/`create`/`list`/`drop` actions.

2. **The RDF projection queryable via SPARQL.** `graph_query`'s `scope`
   parameter documents `'sparql'` as running "a SPARQL 1.1 SELECT/ASK over
   the engine's RDF projection of the graph" (`query_tools.py:324`),
   RLS-governed and fanning out like Cypher/SQL (`query_tools.py:395-400`).

3. **NL → query, dialect-selected.** `nl_query.py:4-9`: given a live KG/table
   schema context, an LLM "emits an executable query in one of the engine's
   query dialects (`cypher`... `sql`... or `sparql`...), and we execute it
   through the matching engine surface." The `graph_ask` tool
   (`query_tools.py:845-848`) is the caller-facing wrapper: it "translates
   your question... into a single read-only query in the best dialect...
   Returns the GENERATED query (auditable), the result rows, and citations...
   so the answer is grounded and verifiable, not a black box."

**The rejected alternative, common to all three, is Cypher-only access** —
every caller, connector, and ETL sink translates into (or is limited to) the
property-graph query language. It loses for callers whose native shape is
tabular (a BI tool expecting SQL over `pg-wire`), semantic-web (a SPARQL
consumer expecting the RDF projection), or natural language (a user who
doesn't know any query language at all). Each of the three mirrors is
scoped, not a full alternate write path: SQL tables are a *mirror* of
connector/ETL data (via `graph_etl sink="table"`,
`ontology_tools.py:1355-1358`, "mirror the inbound `source` connector's data
into an engine SQL table"), SPARQL is a *projection* of the existing graph
(read-only, RLS-governed), and NL→query *generates* one of the other two
dialects rather than defining a fourth independent execution path — so none
of the three duplicates the graph's actual write/storage model.

## Risk Assessment

- **Blast Radius**: `core/table_ingest.py`, `core/nl_query.py`,
  `etl/pipeline.py`, `mcp/tools/query_tools.py`, `mcp/tools/ontology_tools.py`.
- **Backward Compatible**: Yes — Cypher remains the primary/default `scope`;
  the other dialects are additive.
- **Breaking Changes**: None.
- **Known weak point**: three separate query surfaces (SQL tables, SPARQL,
  NL-generated) triple the paths that must independently enforce RLS/
  visibility filtering correctly — a gap in any one dialect's enforcement is
  a distinct vulnerability from the others, not a single chokepoint to audit.
