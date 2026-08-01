# Design Document: SQL DDL files emit database-ontology entities through the SAME generic parse phase code files already use

CONCEPT:AU-KG.ontology.emits-database-ontology-entities

> `agent_utilities/knowledge_graph/pipeline/phases/parse.py`,
> `agent_utilities/models/knowledge_graph.py:34`
> (`DATABASE_TABLE`/`DATABASE_COLUMN`/`DATABASE_VIEW`).

## Decision — `.sql`/`.ddl` are parseable file extensions in the SAME pipeline phase, and the engine's `DatabaseTable`/`DatabaseColumn`/`DatabaseView` node types replay through the identical result-mapping path as `SYMBOL` nodes

`parse.py:54-56` adds `.sql`/`.ddl` to the set of parseable extensions
alongside every general-purpose language extension already handled.
`_replay_parse_result` (`parse.py:169-185`) states the mapping decision: "SQL
DDL extraction emits database-ontology entities **alongside** the code
`SYMBOL` path; map each engine node_type to its registry type" —
`DatabaseTable`/`DatabaseColumn`/`DatabaseView` engine node types are mapped
to their `RegistryNodeType` counterparts through the SAME per-node-type
dispatch loop that already handles `SYMBOL` nodes, with the same numeric
coercion (`props["line"]`) applied uniformly.

**The rejected alternative is a separate SQL-schema ingestion pipeline** — a
plausible design, since a database table/column is conceptually different
from a code symbol. Instead, DDL parsing is folded into the SAME generic
parse phase every other source file goes through: one file walk, one engine
call, one result-replay loop that simply recognizes more node_type values.
The practical payoff is that a `.sql` migration file sitting in a repository
alongside application code is indexed in the SAME pass as that code — a
table/column becomes a first-class, queryable graph entity without a second
ingestion pipeline to keep in sync with the first, and `tests/unit/test_sql_ddl_replay.py`
regression-tests exactly this replay path.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/pipeline/phases/parse.py`
  (`_replay_parse_result`), `models/knowledge_graph.py`
  (`RegistryNodeType.DATABASE_TABLE`/`DATABASE_COLUMN`/`DATABASE_VIEW`).
- **Backward Compatible**: Yes — additive node-type handling in an existing
  dispatch loop; a repo with no `.sql`/`.ddl` files is unaffected.
- **Known weak point**: the db_types dispatch is a small, fixed mapping
  (three types) inside a function whose primary responsibility is code-symbol
  replay — a new engine-emitted database entity kind would need this mapping
  extended by hand rather than being picked up automatically.
