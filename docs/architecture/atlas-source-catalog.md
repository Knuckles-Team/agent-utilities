# Atlas source catalogue

`graph_catalog` is the canonical read surface for Atlas's provider picker and
connection-profile status. The same operation is available through MCP and the
Graph-OS REST route `POST /graph/catalog`; the REST adapter dispatches through
the same `graph_catalog` implementation as MCP.

The default request (`action: "list"`) adds a `sources` projection to the
existing modality catalogue. Each source descriptor has `availability`,
`reason`, `queryMode`, `dialects`, `sync`, and `capabilities`. Connection names
and profile references are metadata only. Endpoint/DSN values, users,
credentials, and probe exception text never cross this boundary.

The initial governed provider set is:

- PostgreSQL and generic relational database (`queryMode: sql`)
- registered external graph kinds (Neo4j, Apache AGE, Ladybug, Epistemic Graph,
  GraphQL; `queryMode: cypher`/`graphql`), with generic OpenCypher and PuppyGraph
  remaining `unverified` until an explicit read probe
- virtual graphs (`queryMode: uql`), unverified until probed
- Spark (`queryMode: compute`, compute-only)
- Iceberg and Trino (`queryMode: sql`)
- S3 and generic object stores (`queryMode: object_store`)
- Teradata (`availability: unsupported` until a certified connector exists)

`action: "preview_sync"` accepts one source, one mode (`delta`, `full`, or
`reconcile`), bounded ids, and optional neutral connection/graph selectors. It
returns `atlas-source-sync-preview.v1` with `wouldExecute: false`; execution
remains exclusively through the existing `source_sync` tool.

The `querySurfaces` projection points Atlas at the existing `graph_ask`/`nl_query`
natural-language paths and the `engine_query(action="uql")` path. Provider
catalogue discovery reuses `graph_configure`'s process-owned connection
registry and the executable source-connector registry; it does not maintain a
WebUI-local provider list. Graph OS owns the typed connector/source
control-plane contracts, which are not a live process registry here, so no
unbacked control-plane state is claimed; their runtime adapter remains a
follow-up integration seam.
