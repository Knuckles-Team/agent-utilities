# Design Document: Route ingestion off `__commons__` onto many graph names, then write and read across them as one KG

CONCEPT:AU-KG.ingest.unified-query-routing · CONCEPT:AU-KG.ingest.batched-cross-graph-writer

> `agent_utilities/knowledge_graph/core/ingest_routing.py` (primary — both
> decisions live in this one module), `agent_utilities/knowledge_graph/core/graph_compute.py`
> (the batched multi-graph write facade), `agent_utilities/core/config.py`
> (the `KG_INGEST_SHARD_FANOUT` toggle), `agent_utilities/mcp/tools/query_tools.py`
> (the read-side union), `agent_utilities/knowledge_graph/core/worker_scheduler.py`
> (`durable_shard_writers`).

## The shared problem

The durable engine shards its redb writer `K` ways by `FNV-1a(graph_name) % K`
(EG-026), so each *graph name* pins to exactly one writer thread/core. Almost
all ingestion wrote the single `__commons__` graph, so `K-1` of the `K` shard
writers sat idle while one did every commit
(`ingest_routing.py:4-11`). Both concepts below are the two halves of fixing
that: **where does a write land** (routing) and **how does a hot single
source spread its writes across writers once it's landed** (fanout + batched
commit). They live in the same module and the same docstrings cross-reference
each other by concept id (`ingest_routing.py:148`, `graph_compute.py:3827-3835`),
which is why this is one document rather than two.

## Decision 1 — a single deterministic policy maps every ingestion item to a destination graph, and reads fan out to stay unified

`CONCEPT:AU-KG.ingest.unified-query-routing`

`route_graph()` (`ingest_routing.py:123-192`) is the one routing seam every
ingest adaptor calls, not string literals scattered through them:

- codebase repo `agent-utilities` → `code:agent-utilities`
- connector `servicenow` → `src:servicenow`
- chat for agent `planner` → `chat:planner`
- research source `arxiv` → `research:arxiv`
- a tenant-scoped item → the existing per-tenant graph (tenant always wins —
  a tenant must stay whole, `ingest_routing.py:162-164`)
- anything with no natural owner → the configured default (`__commons__`)

**The rejected alternative is per-adaptor hardcoded graph-name strings** —
what existed before. It loses because the naming policy would drift adaptor
by adaptor and nothing could reason about "which graphs are content graphs"
centrally (`is_content_graph`, the read path's union predicate, would have no
single source of truth).

**The correctness point this decision is actually graded on**: a node
written to `code:X` lives in a *different* engine graph than `__commons__`,
so a naive single-graph read would silently miss it. Routing therefore
maintains an in-process registry of active content graphs
(`register_content_graph`, seeded once from the engine's tenant list,
`ingest_routing.py:209-246`), and the read tools fan a default/implicit-target
query across `{default + active content graphs}` and merge
(`query_tools.py:658-663`, `query_tools.py:849-854` for `graph_ask`/`graph_table`
adjacent surfaces). `read_graph_targets()` returns just `[default]` until
something has actually been routed, so the common case stays on the fast
single-graph path (`ingest_routing.py:257-271`).

A documented cost of this fan-out, found in `query_tools.py:658-670`: before
a per-fan-out-entry timeout budget was added, an implicit-default `graph_query`
shared ONE `DEFAULT_FANOUT_TIMEOUT_S` (30s) budget across every resolved
content-graph entry including the primary connection — under a wide fan-out
(dozens of idle/unreachable `code:*`/`src:*` graphs) the primary got queued
behind hung ones on the same 8-worker pool and timed out too. That was fixed,
but it is the direct, paid-for cost of unified query over N graphs instead of
one.

## Decision 2 — a hot single source is fanned across K shard-keyed sub-graphs, and multi-graph writes commit in one round-trip

`CONCEPT:AU-KG.ingest.batched-cross-graph-writer`

Routing alone (Decision 1) still pins ONE high-volume source (e.g. a large
FreshRSS backlog) to ONE graph name = one shard writer. `KG_INGEST_SHARD_FANOUT`
(`config.py:2591-2600` region for the sibling `keys-off` doc; the field itself
is `config.py` `kg_ingest_shard_fanout`, default `False`) turns on a second
layer: when a `content_key` is supplied and the resolved graph is a routed
*content* graph (never a tenant/default graph), `route_graph()` appends a
`#<bucket>` suffix keyed by `shard_bucket_for(content_key, K)` — an FNV-1a
hash mirroring the engine's own shard key function so bucketing is
deterministic with zero engine round-trip (`ingest_routing.py:100-115,184-192`).
Codebase graphs are already per-repo (naturally sharded), so fanout applies
to `src:`/`research:`/`chat:` sources. The `#n` suffix keeps the source
prefix, so `is_content_graph` still recognises the sub-graph and Decision 1's
unified read still unions it — the two decisions compose rather than compete.

**The rejected alternative is one graph per source, unconditionally** — the
Decision-1-only state. It is simple and correct but caps a single hot
source's write throughput at one shard writer's rate regardless of `K`. The
tradeoff accepted by turning fanout ON: `#0..#K-1` are K *distinct* graph
identities for what is conceptually one source, so anything that lists
"sources" by graph-name prefix must know to collapse the suffix back — a
cost paid deliberately for parallelism (`config.py` field docstring, the
`kg_ingest_shard_fanout` Field help text).

The companion mechanism, `GraphComputeEngine.multi_graph_batch_update`
(`graph_compute.py:3823-3852`), ships a `graph_name → ops` map to the
engine's `MultiGraphBatchUpdate` op in ONE round-trip; the engine applies
each graph's sub-batch concurrently across the K redb shard writers, so the
write stage scales with the number of *distinct destination graphs* instead
of pinning one lock. It is the write-side companion that makes the K
sub-graphs from the fanout actually pay off in one commit instead of K
serialized round-trips, and it registers every touched graph back into
Decision 1's active-content-graph registry so reads see the result
immediately (`graph_compute.py:3855-3862`).

**The rejected alternative here is K sequential `batch_update` calls, one per
sub-graph.** It works but serializes what the engine can do concurrently and
multiplies round-trip latency by K — exactly the cost the whole fanout
mechanism exists to avoid.

## Risk Assessment

- **Blast Radius**: `ingest_routing.py`, `graph_compute.py`, `core/config.py`,
  `worker_scheduler.py`, `mcp/tools/query_tools.py`, `mcp/kg_server.py`
  (fan-out target resolution).
- **Backward Compatible**: Yes — `KG_INGEST_SHARD_FANOUT` defaults off (one
  graph per source, matching pre-fanout behavior); unified-query-routing
  itself degrades to `[default]` reads until content is actually routed.
- **Breaking Changes**: None currently shipped.
- **Known weak point**: the active-content-graph registry is in-process and
  best-effort seeded from the engine's tenant list on first read
  (`_seed_from_engine`, `ingest_routing.py:220-246`); a multi-process
  deployment where one process ingests and a different process reads before
  either has seeded/registered the graph can miss it until the read process's
  own seed catches up. The fan-out timeout-budget bug (Decision 1) shows this
  class of correctness/latency tradeoff has already bitten once in production
  code and was patched reactively, not designed away.
