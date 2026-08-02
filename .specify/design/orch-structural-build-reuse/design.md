# Design Document: Cache the graph TOPOLOGY per routing-config, rebuild only the cheap per-run config — and skip the cache entirely when per-run toolsets are bound

CONCEPT:AU-ORCH.routing.structural-build-reuse

> Realised by `agent_utilities/graph/builder.py:79-117` (`_BuiltGraphCache`,
> `_GRAPH_CACHE`), `:120-149` (`_graph_cache_key`), `:458-476` (cache lookup in
> `create_graph_agent`), `:477-517` (warm-hit return path), `:879-888`
> (store-on-miss) and `:893+` (`_build_graph_config`). Introduced by commit
> `b794e6af` ("perf(orchestration): chat execution profile + non-blocking reply
> path (P0/P1)").

## Decision — split the built graph into a structural half that is cacheable and a config half that is not

`create_graph_agent` previously *"rebuilt the entire topology +
`discover_agents()` on EVERY turn"* (`builder.py:458`). Agent discovery is a
round-trip to the registry and the topology construction is pure CPU, and
neither depends on anything that varies turn to turn — the same routing
configuration always produces the same nodes and edges.

The decision is to memoize only that structural half. `_graph_cache_key`
(`:120-149`) hashes exactly the inputs that change the *shape* of the graph:
`name`, the `tag_prompts` keys, `router_model`, `agent_model`,
`routing_strategy`, and the `sub_agents` keys. The result lands in a
process-local bounded LRU (`_BuiltGraphCache`, max 64 entries). Everything
else — the per-run configuration — is rebuilt every turn, because
`_build_graph_config` is cheap and genuinely per-run.

**The rejected alternative is caching the whole built agent, and it was
rejected for correctness rather than performance.** A run that binds per-run
MCP toolsets holds live connections whose lifetime is the run, not the process.
Caching an agent with those connections attached would hand a later turn a
toolset belonging to an earlier, finished run — stale credentials, closed
transports, and cross-run bleed of tool availability. So the cache is bypassed
outright whenever per-run toolsets are bound: `builder.py:893+` states the
split directly, that the config is cheap and per-run and *only the graph
TOPOLOGY is cached*. The optimization was deliberately given up on exactly the
runs where it would have been unsafe, rather than made conditional on a flag
someone has to remember to set.

A second, smaller alternative — an unbounded cache — was rejected by the
64-entry LRU bound. The key includes `sub_agents` keys, so a deployment that
constructs many distinct agent rosters would otherwise grow the cache without
limit for the lifetime of the process.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/builder.py`. Process-local only —
  nothing is shared across processes, so no cache-coherency protocol is needed
  and a restart simply starts cold.
- **Backward Compatible**: Yes — a cache miss is the previous behaviour, and
  toolset-bound runs always take the previous behaviour.
- **Known weak point**: the key is a hash of *declared* structural inputs, so
  anything that changes the topology without changing those inputs is invisible
  to it. Agent discovery results in particular are captured into the cached
  topology: if the registry gains or loses a specialist while a warm entry is
  live, the process keeps routing against the roster it discovered at build
  time until the entry is evicted.
