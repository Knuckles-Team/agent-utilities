# Design Document: Fleet catalog probe cache — honest, self-refreshing staleness

CONCEPT:AU-ECO.multiplexer.catalog-probe-ttl-staleness

> `agent_utilities/mcp/multiplexer.py` (`_probe_cache`, `_probe_cache_hit`,
> `probe_server`, `probe_catalog`), `agent_utilities/core/config.py`
> (`mcp_catalog_probe_ttl`), pinned by
> `tests/unit/mcp/test_multiplexer_dynamic_gateway.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-ECO.multiplexer.tool-gateway-catalog` | the dynamic fleet tool catalog itself | med | ECO |
| `AU-ECO.multiplexer.running-vs-dispatchable-metrics` | child-health gauges sampled from a DIFFERENT snapshot (process/breaker state, not the probe cache) | low | ECO |

### Extension Analysis

- **Primary Extension Point**: `MCPMultiplexer._probe_cache`, the per-server
  probe result cache backing `probe_server`/`probe_catalog`.
- **Extension Strategy**: augment — fix the cache's own staleness accounting,
  not a new cache or a new tool.
- **New Concept Required?**: Yes — the fix changed the cache's behavioral
  contract (what counts as "fresh"), not just its internals.

## Problem

The fleet catalog probe cache (`MCPMultiplexer._probe_cache`) serves a
server's last-known tool/status result to avoid re-probing every child on
every call. Before this fix, "is this entry stale" was answered by a narrow
`stale` flag `probe_catalog` set only when a result was *recycled mid-call* —
that flag stays `False` forever on a cache entry nobody ever re-probes, so an
entry could silently outlive `mcp_catalog_probe_ttl` (`MCP_CATALOG_PROBE_TTL`)
by an unbounded amount with nothing to notice.

## Decision

Make staleness a computed, honest fact instead of a flag that can go stale
itself: `_probe_cache_hit`'s `(age_s, is_stale)` is derived from wall-clock
age against `ttl` on every read, never a bare echo of the narrow `stale`
flag. `probe_server`'s single-server short-circuit and `probe_catalog`'s
whole-fleet re-probe targeting both call the same helper, so they can never
disagree about what counts as fresh. A cache hit is honored only inside the
TTL; an entry aged past `mcp_catalog_probe_ttl` is treated as a miss and
re-probed — the same path a `force=True` caller takes — so an entry can never
go stale forever. The re-probe runs through the existing
`_ensure_probing`/budget machinery, so a caller that hits a stale entry is
still bounded by its own `budget` rather than blocking on a full fleet probe.

## Wire-First

Fixed by `fix(mcp): make the fleet catalog probe cache's staleness honest and
self-refreshing` (`30c00433`); pinned by
`tests/unit/mcp/test_multiplexer_dynamic_gateway.py`.
