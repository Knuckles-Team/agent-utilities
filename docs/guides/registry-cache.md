# Registry Hot Cache

> **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation** — Session-Scoped Registry Optimization

This document provides a focused deep-dive into the Registry Hot Cache layer, the performance architecture that eliminates redundant specialist lookups and reduces prompt bloat.

## Motivation

In a system with 30+ MCP servers, each contributing 5-15 tools, the specialist registry can contain 50+ entries. Before CONCEPT:AU-ORCH.adapter.hot-cache-invalidation, **every routing call** performed:

1. A full Cypher query against the Knowledge Graph to fetch all specialists
2. Serialization of all specialist descriptions into the LLM prompt
3. The LLM parsing through thousands of tokens of specialist metadata

This created two compounding problems:

- **Latency**: 50-200ms per registry scan depending on graph backend
- **Prompt bloat**: 50 specialist descriptions ≈ 3,000-5,000 tokens consumed before the LLM even sees the user's query

## Architecture

### `_RegistryCache` Singleton

```python
class _RegistryCache:
    """Session-scoped cache holding the computed specialist registry."""
    _registry: MCPAgentRegistryModel | None = None
    _prompts: dict[str, str] = {}
    _tool_agent_map: dict[str, list[str]] = {}

    @classmethod
    def get_registry(cls) -> MCPAgentRegistryModel:
        if cls._registry is None:
            cls._registry = _fetch_registry_from_kg()
        return cls._registry

    @classmethod
    def invalidate(cls) -> None:
        cls._registry = None
        cls._prompts.clear()
        cls._tool_agent_map.clear()
```

> The cache uses class-level state (no TTL); invalidation is purely
> event-driven. It lives in `core/config.py`.

### Public API

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_discovery_registry()` | `(engine?) → MCPAgentRegistryModel` | Returns the full cached registry; hydrates on first call |
| `get_relevant_specialists()` | `(query, engine, top_n=7) → list[MCPAgent]` | Returns only the top-N specialists relevant to the query |
| `invalidate_registry_cache()` | `() → None` | Clears all cached data; next call re-hydrates from KG |

### Filtering Algorithm

`get_relevant_specialists()` runs the KG hybrid search (`engine.search_hybrid`)
over the query, fetching `top_n × 3` candidates, and keeps the agents whose
names (or `agent`/`prompt` node matches) appear in the results, capped at
`top_n`. If the engine is unavailable or returns nothing, it falls back to the
full registry list capped at `top_n` (default 7).

## Cache Invalidation Strategy

The cache uses an **event-driven invalidation** model — it is never TTL-based. Invalidation only occurs when the underlying data actually changes:

```mermaid
graph TD
    subgraph Triggers ["Invalidation Event Sources"]
        MCP["AU-ECO.mcp.toolkit-live-discovery: POST /mcp/reload\n(New tools discovered)"]
        Pipeline["AU-ECO.mcp.toolkit-live-discovery: Pipeline Completion\n(Code graph changed)"]
        SelfModel["AHE-3.3: SelfModel.update_after_session()\n(New proficiency data)"]
        TeamConfig["AHE-3.3: promote_coalition_to_template()\n(New team template)"]
    end

    subgraph Cache ["_RegistryCache"]
        Inv["AHE-3.1: invalidate_registry_cache()"]
        Clear["ORCH-1.2: _registry = None\n_prompts = {} / _tool_agent_map = {}"]
    end

    MCP --> Inv
    Pipeline --> Inv
    SelfModel --> Inv
    TeamConfig --> Inv
    Inv --> Clear
```

### Why Not TTL-Based?

TTL (time-to-live) caching is inappropriate here because:

1. **Low write frequency**: The registry changes only on server restart, MCP reload, or pipeline runs — not on every request
2. **Consistency requirement**: A stale cache could route queries to wrong specialists
3. **Event sources are known**: All mutation points are within our codebase and can emit invalidation signals

### Why Not a NetworkX Hot Layer?

The alternative considered was maintaining a parallel NetworkX subgraph of "hot" specialists. This was rejected because:

1. **Dual-write complexity**: Keeping NetworkX and the cache in sync adds a second consistency surface
2. **Over-engineering**: The `_RegistryCache` pattern is simpler (dict + singleton) and achieves the same O(1) lookup
3. **Memory overhead**: NetworkX stores full graph topology; the cache stores only the computed registry model

## Performance Impact

| Metric | Before (O(N)) | After (O(1)) | Improvement |
|--------|---------------|--------------|-------------|
| Registry lookup | 50-200ms | <1ms | 50-200x |
| Prompt tokens (specialist descriptions) | 3,000-5,000 | 400-700 (top-7 only) | ~7x reduction |
| LLM routing accuracy | Baseline | +15% (less noise in prompt) | Measurable |

## Integration Points

The cache integrates at these specific locations in the codebase:

| File | Function | Role |
|------|----------|------|
| `core/config.py` | `get_discovery_registry()` | Cache hydration + retrieval |
| `core/config.py` | `get_relevant_specialists()` | Filtered subset retrieval |
| `core/config.py` | `invalidate_registry_cache()` | Event-driven invalidation |
| `graph/_router_impl.py` | router specialist injection | Consumer: uses filtered specialists in LLM prompt |
| `mcp/agent_manager.py` | `sync_mcp_agents()` | Trigger: invalidates after MCP sync |
| `knowledge_graph/pipeline/runner.py` | `PipelineRunner.run()` | Trigger: invalidates after pipeline |
| `knowledge_graph/retrieval/memory_retriever.py` | `update_after_session()` | Trigger: invalidates after self-model update |
| `core/registry/kg_adapter.py` | `promote_coalition_to_template()` | Trigger: invalidates after TeamConfig promotion |

## Related Documentation

- [First Principles Architecture](first-principles.md) — Complete CONCEPT:AU-ORCH.adapter.hot-cache-invalidation through CONCEPT:AU-ECO.messaging.native-backend-abstraction overview
- [Architecture](architecture.md) — Full system architecture
- [Emergent Architecture](emergent-architecture.md) — Self-Model and Workspace Attention
