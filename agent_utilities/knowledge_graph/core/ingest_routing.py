# CONCEPT:AU-KG.ingest.unified-query-routing - Ingestion graph routing: map each ingestion item to a per-source/per-repo/per-tenant destination graph so writes spread across the engine's K-way redb shard writers (EG-026) instead of funnelling into the single __commons__ graph; unified query preserved by fanning reads across the active content-graph set.
"""Ingestion graph routing (CONCEPT:AU-KG.ingest.unified-query-routing).

The problem this removes (north-star roadmap item A). The durable engine shards
its redb writer K ways by ``FNV-1a(graph_name) % K`` (epistemic-graph EG-026), so
each *graph name* pins to exactly one writer thread / core. Almost all ingestion
wrote the single ``__commons__`` graph, so K-1 of the K shard writers sat idle
while one did every commit. Spreading content across **per-source / per-repo /
per-tenant** graph names makes those names hash to different shards, so K cores
commit in parallel — the same "shard the bottleneck" move EG-026 applied to the
writer, now applied to the graph axis it keys on.

The routing is a single deterministic policy (:func:`route_graph`) — a stable,
configurable seam, not string literals scattered through the ingest adaptors:

* codebase repo ``agent-utilities`` → ``code:agent-utilities``
* connector ``servicenow``         → ``src:servicenow``
* chat for agent ``planner``       → ``chat:planner``
* research source ``arxiv``        → ``research:arxiv``
* a tenant-scoped item             → the existing per-tenant graph
  (:func:`~agent_utilities.knowledge_graph.core.shard_topology.tenant_graph_name`)
* anything with no natural owner    → the configured default graph (``__commons__``)

**Preserving unified query (the correctness point).** A node written to
``code:X`` lives in a *different* engine graph than ``__commons__``, so a default
single-graph read would not see it. Routing therefore maintains a lightweight
in-process **registry of the active content graphs** (seeded from the engine's
tenant list) and the read tools fan a default/implicit-target query across
``{default + active content graphs}`` and merge — so split content stays
queryable as one KG. See ``mcp/tools/query_tools.py``.

**Opt-in.** Gated by ``KG_INGEST_GRAPH_ROUTING`` (default OFF). When OFF every
path is byte-for-byte today's behaviour: ingestion writes ``__commons__`` (or the
ambient tenant graph) and reads hit the single default graph. Existing
``__commons__`` content is never moved — flipping the flag only changes where NEW
data lands; the EG-030 shard-migration tool handles relocating old data if ever
wanted.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from .shard_topology import default_graph_name, tenant_graph_name

logger = logging.getLogger(__name__)

__all__ = [
    "CONTENT_GRAPH_PREFIXES",
    "active_content_graphs",
    "engine_for_graph",
    "is_content_graph",
    "read_graph_targets",
    "register_content_graph",
    "route_graph",
    "routing_enabled",
    "safe_engine_for_graph",
    "shard_bucket_for",
    "shard_fanout_enabled",
]

#: Destination-graph prefixes by ingestion class. A graph whose name starts with
#: one of these is a routed "content graph" the read path must union over.
CODE_PREFIX = "code:"
SOURCE_PREFIX = "src:"
CHAT_PREFIX = "chat:"
RESEARCH_PREFIX = "research:"
CONTENT_GRAPH_PREFIXES: tuple[str, ...] = (
    CODE_PREFIX,
    SOURCE_PREFIX,
    CHAT_PREFIX,
    RESEARCH_PREFIX,
)

# Graph names should stay shell/file/metric friendly (engine tenants are string
# keyed). Mirrors the tenant slug discipline in shard_topology.
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _config(config: Any = None) -> Any:
    if config is not None:
        return config
    from agent_utilities.core.config import AgentConfig

    return AgentConfig()


def _slug(value: Any) -> str:
    """Filesystem/metric-safe lowercase slug, or ``""`` when empty."""
    if not value:
        return ""
    return _SLUG_RE.sub("-", str(value).strip()).strip("-_.").lower()


def routing_enabled(config: Any = None) -> bool:
    """Whether per-source ingestion graph routing is active (``KG_INGEST_GRAPH_ROUTING``)."""
    return bool(getattr(_config(config), "kg_ingest_graph_routing", False))


def shard_fanout_enabled(config: Any = None) -> bool:
    """Whether per-shard content-keyed sub-graph fanout is active
    (``KG_INGEST_SHARD_FANOUT``; requires routing; CONCEPT:AU-KG.ingest.batched-cross-graph-writer)."""
    cfg = _config(config)
    return bool(getattr(cfg, "kg_ingest_shard_fanout", False)) and routing_enabled(cfg)


def _fnv1a(text: str) -> int:
    """32-bit FNV-1a hash — mirrors the engine's ``FNV-1a(graph_name) % K`` shard
    keying (EG-026) so a content key buckets deterministically and cheaply, with
    no engine round-trip (CONCEPT:AU-KG.ingest.batched-cross-graph-writer)."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def shard_bucket_for(content_key: str, k: int) -> int:
    """The shard bucket ``[0, k)`` for ``content_key`` (deterministic; CONCEPT:AU-KG.ingest.batched-cross-graph-writer)."""
    if k <= 1:
        return 0
    return _fnv1a(str(content_key)) % k


def is_content_graph(name: str | None) -> bool:
    """True for a routed content graph (one of :data:`CONTENT_GRAPH_PREFIXES`)."""
    return bool(name) and str(name).startswith(CONTENT_GRAPH_PREFIXES)


def route_graph(
    *,
    kind: str | None = None,
    source_type: str | None = None,
    repo: str | None = None,
    tenant: str | None = None,
    agent: str | None = None,
    content_key: str | None = None,
    config: Any = None,
) -> str:
    """Map an ingestion item to its destination graph name (deterministic).

    The single routing policy. Resolution order:

    1. **routing disabled** → the tenant graph (if a tenant is in scope) else the
       configured default graph — byte-for-byte today's behaviour.
    2. **tenant in scope** → the per-tenant graph (a tenant stays wholly on one
       graph, as today; CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw).
    3. **codebase** (``kind="code"`` or a ``repo``) → ``code:<repo>``.
    4. **chat** (``kind="chat"``) → ``chat:<agent>``.
    5. **research** (``kind="research"``) → ``research:<source_type|repo>``.
    6. **connector / external source** (a ``source_type``) → ``src:<source_type>``.
    7. otherwise → the configured default graph.

    A slug that comes out empty falls through to the default graph rather than
    emitting a degenerate ``code:`` name.

    **Per-shard content-keyed fanout** (CONCEPT:AU-KG.ingest.batched-cross-graph-writer). When
    ``KG_INGEST_SHARD_FANOUT`` is on AND a ``content_key`` is supplied AND the
    resolved graph is a routed *content* graph (not a tenant/default graph — a
    tenant must stay whole), a ``#<bucket>`` suffix keyed by ``content_key`` fans a
    single high-volume source across ``K`` distinct sub-graph names so the
    memory-gen write stage spreads over all K redb shard writers instead of
    pinning one. Codebase graphs are already per-repo (naturally sharded), so the
    fanout applies to ``src:`` / ``research:`` / ``chat:`` sources. The suffix keeps
    the source prefix, so :func:`is_content_graph` still recognises the sub-graph
    and the unified read unions it.
    """
    cfg = _config(config)
    default = default_graph_name(cfg)

    if not routing_enabled(cfg):
        return tenant_graph_name(tenant, base=default) if tenant else default

    # A tenant owns its whole graph regardless of source kind (never fanned out).
    if tenant:
        return tenant_graph_name(tenant, base=default)

    graph = default
    if kind == "code" or repo:
        slug = _slug(repo)
        if slug:
            # Codebase is already per-repo sharded; no content fanout.
            return f"{CODE_PREFIX}{slug}"
    elif kind == "chat":
        slug = _slug(agent)
        if slug:
            graph = f"{CHAT_PREFIX}{slug}"
    elif kind == "research":
        slug = _slug(source_type or repo or "papers")
        graph = f"{RESEARCH_PREFIX}{slug}"
    elif source_type:
        slug = _slug(source_type)
        if slug:
            graph = f"{SOURCE_PREFIX}{slug}"

    # Fan a hot single source across K shard-keyed sub-graphs when enabled.
    if content_key and is_content_graph(graph) and shard_fanout_enabled(cfg):
        from .worker_scheduler import durable_shard_writers

        k = max(1, durable_shard_writers())
        if k > 1:
            return f"{graph}#{shard_bucket_for(content_key, k)}"

    return graph


# ---------------------------------------------------------------------------
# Active content-graph registry (preserves unified query)
# ---------------------------------------------------------------------------

_active_lock = threading.RLock()
_active_graphs: set[str] = set()
_seeded = False

# One cached read engine per content graph (built lazily). Keyed by graph name.
_engine_cache_lock = threading.RLock()
_engine_cache: dict[str, Any] = {}


def register_content_graph(name: str | None) -> None:
    """Record that ``name`` is an active routed content graph (idempotent).

    Called by the ingest write path each time it routes content to a graph, so
    the read path can later fan a unified query across the live set.
    """
    if is_content_graph(name):
        with _active_lock:
            _active_graphs.add(str(name))


def _seed_from_engine() -> None:
    """Best-effort one-time seed of the active set from the engine's tenant list.

    After a restart the in-process set is empty until ingestion re-registers a
    graph; seeding from ``client.tenants.list()`` lets reads see already-written
    content graphs immediately.
    """
    global _seeded
    with _active_lock:
        if _seeded:
            return
        _seeded = True
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active()
        gc = getattr(engine, "graph_compute", None)
        client = getattr(gc, "_client", None)
        if client is None:
            return
        for entry in client.tenants.list() or []:
            gname = entry.get("name") if isinstance(entry, dict) else entry
            if is_content_graph(gname):
                with _active_lock:
                    _active_graphs.add(str(gname))
    except Exception:  # noqa: BLE001 — seeding is best-effort, never fatal to a read
        logger.debug("content-graph seed from engine skipped", exc_info=True)


def active_content_graphs(*, seed: bool = True) -> list[str]:
    """Sorted list of the active routed content graphs (optionally seeded once)."""
    if seed:
        _seed_from_engine()
    with _active_lock:
        return sorted(_active_graphs)


def read_graph_targets(config: Any = None) -> list[str]:
    """The graphs a unified default read must union over: default + content graphs.

    Returns ``[default]`` (single, legacy) when routing is disabled or nothing has
    been routed yet, so the read path stays on the fast single-graph path until
    content is actually spread.
    """
    default = default_graph_name(_config(config))
    if not routing_enabled(config):
        return [default]
    graphs = active_content_graphs()
    if not graphs:
        return [default]
    # Default first (legacy __commons__ content + control plane), then the
    # routed content graphs; de-duplicated, order-stable.
    out = [default]
    out.extend(g for g in graphs if g != default)
    return out


def engine_for_graph(name: str) -> Any:
    """A cached read engine bound to content graph ``name``.

    Builds an ``IntelligenceGraphEngine`` over an ``EpistemicGraphBackend`` bound
    to ``name`` (CONCEPT:AU-KG.backend.schedule-on-control-graph) so the standard query/search surface runs
    against that one graph. Cached per graph — connections are reused across
    fan-out reads.
    """
    with _engine_cache_lock:
        eng = _engine_cache.get(name)
        if eng is not None:
            return eng
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine(backend=EpistemicGraphBackend(graph_name=name))
    with _engine_cache_lock:
        existing = _engine_cache.get(name)
        if existing is not None:
            return existing
        _engine_cache[name] = engine
        return engine


def safe_engine_for_graph(name: str) -> tuple[Any, str | None]:
    """``engine_for_graph`` variant for fan-out: ``(engine, error)`` (never raises)."""
    try:
        return engine_for_graph(name), None
    except Exception as e:  # noqa: BLE001 — partial-success fan-out contract
        return None, str(e)


def _reset_for_tests() -> None:
    """Test hook: drop the active set + cached engines (CONCEPT:AU-KG.ingest.unified-query-routing)."""
    global _seeded
    with _active_lock:
        _active_graphs.clear()
        _seeded = False
    with _engine_cache_lock:
        _engine_cache.clear()
