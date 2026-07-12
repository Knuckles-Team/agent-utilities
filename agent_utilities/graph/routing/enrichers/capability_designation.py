"""KG-driven specialist designation (Plan 08 Synergy 1 -> AU-P1-3 engine-native capability index).

Wires capability-aware designation into the live router. The engine's OWN native
filtered ANN (:mod:`agent_utilities.knowledge_graph.retrieval.engine_capability_search`
— a ``query.unified`` ``Scan``/``Filter``/``Rank``/``Limit`` plan, or the native
``semantic_search`` primitive) is now the DEFAULT, authoritative candidate-selection
path: policy/tenant/capability filters are pushed down to the engine and composed
with the vector ``Rank`` in one round-trip, instead of an in-process hnswlib/numpy
scan.

The in-process :class:`~agent_utilities.knowledge_graph.retrieval.capability_index.
CapabilityIndex` this module builds is now a *bounded, non-authoritative cache*
(``_DEFAULT_BOUND`` resident ids, LRU-evicted) used only:

* as the **fallback** when the engine has no vector surface reachable at all (dev,
  or a build without embeddings/``query``/ANN) — back-compat with the original
  behaviour, and
* as a **CDC-maintained** fast path (:class:`CapabilityIndexWatcher`) that bootstraps
  ONCE via a full engine scan and thereafter upserts/evicts only the ids the
  engine's committed-change feed (:mod:`agent_utilities.graph.reactive.
  engine_subscription`) reports as changed — never a periodic full rebuild.

Reward outcomes are made durable: :func:`record_capability_outcome` updates the
in-process EMA (fast, same-process ranking) AND persists it onto the engine node
(:mod:`agent_utilities.knowledge_graph.retrieval.durable_outcome_store`), so a
routing preference survives a process restart instead of resetting to the neutral
prior.

Every entry point is fully guarded: if embeddings, an embedding model, or any node
data are unavailable, the functions return ``None`` so the router falls back to its
existing keyword scan. This wiring therefore never breaks routing — it strictly
augments it when the KG (or the engine) is rich enough.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_CALLABLE_TYPES = {
    "tool",
    "skill",
    "agent",
    "mcp_tool",
    "a2a_agent",
    "callable_resource",
    "internal_skill",
    "agent_skill",
}

# CONCEPT:AU-P1-3 — the in-process index is a bounded CACHE, not the authority (the
# engine is). This caps its resident id count; LRU-evicted beyond it.
_DEFAULT_BOUND = 4096


def _is_callable_node(props: dict[str, Any]) -> bool:
    ntype = str(props.get("type", "")).lower()
    rtype = str(props.get("resource_type", "")).lower()
    return ntype in _CALLABLE_TYPES or rtype in _CALLABLE_TYPES


def _extract_capability_fields(
    nid: str, props: dict[str, Any], backend_embeddings: dict[str, Any]
) -> dict[str, Any] | None:
    """Project one node's properties to the fields :class:`CapabilityIndex.add` needs.

    Returns ``None`` when the node is not a callable resource, is gated by the
    active release channel, or carries no embedding — the caller should then
    ensure ``nid`` is absent from the index (see :meth:`CapabilityIndexWatcher.
    _on_change`) rather than indexing it.
    """
    if not _is_callable_node(props):
        return None

    from agent_utilities.core.release_channel import active_channel, channel_visible

    node_channel = props.get("release_channel") or props.get("channel")
    if node_channel and not channel_visible(node_channel, active_channel()):
        return None

    emb = props.get("embedding") or backend_embeddings.get(nid)
    if not emb:
        return None

    caps = (
        props.get("capabilities")
        or props.get("providesCapability")
        or props.get("provides")
        or []
    )
    if isinstance(caps, str):
        caps = [caps]
    swap = props.get("swappable_with") or props.get("swappableWith") or None
    node_type = props.get("type") or props.get("node_type") or None
    tenant = props.get("tenant")
    policy_tags = props.get("policy_tags") or props.get("policyTags")
    reward = props.get("capability_reward")

    return {
        "id": nid,
        "embedding": list(emb),
        "capabilities": [str(c) for c in caps],
        "swappable_with": swap,
        "node_type": node_type,
        "tenant": tenant,
        "policy_tags": policy_tags,
        "reward": reward,
    }


def _callable_nodes_with_embeddings(engine: Any) -> list[dict[str, Any]]:
    """Best-effort enumeration of callable nodes that carry an embedding.

    Embeddings may live on the node properties (``embedding``) or in the
    backend's embedding store (``backend._embeddings``); both are checked. This is
    the ONE full-graph scan in this module — reserved for the single bootstrap of
    :class:`CapabilityIndexWatcher` (or the legacy no-CDC fallback); every
    subsequent refresh is incremental (CONCEPT:AU-P1-3).
    """
    graph = getattr(engine, "graph", None)
    if graph is None or not hasattr(graph, "node_ids"):
        return []
    backend = getattr(engine, "backend", None)
    backend_embeddings = getattr(backend, "_embeddings", {}) or {}

    try:
        node_ids = graph.node_ids()
    except Exception:
        return []

    nodes: list[dict[str, Any]] = []
    for nid in node_ids:
        try:
            props = graph._get_node_properties(nid) or {}
        except Exception:  # nosec B112 — skip malformed/unreadable nodes during best-effort scan
            continue
        fields = _extract_capability_fields(nid, props, backend_embeddings)
        if fields is not None:
            nodes.append(fields)
    return nodes


def _add_fields(index: Any, fields: dict[str, Any]) -> None:
    index.add(
        fields["id"],
        fields["embedding"],
        capabilities=set(fields["capabilities"]),
        swappable_with=fields.get("swappable_with"),
        node_type=fields.get("node_type"),
        tenant=fields.get("tenant"),
        policy_tags=fields.get("policy_tags"),
        reward=fields.get("reward"),
    )


def build_designation_index(
    engine: Any, *, capability_hierarchy: Any | None = None
) -> Any | None:
    """Build a bounded CapabilityIndex cache from the engine's callable nodes, or None.

    This is a full scan — reserved for a one-time bootstrap (see the module
    docstring). Ongoing maintenance is CDC-driven via
    :class:`CapabilityIndexWatcher`, not a repeat of this function.

    ``capability_hierarchy`` (X-4) is passed straight through to the constructed
    :class:`CapabilityIndex` — see its docstring for the ontology-subsumption-aware
    filtering this enables. ``None`` (default) is the pre-X-4 exact-match behaviour.
    """
    from agent_utilities.knowledge_graph.retrieval.capability_index import (
        CapabilityIndex,
    )

    nodes = _callable_nodes_with_embeddings(engine)
    if not nodes:
        return None
    index = CapabilityIndex(
        bounded_cache_size=_DEFAULT_BOUND, capability_hierarchy=capability_hierarchy
    )
    for fields in nodes:
        try:
            _add_fields(index, fields)
        except Exception:  # nosec B112 — skip nodes that fail to index; index remains usable
            continue
    return index if len(index) else None


class CapabilityIndexWatcher:
    """CDC-driven incremental capability-index cache (CONCEPT:AU-P1-3).

    Bootstraps ONCE via :func:`build_designation_index` (a full engine scan), then
    maintains itself purely from the engine's committed-change feed
    (:mod:`agent_utilities.graph.reactive.engine_subscription`, ``label=""`` — every
    change, since callable nodes span several labels/types): each delivered event
    upserts (or evicts) exactly the node that changed in the bounded in-process
    cache — never a re-scan of the whole graph. Construct ONCE per engine (cached
    on the engine object, mirroring ``AgentTaskDepWatcher``) and call
    :meth:`refresh` on each use.

    Degrades to a full rebuild on every :meth:`refresh` when the engine has no
    streaming surface at all (non-engine backend, or a build without the
    ``streaming`` feature) — identical to the pre-AU-P1-3 behaviour, so a dev
    environment without CDC still gets a (bounded) working cache.
    """

    def __init__(self, engine: Any, *, capability_hierarchy: Any | None = None) -> None:
        self.engine = engine
        # CONCEPT:AU-P1-3 (X-4) — baked in at construction, like bounded_cache_size;
        # a later call with a different hierarchy does not retroactively rebuild an
        # already-cached watcher (same known limitation as other construction-time
        # index parameters in this module).
        self._capability_hierarchy = capability_hierarchy
        built = build_designation_index(
            engine, capability_hierarchy=capability_hierarchy
        )
        if built is None:
            from agent_utilities.knowledge_graph.retrieval.capability_index import (
                CapabilityIndex,
            )

            built = CapabilityIndex(
                bounded_cache_size=_DEFAULT_BOUND,
                capability_hierarchy=capability_hierarchy,
            )
        self._index: Any = built
        self._subscription = self._build_subscription(engine)
        if self._subscription is not None and getattr(
            self._subscription, "available", False
        ):
            try:
                self._subscription.catch_up()
            except Exception as e:  # noqa: BLE001 — cold-start catch-up is best-effort
                logger.debug("CapabilityIndexWatcher: catch_up failed: %s", e)

    def _build_subscription(self, engine: Any) -> Any | None:
        try:
            from agent_utilities.graph.reactive.engine_subscription import subscribe
        except Exception as e:  # noqa: BLE001 — subsystem unimportable -> full-rebuild fallback
            logger.debug(
                "CapabilityIndexWatcher: engine_subscription unavailable: %s", e
            )
            return None
        try:
            return subscribe(engine, "", self._on_change)
        except Exception as e:  # noqa: BLE001
            logger.debug("CapabilityIndexWatcher: subscribe failed: %s", e)
            return None

    def _on_change(self, event: dict[str, Any]) -> None:
        """Incrementally upsert/evict exactly the node this CDC event names."""
        node_id = event.get("node_id")
        if not node_id:
            return
        node_id = str(node_id)
        kind = str(event.get("kind", "")).lower()
        after = event.get("after")

        if "delete" in kind or "remove" in kind or after is None:
            try:
                self._index.remove(node_id)
            except Exception as e:  # noqa: BLE001 — a bad event never wedges the cache
                logger.debug(
                    "CapabilityIndexWatcher: remove failed for %r: %s", node_id, e
                )
            return

        props = after if isinstance(after, dict) else {}
        backend = getattr(self.engine, "backend", None)
        backend_embeddings = getattr(backend, "_embeddings", {}) or {}
        fields = _extract_capability_fields(node_id, props, backend_embeddings)
        if fields is None:
            # No longer callable / no embedding yet — ensure it's absent so a stale
            # entry never lingers past the change that invalidated it.
            try:
                self._index.remove(node_id)
            except Exception:  # noqa: BLE001 — best-effort
                pass
            return
        try:
            _add_fields(self._index, fields)
        except Exception as e:  # noqa: BLE001 — a bad event never wedges the cache
            logger.debug("CapabilityIndexWatcher: upsert failed for %r: %s", node_id, e)

    def refresh(self, *, force_full: bool = False) -> Any | None:
        """Deliver pending CDC changes (or, without CDC, a full rebuild).

        ``force_full=True`` bypasses CDC gating entirely (an explicit caller
        ``refresh``) and always re-scans the engine.
        """
        if force_full:
            rebuilt = build_designation_index(
                self.engine, capability_hierarchy=self._capability_hierarchy
            )
            if rebuilt is not None:
                self._index = rebuilt
            return self._index

        sub = self._subscription
        if sub is None or not getattr(sub, "available", False):
            # No CDC surface reachable — the only correctness-preserving option is
            # a full rebuild each time (matches the pre-AU-P1-3 behaviour exactly).
            rebuilt = build_designation_index(
                self.engine, capability_hierarchy=self._capability_hierarchy
            )
            if rebuilt is not None:
                self._index = rebuilt
            return self._index

        try:
            sub.poll(
                block_ms=0
            )  # delivers only changed events -> _on_change upserts in place
        except Exception as e:  # noqa: BLE001 — a feed hiccup keeps the last-known cache
            logger.debug("CapabilityIndexWatcher: poll failed: %s", e)
        return self._index

    @property
    def index(self) -> Any:
        return self._index


def get_designation_index(
    engine: Any, *, refresh: bool = False, capability_hierarchy: Any | None = None
) -> Any | None:
    """Return the CDC-maintained bounded CapabilityIndex cache for ``engine``.

    Builds (and caches on ``engine``) a :class:`CapabilityIndexWatcher` on first
    use; every call thereafter is O(new-CDC-changes), not a rebuild. ``refresh=True``
    forces one explicit full rebuild (bypassing CDC gating) — the pre-AU-P1-3
    semantics of this flag are preserved for existing callers. ``capability_hierarchy``
    (X-4) only takes effect on the FIRST call for a given engine (it is baked into
    the watcher at construction, like ``bounded_cache_size``); later calls reuse
    the already-cached watcher's hierarchy.
    """
    watcher = getattr(engine, "_capability_index_watcher", None)
    if watcher is None:
        try:
            watcher = CapabilityIndexWatcher(
                engine, capability_hierarchy=capability_hierarchy
            )
        except Exception as e:  # noqa: BLE001 — degrade to the legacy one-shot build
            logger.debug("get_designation_index: watcher construction failed: %s", e)
            index = build_designation_index(
                engine, capability_hierarchy=capability_hierarchy
            )
            try:
                engine._designation_index = index
            except Exception:
                pass
            return index
        try:
            engine._capability_index_watcher = watcher
        except Exception:
            pass  # engine may not allow attribute assignment; watcher just isn't cached

    index = watcher.refresh(force_full=refresh)
    try:
        engine._designation_index = (
            index  # back-compat: callers/tests read this attribute
        )
    except Exception:
        pass
    return index


def embed_query(query: str, embed_fn: Any = None) -> Any | None:
    """Resolve ``query`` to an embedding vector via ``embed_fn``, or the default model.

    Shared by every designation entry point in this module (and by the X-4
    ``capability_routing`` module) so embedding-model resolution has exactly one
    implementation. Returns ``None`` (never raises) when no model is configured.
    """
    if embed_fn is None:
        from agent_utilities.core.embedding_utilities import create_embedding_model

        model = create_embedding_model()
        if model is None:
            return None
        embed_fn = model.get_text_embedding
    return embed_fn(query)


def designate_specialists(
    engine: Any,
    query: str,
    *,
    k: int = 5,
    required_caps: list[str] | None = None,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    embed_fn: Any = None,
    capability_hierarchy: Any | None = None,
) -> list[str] | None:
    """Designate the top-``k`` callable resource ids for ``query``.

    Tries the engine's native filtered ANN first (CONCEPT:AU-P1-3 — the engine is
    the authority); falls back to the bounded, CDC-maintained in-process cache when
    no engine vector surface is reachable. Returns a list of ids, or ``None`` if
    designation is unavailable at all (no embeddings, no model, empty index) —
    signalling the caller to fall back to its keyword scan.

    ``capability_hierarchy`` (X-4) makes ``required_caps`` ontology-subsumption-aware
    on BOTH paths (engine-native push-down + in-process fallback). ``None``
    (default) is the pre-X-4 exact-match behaviour, unchanged for every existing
    caller.
    """
    try:
        embedding = embed_query(query, embed_fn)
        if embedding is None:
            return None

        from agent_utilities.knowledge_graph.retrieval.engine_capability_search import (
            engine_filtered_search,
        )

        engine_hits = engine_filtered_search(
            engine,
            embedding,
            k=k,
            required_caps=required_caps,
            tenant=tenant,
            policy_tags=policy_tags,
            capability_hierarchy=capability_hierarchy,
        )
        if engine_hits is not None:
            return [nid for nid, _score in engine_hits]

        # No engine vector surface reachable at all -> bounded in-process fallback.
        index = get_designation_index(engine, capability_hierarchy=capability_hierarchy)
        if index is None or len(index) == 0:
            return None

        designations = index.designate(
            embedding,
            required_caps=required_caps,
            k=k,
            tenant=tenant,
            required_policy_tags=policy_tags,
        )
        return [d.id for d in designations]
    except Exception as e:  # never break routing
        logger.debug("KG-driven designation unavailable, falling back: %s", e)
        return None


def record_capability_outcome(
    engine: Any,
    entity_id: str,
    *,
    success: bool | None = None,
    reward: float | None = None,
    alpha: float = 0.3,
    source_ids: list[str] | None = None,
) -> float:
    """Record a routing outcome DURABLY (CONCEPT:AU-P1-3 — durable contextual-bandit outcomes).

    Updates the in-process bounded cache's reward EMA (so the very next
    same-process ``designate_specialists`` call already reflects it) AND persists
    the EMA onto the engine node (:func:`~...durable_outcome_store.
    persist_capability_reward`), so the learned preference survives a process
    restart instead of resetting to the neutral 0.5 prior. Never raises — a
    failure on either side falls back to the other's result.

    ``source_ids`` (X-6 / Seam 3, CONCEPT:EG-KG.epistemic.truth-maintenance) is
    the SHARED writeback seam for THIS module's derived-data writeback: when a
    caller passes the real id(s) of the observation the reward was just computed
    from (e.g. :class:`~agent_utilities.knowledge_graph.research.claim_flywheel.
    ClaimFlywheel.record_outcome`'s freshly-written ``ClaimOutcome`` node id —
    never fabricated), and the durable persist above actually landed, this stamps
    a ``:DerivedFrom`` edge from ``entity_id`` to each id and registers
    ``entity_id`` as a live engine-side TruthMaintenance materialization off that
    provenance — so a later change to the observation marks the durable reward
    stale. Omitted (default) by callers with no discrete observation node to
    point at — the durable write still lands, just unregistered, same as before
    this parameter existed. Best-effort, never gates the write above.
    """
    updated: float | None = None

    watcher = getattr(engine, "_capability_index_watcher", None)
    if watcher is not None:
        try:
            updated = watcher.index.record_outcome(
                entity_id, success=success, reward=reward, alpha=alpha
            )
        except Exception as e:  # noqa: BLE001 — durable persistence below still applies
            logger.debug(
                "record_capability_outcome: in-process update failed for %r: %s",
                entity_id,
                e,
            )

    try:
        from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
            persist_capability_reward,
        )

        durable = persist_capability_reward(
            engine, entity_id, success=success, reward=reward, alpha=alpha
        )
        if durable is not None:
            updated = durable
            if source_ids:
                _register_capability_reward_materialization(
                    engine, entity_id, source_ids
                )
    except Exception as e:  # noqa: BLE001 — durability is an augmentation, never load-bearing
        logger.debug(
            "record_capability_outcome: durable persistence failed for %r: %s",
            entity_id,
            e,
        )

    return updated if updated is not None else 0.5


def _register_capability_reward_materialization(
    engine: Any, entity_id: str, source_ids: list[str]
) -> None:
    """X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance) — see
    :func:`record_capability_outcome`'s ``source_ids`` docstring. Best-effort;
    never raises into the caller's durable-write path."""
    for source_id in source_ids:
        try:
            engine.add_edge(entity_id, source_id, relationship_type="DERIVED_FROM")
        except Exception as e:  # noqa: BLE001 — provenance edges are best-effort
            logger.debug(
                "record_capability_outcome: derived_from edge %r->%r failed: %s",
                entity_id,
                source_id,
                e,
            )
    try:
        engine.register_materialization(entity_id)
    except Exception as e:  # noqa: BLE001 — TMS registration is best-effort
        logger.debug(
            "record_capability_outcome: register_materialization failed for %r: %s",
            entity_id,
            e,
        )


def explain_capability_eligibility(
    engine: Any,
    entity_id: str,
    *,
    required_caps: list[str] | None = None,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    capability_hierarchy: Any | None = None,
) -> dict[str, Any] | None:
    """Explain WHY ``entity_id`` is (or isn't) eligible under the given filters.

    CONCEPT:AU-P1-3 — explainable routing. Reads the bounded in-process cache (built/
    maintained the same way :func:`designate_specialists` maintains it); returns
    ``None`` when the cache is unavailable or the entity was never indexed, so the
    caller can distinguish "not eligible" from "not known". ``capability_hierarchy``
    (X-4) makes the returned ``missing_caps``/``eligible`` verdict subsumption-aware
    and adds a ``subsumption_paths`` entry — see
    :func:`~agent_utilities.knowledge_graph.retrieval.capability_index.compute_eligibility`.
    For a candidate the in-process cache never resident (the common case when the
    engine's own filtered ANN is authoritative), prefer
    ``graph.routing.enrichers.capability_routing.explain_routing_eligibility``,
    which computes the same shape directly from the engine's node properties.
    """
    index = get_designation_index(engine, capability_hierarchy=capability_hierarchy)
    if index is None or len(index) == 0:
        return None
    if entity_id not in index:
        return None
    return index.explain(
        entity_id,
        required_caps=required_caps,
        tenant=tenant,
        required_policy_tags=policy_tags,
    )
