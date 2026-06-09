#!/usr/bin/python
from __future__ import annotations

"""Object Data Funnel — graph-to-search-index sync (CONCEPT:KG-2.44).

Palantir provenance: *object-indexing/overview* (the **Object Data Funnel**) and
*object-backend*. Foundry funnels objects from the source-of-truth store into a
search/serving index through two complementary pipelines:

* a **batch** pipeline that rebuilds the index in bulk, and
* a **streaming/incremental** pipeline that applies a delta of upserts/deletes to
  the *live* index without a full rebuild,

both subject to **data restrictions** (security/eligibility filters that decide
which objects may enter the index) and **metadata sync** (the staleness/reindex
half, see :mod:`.staleness`).

This module is that funnel on top of the existing
:class:`~agent_utilities.knowledge_graph.retrieval.capability_index.CapabilityIndex`
(the live HNSW/numpy search structure the router already calls via
``KnowledgeGraph.designate``). It does **not** reinvent the index; it drives it:

* :class:`DataRestriction` — a composable predicate over a node's type/props that
  decides index-eligibility (the funnel's data-restriction stage).
* :class:`ObjectIndexFunnel` — owns one ``CapabilityIndex`` + one
  :class:`~...staleness.StalenessLedger` and exposes:
    - :meth:`batch_sync` — full rebuild via ``CapabilityIndex.build_from_edges``.
    - :meth:`incremental_sync` — apply an upsert/delete delta to the live index.
      Upserts use the index's native in-place ``add`` (HNSW ``add_items`` when
      that backend is active, else the numpy map). Deletes are applied as a real
      delta overlay: a tombstone set that filters live ``designate`` results and
      drops the vector/capability/ledger entries, with automatic compaction
      (a real rebuild) once the tombstone ratio crosses a threshold so HNSW does
      not accumulate unreachable labels forever.
    - :meth:`reconcile` — observe the current source, compute drift via the
      ledger, and apply exactly the needed incremental delta.

Wire-First: the funnel is consumed by the live retrieval plane — it produces and
maintains the same ``CapabilityIndex`` that ``KnowledgeGraph.designate`` ranks
against, and its :meth:`search` delegates straight to ``designate`` so callers
get tombstone-filtered, restriction-respecting results on the live path.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...retrieval.capability_index import CapabilityIndex, Designation
from .staleness import StalenessLedger, content_hash

logger = logging.getLogger(__name__)

__all__ = [
    "DataRestriction",
    "FunnelDelta",
    "SyncResult",
    "ObjectIndexFunnel",
    "index_payload_of",
    "id_of",
]

# Keys that constitute an object's index-relevant payload for hashing/staleness.
_PAYLOAD_KEYS = ("embedding", "capabilities", "swappable_with")


def _getter_for(node: Any) -> Callable[..., Any]:
    """Return a uniform ``get(key, default)`` over a dict or attribute object."""
    if isinstance(node, Mapping):
        return node.get

    def getter(key: str, default: Any = None, _n: Any = node) -> Any:
        return getattr(_n, key, default)

    return getter


def id_of(node: Any) -> str | None:
    """Extract the entity id from a node mapping/object (``None`` if absent)."""
    nid = _getter_for(node)("id")
    return None if nid is None else str(nid)


def _caps_of(node: Any) -> list[str]:
    g = _getter_for(node)
    caps = g("capabilities") or g("provides") or g("providesCapability") or []
    return [str(c) for c in caps]


def _swap_of(node: Any) -> list[str]:
    g = _getter_for(node)
    swap = g("swappable_with") or g("swappableWith") or []
    return [str(s) for s in swap]


def index_payload_of(node: Any) -> dict[str, Any]:
    """Extract the index-relevant payload used for content hashing/staleness.

    Only the fields that actually affect retrieval (embedding, capabilities,
    swappable adjacency) are included, so a touch that changes an unrelated
    property does not register as index drift.
    """
    g = _getter_for(node)
    emb = g("embedding")
    payload: dict[str, Any] = {
        "embedding": list(emb) if emb is not None else None,
        "capabilities": sorted(_caps_of(node)),
        "swappable_with": sorted(_swap_of(node)),
    }
    return payload


@dataclass
class DataRestriction:
    """Index-eligibility predicate (the funnel's data-restriction stage).

    A restriction decides whether an object may enter the search index. It is
    composed of:

    * ``allowed_types`` — if non-empty, the node's ``type``/``node_type`` must be
      in this set (case-insensitive).
    * ``denied_types`` — node types that are always excluded.
    * ``predicate`` — an arbitrary ``(node) -> bool`` for property-level rules
      (e.g. exclude ``restricted`` classification, require an embedding present).

    An object is eligible iff it passes all configured stages. With no stages
    configured the restriction admits everything (the open default).

    Provenance: Palantir *object-indexing* data restrictions — objects excluded
    from indexing for security/eligibility reasons never reach the serving index.
    """

    allowed_types: set[str] = field(default_factory=set)
    denied_types: set[str] = field(default_factory=set)
    predicate: Callable[[Any], bool] | None = None
    name: str = "data_restriction"

    @staticmethod
    def _type_of(node: Any) -> str | None:
        g = _getter_for(node)
        t = g("type") or g("node_type") or g("nodeType")
        return None if t is None else str(t).lower()

    def admits(self, node: Any) -> bool:
        """Whether ``node`` is eligible for the search index."""
        node_type = self._type_of(node)
        if self.denied_types and node_type in {t.lower() for t in self.denied_types}:
            return False
        if self.allowed_types:
            if node_type is None:
                return False
            if node_type not in {t.lower() for t in self.allowed_types}:
                return False
        if self.predicate is not None:
            try:
                if not self.predicate(node):
                    return False
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("DataRestriction predicate raised for node: %s", exc)
                return False
        return True

    def filter(self, nodes: Iterable[Any]) -> list[Any]:
        """Return only the nodes this restriction admits."""
        return [n for n in nodes if self.admits(n)]


@dataclass
class FunnelDelta:
    """A streaming-sync delta: objects to upsert and ids to delete.

    Attributes:
        upserts: Node mappings/objects to add-or-replace in the index.
        deletes: Object ids to remove from the index.
        source_watermark: Monotonic source revision the delta corresponds to
            (recorded in the staleness ledger for upserted objects).
    """

    upserts: list[Any] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    source_watermark: float = 0.0


@dataclass
class SyncResult:
    """Outcome of a sync operation, for audit / MCP transport."""

    mode: str
    upserted: int = 0
    deleted: int = 0
    skipped_restricted: int = 0
    rebuilt: bool = False
    live_size: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "upserted": self.upserted,
            "deleted": self.deleted,
            "skipped_restricted": self.skipped_restricted,
            "rebuilt": self.rebuilt,
            "live_size": self.live_size,
        }


class ObjectIndexFunnel:
    """Sync objects from the source-of-truth graph into the live search index.

    Owns a single :class:`CapabilityIndex` (the structure
    ``KnowledgeGraph.designate`` ranks against), a :class:`DataRestriction`
    eligibility gate, and a :class:`StalenessLedger` tracking per-object index
    freshness. Provides batch (full rebuild) and incremental/streaming (live
    delta) sync, with delete implemented as a real tombstone overlay that is
    compacted by an actual rebuild once it grows past ``compaction_threshold``.

    Args:
        index: An existing :class:`CapabilityIndex` to drive (one is created if
            omitted, inheriting ``dim``/``prefer_backend``).
        restriction: Index-eligibility gate (open default if omitted).
        dim: Embedding dimensionality for a freshly created index.
        prefer_backend: Force ``"hnsw"``/``"numpy"`` for a created index.
        compaction_threshold: When tombstoned-fraction of the live index exceeds
            this, an incremental sync triggers a real compacting rebuild that
            physically evicts deleted objects from the underlying ANN structure.
    """

    def __init__(
        self,
        index: CapabilityIndex | None = None,
        *,
        restriction: DataRestriction | None = None,
        dim: int | None = None,
        prefer_backend: str | None = None,
        compaction_threshold: float = 0.25,
    ) -> None:
        # NB: an *empty* CapabilityIndex is falsy (``__len__`` == 0), so this must
        # be an explicit ``is None`` check — ``index or CapabilityIndex(...)``
        # would wrongly create a second index when handed the live (empty) one,
        # silently decoupling the funnel from the index the router ranks against.
        self.index: CapabilityIndex = (
            index
            if index is not None
            else CapabilityIndex(dim=dim, prefer_backend=prefer_backend)
        )
        self.restriction: DataRestriction = restriction or DataRestriction()
        self.ledger: StalenessLedger = StalenessLedger()
        self.compaction_threshold = float(compaction_threshold)
        # Live source-of-truth payloads for admitted objects, keyed by id. Kept
        # so a tombstone-compaction rebuild can reconstruct the index without a
        # round-trip to the graph, and so reconcile can diff against source.
        self._live_nodes: dict[str, Any] = {}
        # Tombstoned ids — physically still in the underlying ANN structure (for
        # the HNSW backend) until compaction, but excluded from every result.
        self._tombstones: set[str] = set()

    # ── introspection ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        """Number of *live* (non-tombstoned) indexed objects."""
        return len(self._live_nodes)

    @property
    def tombstone_count(self) -> int:
        return len(self._tombstones)

    def live_ids(self) -> set[str]:
        return set(self._live_nodes)

    # ── batch pipeline ───────────────────────────────────────────────────────
    def batch_sync(self, nodes: Iterable[Any]) -> SyncResult:
        """Full rebuild of the index from the source-of-truth node set.

        Applies the data restriction, rebuilds the underlying index via
        ``CapabilityIndex.build_from_edges`` (the canonical bulk loader), and
        resets the staleness ledger to the freshly indexed versions. Clears any
        outstanding tombstones — a full rebuild is the ultimate compaction.
        """
        all_nodes = list(nodes)
        admitted = self.restriction.filter(all_nodes)
        # Restriction skips = inputs that carried an id but were excluded.
        admitted_ids = {i for n in admitted if (i := id_of(n)) is not None}
        skipped = sum(
            1
            for n in all_nodes
            if (i := id_of(n)) is not None and i not in admitted_ids
        )

        # Build a fresh index of the SAME shape (backend/dim) to guarantee a
        # clean structure with no tombstone residue.
        fresh = CapabilityIndex(
            dim=self.index.dim,
            prefer_backend=self.index.backend,
        )
        live: dict[str, Any] = {}
        for node in admitted:
            nid = id_of(node)
            if nid is None:
                continue
            g = _getter_for(node)
            if g("embedding") is None:
                continue
            live[nid] = node

        fresh.build_from_edges(live.values())

        self.index = fresh
        self._live_nodes = live
        self._tombstones = set()
        self.ledger.clear()
        for nid, node in live.items():
            self.ledger.record_payload(nid, index_payload_of(node))

        return SyncResult(
            mode="batch",
            upserted=len(live),
            deleted=0,
            skipped_restricted=skipped,
            rebuilt=True,
            live_size=len(live),
        )

    # ── streaming / incremental pipeline ─────────────────────────────────────
    def upsert(self, node: Any, source_watermark: float = 0.0) -> bool:
        """Add-or-replace a single object in the live index (no full rebuild).

        Returns ``True`` if the object was indexed, ``False`` if it was excluded
        by the data restriction or lacked an embedding.
        """
        if not self.restriction.admits(node):
            return False
        nid = id_of(node)
        if nid is None:
            return False
        g = _getter_for(node)
        emb = g("embedding")
        if emb is None:
            return False
        try:
            if len(emb) == 0:
                return False
        except TypeError:
            pass
        # Native in-place upsert: CapabilityIndex.add replaces an existing id and
        # uses hnsw add_items when the HNSW backend is active.
        self.index.add(nid, emb, _caps_of(node), swappable_with=_swap_of(node))
        self._live_nodes[nid] = node
        self._tombstones.discard(nid)
        self.ledger.record_payload(nid, index_payload_of(node), source_watermark)
        return True

    def delete(self, object_id: str) -> bool:
        """Remove a single object from the live index via the tombstone overlay.

        The object is dropped from the live node map, the staleness ledger, and
        the index's own capability/vector maps so it never ranks again. The HNSW
        label (if any) is tombstoned and physically evicted at the next
        compaction. Returns whether the object was present.
        """
        object_id = str(object_id)
        present = object_id in self._live_nodes
        self._live_nodes.pop(object_id, None)
        self.ledger.record_deleted(object_id)
        self._evict_from_index(object_id)
        return present

    def _evict_from_index(self, object_id: str) -> None:
        """Drop an id from the index's maps; tombstone its HNSW label if present.

        For the numpy backend this fully removes the vector (no residue). For the
        HNSW backend the vector map and capability maps are removed immediately
        (so ranking can never return it), and the underlying hnsw label is
        marked for compaction — hnswlib cannot cheaply delete a single label, so
        the physical eviction happens at the next rebuild.
        """
        idx = self.index
        # Remove from capability inverted index + id->caps map.
        for cap in idx._id_to_caps.pop(object_id, set()):
            providers = idx._cap_to_ids.get(cap)
            if providers is not None:
                providers.discard(object_id)
                if not providers:
                    idx._cap_to_ids.pop(cap, None)
        # Remove from swappable adjacency (symmetric).
        for partner in idx._swappable.pop(object_id, set()):
            peers = idx._swappable.get(partner)
            if peers is not None:
                peers.discard(object_id)
                if not peers:
                    idx._swappable.pop(partner, None)
        idx._reward.pop(object_id, None)

        if idx.backend == "hnsw" and object_id in idx._id_to_label:
            # Physically unreachable from rank only once the vector is gone; keep
            # the label reserved and tombstone it for compaction.
            label = idx._id_to_label.get(object_id)
            if label is not None and idx._hnsw is not None:
                try:
                    idx._hnsw.mark_deleted(label)
                except Exception:  # pragma: no cover - backend variance
                    # Older hnswlib without mark_deleted: removing the vector map
                    # entry + post-filter (below) still guarantees correctness.
                    pass
            self._tombstones.add(object_id)
        # Remove the vector last (source of truth for numpy ranking + rebuilds).
        idx._id_to_vec.pop(object_id, None)

    def incremental_sync(self, delta: FunnelDelta) -> SyncResult:
        """Apply a streaming delta (upserts + deletes) to the live index.

        Real incremental update — only the delta's objects are touched; the rest
        of the index is untouched (no full rebuild) unless the accumulated
        tombstone fraction crosses ``compaction_threshold``, in which case a
        compacting rebuild physically evicts deleted objects from the underlying
        ANN structure.
        """
        upserted = 0
        skipped = 0
        for node in delta.upserts:
            if self.upsert(node, delta.source_watermark):
                upserted += 1
            else:
                skipped += 1

        deleted = 0
        for oid in delta.deletes:
            if self.delete(oid):
                deleted += 1

        rebuilt = False
        if self._should_compact():
            self._compact()
            rebuilt = True

        return SyncResult(
            mode="incremental",
            upserted=upserted,
            deleted=deleted,
            skipped_restricted=skipped,
            rebuilt=rebuilt,
            live_size=len(self._live_nodes),
        )

    def _should_compact(self) -> bool:
        """Whether the tombstone overlay is large enough to warrant a rebuild."""
        if not self._tombstones:
            return False
        live = len(self._live_nodes)
        total = live + len(self._tombstones)
        if total == 0:
            return False
        return (len(self._tombstones) / total) >= self.compaction_threshold

    def _compact(self) -> None:
        """Physically rebuild the index from live nodes, evicting tombstones."""
        fresh = CapabilityIndex(dim=self.index.dim, prefer_backend=self.index.backend)
        fresh.build_from_edges(self._live_nodes.values())
        # Preserve learned rewards for surviving ids.
        for nid, r in list(self.index._reward.items()):
            if nid in self._live_nodes:
                fresh._reward[nid] = r
        self.index = fresh
        self._tombstones.clear()

    # ── reconcile against live source (closes the staleness loop) ────────────
    def reconcile(
        self,
        source_nodes: Iterable[Any],
        source_watermark: float = 0.0,
    ) -> SyncResult:
        """Observe the current source and apply exactly the needed delta.

        Computes drift via the staleness ledger (new/changed/orphaned objects
        among the *admitted* source set) and incrementally upserts the changed
        objects and deletes the orphaned ones — without a full rebuild. This is
        the metadata-sync driver: source state in, consistent index out.
        """
        admitted = {
            nid: node
            for node in self.restriction.filter(list(source_nodes))
            if (nid := id_of(node)) is not None
            and _getter_for(node)("embedding") is not None
        }
        # Build the source hash view the ledger compares against.
        source_view = {
            nid: content_hash(index_payload_of(node)) for nid, node in admitted.items()
        }
        report = self.ledger.compare(source_view)

        delta = FunnelDelta(source_watermark=source_watermark)
        for nid in report.stale | report.missing:
            delta.upserts.append(admitted[nid])
        delta.deletes = sorted(report.orphaned)
        return self.incremental_sync(delta)

    def needs_reindex(self, source_nodes: Iterable[Any]) -> bool:
        """Whether the live index has drifted from the admitted source set."""
        admitted = {
            nid: node
            for node in self.restriction.filter(list(source_nodes))
            if (nid := id_of(node)) is not None
            and _getter_for(node)("embedding") is not None
        }
        source_view = {
            nid: content_hash(index_payload_of(node)) for nid, node in admitted.items()
        }
        return self.ledger.needs_reindex(source_view)

    # ── live search (delegates to the router's designate path) ───────────────
    def search(
        self,
        prompt_embedding: Any,
        required_caps: Any = None,
        k: int = 5,
    ) -> list[Designation]:
        """Rank live objects for a task via the underlying ``designate`` path.

        Tombstoned objects are excluded post-rank so deletes are honoured even
        before a compaction physically evicts them from the HNSW structure.
        """
        if not self._tombstones:
            return self.index.designate(prompt_embedding, required_caps, k=k)
        # Oversample to absorb tombstones still present in the HNSW structure,
        # then drop them and trim back to k.
        pad = len(self._tombstones)
        raw = self.index.designate(prompt_embedding, required_caps, k=k + pad)
        live = [d for d in raw if d.id not in self._tombstones]
        return live[:k]
