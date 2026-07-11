#!/usr/bin/python
from __future__ import annotations

"""Tests for the Object Index Lifecycle / Object Data Funnel (CONCEPT:AU-KG.ontology.batch-incremental-sync-live).

Covers: batch full-build, incremental upsert + delete reflected on the live
index without a full rebuild, DataRestriction excluding ineligible objects,
content-based staleness drift detection on source mutation, reconcile clearing
that drift, and tombstone compaction physically evicting deletes.

Self-contained against existing stable code (CapabilityIndex) only.
"""

import types

import pytest

from agent_utilities.knowledge_graph.ontology.indexing import (
    DataRestriction,
    FunnelDelta,
    ObjectIndexFunnel,
    StalenessLedger,
    content_hash,
)
from agent_utilities.numeric import xp as np

DIM = 8


def _vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(DIM).astype(np.float32).tolist()


def _node(nid: str, seed: int, caps=None, node_type="tool", **extra):
    n = {
        "id": nid,
        "embedding": _vec(seed),
        "capabilities": list(caps or []),
        "type": node_type,
    }
    n.update(extra)
    return n


# ── batch ────────────────────────────────────────────────────────────────────
def test_batch_sync_full_build() -> None:
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    nodes = [_node(f"t{i}", seed=i, caps=["search"]) for i in range(5)]
    result = funnel.batch_sync(nodes)

    assert result.mode == "batch"
    assert result.rebuilt is True
    assert result.upserted == 5
    assert len(funnel) == 5
    assert funnel.live_ids() == {f"t{i}" for i in range(5)}
    # Designate via the live search path returns indexed objects.
    hits = funnel.search(_vec(1), required_caps=["search"], k=3)
    assert hits and all(h.id.startswith("t") for h in hits)


# ── incremental upsert + delete, no full rebuild ─────────────────────────────
def test_incremental_upsert_and_delete_live() -> None:
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    funnel.batch_sync([_node(f"t{i}", seed=i, caps=["search"]) for i in range(3)])
    index_obj_before = funnel.index

    # Incremental upsert of a brand-new object + delete of an existing one.
    delta = FunnelDelta(
        upserts=[_node("t99", seed=99, caps=["search"])],
        deletes=["t0"],
        source_watermark=10.0,
    )
    result = funnel.incremental_sync(delta)

    assert result.mode == "incremental"
    assert result.upserted == 1
    assert result.deleted == 1
    # No full rebuild for a small delta (tombstone fraction below threshold).
    assert result.rebuilt is False
    assert funnel.index is index_obj_before  # same live index object mutated in place

    assert "t99" in funnel.live_ids()
    assert "t0" not in funnel.live_ids()

    # Deleted object never appears in search results, even before compaction.
    for seed in range(20):
        hits = funnel.search(_vec(seed), k=10)
        assert all(h.id != "t0" for h in hits)
    # The new object is retrievable.
    all_ids = {h.id for s in range(20) for h in funnel.search(_vec(s), k=10)}
    assert "t99" in all_ids


def test_incremental_delete_on_hnsw_if_available() -> None:
    from agent_utilities.knowledge_graph.retrieval.capability_index import (
        _HNSW_AVAILABLE,
    )

    if not _HNSW_AVAILABLE:
        pytest.skip("hnswlib not installed; numpy delete path covered elsewhere")

    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="hnsw")
    funnel.batch_sync([_node(f"t{i}", seed=i, caps=["search"]) for i in range(6)])
    assert funnel.index.backend == "hnsw"
    funnel.incremental_sync(FunnelDelta(deletes=["t2"]))
    for seed in range(30):
        assert all(h.id != "t2" for h in funnel.search(_vec(seed), k=10))


# ── DataRestriction excludes ineligible objects ──────────────────────────────
def test_data_restriction_excludes_objects() -> None:
    restriction = DataRestriction(
        allowed_types={"tool"},
        denied_types={"secret"},
        predicate=lambda n: n.get("classification") != "restricted",
    )
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy", restriction=restriction)
    nodes = [
        _node("ok", seed=1, caps=["search"], node_type="tool"),
        _node("wrong_type", seed=2, node_type="dataset"),  # not allowed type
        _node("denied", seed=3, node_type="secret"),  # denied type
        _node("restricted", seed=4, node_type="tool", classification="restricted"),
    ]
    result = funnel.batch_sync(nodes)

    assert funnel.live_ids() == {"ok"}
    assert result.skipped_restricted == 3

    # Restriction also gates the incremental path.
    r2 = funnel.incremental_sync(
        FunnelDelta(upserts=[_node("denied2", seed=5, node_type="secret")])
    )
    assert r2.upserted == 0
    assert r2.skipped_restricted == 1
    assert "denied2" not in funnel.live_ids()


# ── staleness drift detection + reindex clears it ────────────────────────────
def test_staleness_detects_drift_and_reindex_clears() -> None:
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    source = [_node(f"t{i}", seed=i, caps=["search"]) for i in range(3)]
    funnel.batch_sync(source)

    # No mutation -> no drift.
    assert funnel.needs_reindex(source) is False

    # Mutate one object's index-relevant payload (new embedding) -> drift.
    mutated = _node("t1", seed=999, caps=["search"])
    new_source = [source[0], mutated, source[2]]
    assert funnel.needs_reindex(new_source) is True

    # Reconcile applies exactly the needed delta and clears drift.
    result = funnel.reconcile(new_source, source_watermark=5.0)
    assert result.upserted == 1
    assert result.deleted == 0
    assert funnel.needs_reindex(new_source) is False


def test_reconcile_handles_new_and_orphaned() -> None:
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    funnel.batch_sync([_node(f"t{i}", seed=i) for i in range(3)])

    # Source drops t0 (orphan) and adds t9 (new).
    new_source = [_node("t1", seed=1), _node("t2", seed=2), _node("t9", seed=9)]
    assert funnel.needs_reindex(new_source) is True

    result = funnel.reconcile(new_source)
    assert result.upserted == 1  # t9
    assert result.deleted == 1  # t0
    assert funnel.live_ids() == {"t1", "t2", "t9"}
    assert funnel.needs_reindex(new_source) is False


# ── delete eviction semantics per backend ────────────────────────────────────
def test_numpy_delete_is_immediate_physical_eviction() -> None:
    # The numpy backend supports true single-vector removal, so deletes are
    # physically evicted at once — no tombstone residue, no compaction needed.
    funnel = ObjectIndexFunnel(
        dim=DIM, prefer_backend="numpy", compaction_threshold=0.25
    )
    funnel.batch_sync([_node(f"t{i}", seed=i) for i in range(8)])
    index_before = funnel.index

    result = funnel.incremental_sync(FunnelDelta(deletes=["t0", "t1", "t2"]))
    assert result.deleted == 3
    assert result.rebuilt is False  # numpy needs no compaction
    assert funnel.index is index_before
    assert funnel.tombstone_count == 0
    assert funnel.live_ids() == {f"t{i}" for i in range(3, 8)}
    # Underlying index physically dropped the vectors.
    assert "t0" not in funnel.index._id_to_vec


def test_hnsw_tombstone_compaction_rebuilds() -> None:
    # hnswlib cannot cheaply delete a single label, so deletes accumulate as a
    # tombstone overlay until the fraction crosses the threshold -> rebuild.
    from agent_utilities.knowledge_graph.retrieval.capability_index import (
        _HNSW_AVAILABLE,
    )

    if not _HNSW_AVAILABLE:
        pytest.skip("hnswlib not installed; numpy eviction covered separately")

    funnel = ObjectIndexFunnel(
        dim=DIM, prefer_backend="hnsw", compaction_threshold=0.25
    )
    funnel.batch_sync([_node(f"t{i}", seed=i) for i in range(8)])
    index_before = funnel.index

    # Delete 3 of 8 -> tombstone fraction 3/8 = 0.375 >= 0.25 -> compaction.
    result = funnel.incremental_sync(FunnelDelta(deletes=["t0", "t1", "t2"]))
    assert result.deleted == 3
    assert result.rebuilt is True
    assert funnel.index is not index_before  # physically rebuilt
    assert funnel.tombstone_count == 0
    assert funnel.live_ids() == {f"t{i}" for i in range(3, 8)}


# ── staleness ledger unit (content-based, not timestamp) ─────────────────────
def test_content_hash_order_independent() -> None:
    a = {"capabilities": ["x", "y"], "embedding": [1.0, 2.0]}
    b = {"embedding": [1.0, 2.0], "capabilities": ["x", "y"]}
    assert content_hash(a) == content_hash(b)
    c = {"embedding": [1.0, 2.0], "capabilities": ["x", "z"]}
    assert content_hash(a) != content_hash(c)


def test_staleness_ledger_classifies() -> None:
    ledger = StalenessLedger()
    ledger.record_payload("a", {"v": 1})
    ledger.record_payload("b", {"v": 2})

    # Source: a unchanged, b changed, c new, (no a-removal).
    report = ledger.compare({"a": {"v": 1}, "b": {"v": 99}, "c": {"v": 3}})
    assert report.fresh == {"a"}
    assert report.stale == {"b"}
    assert report.missing == {"c"}
    assert report.orphaned == set()
    assert report.needs_reindex is True

    # mark_reindexed clears the drift.
    ledger.mark_reindexed(
        report.drift_ids, {"a": {"v": 1}, "b": {"v": 99}, "c": {"v": 3}}
    )
    report2 = ledger.compare({"a": {"v": 1}, "b": {"v": 99}, "c": {"v": 3}})
    assert report2.needs_reindex is False


# ── CDC-driven lifecycle (CONCEPT:AU-P1-3 pattern reuse — L33) ───────────────
#
# Mirrors the fake-streaming/CDC-capable-graph pattern from
# tests/unit/graph/test_capability_designation.py (the AU-P1-3
# CapabilityIndexWatcher suite) to prove ObjectIndexFunnel adopts the SAME
# bootstrap-once + engine_subscription.subscribe pattern rather than its own
# rebuild loop.
class _FakeStreaming:
    """Minimal stand-in for ``client.streaming`` (``cdc_read`` + ``watch``)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def cdc_read(self, _graph_name, cursor, limit=4096):
        pending = [e for e in self.events if e["seq"] >= cursor]
        return pending[:limit]

    def watch(self, _graph_name, cursor, label="", timeout_ms=0):
        pending = [e for e in self.events if e["seq"] >= cursor]
        next_seq = (pending[-1]["seq"] + 1) if pending else cursor
        return {"events": pending, "next_seq": next_seq}


class _CdcCapableGraph:
    """A node-scannable graph that ALSO resolves an engine streaming surface."""

    def __init__(self, nodes: dict, streaming: _FakeStreaming) -> None:
        self._nodes = dict(nodes)
        self._client = types.SimpleNamespace(streaming=streaming)
        self.graph_name = "engtest"
        self.node_ids_calls = 0

    def node_ids(self):
        self.node_ids_calls += 1
        return list(self._nodes.keys())

    def _get_node_properties(self, nid):
        return self._nodes.get(nid, {})


class _PlainGraph:
    """A node-scannable graph with NO streaming surface (the CDC-unavailable case)."""

    def __init__(self, nodes: dict) -> None:
        self._nodes = dict(nodes)
        self.node_ids_calls = 0

    def node_ids(self):
        self.node_ids_calls += 1
        return list(self._nodes.keys())

    def _get_node_properties(self, nid):
        return self._nodes.get(nid, {})


def _engine_nodes(n: int, seed_offset: int = 0) -> dict:
    return {
        f"t{i}": {
            "id": f"t{i}",
            "type": "tool",
            "embedding": _vec(i + seed_offset),
            "capabilities": ["search"],
        }
        for i in range(n)
    }


def test_bind_engine_bootstraps_once_via_full_scan() -> None:
    streaming = _FakeStreaming()
    graph = _CdcCapableGraph(_engine_nodes(3), streaming)
    engine = types.SimpleNamespace(graph=graph)

    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    result = funnel.bind_engine(engine)

    assert result.mode == "batch"
    assert result.rebuilt is True
    assert graph.node_ids_calls == 1  # exactly one bootstrap full scan
    assert funnel.live_ids() == {"t0", "t1", "t2"}
    assert funnel.cdc_available is True
    assert funnel.bound_engine is engine


def test_poll_delivers_cdc_upsert_without_a_full_rescan() -> None:
    streaming = _FakeStreaming()
    graph = _CdcCapableGraph(_engine_nodes(3), streaming)
    engine = types.SimpleNamespace(graph=graph)

    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    funnel.bind_engine(engine)
    assert graph.node_ids_calls == 1

    # A brand-new object arrives purely via the CDC feed.
    streaming.events.append(
        {
            "seq": 1,
            "kind": "upsert",
            "node_id": "t99",
            "label": "",
            "after": {
                "id": "t99",
                "type": "tool",
                "embedding": _vec(99),
                "capabilities": ["search"],
            },
        }
    )
    result = funnel.poll()

    assert graph.node_ids_calls == 1  # STILL just the one bootstrap scan
    assert result.mode == "cdc"
    assert result.upserted == 1
    assert result.rebuilt is False
    assert "t99" in funnel.live_ids()
    assert len(funnel) == 4


def test_poll_delivers_cdc_delete_without_a_full_rescan() -> None:
    streaming = _FakeStreaming()
    graph = _CdcCapableGraph(_engine_nodes(3), streaming)
    engine = types.SimpleNamespace(graph=graph)

    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    funnel.bind_engine(engine)

    streaming.events.append(
        {"seq": 1, "kind": "delete", "node_id": "t0", "label": "", "after": None}
    )
    result = funnel.poll()

    assert graph.node_ids_calls == 1  # eviction was incremental, not a rescan
    assert result.mode == "cdc"
    assert result.deleted == 1
    assert "t0" not in funnel.live_ids()
    # Deleted object never appears in search, even without a compaction.
    for seed in range(20):
        assert all(h.id != "t0" for h in funnel.search(_vec(seed), k=10))


def test_poll_respects_data_restriction_on_cdc_events() -> None:
    streaming = _FakeStreaming()
    graph = _CdcCapableGraph(_engine_nodes(2), streaming)
    engine = types.SimpleNamespace(graph=graph)

    restriction = DataRestriction(allowed_types={"tool"})
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy", restriction=restriction)
    funnel.bind_engine(engine)
    assert graph.node_ids_calls == 1

    # A non-admitted node type arrives via CDC -> the funnel's OWN
    # DataRestriction still applies on the CDC path (via upsert()).
    streaming.events.append(
        {
            "seq": 1,
            "kind": "upsert",
            "node_id": "dataset1",
            "label": "",
            "after": {"id": "dataset1", "type": "dataset", "embedding": _vec(50)},
        }
    )
    result = funnel.poll()

    assert graph.node_ids_calls == 1
    assert result.upserted == 0
    assert "dataset1" not in funnel.live_ids()


def test_poll_without_cdc_surface_falls_back_to_reconcile() -> None:
    """No engine streaming surface reachable -> poll degrades to one full
    reconcile pass over a fresh scan (mirrors CapabilityIndexWatcher.refresh's
    degrade-to-full-rebuild behaviour for a dev/non-engine backend)."""
    graph = _PlainGraph(_engine_nodes(2))
    engine = types.SimpleNamespace(graph=graph)

    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    funnel.bind_engine(engine)
    assert funnel.cdc_available is False
    assert graph.node_ids_calls == 1

    result = funnel.poll()
    assert graph.node_ids_calls == 2  # degraded: one extra full rescan
    assert result.mode == "incremental"  # reconcile's SyncResult.mode


def test_poll_before_bind_engine_raises() -> None:
    funnel = ObjectIndexFunnel(dim=DIM, prefer_backend="numpy")
    with pytest.raises(RuntimeError):
        funnel.poll()
