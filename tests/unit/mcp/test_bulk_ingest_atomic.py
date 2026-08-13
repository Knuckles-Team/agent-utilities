"""B-11 — `graph_write(action="bulk_ingest")` onto the engine's atomic primitives.

Was: a Python `for` loop calling `engine.add_node()` once per element (nodes
only, non-atomic, no idempotency key — `write_ingest_tools.py:236-244` in the
pre-fix revision). Rewritten onto `IntelligenceGraphEngine.batch_typed_mutations`
(one native `BatchUpdate` transaction, real `upsert: bool` insert-or-merge) as
the light path, and `envelope_ingest.ingest_graph_slice` (one
`ApplyChangeEnvelope(s)` transaction, durably idempotent, evidence/lineage/
policy-carrying) when the caller supplies evidence or an idempotency key.

These tests prove: (1) the light path calls the batch primitive ONCE per chunk,
never per element; (2) chunking at the engine's documented bounds is
deterministic and never drops an element; (3) edges are accepted; (4) a caller
idempotency key routes onto ApplyChangeEnvelopes and its replay outcome
(`applied` vs `idempotent_skip`) is surfaced honestly, not collapsed to a
generic "ok"; (5) `upsert=False` selects real INSERT (not MERGE) semantics on
the wire op.
"""

from __future__ import annotations

import json
from typing import Any

from agent_utilities.mcp.tools import write_ingest_tools as wit


class _FakeBatchEngine:
    """Fakes `IntelligenceGraphEngine.batch_typed_mutations` — the ONE atomic
    call the light path must use instead of a per-element loop."""

    def __init__(self, *, ok: bool = True) -> None:
        self.batch_calls: list[list[dict[str, Any]]] = []
        self.upsert_flags: list[bool] = []
        self.add_node_calls: list[Any] = []
        self.add_edge_calls: list[Any] = []
        self.link_nodes_calls: list[Any] = []
        self._ok = ok

    def batch_typed_mutations(self, mutations, *, upsert: bool = True) -> bool:
        self.batch_calls.append(list(mutations))
        self.upsert_flags.append(upsert)
        return self._ok

    # Present so a regression back to the per-element loop would be visible in
    # the assertions below (never expected to be called by the light path).
    def add_node(self, *a, **kw):
        self.add_node_calls.append((a, kw))

    def add_edge(self, *a, **kw):
        self.add_edge_calls.append((a, kw))

    def link_nodes(self, *a, **kw):
        self.link_nodes_calls.append((a, kw))


def _nodes_json(n: int) -> str:
    return json.dumps(
        [{"id": f"n{i}", "type": "Thing", "properties": {"i": i}} for i in range(n)]
    )


# ── (1) light path is ONE batch call, never a per-element loop ─────────────


def test_bulk_ingest_light_path_batches_once_not_per_element():
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine,
            _nodes_json(25),
            idempotency_key="",
            evidence="[]",
            upsert=True,
        )
    )
    assert out["mode"] == "batch_update"
    assert out["nodes_ingested"] == 25
    assert out["edges_ingested"] == 0
    assert out["chunks"] == 1
    assert out["applied_ops"] == 25
    # The defining assertion: ONE atomic call, not 25 per-element calls.
    assert len(engine.batch_calls) == 1
    assert len(engine.batch_calls[0]) == 25
    assert engine.add_node_calls == []
    assert engine.upsert_flags == [True]


def test_bulk_ingest_accepts_edges_alongside_nodes():
    payload = json.dumps(
        [
            {"id": "a", "type": "Thing", "properties": {}},
            {"id": "b", "type": "Thing", "properties": {}},
            {
                "kind": "edge",
                "source_id": "a",
                "target_id": "b",
                "rel_type": "RELATES_TO",
                "properties": {"weight": 1.0},
            },
            # Implicit edge kind: source_id/target_id present, no explicit "kind".
            {"source_id": "b", "target_id": "a", "rel_type": "BACK_TO"},
        ]
    )
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine, payload, idempotency_key="", evidence="[]", upsert=True
        )
    )
    assert out["nodes_ingested"] == 2
    assert out["edges_ingested"] == 2
    assert len(engine.batch_calls) == 1
    kinds = [m["kind"] for m in engine.batch_calls[0]]
    assert kinds == ["node", "node", "edge", "edge"]
    edge_ops = [m for m in engine.batch_calls[0] if m["kind"] == "edge"]
    assert edge_ops[0]["source"] == "a"
    assert edge_ops[0]["target"] == "b"
    assert edge_ops[0]["rel_type"] == "RELATES_TO"
    assert engine.add_edge_calls == []
    assert engine.link_nodes_calls == []


def test_bulk_ingest_upsert_false_maps_to_insert_semantics():
    engine = _FakeBatchEngine()
    wit._run_bulk_ingest(
        engine, _nodes_json(3), idempotency_key="", evidence="[]", upsert=False
    )
    assert engine.upsert_flags == [False]


def test_bulk_ingest_unavailable_capability_fails_loud_never_falls_back():
    """`batch_typed_mutations` returning False means the backend has no native
    typed-batch capability -- must be a typed error, never a silent fallback
    to the per-element loop this rewrite exists to remove."""
    engine = _FakeBatchEngine(ok=False)
    out = json.loads(
        wit._run_bulk_ingest(
            engine, _nodes_json(3), idempotency_key="", evidence="[]", upsert=True
        )
    )
    assert out["error"]["code"] == "dependency_unavailable"
    assert engine.add_node_calls == []


# ── (2) deterministic chunking at the engine's documented bounds, never drops ──


def test_chunk_batch_mutations_splits_at_max_ops_and_preserves_everything():
    mutations = [{"kind": "node", "id": f"n{i}"} for i in range(250)]
    chunks = wit._chunk_batch_mutations(mutations, max_ops=100, max_bytes=10**9)
    assert [len(c) for c in chunks] == [100, 100, 50]
    # Order preserved, nothing dropped or duplicated.
    flattened = [m["id"] for chunk in chunks for m in chunk]
    assert flattened == [f"n{i}" for i in range(250)]


def test_chunk_batch_mutations_splits_at_byte_budget():
    big_props = {"blob": "x" * 1000}
    mutations = [
        {"kind": "node", "id": f"n{i}", "properties": big_props} for i in range(20)
    ]
    # Force a tiny byte budget so several mutations must split into many chunks.
    chunks = wit._chunk_batch_mutations(mutations, max_ops=10_000, max_bytes=4000)
    assert len(chunks) > 1
    flattened = [m["id"] for chunk in chunks for m in chunk]
    assert flattened == [f"n{i}" for i in range(20)]


def test_chunk_batch_mutations_is_deterministic():
    mutations = [
        {"kind": "node", "id": f"n{i}", "properties": {"i": i}} for i in range(137)
    ]
    a = wit._chunk_batch_mutations(mutations, max_ops=17, max_bytes=10**9)
    b = wit._chunk_batch_mutations(mutations, max_ops=17, max_bytes=10**9)
    assert [len(c) for c in a] == [len(c) for c in b]
    assert a == b


def test_bulk_ingest_reports_chunking_when_forced_to_split(monkeypatch):
    """The response HONESTLY reports how many chunks were used, never silently
    collapsing a multi-chunk batch into a single-shot report."""
    monkeypatch.setattr(wit, "_BULK_INGEST_MAX_OPS", 10)
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine, _nodes_json(25), idempotency_key="", evidence="[]", upsert=True
        )
    )
    assert out["chunks"] == 3
    assert out["chunk_sizes"] == [10, 10, 5]
    assert out["applied_ops"] == 25
    assert len(engine.batch_calls) == 3
    # Chunks applied sequentially, in order.
    assert [m["id"] for m in engine.batch_calls[0]] == [f"n{i}" for i in range(10)]
    assert [m["id"] for m in engine.batch_calls[1]] == [f"n{i}" for i in range(10, 20)]
    assert [m["id"] for m in engine.batch_calls[2]] == [f"n{i}" for i in range(20, 25)]


# ── (3) heavy path: evidence/idempotency key -> ApplyChangeEnvelope(s) ─────


def test_bulk_ingest_idempotency_key_routes_to_change_envelope(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_ingest_graph_slice(engine, connector, entities, relationships, **kwargs):
        captured["engine"] = engine
        captured["connector"] = connector
        captured["entities"] = entities
        captured["relationships"] = relationships
        captured["kwargs"] = kwargs
        return {
            "status": "skipped",
            "envelope_id": "envelope:abc",
            "idempotency_key": kwargs["idempotency_key"],
            "watermark_advanced": False,
        }

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest_module

    monkeypatch.setattr(
        envelope_ingest_module, "ingest_graph_slice", _fake_ingest_graph_slice
    )

    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine,
            _nodes_json(2),
            idempotency_key="caller-chosen-key",
            evidence="[]",
            upsert=True,
        )
    )
    assert out["mode"] == "change_envelope"
    # The engine's REAL replay outcome is surfaced honestly -- "skipped" (an
    # idempotent replay), never silently reported as a fresh "success".
    assert out["status"] == "skipped"
    assert out["idempotency_key"] == "caller-chosen-key"
    assert captured["kwargs"]["idempotency_key"] == "caller-chosen-key"
    assert captured["connector"] == "bulk_ingest"
    assert len(captured["entities"]) == 2
    # The light path's batch primitive must NOT have been used.
    assert engine.batch_calls == []


def test_bulk_ingest_evidence_alone_routes_to_change_envelope_and_attaches_it(
    monkeypatch,
):
    captured: dict[str, Any] = {}

    def _fake_ingest_graph_slice(engine, connector, entities, relationships, **kwargs):
        captured["entities"] = entities
        return {"status": "success", "node_id": entities[0]["id"]}

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest_module

    monkeypatch.setattr(
        envelope_ingest_module, "ingest_graph_slice", _fake_ingest_graph_slice
    )

    engine = _FakeBatchEngine()
    evidence = json.dumps(
        [{"object_id": "n0", "modality": "structured", "content_digest": "d"}]
    )
    out = json.loads(
        wit._run_bulk_ingest(
            engine, _nodes_json(1), idempotency_key="", evidence=evidence, upsert=True
        )
    )
    assert out["mode"] == "change_envelope"
    assert out["status"] == "success"
    assert captured["entities"][0]["_evidence"] == json.loads(evidence)
    assert engine.batch_calls == []


def test_bulk_ingest_change_envelope_failure_surfaces_as_data_not_a_crash(monkeypatch):
    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest_module

    def _boom(*a, **kw):
        raise RuntimeError("synthetic engine rejection")

    monkeypatch.setattr(envelope_ingest_module, "ingest_graph_slice", _boom)
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine, _nodes_json(1), idempotency_key="k1", evidence="[]", upsert=True
        )
    )
    assert "error" in out


# ── (4) input validation never silently drops a malformed element ──────────


def test_bulk_ingest_rejects_edge_missing_rel_type():
    payload = json.dumps([{"source_id": "a", "target_id": "b"}])
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine, payload, idempotency_key="", evidence="[]", upsert=True
        )
    )
    assert out["error"]["code"] == "invalid_request"
    assert engine.batch_calls == []


def test_bulk_ingest_rejects_node_missing_id():
    payload = json.dumps([{"type": "Thing", "properties": {}}])
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine, payload, idempotency_key="", evidence="[]", upsert=True
        )
    )
    assert out["error"]["code"] == "invalid_request"


def test_bulk_ingest_empty_batch_is_a_clean_noop():
    engine = _FakeBatchEngine()
    out = json.loads(
        wit._run_bulk_ingest(
            engine, "[]", idempotency_key="", evidence="[]", upsert=True
        )
    )
    assert out == {
        "action": "bulk_ingest",
        "mode": "noop",
        "nodes_ingested": 0,
        "edges_ingested": 0,
        "chunks": 0,
    }
    assert engine.batch_calls == []
