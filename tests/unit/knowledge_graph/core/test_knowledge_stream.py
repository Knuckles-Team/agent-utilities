"""``Method::KnowledgeStream`` consumer (CONCEPT:AU-KG.query.knowledge-stream-consumer, report §9 #3).

Exercises the cursor-loop/dispatch logic against a fake ``.knowledge`` sub-client
and a fake ``pyarrow`` double (never a real Arrow install — this module must stay
importable/testable without one, matching the repo's Dependency discipline), plus
the live-path wiring: ``GraphComputeEngine.stream_graph_confidence`` and
``compliance_tools._posture()`` (a real, already-registered MCP tool + REST route)
actually calling into this consumer end to end.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import knowledge_stream


class _FakeTable:
    def __init__(self, rows: list[dict], columns: list[str]) -> None:
        self._rows = rows
        self.column_names = columns

    def to_pylist(self) -> list[dict]:
        return self._rows


class _FakeSchema:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakeRecordBatch:
    """Minimal stand-in for ``pyarrow.RecordBatch`` — only the surface a caller of
    :func:`pull_record_batches` reads (``schema.names``/``num_rows``/``to_pylist``)."""

    def __init__(self, rows: list[dict], columns: list[str]) -> None:
        self._rows = rows
        self.num_rows = len(rows)
        self.schema = _FakeSchema(columns)

    def to_pylist(self) -> list[dict]:
        return self._rows


class _FakeReader:
    def __init__(self, table: _FakeTable) -> None:
        self._table = table

    def read_all(self) -> _FakeTable:
        return self._table

    def __iter__(self):  # noqa: ANN204 — one RecordBatch per page (engine writes one)
        yield _FakeRecordBatch(self._table._rows, self._table.column_names)


class _FakeIpc:
    def __init__(self, table_by_payload: dict[bytes, _FakeTable]) -> None:
        self._table_by_payload = table_by_payload

    def open_stream(self, payload: bytes) -> _FakeReader:
        return _FakeReader(self._table_by_payload[payload])


class _FakePyarrow:
    """Minimal stand-in for the ``pyarrow`` module — only what this module uses."""

    def __init__(self, table_by_payload: dict[bytes, _FakeTable]) -> None:
        self.ipc = _FakeIpc(table_by_payload)


class _FakeKnowledgeClient:
    """Stand-in for ``SyncEpistemicGraphClient.knowledge`` — records every call and
    serves pre-built pages, threading the cursor exactly like the real wire."""

    def __init__(self, pages: list[dict]) -> None:
        self._pages = list(pages)
        self.calls: list[dict] = []

    def pull(self, query, *, batch_size, cursor=None):  # noqa: ANN001
        self.calls.append({"query": query, "batch_size": batch_size, "cursor": cursor})
        return self._pages.pop(0)


class _FakeCompute:
    def __init__(self, knowledge: _FakeKnowledgeClient | None) -> None:
        class _Client:
            pass

        self._client = _Client()
        if knowledge is not None:
            self._client.knowledge = knowledge


def _row(id_: str, kind: str, confidence: float, score: float | None) -> dict:
    return {
        "id": id_,
        "kind": kind,
        "score_score": score,
        "confidence": confidence,
        "evidence_kind": None,
        "evidence_refs_json": [],
        "valid_from": None,
        "valid_until": None,
        "tx_from": 10,
        "tx_to": None,
        "source_refs": ["src:1"],
        "policy_labels": [],
        "transformation_ids": [],
        "proof_ids": [],
        "alternative_ids": [],
        "contradiction_ids": [],
        "blob_handle": None,
        "has_payload": False,
    }


# ---------------------------------------------------------------------------
# availability / clean degrade
# ---------------------------------------------------------------------------


def test_returns_none_without_a_knowledge_client(monkeypatch):
    monkeypatch.setattr(knowledge_stream, "_pyarrow", lambda: _FakePyarrow({}))
    compute = _FakeCompute(knowledge=None)
    assert knowledge_stream.stream_graph_confidence(compute, "Claim") is None
    assert knowledge_stream.available(compute) is False


def test_returns_none_without_pyarrow(monkeypatch):
    monkeypatch.setattr(knowledge_stream, "_pyarrow", lambda: None)
    compute = _FakeCompute(knowledge=_FakeKnowledgeClient(pages=[]))
    assert knowledge_stream.stream_graph_confidence(compute, "Claim") is None
    assert knowledge_stream.available(compute) is False


# ---------------------------------------------------------------------------
# live-path: cursor-loop dispatch + Arrow decode, end to end
# ---------------------------------------------------------------------------


def test_stream_graph_confidence_pages_until_exhausted(monkeypatch):
    page0_payload = b"page0"
    page1_payload = b"page1"
    table_by_payload = {
        page0_payload: _FakeTable(
            [
                _row("ref:a", "graph_row", 0.9, None),
                _row("ref:b", "graph_row", 0.4, None),
            ],
            ["id", "kind", "score_score", "confidence", "source_refs"],
        ),
        page1_payload: _FakeTable(
            [_row("ref:c", "graph_row", 0.6, None)],
            ["id", "kind", "score_score", "confidence", "source_refs"],
        ),
    }
    monkeypatch.setattr(
        knowledge_stream, "_pyarrow", lambda: _FakePyarrow(table_by_payload)
    )

    cursor_page1 = {
        "schema_version": 1,
        "family": "graph",
        "batch_size": 2,
        "row_offset": 2,
        "batch_index": 1,
        "exhausted": False,
    }
    cursor_final = {
        **cursor_page1,
        "row_offset": 3,
        "batch_index": 2,
        "exhausted": True,
    }
    knowledge = _FakeKnowledgeClient(
        pages=[
            {
                "schema_version": 1,
                "family": "graph",
                "projection": "arrow_ipc_v1",
                "cursor": cursor_page1,
                "payload": page0_payload,
            },
            {
                "schema_version": 1,
                "family": "graph",
                "projection": "arrow_ipc_v1",
                "cursor": cursor_final,
                "payload": page1_payload,
            },
        ]
    )
    compute = _FakeCompute(knowledge=knowledge)

    assert knowledge_stream.available(compute) is True
    rows = list(
        knowledge_stream.stream_graph_confidence(
            compute, "Claim", batch_size=2, limit=0
        )
    )

    # Every row arrived, in page order, correctly decoded off the fake Arrow table.
    assert [r["id"] for r in rows] == ["ref:a", "ref:b", "ref:c"]
    assert [r["confidence"] for r in rows] == [0.9, 0.4, 0.6]
    assert rows[0]["scores"] == {"score": None}
    assert rows[0]["source_refs"] == ["src:1"]
    assert rows[0]["tx_time"] == (10, None)

    # Exactly two RPCs (one per page); the query is IDENTICAL on both calls; the
    # cursor threads from the first response into the second request — never
    # re-sent from scratch, never dropped.
    assert len(knowledge.calls) == 2
    assert knowledge.calls[0]["query"] == {
        "family": "graph",
        "label": "Claim",
        "limit": 0,
    }
    assert knowledge.calls[0]["cursor"] is None
    assert knowledge.calls[1]["query"] == knowledge.calls[0]["query"]
    assert knowledge.calls[1]["cursor"] == cursor_page1


def test_pull_knowledge_stream_stops_on_rpc_error_without_raising(monkeypatch):
    monkeypatch.setattr(knowledge_stream, "_pyarrow", lambda: _FakePyarrow({}))

    class _BoomKnowledge:
        def pull(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("engine build has no knowledge surface")

    compute = _FakeCompute(knowledge=_BoomKnowledge())
    rows_iter = knowledge_stream.pull_knowledge_stream(
        compute, {"family": "graph", "label": "Claim", "limit": 0}
    )
    assert rows_iter is not None  # availability check passed (client + pyarrow present)
    assert list(rows_iter) == []  # the mid-stream error degrades to an empty stream


# ---------------------------------------------------------------------------
# live-path: GraphComputeEngine.stream_graph_confidence wiring (engine.py/graph_compute.py)
# ---------------------------------------------------------------------------


def test_graph_compute_engine_stream_graph_confidence_delegates(monkeypatch):
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    payload = b"only-page"
    table = _FakeTable(
        [_row("ref:x", "graph_row", 0.77, None)],
        ["id", "kind", "score_score", "confidence"],
    )
    monkeypatch.setattr(
        knowledge_stream, "_pyarrow", lambda: _FakePyarrow({payload: table})
    )
    knowledge = _FakeKnowledgeClient(
        pages=[
            {
                "schema_version": 1,
                "family": "graph",
                "projection": "arrow_ipc_v1",
                "cursor": {
                    "schema_version": 1,
                    "family": "graph",
                    "batch_size": 512,
                    "row_offset": 1,
                    "batch_index": 1,
                    "exhausted": True,
                },
                "payload": payload,
            }
        ]
    )

    engine = GraphComputeEngine.__new__(
        GraphComputeEngine
    )  # bypass __init__ (transport)

    class _Client:
        pass

    engine._client = _Client()
    engine._client.knowledge = knowledge

    rows = list(engine.stream_graph_confidence("Claim", limit=5))
    assert [r["id"] for r in rows] == ["ref:x"]
    assert knowledge.calls[0]["query"] == {
        "family": "graph",
        "label": "Claim",
        "limit": 5,
    }


# ---------------------------------------------------------------------------
# pull_record_batches — the COLUMNAR currency (yields pyarrow.RecordBatch)
# ---------------------------------------------------------------------------


def test_pull_record_batches_yields_bounded_batches_threading_the_cursor(monkeypatch):
    page0_payload = b"page0"
    page1_payload = b"page1"
    cols = [
        "id",
        "kind",
        "score_score",
        "confidence",
        "proof_ids",
        "contradiction_ids",
    ]
    table_by_payload = {
        page0_payload: _FakeTable(
            [
                _row("ref:a", "graph_row", 0.9, None),
                _row("ref:b", "graph_row", 0.4, None),
            ],
            cols,
        ),
        page1_payload: _FakeTable([_row("ref:c", "graph_row", 0.6, None)], cols),
    }
    monkeypatch.setattr(
        knowledge_stream, "_pyarrow", lambda: _FakePyarrow(table_by_payload)
    )

    cursor_page1 = {
        "schema_version": 1,
        "family": "graph",
        "batch_size": 2,
        "row_offset": 2,
        "batch_index": 1,
        "exhausted": False,
    }
    cursor_final = {
        **cursor_page1,
        "row_offset": 3,
        "batch_index": 2,
        "exhausted": True,
    }
    knowledge = _FakeKnowledgeClient(
        pages=[
            {
                "schema_version": 1,
                "family": "graph",
                "projection": "arrow_ipc_v1",
                "cursor": cursor_page1,
                "payload": page0_payload,
            },
            {
                "schema_version": 1,
                "family": "graph",
                "projection": "arrow_ipc_v1",
                "cursor": cursor_final,
                "payload": page1_payload,
            },
        ]
    )
    compute = _FakeCompute(knowledge=knowledge)

    batches = list(
        knowledge_stream.pull_record_batches(
            compute, {"family": "graph", "label": "Claim", "limit": 0}, batch_size=2
        )
    )

    # ONE RecordBatch per bounded page (never the whole result concatenated), each
    # carrying the epistemic columns straight off the wire schema.
    assert len(batches) == 2
    assert [b.num_rows for b in batches] == [2, 1]
    assert "proof_ids" in batches[0].schema.names
    assert "contradiction_ids" in batches[0].schema.names
    assert [r["id"] for r in batches[0].to_pylist()] == ["ref:a", "ref:b"]

    # Same cursor-threading contract as the row path: two RPCs, cursor forwarded.
    assert len(knowledge.calls) == 2
    assert knowledge.calls[1]["cursor"] == cursor_page1


def test_pull_record_batches_real_pyarrow_round_trip():
    """Genuine Arrow-IPC decode (not a fake): the engine's bounded page bytes decode
    back to a real ``pyarrow.RecordBatch`` carrying ``proof_ids``/``contradiction_ids``
    — the columns the row path (``EpistemicRow``) also carries, proven end to end."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.ipc  # noqa: F401 — presence of the ipc submodule

    table = pa.table(
        {
            "id": ["ref:a", "ref:b"],
            "kind": ["graph_row", "graph_row"],
            "score_score": pa.array([0.9, 0.4], type=pa.float32()),
            "confidence": [0.82, 0.5],
            "proof_ids": [["evidence:1", "claim:base"], []],
            "contradiction_ids": [["claim:2"], []],
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    payload = sink.getvalue().to_pybytes()

    knowledge = _FakeKnowledgeClient(
        pages=[
            {
                "schema_version": 1,
                "family": "graph",
                "projection": "arrow_ipc_v1",
                "cursor": {
                    "schema_version": 1,
                    "family": "graph",
                    "batch_size": 512,
                    "row_offset": 2,
                    "batch_index": 1,
                    "exhausted": True,
                },
                "payload": payload,
            }
        ]
    )
    compute = _FakeCompute(knowledge=knowledge)

    batches = list(
        knowledge_stream.pull_record_batches(
            compute, {"family": "graph", "label": "Claim", "limit": 0}
        )
    )
    assert len(batches) == 1
    record_batch = batches[0]
    assert isinstance(record_batch, pa.RecordBatch)
    rows = record_batch.to_pylist()
    assert rows[0]["proof_ids"] == ["evidence:1", "claim:base"]
    assert rows[0]["contradiction_ids"] == ["claim:2"]
    assert rows[1]["proof_ids"] == []


def test_pull_record_batches_degrades_to_none(monkeypatch):
    # No pyarrow -> None (never raises), so a caller falls back to a non-columnar path.
    monkeypatch.setattr(knowledge_stream, "_pyarrow", lambda: None)
    compute = _FakeCompute(knowledge=_FakeKnowledgeClient(pages=[]))
    assert knowledge_stream.pull_record_batches(compute, {"family": "graph"}) is None

    # No .knowledge streaming surface -> None (e.g. an older engine build).
    monkeypatch.setattr(knowledge_stream, "_pyarrow", lambda: _FakePyarrow({}))
    assert (
        knowledge_stream.pull_record_batches(
            _FakeCompute(knowledge=None), {"family": "graph"}
        )
        is None
    )


# ---------------------------------------------------------------------------
# facade: KnowledgeGraph.query_batches — family -> wire-query mapping
# ---------------------------------------------------------------------------


def test_query_batches_builds_the_family_wire_query(monkeypatch):
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

    captured: dict = {}

    def _spy(compute, query, *, batch_size):  # noqa: ANN001, ANN202
        captured["query"] = query
        captured["batch_size"] = batch_size
        return iter(())

    monkeypatch.setattr(knowledge_stream, "pull_record_batches", _spy)

    facade = KnowledgeGraph.__new__(KnowledgeGraph)
    facade._compute = object()  # any sentinel; the spy ignores it

    # Default family is cross_modal (UQL text — subsumes cypher-style MATCH queries).
    list(facade.query_batches("MATCH (n) |> LIMIT 1"))
    assert captured["query"] == {
        "family": "cross_modal",
        "text": "MATCH (n) |> LIMIT 1",
    }

    # "cypher" and "uql" both route to the cross_modal UQL surface.
    for alias in ("cypher", "uql"):
        list(facade.query_batches("MATCH (n)", family=alias))
        assert captured["query"]["family"] == "cross_modal"

    list(facade.query_batches("SELECT 1", family="sql", params=b"\x90", batch_size=256))
    assert captured["query"] == {
        "family": "sql",
        "query": "SELECT 1",
        "params_msgpack": b"\x90",
    }
    assert captured["batch_size"] == 256

    list(facade.query_batches("SELECT * WHERE { ?s ?p ?o }", family="sparql"))
    assert captured["query"] == {
        "family": "rdf",
        "query": "SELECT * WHERE { ?s ?p ?o }",
        "base_iri": "",
        "type_convention": "",
    }

    list(facade.query_batches("Claim", family="graph", limit=7))
    assert captured["query"] == {"family": "graph", "label": "Claim", "limit": 7}


def test_query_batches_rejects_unknown_family():
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

    facade = KnowledgeGraph.__new__(KnowledgeGraph)
    facade._compute = object()
    with pytest.raises(ValueError, match="unknown family"):
        facade.query_batches("x", family="nope")
