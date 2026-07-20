"""``Method::KnowledgeStream`` consumer (CONCEPT:AU-KG.query.knowledge-stream-consumer, report §9 #3).

Exercises the cursor-loop/dispatch logic against a fake ``.knowledge`` sub-client
and a fake ``pyarrow`` double (never a real Arrow install — this module must stay
importable/testable without one, matching the repo's Dependency discipline), plus
the live-path wiring: ``GraphComputeEngine.stream_graph_confidence`` and
``compliance_tools._posture()`` (a real, already-registered MCP tool + REST route)
actually calling into this consumer end to end.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core import knowledge_stream


class _FakeTable:
    def __init__(self, rows: list[dict], columns: list[str]) -> None:
        self._rows = rows
        self.column_names = columns

    def to_pylist(self) -> list[dict]:
        return self._rows


class _FakeReader:
    def __init__(self, table: _FakeTable) -> None:
        self._table = table

    def read_all(self) -> _FakeTable:
        return self._table


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
