"""Live-path proof for the ``RowVersion`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, Evidence seam completion).

Pass 1 declined ``DatabaseConnector.poll()``: it has ``row_id`` (the
configured ``id_field``) and a STRING watermark (``updated_field``), but no
``table`` (the connector wraps an arbitrary ``SELECT``, potentially a
join/view — not inferrable without misattributing a multi-table query) and
no guaranteed integer ``version`` (``updated_field`` is documented as
EITHER a timestamp OR an incrementing id). This test proves the honest,
non-fabricating remedy: ``table`` is now an optional, operator-supplied
config field (real information only the query's author has — never derived
from the query text); ``version`` is derived by parsing the row's own
already-fetched watermark value as an int — a real number when
``updated_field`` genuinely is an incrementing-id/revision column, a clean
no-op (never invented) when it is a timestamp.

Layers:
* :func:`test_poll_writes_row_version_evidence_when_table_and_integer_watermark_configured`
  — the positive case: both real facts present -> a real evidence write.
* :func:`test_poll_writes_nothing_without_table_configured` — `table` unset
  (the default) -> genuinely no evidence, never a guessed table name.
* :func:`test_poll_writes_nothing_for_a_non_integer_timestamp_watermark` —
  `updated_field` is an ISO timestamp -> genuinely no version, never invented.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.memory import native_ingest
from agent_utilities.protocols.source_connectors import build_connector


class _FakeStore:
    def __init__(self) -> None:
        self.row_calls: list[tuple[bytes, dict]] = []

    def store_row_version_evidence(self, data: bytes, **kwargs):
        self.row_calls.append((data, kwargs))
        return object()


class _FakeConn:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def read(self, q, p=None, *, max_rows=10_000):
        return self._rows[:max_rows]

    def health_check(self) -> bool:
        return True


def test_poll_writes_row_version_evidence_when_table_and_integer_watermark_configured(
    monkeypatch,
):
    rows = [
        {"id": "row-1", "title": "A", "body": "alpha text", "rev": "3"},
        {"id": "row-2", "title": "B", "body": "beta text", "rev": "5"},
    ]
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    conn = build_connector(
        "database",
        {
            "query": "select *",
            "text_field": "body",
            "updated_field": "rev",
            "table": "orders",
            "conn": _FakeConn(rows),
        },
    )
    batch = conn.poll()

    assert len(batch.documents) == 2
    assert len(store.row_calls) == 2
    (data1, kw1), (data2, kw2) = store.row_calls
    assert data1 == b"alpha text"
    assert kw1["table"] == "orders"
    assert kw1["row_id"] == "row-1"
    assert kw1["version"] == 3
    assert kw1["source"] == "database"
    assert data2 == b"beta text"
    assert kw2["row_id"] == "row-2"
    assert kw2["version"] == 5


def test_poll_writes_nothing_without_table_configured(monkeypatch):
    rows = [{"id": "row-1", "title": "A", "body": "alpha", "rev": "3"}]
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    conn = build_connector(
        "database",
        {
            "query": "select *",
            "text_field": "body",
            "updated_field": "rev",
            # no `table` -> never a guessed table name
            "conn": _FakeConn(rows),
        },
    )
    conn.poll()
    assert store.row_calls == []


def test_poll_writes_nothing_for_a_non_integer_timestamp_watermark(monkeypatch):
    rows = [{"id": "row-1", "title": "A", "body": "alpha", "ts": "2026-01-01"}]
    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    conn = build_connector(
        "database",
        {
            "query": "select *",
            "text_field": "body",
            "updated_field": "ts",
            "table": "orders",  # table IS configured...
            "conn": _FakeConn(rows),
        },
    )
    conn.poll()
    # ...but the watermark is a timestamp, not an integer -> no invented version.
    assert store.row_calls == []


def test_table_config_is_validated_like_other_field_names():
    import pytest

    with pytest.raises(ValueError, match="field mapping"):
        build_connector(
            "database",
            {
                "query": "select *",
                "text_field": "body",
                "table": "orders; DROP TABLE x",
                "conn": _FakeConn([]),
            },
        )
