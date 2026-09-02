"""Application-service contracts for governed tabular projection reads."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_utilities.knowledge_graph.backends.trino_backend import KnowledgeBatch
from agent_utilities.knowledge_graph.core.tabular_query_service import (
    TabularQueryRequest,
    TabularQueryService,
)


class _Backend:
    def __init__(self, pages):
        self.pages = pages
        self.sql = None
        self.closed = False

    def query(self, sql, *, snapshot_id=None):
        self.sql = sql
        return iter(self.pages)

    def close(self):
        self.closed = True


def _page(rows, *, index=0, snapshot_id=None):
    return KnowledgeBatch(
        rows=rows,
        columns=("id",),
        snapshot_id=snapshot_id,
        lsn=snapshot_id,
        row_count=len(rows),
        page_index=index,
    )


def test_service_materializes_rows_and_page_provenance():
    backend = _Backend([_page([{"id": 1}]), _page([{"id": 2}], index=1)])
    service = TabularQueryService(backend)

    payload = service.execute(TabularQueryRequest("SELECT id FROM t")).to_payload()

    assert backend.sql == "SELECT id FROM t"
    assert payload["rows"] == [{"id": 1}, {"id": 2}]
    assert payload["columns"] == ["id"]
    assert payload["provenance"]["pages"] == [
        {"page_index": 0, "snapshot_id": None, "lsn": None, "row_count": 1},
        {"page_index": 1, "snapshot_id": None, "lsn": None, "row_count": 1},
    ]


@pytest.mark.parametrize(
    "query_request",
    [
        TabularQueryRequest("SELECT 1", graph="tenant-a"),
        TabularQueryRequest("SELECT 1", as_of="2026-09-02T00:00:00Z"),
        TabularQueryRequest(""),
    ],
)
def test_service_rejects_unhonored_or_empty_request(query_request):
    backend = _Backend([])
    with pytest.raises(ValueError):
        TabularQueryService(backend).execute(query_request)
    assert backend.sql is None


def test_service_rejects_inconsistent_page_contract():
    backend = _Backend([_page([{"id": 1}], index=1)])
    with pytest.raises(ValueError, match="page indexes"):
        TabularQueryService(backend).execute(TabularQueryRequest("SELECT 1"))


@pytest.mark.parametrize("field", ["snapshot_id", "lsn"])
def test_service_rejects_provenance_change_between_pages(field):
    first = _page([{"id": 1}], snapshot_id="100")
    second = _page([{"id": 2}], index=1, snapshot_id="100")
    second = replace(second, **{field: "101"})

    with pytest.raises(ValueError, match="snapshot provenance"):
        TabularQueryService(_Backend([first, second])).execute(
            TabularQueryRequest("SELECT 1")
        )


def test_service_owns_backend_lifecycle():
    backend = _Backend([])
    service = TabularQueryService(backend)
    service.close()
    assert backend.closed is True
