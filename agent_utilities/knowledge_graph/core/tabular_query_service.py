"""Application service for governed tabular projection reads.

CONCEPT:AU-KG.compute.trino-query-backend

The service owns the Trino-shaped query use case while depending only on the
lower :class:`QueryBackend` port. MCP and REST are transport adapters; identity
and configuration are bound at composition. This keeps tabular projection
semantics out of both the backend adapter and the public transports.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

__all__ = [
    "KnowledgeBatch",
    "QueryBackend",
    "TabularQueryRequest",
    "TabularQueryResult",
    "TabularQueryService",
]

_UNSET = object()


@dataclass(frozen=True)
class KnowledgeBatch:
    """One provenance-carrying page returned by a tabular query backend."""

    rows: list[dict[str, Any]]
    columns: tuple[str, ...]
    snapshot_id: str | None
    lsn: str | None
    row_count: int
    page_index: int


@runtime_checkable
class QueryBackend(Protocol):
    """Injected port for a read-only, provenance-carrying tabular backend."""

    def query(
        self, sql: str, *, snapshot_id: str | None = None
    ) -> Iterator[KnowledgeBatch]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class TabularQueryRequest:
    """One transport-neutral tabular query request."""

    sql: str
    graph: str = ""
    as_of: str = ""


@dataclass(frozen=True)
class TabularQueryResult:
    """Materialized rows plus explicit page and snapshot provenance."""

    rows: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]
    pages: tuple[dict[str, Any], ...]
    snapshot_id: str | None
    lsn: str | None

    def to_payload(self) -> dict[str, Any]:
        """Return the stable public payload consumed by thin transports."""

        return {
            "rows": list(self.rows),
            "columns": list(self.columns),
            "connection": "trino",
            "graph": "",
            "provenance": {
                "pages": list(self.pages),
                "snapshot_id": self.snapshot_id,
                "lsn": self.lsn,
            },
        }


class TabularQueryService:
    """Execute the tabular projection read use case through one injected port."""

    def __init__(self, backend: QueryBackend) -> None:
        if not isinstance(backend, QueryBackend):
            raise TypeError("TabularQueryService requires a QueryBackend")
        self._backend = backend

    def execute(self, request: TabularQueryRequest) -> TabularQueryResult:
        """Execute a query without silently changing unsupported selectors."""

        self._validate_request(request)
        return self._materialize(self._backend.query(request.sql))

    @staticmethod
    def _validate_request(request: TabularQueryRequest) -> None:
        if not isinstance(request, TabularQueryRequest):
            raise TypeError("request must be a TabularQueryRequest")
        if not isinstance(request.graph, str) or request.graph:
            raise ValueError("Trino has no physical graph selector; omit graph")
        if not isinstance(request.as_of, str) or request.as_of.strip():
            raise ValueError(
                "generic as_of is not a Trino snapshot id; use the validated "
                "backend as_of() API"
            )
        if not isinstance(request.sql, str) or not request.sql.strip():
            raise ValueError("Trino SQL must be a non-empty string")

    @staticmethod
    def _stable_page_value(current: Any, value: Any, message: str) -> Any:
        if current is _UNSET:
            return value
        if current != value:
            raise ValueError(message)
        return current

    def _materialize(self, source: Iterator[KnowledgeBatch]) -> TabularQueryResult:
        rows: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        columns: tuple[str, ...] | object = _UNSET
        snapshot_id: str | None | object = _UNSET
        lsn: str | None | object = _UNSET
        for expected_index, page in enumerate(source):
            self._validate_page(page, expected_index)
            columns = self._stable_page_value(
                columns,
                page.columns,
                "tabular query returned inconsistent page columns",
            )
            snapshot_id = self._stable_page_value(
                snapshot_id,
                page.snapshot_id,
                "tabular query returned inconsistent snapshot provenance",
            )
            lsn = self._stable_page_value(
                lsn,
                page.lsn,
                "tabular query returned inconsistent snapshot provenance",
            )
            rows.extend(page.rows)
            pages.append(
                {
                    "page_index": page.page_index,
                    "snapshot_id": page.snapshot_id,
                    "lsn": page.lsn,
                    "row_count": page.row_count,
                }
            )
        if columns is _UNSET:
            return TabularQueryResult((), (), (), None, None)
        return TabularQueryResult(
            rows=tuple(rows),
            columns=cast(tuple[str, ...], columns),
            pages=tuple(pages),
            snapshot_id=cast(str | None, snapshot_id),
            lsn=cast(str | None, lsn),
        )

    @staticmethod
    def _validate_page(page: Any, expected_index: int) -> None:
        if not isinstance(page, KnowledgeBatch):
            raise TypeError("QueryBackend returned an invalid KnowledgeBatch")
        if page.page_index != expected_index:
            raise ValueError("tabular query returned non-contiguous page indexes")
        if page.row_count != len(page.rows):
            raise ValueError("tabular query returned an invalid page row count")
        if any(not isinstance(row, dict) for row in page.rows):
            raise TypeError("tabular query rows must be mappings")

    def close(self) -> None:
        """Release the injected backend's pooled resources."""

        self._backend.close()
