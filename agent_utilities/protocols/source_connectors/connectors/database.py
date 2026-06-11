from __future__ import annotations

"""Relational/NoSQL database document-source connector.

CONCEPT:ECO-4.25 — reference ``Load`` + ``Poll`` connector over a database.
CONCEPT:ECO-4.26 — incremental poll keyed on a monotonic column watermark.

Wraps the existing :class:`UniversalConnector` (CONCEPT:KG-2.9) so a database it
speaks becomes a document source: a query's rows are each mapped to a
:class:`SourceDocument` via a declarative field map. **PostgreSQL is the proven
native path** — other ``UniversalConnector`` dialects (MySQL/MariaDB, MSSQL,
Oracle, Mongo) require their driver installed in-process and are better served
through the :mod:`mcp_tool` source (CONCEPT:KG-2.59) over sql-mcp, which owns
the dialect drivers, the read-only gate, and the row caps. No new driver
dependency is added here; ``UniversalConnector`` lazily imports the appropriate
driver and raises a clear error if it is missing.

This composes rather than competes: ``UniversalConnector`` already does the wire
work + ``introspect()``; this connector adds the *document* projection and the
incremental watermark so DB content lands in the KG as ``Document`` + ``Chunk``
ontology objects (with all the OWL/ACL benefits the framework brings).
"""

from collections.abc import Iterator
from typing import Any

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
)
from ..registry import register_source


@register_source("database")
class DatabaseConnector(LoadConnector, PollConnector):
    """Map a database query's rows to documents.

    CONCEPT:ECO-4.25.

    Config:
        dsn: Connection string (e.g. ``postgresql://user:pw@host/db``,
            ``mysql://…`` (MariaDB), ``mssql://…``, ``oracle://…``,
            ``sqlite:///path.db``, ``mongodb://…``). Required.
        kind: Optional explicit backend kind (inferred from the DSN otherwise).
        query: The SQL/Mongo query whose rows become documents (required).
        id_field / title_field / text_field: row→document field map.
        updated_field: Optional monotonic column (timestamp / incrementing id)
            used as the incremental-poll watermark.
        watermark_param: Named parameter the query binds the prior watermark to
            for incremental polling (e.g. ``:since``). When set, ``poll`` passes
            ``{watermark_param: last_watermark}`` so the query returns only new
            rows; when unset, ``poll`` re-runs the full query and filters in-memory
            by ``updated_field``.
        conn: Optional pre-built ``UniversalConnector`` (injected for tests).
    """

    provider = "Database"
    priority = 45

    def configure(
        self,
        *,
        dsn: str = "",
        kind: str | None = None,
        query: str = "",
        id_field: str = "id",
        title_field: str = "title",
        text_field: str = "text",
        updated_field: str = "",
        watermark_param: str = "",
        conn: Any = None,
        **_: object,
    ) -> None:
        if conn is None and not dsn:
            raise ValueError("DatabaseConnector requires a 'dsn' (or an injected conn)")
        if not query:
            raise ValueError("DatabaseConnector requires a 'query'")
        self.dsn = dsn
        self.kind = kind
        self.query = query
        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field
        self.updated_field = updated_field
        self.watermark_param = watermark_param
        self._conn = conn

    def _connection(self) -> Any:
        if self._conn is None:
            from ...universal_connector import UniversalConnector
            from ..registry import logger  # reuse package logger

            logger.debug("[ECO-4.25] opening database connection kind=%s", self.kind)
            self._conn = UniversalConnector(self.dsn, kind=self.kind)
        return self._conn

    def health_check(self) -> bool:
        try:
            return bool(self._connection().health_check())
        except Exception:  # noqa: BLE001 — unreachable DB → unhealthy, not fatal
            return False

    def _to_document(self, row: dict[str, Any]) -> SourceDocument | None:
        rid = row.get(self.id_field)
        text = row.get(self.text_field)
        if rid is None or not isinstance(text, str) or not text.strip():
            return None
        title = row.get(self.title_field)
        updated = row.get(self.updated_field) if self.updated_field else None
        return SourceDocument(
            id=str(rid),
            source_uri=f"{self.kind or 'db'}://{self.id_field}={rid}",
            title=str(title) if title else str(rid),
            text=text,
            doc_type="record",
            metadata={"row": {k: str(v) for k, v in row.items()}},
            external_access=ExternalAccess.public(),
            updated_at=str(updated) if updated is not None else None,
        )

    def _run(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = self._connection().read(self.query, params or {})
        return [r for r in rows if isinstance(r, dict)]

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        for row in self._run():
            doc = self._to_document(row)
            if doc is not None:
                yield doc

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Return rows newer than the watermark; advance it to the new max.

        CONCEPT:ECO-4.26 — when ``watermark_param`` is set the query binds the
        prior watermark (server-side filtering); otherwise the full result is
        filtered in-memory by ``updated_field``. The watermark advances to the max
        ``updated_at`` observed, so a re-poll with no new rows yields nothing.
        """
        prior = checkpoint.watermark if checkpoint else None
        if self.watermark_param and prior is not None:
            rows = self._run({self.watermark_param: prior})
        else:
            rows = self._run()

        docs: list[SourceDocument] = []
        max_wm = prior
        for row in rows:
            doc = self._to_document(row)
            if doc is None:
                continue
            if self.updated_field and prior is not None and doc.updated_at is not None:
                if str(doc.updated_at) <= str(prior):
                    continue
            docs.append(doc)
            if doc.updated_at is not None and (
                max_wm is None or str(doc.updated_at) > str(max_wm)
            ):
                max_wm = doc.updated_at
        return CheckpointedBatch(
            documents=docs,
            checkpoint=ConnectorCheckpoint(has_more=False, watermark=max_wm),
        )
