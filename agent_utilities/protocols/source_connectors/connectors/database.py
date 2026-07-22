from __future__ import annotations

"""Relational/NoSQL database document-source connector.

CONCEPT:AU-ECO.connector.document-source-framework — reference ``Load`` + ``Poll`` connector over a database.
CONCEPT:AU-ECO.connector.incremental-poll-watermark — incremental poll keyed on a monotonic column watermark.

Wraps the existing :class:`UniversalConnector` (CONCEPT:AU-KG.ingest.enterprise-source-extractor) so a database it
speaks becomes a document source: a query's rows are each mapped to a
:class:`SourceDocument` via a declarative field map. **PostgreSQL is the proven
native path** — other ``UniversalConnector`` dialects (MySQL/MariaDB, MSSQL,
Oracle, Mongo) require their driver installed in-process and are better served
through the :mod:`mcp_tool` source (CONCEPT:AU-KG.ingest.mcp-tool-connector) over sql-mcp, which owns
the dialect drivers, the read-only gate, and the row caps. No new driver
dependency is added here; ``UniversalConnector`` lazily imports the appropriate
driver and raises a clear error if it is missing.

This composes rather than competes: ``UniversalConnector`` already does the wire
work + ``introspect()``; this connector adds the *document* projection and the
incremental watermark so DB content lands in the KG as ``Document`` + ``Chunk``
ontology objects (with all the OWL/ACL benefits the framework brings).
"""

import json
import re
from collections.abc import Iterator
from typing import Any

from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    LoadConnector,
    PollConnector,
    SourceDocument,
    default_external_access,
)
from ..registry import register_source

_SECRET_REF_RE = re.compile(r"^(?:vault|env|secret)://[A-Za-z0-9_./#-]+$")
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,254}$")
_SOURCE_ALIAS_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")


@register_source("database")
class DatabaseConnector(LoadConnector, PollConnector):
    """Map a database query's rows to documents.

    CONCEPT:AU-ECO.connector.document-source-framework.

    Config:
        connection_profile_ref: Runtime secret reference resolving to either a
            DSN string or ``{"dsn": ..., "kind": ...}``. Required unless
            ``conn`` is injected. The resolved profile never enters durable
            connector config, logs, provenance, or graph identifiers.
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
        table: Optional, operator-supplied name of the single table ``query``
            reads (CONCEPT:AU-KG.identity.evidence-spine-convergence). The connector wraps an
            arbitrary ``SELECT`` — potentially a join/view/aggregate — so this is
            NEVER inferred from the query text (that would misattribute a
            multi-table read); it is real information only the operator who
            wrote ``query`` has. Left unset (the default), ``poll`` writes no
            ``RowVersion`` evidence locus — never a guessed table name. Set it
            only when ``query`` genuinely reads one table.
        conn: Optional pre-built ``UniversalConnector`` (injected for tests).
        source_alias: Optional neutral logical alias (never an endpoint/name).
        max_rows: Hard cap per load/poll request; overflow fails closed.
    """

    provider = "Database"

    def configure(
        self,
        *,
        connection_profile_ref: str = "",
        kind: str | None = None,
        query: str = "",
        id_field: str = "id",
        title_field: str = "title",
        text_field: str = "text",
        updated_field: str = "",
        watermark_param: str = "",
        table: str = "",
        source_alias: str = "database-source",
        max_rows: int = 5_000,
        conn: Any = None,
        **_: object,
    ) -> None:
        if "dsn" in _:
            raise ValueError("legacy database dsn configuration is unsupported")
        profile_ref = str(connection_profile_ref or "").strip()
        if conn is None and not _SECRET_REF_RE.fullmatch(profile_ref):
            raise ValueError(
                "DatabaseConnector requires a runtime connection_profile_ref"
            )
        if (
            conn is not None
            and profile_ref
            and not _SECRET_REF_RE.fullmatch(profile_ref)
        ):
            raise ValueError("literal database connection values are not supported")
        if not query:
            raise ValueError("DatabaseConnector requires a 'query'")
        if len(query.encode("utf-8")) > 65_536 or "\x00" in query:
            raise ValueError("DatabaseConnector query is invalid")
        if not _SOURCE_ALIAS_RE.fullmatch(source_alias):
            raise ValueError("DatabaseConnector source_alias is invalid")
        if not isinstance(max_rows, int) or isinstance(max_rows, bool):
            raise ValueError("DatabaseConnector max_rows is invalid")
        if not 1 <= max_rows <= 9_999:
            raise ValueError("DatabaseConnector max_rows is out of range")
        fields = (id_field, title_field, text_field)
        if updated_field:
            fields = (*fields, updated_field)
        if watermark_param:
            fields = (*fields, watermark_param)
        if table:
            fields = (*fields, table)
        if any(not _FIELD_RE.fullmatch(value) for value in fields):
            raise ValueError("DatabaseConnector field mapping is invalid")
        effective_kind = str(kind or "").strip().lower() or None
        self._validate_read_query(query, effective_kind)
        self.connection_profile_ref = profile_ref
        self.kind = effective_kind
        self.query = query
        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field
        self.updated_field = updated_field
        self.watermark_param = watermark_param
        self.table = table
        self.source_alias = source_alias
        self.source_ref = persistence_reference(
            "database_source",
            f"{source_alias}\x00{profile_ref or 'injected-source'}",
        )
        self.max_rows = max_rows
        self._conn = conn
        self._privacy_guard = PersistencePrivacyGuard()
        self.external_access = default_external_access()

    @staticmethod
    def _validate_read_query(query: str, kind: str | None) -> None:
        """Reject obvious mutating or stacked SQL at the ingestion boundary.

        This is defense in depth; production credentials must still be granted
        read-only database permissions. Mongo's ``query`` is a collection name,
        while SQL ingestion deliberately accepts one plain ``SELECT`` only.
        """
        value = query.strip()
        if kind == "mongodb":
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", value):
                raise ValueError("Mongo source query must be a collection name")
            return
        if value.endswith(";"):
            value = value[:-1].rstrip()
        if (
            not re.match(r"(?is)^select\b", value)
            or ";" in value
            or "--" in value
            or "/*" in value
            or re.search(r"(?is)\binto\b", value)
            or re.search(r"(?is)\bfor\s+(update|share)\b", value)
        ):
            raise ValueError("Database ingestion requires one read-only SELECT query")

    def _connection(self) -> Any:
        if self._conn is None:
            from agent_utilities.security.secrets_client import create_secrets_client

            from ...universal_connector import UniversalConnector
            from ..registry import logger  # reuse package logger

            logger.debug("[ECO-4.25] opening database connection kind=%s", self.kind)
            secrets = create_secrets_client()
            raw = secrets.resolve_ref(self.connection_profile_ref)
            rendered = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
            if not rendered or len(rendered.encode("utf-8")) > 65_536:
                raise ValueError("database connection profile is unavailable")
            profile_kind = None
            resolved_dsn = rendered
            try:
                profile = json.loads(rendered)
            except (TypeError, ValueError):
                profile = None
            if isinstance(profile, dict):
                if set(profile).difference(
                    {"dsn", "kind", "tls_service", "tls_profile", "tls_profile_ref"}
                ):
                    raise ValueError(
                        "database connection profile has unsupported fields"
                    )
                resolved_dsn = str(profile.get("dsn") or "")
                profile_kind = str(profile.get("kind") or "").strip().lower() or None
            if (
                not resolved_dsn
                or "\x00" in resolved_dsn
                or len(resolved_dsn.encode("utf-8")) > 8_192
            ):
                raise ValueError("database connection profile is unavailable")
            if self.kind and profile_kind and self.kind != profile_kind:
                raise ValueError("database connection profile kind does not match")
            self._conn = UniversalConnector(
                resolved_dsn,
                kind=self.kind or profile_kind,
                source_alias=self.source_alias,
                tls_service=(
                    str(profile.get("tls_service") or "").strip() or None
                    if isinstance(profile, dict)
                    else None
                ),
                tls_profile=(
                    str(profile.get("tls_profile") or "").strip() or None
                    if isinstance(profile, dict)
                    else None
                ),
                tls_profile_ref=(
                    str(profile.get("tls_profile_ref") or "").strip() or None
                    if isinstance(profile, dict)
                    else None
                ),
                tls_resolver=secrets.resolve_ref,
            )
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
        rendered_id = str(rid)
        if not rendered_id or len(rendered_id.encode("utf-8")) > 4_096:
            return None
        document_id = persistence_reference(
            "source_document",
            rendered_id,
            namespace=self.source_ref,
        )
        title = row.get(self.title_field)
        updated = row.get(self.updated_field) if self.updated_field else None
        clean_title, _ = self._privacy_guard.sanitize_text(
            str(title) if title else "database record"
        )
        clean_text, _ = self._privacy_guard.sanitize_text(text)
        clean_updated = None
        if updated is not None:
            clean_updated, _ = self._privacy_guard.sanitize_text(str(updated))
        return SourceDocument(
            id=document_id,
            source_uri=(f"external-source://{self.source_ref}/{document_id}"),
            title=clean_title,
            text=clean_text,
            doc_type="record",
            metadata={
                "source_system": self.kind or "database",
                "source_ref": self.source_ref,
            },
            external_access=self.external_access.model_copy(deep=True),
            updated_at=clean_updated,
        )

    def _run(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = self._connection().read(
            self.query,
            params or {},
            max_rows=self.max_rows + 1,
        )
        if len(rows) > self.max_rows:
            raise RuntimeError("database source result exceeds the configured row cap")
        return [r for r in rows if isinstance(r, dict)]

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        for row in self._run():
            doc = self._to_document(row)
            if doc is not None:
                yield doc

    # -- PollConnector -----------------------------------------------------

    def _persist_row_version_evidence(
        self, row: dict[str, Any], doc: SourceDocument
    ) -> None:
        """Through-write a ``RowVersion`` evidence locus for this poll
        observation (CONCEPT:AU-KG.identity.evidence-spine-convergence).

        Opt-in on TWO real, non-fabricated facts — never inferred, never
        guessed:

        * ``table`` — only set when the operator explicitly configured it
          (see :meth:`configure`'s docstring: this connector wraps an
          arbitrary ``SELECT``, so a table name is NEVER derived from the
          query text, which would misattribute a join/view/aggregate).
        * ``version`` — the row's own already-fetched ``updated_field``
          value, parsed as an int. When it genuinely is an incrementing-id/
          revision column (a real, supported ``updated_field`` use case per
          :meth:`configure`'s docstring) this is a real number already in
          the row, not synthesized. When it is an ISO timestamp (the other
          supported use case) the parse fails and this cleanly no-ops —
          there IS no integer version to report, so none is invented.

        With either fact missing, this is a silent no-op — genuinely no
        evidence to write, not a forced write. Best-effort: an
        engine-unavailable store never affects polling.
        """
        if not self.table or not doc.updated_at:
            return
        try:
            version = int(str(doc.updated_at))
        except (TypeError, ValueError):
            return
        row_id = row.get(self.id_field)
        if row_id is None:
            return
        try:
            from agent_utilities.knowledge_graph.memory.native_ingest import (
                media_store,
            )

            store = media_store()
            store.store_row_version_evidence(
                doc.text.encode("utf-8"),
                table=self.table,
                row_id=str(row_id),
                version=version,
                mime_type="text/plain",
                source=self.kind or "database",
            )
        except Exception:  # noqa: BLE001 — evidence write never affects polling
            pass

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Return rows newer than the watermark; advance it to the new max.

        CONCEPT:AU-ECO.connector.incremental-poll-watermark — when ``watermark_param`` is set the query binds the
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
            self._persist_row_version_evidence(row, doc)
            if doc.updated_at is not None and (
                max_wm is None or str(doc.updated_at) > str(max_wm)
            ):
                max_wm = doc.updated_at
        return CheckpointedBatch(
            documents=docs,
            checkpoint=ConnectorCheckpoint(has_more=False, watermark=max_wm),
        )
