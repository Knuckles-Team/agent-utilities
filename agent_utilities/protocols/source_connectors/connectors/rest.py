from __future__ import annotations

"""Generic paginated REST/JSON document-source connector.

CONCEPT:ECO-4.25 — reference ``Load`` + ``Poll`` connector.
CONCEPT:ECO-4.26 — cursor-based incremental poll.

Fetches a JSON list endpoint and maps each record to a :class:`SourceDocument`
via a declarative field map (``id_field`` / ``title_field`` / ``text_field`` /
``updated_field``). Pagination follows either a ``next_url`` field or a cursor
field echoed back as a query parameter. ``httpx`` is imported lazily; tests inject
a transport (``httpx.MockTransport``) or a ``fetch_fn`` so no network is required.
"""

from collections.abc import Callable, Iterator
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

# A fetch returns the decoded JSON for a (url, params) request.
JsonFetchFn = Callable[[str, dict[str, Any]], Any]


def _dig(record: dict[str, Any], dotted: str) -> Any:
    """Resolve a dotted path (``a.b.c``) within a nested dict; ``None`` if absent."""
    cur: Any = record
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _make_httpx_fetch(
    transport: Any = None, headers: dict[str, str] | None = None
) -> JsonFetchFn:
    """Build a lazy-``httpx`` JSON fetcher (optionally over an injected transport)."""

    def _fetch(url: str, params: dict[str, Any]) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment without httpx
            raise RuntimeError(
                "RestJsonConnector needs 'httpx'. Install it or pass a fetch_fn."
            ) from exc
        client_kwargs: dict[str, Any] = {"timeout": 60.0}
        if transport is not None:
            client_kwargs["transport"] = transport
        if headers:
            client_kwargs["headers"] = headers
        with httpx.Client(**client_kwargs) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    return _fetch


@register_source("rest")
class RestJsonConnector(LoadConnector, PollConnector):
    """Map a paginated JSON list endpoint to documents.

    CONCEPT:ECO-4.25.

    Config:
        url: List endpoint (required).
        records_field: Dotted path to the array of records (default: the body is
            itself the array).
        id_field / title_field / text_field / updated_field: dotted field maps.
        cursor_field: Record/response field carrying the next-page cursor.
        cursor_param: Query parameter name to send the cursor back as.
        next_url_field: Response field carrying an absolute next-page URL.
        params: Static query params sent on every request.
        max_pages: Page cap (default 50).
        fetch_fn / transport / headers: offline/test injection + auth headers.
    """

    provider = "REST/JSON"

    def configure(
        self,
        *,
        url: str = "",
        records_field: str = "",
        id_field: str = "id",
        title_field: str = "title",
        text_field: str = "text",
        updated_field: str = "",
        cursor_field: str = "",
        cursor_param: str = "cursor",
        next_url_field: str = "",
        params: dict[str, Any] | None = None,
        max_pages: int = 50,
        fetch_fn: JsonFetchFn | None = None,
        transport: Any = None,
        headers: dict[str, str] | None = None,
        **_: object,
    ) -> None:
        if not url:
            raise ValueError("RestJsonConnector requires a 'url'")
        self.url = url
        self.records_field = records_field
        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field
        self.updated_field = updated_field
        self.cursor_field = cursor_field
        self.cursor_param = cursor_param
        self.next_url_field = next_url_field
        self.params = dict(params or {})
        self.max_pages = max(1, int(max_pages))
        self._fetch: JsonFetchFn = fetch_fn or _make_httpx_fetch(transport, headers)

    def health_check(self) -> bool:
        return bool(self.url)

    def _records(self, body: Any) -> list[dict[str, Any]]:
        data = _dig(body, self.records_field) if self.records_field else body
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    def _to_document(self, record: dict[str, Any]) -> SourceDocument | None:
        rid = _dig(record, self.id_field)
        text = _dig(record, self.text_field)
        if rid is None or not isinstance(text, str) or not text.strip():
            return None
        title = _dig(record, self.title_field)
        updated = _dig(record, self.updated_field) if self.updated_field else None
        return SourceDocument(
            id=str(rid),
            source_uri=self.url,
            title=str(title) if title else str(rid),
            text=text,
            doc_type="record",
            metadata={"raw": record},
            external_access=ExternalAccess.public(),
            updated_at=str(updated) if updated is not None else None,
        )

    def _next_cursor(self, body: Any, records: list[dict[str, Any]]) -> str | None:
        """Resolve the next-page cursor from the response or the last record."""
        if self.cursor_field:
            top = _dig(body, self.cursor_field)
            if top:
                return str(top)
            if records:
                last = _dig(records[-1], self.cursor_field)
                if last:
                    return str(last)
        return None

    def _page(
        self, url: str, cursor: str | None
    ) -> tuple[list[SourceDocument], str | None, str | None]:
        """Fetch one page → ``(documents, next_url, next_cursor)``."""
        params = dict(self.params)
        if cursor and self.cursor_param:
            params[self.cursor_param] = cursor
        body = self._fetch(url, params)
        records = self._records(body)
        docs = [d for d in (self._to_document(r) for r in records) if d is not None]
        next_url = _dig(body, self.next_url_field) if self.next_url_field else None
        next_cursor = self._next_cursor(body, records)
        return docs, (str(next_url) if next_url else None), next_cursor

    # -- LoadConnector -----------------------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        url: str | None = self.url
        cursor: str | None = None
        pages = 0
        while url and pages < self.max_pages:
            docs, next_url, next_cursor = self._page(url, cursor)
            yield from docs
            pages += 1
            if next_url:
                url, cursor = next_url, None
            elif next_cursor:
                url, cursor = self.url, next_cursor
            else:
                break

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Fetch the next page only, carrying the cursor across calls.

        CONCEPT:ECO-4.26 — each ``poll`` advances one page; ``has_more`` and the
        ``cursor`` in the returned checkpoint drive the drain loop.
        """
        cursor = checkpoint.cursor if checkpoint else None
        url = (checkpoint.state.get("url") if checkpoint else None) or self.url
        docs, next_url, next_cursor = self._page(url, cursor)
        if next_url:
            cp = ConnectorCheckpoint(
                has_more=True, cursor=None, state={"url": next_url}
            )
        elif next_cursor:
            cp = ConnectorCheckpoint(
                has_more=True, cursor=next_cursor, state={"url": self.url}
            )
        else:
            cp = ConnectorCheckpoint(has_more=False)
        return CheckpointedBatch(documents=docs, checkpoint=cp)
