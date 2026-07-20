"""Pagination iterators for fleet HTTP clients.

CONCEPT:AU-ECO.ui.fleet-http-client-library — Fleet HTTP Client Library

Five pagination dialects cover the fleet's upstream APIs:

* ``cursor`` — a cursor token sent as a query param, with the next cursor at
  a dotted path in the response body;
* ``page`` — page-number / page-size params, stopping on a short page;
* ``offset`` — offset / limit params, advancing by the items received;
* ``link`` — RFC 5988 ``Link: <...>; rel="next"`` headers (Okta, GitHub);
* ``since_id`` — keyset pagination resuming from the last record's id.

Relationship to the KG-2.59 ``mcp_tool`` source connector: that connector
also paginates (``cursor``/``page`` modes), but its machinery is interwoven
with MCP tool-argument rendering (dotted *param* paths, ``params_style``
JSON encoding), SQL keyset bootstrapping, and record→document conversion —
none of which exist at the HTTP layer — while this module needs HTTP-only
affordances (``Link`` response headers, absolute next-URLs) that MCP tool
results never carry. Extracting a shared core would reduce both sides to a
trivial advance-loop wrapped in adapter layers, so this is a documented
parallel implementation; the cursor semantics (param + dotted response
path, stop-on-falsy) deliberately match ``mcp_tool`` so configurations
translate 1:1.

The iterators are transport-agnostic: they drive a ``fetch_page`` callable
``(endpoint, params) -> (data, headers)`` supplied by
:meth:`agent_utilities.http.BaseApiClient.paginate` (sync) or
:meth:`agent_utilities.http.AsyncBaseApiClient.paginate` (async), and are
trivially fed by a fake in tests.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

__all__ = [
    "AsyncPaginationIterator",
    "PaginationIterator",
]

#: Pagination dialects supported by the iterators.
PAGINATION_MODES = ("cursor", "page", "offset", "link", "since_id")

_LINK_NEXT_RE = re.compile(r'<(?P<url>[^>]+)>\s*;[^,]*\brel="?next"?')

#: A page fetch: ``(endpoint_or_url, params) -> (parsed_data, headers)``.
FetchPage = Callable[[str, dict[str, Any] | None], tuple[Any, Mapping[str, str]]]
AsyncFetchPage = Callable[
    [str, dict[str, Any] | None], Awaitable[tuple[Any, Mapping[str, str]]]
]


def _dig(obj: Any, path: str) -> Any:
    """Resolve a dotted path inside nested dicts; ``None`` when absent."""
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _link_next_url(headers: Mapping[str, str]) -> str | None:
    """Extract the ``rel="next"`` target from a ``Link`` header."""
    raw = None
    for key, value in headers.items():
        if str(key).lower() == "link":
            raw = str(value)
            break
    if not raw:
        return None
    match = _LINK_NEXT_RE.search(raw)
    return match.group("url") if match else None


class _PageWalk:
    """Pure pagination state machine shared by the sync and async iterators.

    Holds no I/O: :meth:`start` yields the first ``(endpoint, params)`` and
    :meth:`advance` consumes one page's ``(items, data, headers)`` to produce
    the next request — or ``None`` when the collection is exhausted.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        mode: str,
        params: dict[str, Any] | None,
        cursor_param: str,
        cursor_path: str,
        page_param: str,
        page_size_param: str,
        page_size: int | None,
        start_page: int,
        offset_param: str,
        limit_param: str,
        since_param: str,
        id_field: str,
    ) -> None:
        if mode not in PAGINATION_MODES:
            raise ValueError(f"mode must be one of {PAGINATION_MODES}, got {mode!r}")
        self.mode = mode
        self.endpoint = endpoint
        self.base_params = dict(params or {})
        self.cursor_param = cursor_param
        self.cursor_path = cursor_path
        self.page_param = page_param
        self.page_size_param = page_size_param
        self.page_size = page_size
        self.offset_param = offset_param
        self.limit_param = limit_param
        self.since_param = since_param
        self.id_field = id_field
        self._page = start_page
        self._offset = 0
        #: Resume token after iteration (cursor / link / since_id modes).
        self.next_cursor: str | None = None

    def start(self) -> tuple[str, dict[str, Any]]:
        params = dict(self.base_params)
        if self.mode == "page":
            params[self.page_param] = self._page
            if self.page_size is not None:
                params[self.page_size_param] = self.page_size
        elif self.mode == "offset":
            params[self.offset_param] = self._offset
            if self.page_size is not None:
                params[self.limit_param] = self.page_size
        return self.endpoint, params

    def advance(
        self,
        items: list[Any],
        data: Any,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, Any]] | None:
        if self.mode == "cursor":
            cursor = _dig(data, self.cursor_path) if self.cursor_path else None
            self.next_cursor = str(cursor) if cursor else None
            if not cursor:
                return None
            params = dict(self.base_params)
            params[self.cursor_param] = cursor
            return self.endpoint, params

        if self.mode == "link":
            next_url = _link_next_url(headers)
            self.next_cursor = next_url
            if not next_url:
                return None
            # The next URL already carries the full query string.
            return next_url, {}

        if self.mode == "since_id":
            if not items:
                self.next_cursor = None
                return None
            last = items[-1]
            last_id = last.get(self.id_field) if isinstance(last, dict) else None
            if last_id is None:
                self.next_cursor = None
                return None
            self.next_cursor = str(last_id)
            params = dict(self.base_params)
            params[self.since_param] = last_id
            return self.endpoint, params

        # page / offset: a short or empty page means the collection is done.
        if not items or (self.page_size is not None and len(items) < self.page_size):
            return None
        if self.mode == "page":
            self._page += 1
        else:
            self._offset += len(items)
        return self.start()


class _PaginationConfig:
    """Shared configuration / bookkeeping for both iterator variants."""

    def __init__(
        self,
        endpoint: str,
        *,
        mode: str = "cursor",
        params: dict[str, Any] | None = None,
        items_path: str = "",
        max_items: int = 1000,
        max_pages: int = 100,
        cursor_param: str = "cursor",
        cursor_path: str = "next_cursor",
        page_param: str = "page",
        page_size_param: str = "page_size",
        page_size: int | None = None,
        start_page: int = 1,
        offset_param: str = "offset",
        limit_param: str = "limit",
        since_param: str = "since_id",
        id_field: str = "id",
    ) -> None:
        if mode not in PAGINATION_MODES:
            raise ValueError(f"mode must be one of {PAGINATION_MODES}, got {mode!r}")
        self._mode = mode
        self._params = params
        self._cursor_param = cursor_param
        self._cursor_path = cursor_path
        self._page_param = page_param
        self._page_size_param = page_size_param
        self._page_size = page_size
        self._start_page = start_page
        self._offset_param = offset_param
        self._limit_param = limit_param
        self._since_param = since_param
        self._id_field = id_field
        self.endpoint = endpoint
        self.items_path = items_path
        self.max_items = max(0, int(max_items))
        self.max_pages = max(1, int(max_pages))
        #: Set during iteration; inspect after the loop completes.
        self.truncated = False
        self.pages_fetched = 0
        self.items_yielded = 0
        self.next_cursor: str | None = None

    def _new_walk(self) -> _PageWalk:
        return _PageWalk(
            self.endpoint,
            mode=self._mode,
            params=self._params,
            cursor_param=self._cursor_param,
            cursor_path=self._cursor_path,
            page_param=self._page_param,
            page_size_param=self._page_size_param,
            page_size=self._page_size,
            start_page=self._start_page,
            offset_param=self._offset_param,
            limit_param=self._limit_param,
            since_param=self._since_param,
            id_field=self._id_field,
        )

    def _extract_items(self, data: Any) -> list[Any]:
        payload = _dig(data, self.items_path) if self.items_path else data
        if isinstance(payload, list):
            return payload
        return []

    def _begin(self, walk: _PageWalk) -> None:
        self.truncated = False
        self.pages_fetched = 0
        self.items_yielded = 0
        self.next_cursor = None
        self._budget = self.max_items
        self._walk = walk

    def _consume_page(
        self, items: list[Any], data: Any, headers: Mapping[str, str]
    ) -> tuple[list[Any], tuple[str, dict[str, Any]] | None]:
        """Apply budgets to one fetched page; returns (to_yield, next_request)."""
        self.pages_fetched += 1
        next_request = self._walk.advance(items, data, headers)
        to_yield = items
        if len(items) > self._budget:
            to_yield = items[: self._budget]
            self.truncated = True
            next_request = None
        self._budget -= len(to_yield)
        self.items_yielded += len(to_yield)
        if next_request is not None and self._budget <= 0:
            self.truncated = True
            next_request = None
        if next_request is not None and self.pages_fetched >= self.max_pages:
            self.truncated = True
            next_request = None
        self.next_cursor = self._walk.next_cursor
        return to_yield, next_request


class PaginationIterator(_PaginationConfig):
    """Iterate records across pages of a paginated HTTP collection.

    Built by :meth:`agent_utilities.http.BaseApiClient.paginate`; iterating
    yields individual records lazily. After iteration, ``truncated``,
    ``pages_fetched``, ``items_yielded`` and ``next_cursor`` describe the
    sweep (Okta-style resume semantics).

    Args:
        fetch_page: ``(endpoint, params) -> (data, headers)`` page fetcher.
        endpoint: Collection endpoint (relative to the client's base URL).
        mode: One of ``cursor`` / ``page`` / ``offset`` / ``link`` /
            ``since_id`` (see module docstring).
        params: Base query parameters sent with every page.
        items_path: Dotted path to the record list inside the response body;
            empty when the body *is* the list.
        max_items: Overall record budget (yields stop and ``truncated`` is
            set when reached).
        max_pages: Page-fetch budget.
        cursor_param / cursor_path: Cursor request param and dotted response
            path (``cursor`` mode).
        page_param / page_size_param / page_size / start_page: ``page`` mode.
        offset_param / limit_param: ``offset`` mode (shares ``page_size``).
        since_param / id_field: ``since_id`` keyset mode.
    """

    def __init__(self, fetch_page: FetchPage, endpoint: str, **options: Any) -> None:
        super().__init__(endpoint, **options)
        self._fetch_page = fetch_page

    def __iter__(self) -> Any:
        self._begin(self._new_walk())
        request: tuple[str, dict[str, Any]] | None = self._walk.start()
        while request is not None:
            url, params = request
            data, headers = self._fetch_page(url, params or None)
            items = self._extract_items(data)
            to_yield, request = self._consume_page(items, data, headers)
            yield from to_yield

    def collect(self) -> dict[str, Any]:
        """Drain the iterator into an Okta-style collection envelope."""
        items = list(self)
        return {
            "data": items,
            "count": len(items),
            "truncated": self.truncated,
            "next_cursor": self.next_cursor,
        }


class AsyncPaginationIterator(_PaginationConfig):
    """Async twin of :class:`PaginationIterator` (``async for`` records)."""

    def __init__(
        self, fetch_page: AsyncFetchPage, endpoint: str, **options: Any
    ) -> None:
        super().__init__(endpoint, **options)
        self._fetch_page = fetch_page

    async def __aiter__(self) -> Any:
        self._begin(self._new_walk())
        request: tuple[str, dict[str, Any]] | None = self._walk.start()
        while request is not None:
            url, params = request
            data, headers = await self._fetch_page(url, params or None)
            items = self._extract_items(data)
            to_yield, request = self._consume_page(items, data, headers)
            for item in to_yield:
                yield item

    async def collect(self) -> dict[str, Any]:
        """Drain the iterator into an Okta-style collection envelope."""
        items = [item async for item in self]
        return {
            "data": items,
            "count": len(items),
            "truncated": self.truncated,
            "next_cursor": self.next_cursor,
        }
