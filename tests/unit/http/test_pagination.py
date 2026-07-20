"""Tests for agent_utilities.http.pagination (CONCEPT:AU-ECO.ui.fleet-http-client-library).

Drives the iterators with fake page fetchers across all five dialects:
cursor, page/page_size, offset, Link-header, since-id.
"""

from __future__ import annotations

import pytest

from agent_utilities.http.pagination import AsyncPaginationIterator, PaginationIterator

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _record(n: int) -> dict:
    return {"id": n, "name": f"item-{n}"}


# --------------------------------------------------------------------------- #
# cursor mode
# --------------------------------------------------------------------------- #


def test_cursor_pagination_follows_dotted_response_path():
    pages = {
        None: {"items": [_record(1), _record(2)], "meta": {"next": "c2"}},
        "c2": {"items": [_record(3)], "meta": {"next": None}},
    }
    seen_params = []

    def fetch(url, params):
        seen_params.append(params)
        cursor = (params or {}).get("after")
        return pages[cursor], {}

    iterator = PaginationIterator(
        fetch,
        "/widgets",
        mode="cursor",
        cursor_param="after",
        cursor_path="meta.next",
        items_path="items",
        params={"q": "x"},
    )
    assert [r["id"] for r in iterator] == [1, 2, 3]
    assert iterator.pages_fetched == 2
    assert iterator.truncated is False
    assert iterator.next_cursor is None
    assert seen_params[1] == {"q": "x", "after": "c2"}


def test_cursor_max_items_truncates_and_reports_resume_cursor():
    def fetch(url, params):
        start = int((params or {}).get("cursor", 0))
        items = [_record(start + i) for i in range(3)]
        return {"items": items, "next_cursor": str(start + 3)}, {}

    iterator = PaginationIterator(
        fetch, "/widgets", mode="cursor", items_path="items", max_items=4
    )
    assert len(list(iterator)) == 4
    assert iterator.truncated is True
    assert iterator.next_cursor is not None


# --------------------------------------------------------------------------- #
# page mode
# --------------------------------------------------------------------------- #


def test_page_pagination_stops_on_short_page():
    requested = []

    def fetch(url, params):
        requested.append(dict(params or {}))
        page = params["page"]
        size = params["page_size"]
        count = size if page < 3 else 1  # third page is short
        return [_record(page * 10 + i) for i in range(count)], {}

    iterator = PaginationIterator(fetch, "/things", mode="page", page_size=2)
    items = list(iterator)
    assert len(items) == 5
    assert requested[0] == {"page": 1, "page_size": 2}
    assert requested[-1] == {"page": 3, "page_size": 2}
    assert iterator.truncated is False


def test_page_pagination_respects_max_pages():
    def fetch(url, params):
        return [_record(params["page"])], {}

    iterator = PaginationIterator(
        fetch, "/things", mode="page", page_size=1, max_pages=3
    )
    assert len(list(iterator)) == 3
    assert iterator.truncated is True


# --------------------------------------------------------------------------- #
# offset mode
# --------------------------------------------------------------------------- #


def test_offset_pagination_advances_by_items_received():
    dataset = [_record(i) for i in range(7)]

    def fetch(url, params):
        offset, limit = params["offset"], params["limit"]
        return dataset[offset : offset + limit], {}

    iterator = PaginationIterator(fetch, "/rows", mode="offset", page_size=3)
    assert [r["id"] for r in iterator] == list(range(7))
    assert iterator.pages_fetched == 3


# --------------------------------------------------------------------------- #
# link mode
# --------------------------------------------------------------------------- #


def test_link_pagination_follows_rel_next_absolute_urls():
    urls_fetched = []

    def fetch(url, params):
        urls_fetched.append(url)
        if url == "/users":
            headers = {"Link": '<https://api.test/users?after=a2>; rel="next"'}
            return [_record(1)], headers
        return [_record(2)], {"Link": '<https://api.test/x>; rel="self"'}

    iterator = PaginationIterator(fetch, "/users", mode="link")
    assert [r["id"] for r in iterator] == [1, 2]
    assert urls_fetched == ["/users", "https://api.test/users?after=a2"]
    assert iterator.next_cursor is None


# --------------------------------------------------------------------------- #
# since_id mode
# --------------------------------------------------------------------------- #


def test_since_id_pagination_resumes_from_last_record():
    def fetch(url, params):
        since = (params or {}).get("since_id", 0)
        if since >= 4:
            return [], {}
        return [_record(since + 1), _record(since + 2)], {}

    iterator = PaginationIterator(fetch, "/events", mode="since_id")
    assert [r["id"] for r in iterator] == [1, 2, 3, 4]
    # An empty page means the collection is exhausted — nothing to resume.
    assert iterator.next_cursor is None


# --------------------------------------------------------------------------- #
# shared behavior
# --------------------------------------------------------------------------- #


def test_invalid_mode_rejected_at_construction():
    with pytest.raises(ValueError, match="mode must be one of"):
        PaginationIterator(lambda url, params: ([], {}), "/x", mode="scroll")


def test_non_list_payload_yields_nothing():
    iterator = PaginationIterator(
        lambda url, params: ({"unexpected": "shape"}, {}), "/x", mode="page"
    )
    assert list(iterator) == []


def test_collect_returns_okta_style_envelope():
    def fetch(url, params):
        return ([_record(1), _record(2)] if params["page"] == 1 else []), {}

    iterator = PaginationIterator(fetch, "/x", mode="page")
    envelope = iterator.collect()
    assert envelope["count"] == 2
    assert envelope["truncated"] is False
    assert envelope["next_cursor"] is None
    assert [r["id"] for r in envelope["data"]] == [1, 2]


def test_iterator_is_restartable_with_fresh_state():
    def fetch(url, params):
        page = params["page"]
        return ([_record(page)] if page <= 2 else []), {}

    iterator = PaginationIterator(fetch, "/x", mode="page", page_size=1)
    assert len(list(iterator)) == 2
    assert len(list(iterator)) == 2  # restart yields the same sweep


# --------------------------------------------------------------------------- #
# async iterator
# --------------------------------------------------------------------------- #


async def test_async_cursor_pagination():
    async def fetch(url, params):
        cursor = (params or {}).get("cursor")
        if cursor is None:
            return {"items": [_record(1)], "next_cursor": "c2"}, {}
        return {"items": [_record(2)], "next_cursor": None}, {}

    iterator = AsyncPaginationIterator(
        fetch, "/widgets", mode="cursor", items_path="items"
    )
    assert [r["id"] async for r in iterator] == [1, 2]
    assert iterator.pages_fetched == 2


async def test_async_collect_envelope():
    async def fetch(url, params):
        return ([_record(1)] if params["page"] == 1 else []), {}

    iterator = AsyncPaginationIterator(fetch, "/x", mode="page")
    envelope = await iterator.collect()
    assert envelope["count"] == 1
    assert envelope["truncated"] is False
