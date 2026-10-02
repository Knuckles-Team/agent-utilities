"""Shared two-page CDC ingestion scenario for
``test_external_graph_ingestion.py`` (NOT shared with the frozen
``tests/characterization/knowledge_graph/ingestion/
test_ingest_registered_graph_characterization.py``, which pins its own
inline copy of an equivalent scenario verbatim and must not change).

The stub serves pages from a small table, and the helper only drives the
ingestion; each caller asserts what it adds beyond the characterized result.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agent_utilities.knowledge_graph.ingestion.external_graph import (
    ExternalGraphIngestionRequest,
    ingest_registered_graph,
)


def _upsert(node_id: str, kind: str, version: str, title: str) -> dict[str, Any]:
    record = {"id": node_id, "kind": kind, "version": version}
    return {
        "operation": "upsert",
        "entity": "node",
        "record": {**record, "properties": {"title": title}},
    }


def _delete(node_id: str) -> dict[str, Any]:
    return {"operation": "delete", "entity": "node", "id": node_id}


# cursor presented -> (events, next cursor, more pages follow)
_TWO_PAGES: dict[str, tuple[list[dict[str, Any]], str, bool]] = {
    "cursor-1": (
        [
            _upsert("raw-node-a", "Capability", "1", "Synthetic A"),
            _delete("raw-node-old"),
        ],
        "cursor-2",
        True,
    ),
    "cursor-2": (
        [_upsert("raw-node-b", "Process", "2", "Synthetic B")],
        "cursor-3",
        False,
    ),
}


class PagedChangeGraph:
    """A change-feed stub that serves a fixed table of pages keyed by the
    cursor presented, and records every cursor it was asked for. A snapshot
    read must never be reached while a change feed is available."""

    def __init__(
        self, pages: dict[str, tuple[list[dict[str, Any]], str, bool]]
    ) -> None:
        self._pages = pages
        self.cursors: list[str | None] = []

    def execute_read(self, _query: str, _params: dict):
        raise AssertionError("snapshot query must not run when CDC is available")

    def read_change_page(self, *, cursor: str | None, limit: int):
        assert limit == 2
        self.cursors.append(cursor)
        events, next_cursor, has_more = self._pages[str(cursor)]
        return {"events": events, "next_cursor": next_cursor, "has_more": has_more}


def run_two_page_cdc(
    monkeypatch: Any,
    *,
    registry_cls: Any,
    profile: Any,
    patch_capture: Any,
    base_request: ExternalGraphIngestionRequest,
) -> tuple[PagedChangeGraph, dict[str, Any], list]:
    """Drive ``ingest_registered_graph`` through one two-page change-feed
    cycle starting from a stored cursor and return the stub graph, the
    result and the captured envelopes for the caller to assert on.

    The end-to-end result of this scenario is pinned by the characterization
    suite; callers here assert only what they add to it.
    """
    captured: list = []
    graph = PagedChangeGraph(_TWO_PAGES)
    start = next(iter(_TWO_PAGES))
    monkeypatch.setattr(
        f"{ingest_registered_graph.__module__}.read_change_cursor",
        lambda *_args, **_kwargs: start,
    )
    patch_capture(monkeypatch, captured)
    paged = dataclasses.replace(base_request, page_size=2, max_pages=len(_TWO_PAGES))
    result = ingest_registered_graph(
        object(), registry_cls(graph), paged, profile=profile()
    )
    return graph, result, captured
