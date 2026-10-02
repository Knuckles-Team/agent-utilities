"""Shared two-page CDC ingestion scenario for
``test_external_graph_ingestion.py`` (NOT shared with the frozen
``tests/characterization/knowledge_graph/ingestion/
test_ingest_registered_graph_characterization.py``, which pins its own
inline copy of an equivalent scenario verbatim and must not change).

Extracting this out of the unit test's own body means that file's test no
longer carries a full copy of the stub graph + scenario wiring inline —
only the call and its own assertions do.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.ingestion.external_graph import (
    ExternalGraphIngestionRequest,
    ingest_registered_graph,
)


class CDCGraph:
    """A two-page CDC stub: page 1 upserts + deletes one node, page 2 upserts
    another. ``execute_read`` must never be reached while CDC is available."""

    def __init__(self) -> None:
        self.cursors: list[str | None] = []

    def execute_read(self, _query: str, _params: dict):
        raise AssertionError("snapshot query must not run when CDC is available")

    def read_change_page(self, *, cursor: str | None, limit: int):
        assert limit == 2
        self.cursors.append(cursor)
        if cursor == "cursor-1":
            return {
                "events": [
                    {
                        "operation": "upsert",
                        "entity": "node",
                        "record": {
                            "id": "raw-node-a",
                            "kind": "Capability",
                            "version": "1",
                            "properties": {"title": "Synthetic A"},
                        },
                    },
                    {
                        "operation": "delete",
                        "entity": "node",
                        "id": "raw-node-old",
                    },
                ],
                "next_cursor": "cursor-2",
                "has_more": True,
            }
        return {
            "events": [
                {
                    "operation": "upsert",
                    "entity": "node",
                    "record": {
                        "id": "raw-node-b",
                        "kind": "Process",
                        "version": "2",
                        "properties": {"title": "Synthetic B"},
                    },
                }
            ],
            "next_cursor": "cursor-3",
            "has_more": False,
        }


def assert_two_page_cdc_scenario(
    monkeypatch: Any,
    *,
    registry_cls: Any,
    profile: Any,
    patch_capture: Any,
    base_request: Any,
) -> list:
    """Drive ``ingest_registered_graph`` through one two-page CDC cycle and
    assert the full set of invariants this scenario proves: the cursor
    advances exactly once, the sync strategy/node/delete counts are right,
    every envelope carries the correct operation in order, and only the
    LAST envelope (the trailing snapshot-complete marker) carries a
    checkpoint.

    ``registry_cls``/``profile``/``patch_capture``/``base_request`` are the
    calling test module's own doubles (``_Registry``, ``_profile``,
    ``_patch_ingest_capture``, ``_request()``) — this helper stays agnostic
    of which test file supplies them. Returns ``captured`` for any
    additional, caller-specific assertions (e.g. privacy/provenance checks)
    beyond this shared scenario's own invariants.
    """
    captured: list = []
    graph = CDCGraph()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.read_change_cursor",
        lambda _engine, _connector, *, source_instance: "cursor-1",
    )
    patch_capture(monkeypatch, captured)
    request = ExternalGraphIngestionRequest(
        **{**base_request.__dict__, "page_size": 2, "max_pages": 2}
    )
    result = ingest_registered_graph(
        object(), registry_cls(graph), request, profile=profile()
    )

    assert graph.cursors == ["cursor-1", "cursor-2"]
    assert result["sync_strategy"] == "cdc"
    assert result["nodes"] == 2
    assert result["deletes"] == 1
    assert [envelope.operation for envelope in captured] == [
        "upsert",
        "upsert",
        "delete",
        "snapshot_complete",
    ]
    assert all(envelope.checkpoint is None for envelope in captured[:-1])
    assert captured[-1].checkpoint == "cursor-3"
    return captured
