"""BUG-047 regression: the WorkItem poll/admission path must stay BOUNDED.

The engine's sustained 8-21s slow queries (against a 500ms threshold) traced to
``TaskManagerMixin._pending_by_lane`` — called on EVERY non-hydration claim
attempt, by every worker, on every poll cycle (idle backoff is only 2-15s) —
which used to fan out through ``_ingest_work_item_index``'s
``MATCH (w:WorkItem) RETURN <15 wide fields>`` with no ``WHERE``/``LIMIT``,
returning every WorkItem ever created. This locks the fix in place: a
regression back to the unbounded full scan must fail this test, not just show
up as a live incident three weeks later.
"""

from __future__ import annotations

import threading
from typing import Any

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.knowledge_graph.core.task_lanes import LANE_NAMES


class _RecordingWorkItemEngine:
    """Fake ``_work_item_engine`` that records every Cypher call it receives."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        return self.rows


class Harness:
    """Mixes in the REAL ``_pending_by_lane`` implementation over a fake engine."""

    _pending_by_lane = TaskManagerMixin._pending_by_lane

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._engine = _RecordingWorkItemEngine(rows)
        self._active_work_item_claims_lock = threading.Lock()

    @property
    def _work_item_engine(self) -> _RecordingWorkItemEngine:
        return self._engine


def test_pending_by_lane_issues_exactly_one_bounded_aggregate_query() -> None:
    """Exactly one Cypher call, and it must not be the unbounded full scan."""
    harness = Harness(rows=[{"resource_class": "ingestion", "n": 3}])

    result = harness._pending_by_lane()

    assert len(harness._engine.calls) == 1, (
        "one poll must issue exactly one engine call, not fan out into "
        "_ingest_work_item_index's per-row materialization"
    )
    cypher, params = harness._engine.calls[0]

    # It's a WHERE-filtered, GROUP-BY-style aggregate (mirrors
    # work_item.machine_state_distribution's convention) — never the old
    # unbounded field-by-field row scan.
    assert "WHERE" in cypher
    assert "count(w)" in cypher
    assert "w.status" in cypher
    assert "w.next_retry_at" in cypher
    assert "w.kind" in cypher  # only ingest_task WorkItems feed lane admission

    # The old unbounded projection is GONE from this call site: it pulled 15
    # wide fields (id/payload_ref/metadata/error_ref/...) with no WHERE/LIMIT.
    assert "w.payload_ref AS payload_ref" not in cypher
    assert "w.metadata AS metadata" not in cypher
    assert "w.attempt AS attempt" not in cypher

    # A time predicate is present and bound as a real parameter, not inlined —
    # proves the "ready AND due" predicate (mirrors
    # _task_status_from_work_item's "pending" classification) is actually wired,
    # not just present in a docstring.
    assert params is not None
    assert "now" in params
    assert isinstance(params["now"], float)

    assert result["ingestion"] == 3
    for lane in LANE_NAMES:
        if lane != "ingestion":
            assert result[lane] == 0


def test_pending_by_lane_aggregates_multiple_lanes_from_one_call() -> None:
    harness = Harness(
        rows=[
            {"resource_class": "ingestion", "n": 2},
            {"resource_class": "connectors", "n": 5},
        ]
    )

    result = harness._pending_by_lane()

    assert len(harness._engine.calls) == 1
    assert result["ingestion"] == 2
    assert result["connectors"] == 5


def test_pending_by_lane_ignores_unknown_lane_names_defensively() -> None:
    """A resource_class outside the known lane taxonomy must not raise or
    silently grow the result dict — only ``LANE_NAMES`` keys are ever present."""
    harness = Harness(rows=[{"resource_class": "not_a_real_lane", "n": 99}])

    result = harness._pending_by_lane()

    assert set(result) == set(LANE_NAMES)
    assert sum(result.values()) == 0
