"""ADR-5 / W2.2 — the WorkItem statechart-mirror read helpers.

The engine writes a co-located ``machine_state`` (its phase-1 statechart mirror) onto each
WorkItem node atomically with the authoritative ``status``.
:func:`~agent_utilities.orchestration.work_item.machine_state_distribution` surfaces the
lifecycle-state distribution over that property, and
:func:`~agent_utilities.orchestration.work_item.find_status_machine_divergences` sweeps for
disagreements and raises the au-side divergence alarm via the module logger.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.orchestration import work_item


class _CypherEngine:
    """Engine double: returns canned Cypher rows keyed by a marker substring."""

    def __init__(self, rows_by_marker: dict[str, list[dict[str, Any]]]) -> None:
        self._rows_by_marker = rows_by_marker
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def query_cypher(
        self, q: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append((q, params or {}))
        for marker, rows in self._rows_by_marker.items():
            if marker in q:
                return rows
        return []


def test_machine_state_distribution_groups_by_state() -> None:
    engine = _CypherEngine(
        {
            "count(w) AS n": [
                {"state": "leased", "n": 2},
                {"state": "running", "n": 1},
                {"state": "succeeded", "n": 1},
            ]
        }
    )
    assert work_item.machine_state_distribution(engine) == {
        "leased": 2,
        "running": 1,
        "succeeded": 1,
    }


def test_machine_state_distribution_scopes_by_tenant() -> None:
    engine = _CypherEngine({"count(w) AS n": [{"state": "ready", "n": 3}]})
    assert work_item.machine_state_distribution(engine, tenant="t1") == {"ready": 3}
    assert engine.queries[0][1] == {"tenant": "t1"}


def test_find_divergences_flags_and_alarms(caplog: Any) -> None:
    engine = _CypherEngine(
        {
            "machine_state AS machine_state": [
                {"id": "wi-1", "status": "leased", "machine_state": "leased"},
                {"id": "wi-2", "status": "succeeded", "machine_state": "ready"},
            ]
        }
    )
    with caplog.at_level(
        logging.WARNING, logger="agent_utilities.orchestration.work_item"
    ):
        divergences = work_item.find_status_machine_divergences(engine)
    assert divergences == [
        {"id": "wi-2", "status": "succeeded", "machine_state": "ready"}
    ]
    # The au-side divergence alarm fired for the divergent item (and only it).
    alarms = [m for m in caplog.messages if "statechart divergence" in m]
    assert len(alarms) == 1
    assert "wi-2" in alarms[0]


def test_find_divergences_clean_when_mirror_tracks_authority(caplog: Any) -> None:
    engine = _CypherEngine(
        {
            "machine_state AS machine_state": [
                {"id": "wi-1", "status": "leased", "machine_state": "leased"},
                {"id": "wi-2", "status": "running", "machine_state": "running"},
            ]
        }
    )
    with caplog.at_level(
        logging.WARNING, logger="agent_utilities.orchestration.work_item"
    ):
        assert work_item.find_status_machine_divergences(engine) == []
    assert not any("statechart divergence" in m for m in caplog.messages)
