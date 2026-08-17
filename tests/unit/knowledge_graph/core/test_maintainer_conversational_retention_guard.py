"""BUG-041 guard: automated low-signal pruning must not silently delete conversational
node types that have no reingest/backfill path (CONCEPT:AU-ECO.messaging.conversational-retention-guard).

Known-bad proof: an ``InboundMessage`` node below the importance threshold is NOT sent to
the ``DETACH DELETE`` sweep by default, and a loud warning fires citing BUG-041.
Known-good proof: an ordinary node type (e.g. ``WorkItem``) below the threshold is pruned
exactly as before, and the guard stays silent — the change is scoped to the four
unrecoverable conversational types only, never a general behavior change.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.knowledge_graph.core.maintainer import (
    UNRECOVERABLE_CONVERSATIONAL_NODE_TYPES,
    GraphMaintainer,
)

_LOGGER_NAME = "agent_utilities.knowledge_graph.core.maintainer"


class _FakeBackend:
    """Records every query passed to ``execute`` and returns a scripted count."""

    def __init__(self, protected_count: int = 1) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._protected_count = protected_count

    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
        self.queries.append((query, params or {}))
        if "count(n)" in query:
            return [{"c": self._protected_count}]
        return []


class _FakeEngine:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend


def test_unrecoverable_conversational_types_are_the_four_confirmed_by_bug041():
    assert UNRECOVERABLE_CONVERSATIONAL_NODE_TYPES == {
        "InboundMessage",
        "Thread",
        "Memory",
        "Memento",
    }


def test_default_sweep_excludes_unrecoverable_conversational_types_and_warns(caplog):
    """Known-bad proof: the guard fires for InboundMessage/Thread/Memory/Memento."""
    backend = _FakeBackend(protected_count=3)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = maintainer.prune_low_importance_nodes(threshold=0.05)

    assert result == 1
    # The DETACH DELETE query must exclude every unrecoverable conversational type.
    delete_query = next(q for q, _ in backend.queries if "DETACH DELETE" in q)
    for node_type in UNRECOVERABLE_CONVERSATIONAL_NODE_TYPES:
        assert f"n:{node_type}" in delete_query

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("BUG-041" in r.getMessage() for r in warnings)
    assert any("SKIPPED 3 node(s)" in r.getMessage() for r in warnings)


def test_ordinary_node_type_prune_is_unaffected_and_guard_stays_quiet(caplog):
    """Known-good proof: a non-conversational type below threshold is still swept, and the
    BUG-041 guard emits no warning (protected_count == 0 for a corpus with no matching
    conversational rows)."""
    backend = _FakeBackend(protected_count=0)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = maintainer.prune_low_importance_nodes(threshold=0.05)

    assert result == 1
    delete_query = next(q for q, _ in backend.queries if "DETACH DELETE" in q)
    # The sweep still runs (WorkItem/Log/etc. below threshold are unaffected by this guard).
    assert "importance_score < $threshold" in delete_query
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("BUG-041" in r.getMessage() for r in warnings)


def test_explicit_opt_in_removes_the_protection_and_the_warning(caplog):
    """An operator who has confirmed recoverability (or accepted the loss) can opt back in;
    the resulting query must NOT exclude the four types, and no protection warning fires."""
    backend = _FakeBackend(protected_count=5)
    maintainer = GraphMaintainer(_FakeEngine(backend))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        maintainer.prune_low_importance_nodes(
            threshold=0.05, include_unrecoverable_conversational=True
        )

    delete_query = next(q for q, _ in backend.queries if "DETACH DELETE" in q)
    for node_type in UNRECOVERABLE_CONVERSATIONAL_NODE_TYPES:
        assert f"n:{node_type}" not in delete_query
    # No count query is even issued when the caller explicitly opted in.
    assert not any("count(n)" in q for q, _ in backend.queries)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("BUG-041" in r.getMessage() for r in warnings)


def test_count_query_failure_never_blocks_the_actual_prune(caplog):
    """Fail-closed on the PROTECTION, fail-open on the best-effort COUNT: a broken count
    query must not prevent the exclusion clause from still being applied."""

    class _BrokenCountBackend(_FakeBackend):
        def execute(self, query: str, params: dict[str, Any] | None = None) -> Any:
            if "count(n)" in query:
                raise RuntimeError("backend unavailable")
            return super().execute(query, params)

    backend = _BrokenCountBackend()
    maintainer = GraphMaintainer(_FakeEngine(backend))

    result = maintainer.prune_low_importance_nodes(threshold=0.05)

    assert result == 1
    delete_query = next(q for q, _ in backend.queries if "DETACH DELETE" in q)
    for node_type in UNRECOVERABLE_CONVERSATIONAL_NODE_TYPES:
        assert f"n:{node_type}" in delete_query
