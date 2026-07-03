"""_checkpoint_db must never run a raw query on a non-SQLite backend.

Regression: the live ``TieredGraphBackend`` (and the epistemic/Postgres
backends) route ``execute()`` through the Cypher engine, so the old raw
``execute("CHECKPOINT;")`` fallback misparsed that string and blocked
indefinitely on the engine — deadlocking every task worker after each
``_update_task_status``. (CONCEPT:KG-2.8)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _GraphBackend:
    """A tiered/graph backend: it has no WAL, and execute() hits the engine."""

    def execute(self, *_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError(
            "_checkpoint_db must not call execute() on a graph backend"
        )


class _SqliteBackend:
    def __init__(self) -> None:
        self.checkpointed = False

    def wal_checkpoint(self) -> bool:
        self.checkpointed = True
        return True


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_checkpoint_db_skips_backend_without_wal_checkpoint():
    stub = SimpleNamespace(backend=_GraphBackend())
    # No exception, and crucially no execute() call (which would block forever).
    TaskManagerMixin._checkpoint_db(stub)


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_checkpoint_db_uses_wal_checkpoint_when_present():
    be = _SqliteBackend()
    TaskManagerMixin._checkpoint_db(SimpleNamespace(backend=be))
    assert be.checkpointed is True


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_checkpoint_db_tolerates_missing_backend():
    TaskManagerMixin._checkpoint_db(SimpleNamespace(backend=None))
