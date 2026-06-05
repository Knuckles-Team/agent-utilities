"""CONCEPT:KG-2.17 — hygiene registered as a recurring maintenance daemon job.

Verifies the hygiene tick is in the consolidated maintenance-job registry (gated by
KG_HYGIENE_DAEMON) and that the tick runs without raising.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


def _bare_mixin():
    # Construct without __init__ — _maintenance_jobs / _tick_hygiene only need the class methods.
    return TaskManagerMixin.__new__(TaskManagerMixin)


@pytest.mark.concept(id="KG-2.17")
def test_hygiene_job_registered_by_default(monkeypatch):
    monkeypatch.delenv("KG_HYGIENE_DAEMON", raising=False)
    inst = _bare_mixin()
    jobs = inst._maintenance_jobs()
    names = [n for n, _, _ in jobs]
    assert "hygiene" in names
    # Default interval is daily.
    interval = next(i for n, i, _ in jobs if n == "hygiene")
    assert interval == 86400.0


@pytest.mark.concept(id="KG-2.17")
def test_hygiene_job_can_be_disabled(monkeypatch):
    monkeypatch.setenv("KG_HYGIENE_DAEMON", "0")
    inst = _bare_mixin()
    names = [n for n, _, _ in inst._maintenance_jobs()]
    assert "hygiene" not in names


@pytest.mark.concept(id="KG-2.17")
def test_tick_hygiene_runs_without_raising():
    inst = _bare_mixin()

    class _Backend:
        def __init__(self):
            self.calls = 0

        def execute(self, q, params=None):
            self.calls += 1
            return []  # empty memory set

    inst.backend = _Backend()
    inst._tick_hygiene()  # must not raise
    assert inst.backend.calls >= 1  # the scan query ran
