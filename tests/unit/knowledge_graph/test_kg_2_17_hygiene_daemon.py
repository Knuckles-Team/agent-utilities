"""CONCEPT:KG-2.17 — hygiene registered as a recurring maintenance daemon job.

Verifies the hygiene tick is in the consolidated maintenance-job registry and that
the tick runs without raising. The per-daemon ``KG_HYGIENE_DAEMON`` toggle was
collapsed into the single ``KG_DEV_MODE`` switch (config discipline, Phase 3): the
registry always lists hygiene; disabling happens once at the scheduler loop via
``KG_DEV_MODE``, not per job.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _kg_dev_mode,
)


def _bare_mixin():
    # Construct without __init__ — _maintenance_jobs / _tick_hygiene only need the class methods.
    return TaskManagerMixin.__new__(TaskManagerMixin)  # type: ignore[type-abstract]


@pytest.mark.concept(id="KG-2.17")
def test_hygiene_job_registered_by_default(monkeypatch):
    monkeypatch.delenv("KG_DEV_MODE", raising=False)
    inst = _bare_mixin()
    jobs = inst._maintenance_jobs()
    names = [n for n, _, _ in jobs]
    assert "hygiene" in names
    # Default interval is daily.
    interval = next(i for n, i, _ in jobs if n == "hygiene")
    assert interval == 86400.0


@pytest.mark.concept(id="KG-2.17")
def test_all_maintenance_daemons_disabled_via_dev_mode(monkeypatch):
    # KG_HYGIENE_DAEMON (and every other per-daemon KG_*_DAEMON toggle) was
    # collapsed into one KG_DEV_MODE switch. The registry still lists hygiene —
    # gating is at the scheduler loop (`_kg_dev_mode()`), not the registry.
    from agent_utilities.core import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", True, raising=False)
    assert _kg_dev_mode() is True
    names = [n for n, _, _ in _bare_mixin()._maintenance_jobs()]
    assert "hygiene" in names

    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", False, raising=False)
    assert _kg_dev_mode() is False


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
