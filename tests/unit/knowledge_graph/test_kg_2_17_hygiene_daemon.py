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


def _maint_specs():
    """Register maintenance :Schedule nodes for the current config (OS-5.44)."""
    from agent_utilities.core import schedule_engine as _se
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    inst = _bare_mixin()
    inst.backend = EpistemicGraphBackend()
    inst._register_maintenance_schedules()
    return {s.name: s for s in _se._load_all(inst)}


@pytest.mark.concept(id="KG-2.17")
def test_hygiene_job_registered_by_default(monkeypatch):
    monkeypatch.delenv("KG_DEV_MODE", raising=False)
    # hygiene is now a durable :Schedule the unified scheduler enqueues (OS-5.44).
    spec = _maint_specs()["hygiene"]
    assert spec.enabled
    # Default interval is daily.
    assert spec.interval_s == 86400.0


@pytest.mark.concept(id="KG-2.17")
def test_all_maintenance_daemons_disabled_via_dev_mode(monkeypatch):
    # KG_HYGIENE_DAEMON (and every other per-daemon KG_*_DAEMON toggle) was
    # collapsed into one KG_DEV_MODE switch. The registry still lists hygiene —
    # gating is at the scheduler loop (`_kg_dev_mode()`), not the registry.
    from agent_utilities.core import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", True, raising=False)
    assert _kg_dev_mode() is True
    assert _maint_specs()["hygiene"].enabled

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
