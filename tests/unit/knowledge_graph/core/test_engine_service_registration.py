"""Regression: `register_services` must not latch its once-only guard on failure.

D-DSTO-2 (reports/deferred/lane-dst-orch.md): `IntelligenceGraphEngine.register_services`
set `self._services_registered = True` unconditionally, even when every per-service
`ServiceRegistry.register_with_kg` write failed and returned `count == 0`. Combined
with the early-return guard (`if self._services_registered: return 0`), a single
transient failure (e.g. backend down at startup) permanently disabled KG service
discovery for the life of the process — a write-then-mark-seen bug.

The fix: `self._services_registered = count > 0`, so a zero-count pass is retried
on the next call, and only a genuine partial-or-full success latches the guard.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.core.registry import service_adapter as service_adapter_module
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


class _FakeServiceRegistry:
    """Stand-in for `ServiceRegistry.instance()` with a scripted return count."""

    def __init__(self, register_count: int) -> None:
        self._register_count = register_count
        self.register_calls = 0

    def initialize(self) -> int:
        return 0

    def register_with_kg(self, engine: Any) -> int:
        self.register_calls += 1
        return self._register_count


def _bare_engine() -> IntelligenceGraphEngine:
    engine = IntelligenceGraphEngine.__new__(IntelligenceGraphEngine)
    engine._services_registered = False
    return engine


def test_all_services_failing_does_not_latch_the_registered_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """count == 0 (every per-service write failed) must leave the guard open."""
    fake_registry = _FakeServiceRegistry(register_count=0)
    monkeypatch.setattr(
        service_adapter_module.ServiceRegistry, "instance", lambda: fake_registry
    )
    engine = _bare_engine()

    result = engine.register_services()

    assert result == 0
    assert engine._services_registered is False, (
        "a 0-count registration pass must NOT set _services_registered=True — "
        "doing so permanently skips retry (D-DSTO-2)"
    )

    # Retry must actually be attempted next call, not short-circuited.
    fake_registry_2 = _FakeServiceRegistry(register_count=3)
    monkeypatch.setattr(
        service_adapter_module.ServiceRegistry, "instance", lambda: fake_registry_2
    )
    result2 = engine.register_services()
    assert result2 == 3
    assert fake_registry_2.register_calls == 1
    assert engine._services_registered is True


def test_successful_registration_latches_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine (count > 0) success still latches the once-only guard."""
    fake_registry = _FakeServiceRegistry(register_count=5)
    monkeypatch.setattr(
        service_adapter_module.ServiceRegistry, "instance", lambda: fake_registry
    )
    engine = _bare_engine()

    result = engine.register_services()

    assert result == 5
    assert engine._services_registered is True

    # A second call must short-circuit (no re-registration).
    result2 = engine.register_services()
    assert result2 == 0
    assert fake_registry.register_calls == 1
