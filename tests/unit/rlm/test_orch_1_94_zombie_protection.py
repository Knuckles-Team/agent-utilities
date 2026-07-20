"""CONCEPT:AU-ORCH.sandbox.warm-parent-lifetime-cap — hard age protection for warm parents."""

from __future__ import annotations

import time

from agent_utilities.runtime.warm_registry import (
    DEFAULT_MAX_AGE_SECS,
    WarmParentRegistry,
)


def test_registry_age_reaps_busy_non_idle_parent() -> None:
    """A parent whose idle clock is fresh (busy) but whose age exceeds the cap is still reaped."""
    reg = WarmParentRegistry(max_parents=4)
    closed: list[str] = []
    reg.register(
        "busy", object(), close=lambda: closed.append("busy"), kind="firecracker"
    )
    entry = reg._entries["busy"]  # noqa: SLF001 — white-box: simulate a long-lived, just-used parent
    now = time.time()
    entry.last_used = now  # NOT idle
    entry.created = now - (DEFAULT_MAX_AGE_SECS + 60)  # but past the hard age cap

    reaped = reg.reap(max_idle_secs=10_000_000, max_age_secs=DEFAULT_MAX_AGE_SECS)

    assert reaped == ["busy"]
    assert closed == ["busy"]  # close() was actually invoked (container torn down)
    assert "busy" not in reg._entries  # noqa: SLF001


def test_registry_keeps_young_busy_parent() -> None:
    reg = WarmParentRegistry(max_parents=4)
    reg.register("fresh", object(), close=lambda: None, kind="firecracker")
    reaped = reg.reap(max_idle_secs=10_000_000, max_age_secs=DEFAULT_MAX_AGE_SECS)
    assert reaped == []
    assert "fresh" in reg._entries  # noqa: SLF001


if __name__ == "__main__":  # script-mode fallback when pytest is unavailable
    test_registry_age_reaps_busy_non_idle_parent()
    test_registry_keeps_young_busy_parent()
    print("zombie-protection core tests passed (pytest skipped)")
