"""CONCEPT:ORCH-1.94 — strict zombie protection for warm-fork sandboxes.

Covers the three layers that close the 3-day, 100%-CPU runaway-sandbox gap:
1. WarmParentRegistry age-reaps a busy (non-idle) parent past the hard age cap.
2. The orphan sweep parses Docker timestamps and removes Exited + over-age containers,
   while failing SAFE on an undated/young one.
3. The warm-container command self-expires via PID-1 ``timeout`` (constants are wired).
"""

from __future__ import annotations

import time
from datetime import UTC

from agent_utilities.rlm.sandboxes import container_fork_backend as cfb
from agent_utilities.runtime.warm_registry import (
    DEFAULT_MAX_AGE_SECS,
    WarmParentRegistry,
)


def test_registry_age_reaps_busy_non_idle_parent() -> None:
    """A parent whose idle clock is fresh (busy) but whose age exceeds the cap is still reaped."""
    reg = WarmParentRegistry(max_parents=4)
    closed: list[str] = []
    reg.register(
        "busy", object(), close=lambda: closed.append("busy"), kind="container_fork"
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
    reg.register("fresh", object(), close=lambda: None, kind="container_fork")
    reaped = reg.reap(max_idle_secs=10_000_000, max_age_secs=DEFAULT_MAX_AGE_SECS)
    assert reaped == []
    assert "fresh" in reg._entries  # noqa: SLF001


def test_parse_docker_started_at() -> None:
    # RFC3339 with 9 fractional digits + Z (Docker's native format)
    epoch = cfb._parse_docker_started_at("2026-06-25T19:00:39.490722832Z")
    assert epoch is not None and epoch > 0
    # Docker's zero value → None (fail safe: never reaped on age)
    assert cfb._parse_docker_started_at("0001-01-01T00:00:00Z") is None
    assert cfb._parse_docker_started_at("") is None
    assert cfb._parse_docker_started_at("not-a-timestamp") is None


def test_orphan_sweep_removes_exited_and_over_age_skips_young(monkeypatch) -> None:
    now = time.time()
    started = {
        "old_running": _iso(now - (cfb.WARM_CONTAINER_MAX_AGE_S + 600)),
        "young_running": _iso(now - 30),
    }
    removed: list[str] = []

    class _Res:
        def __init__(self, stdout="", rc=0):
            self.stdout = stdout
            self.returncode = rc

    def fake_run(argv, **kw):  # noqa: ANN001
        if argv[:2] == ["docker", "ps"]:
            return _Res(
                "c1\trlm-cfork-exited\texited\n"
                "c2\trlm-cfork-old\trunning\n"
                "c3\trlm-cfork-young\trunning\n"
            )
        if argv[1] == "inspect":
            cid = argv[-1]
            return _Res(
                {"c2": started["old_running"], "c3": started["young_running"]}.get(
                    cid, ""
                )
            )
        if argv[1] == "rm":
            removed.append(argv[-1])
            return _Res()
        return _Res()

    monkeypatch.setattr(cfb.subprocess, "run", fake_run)

    swept = cfb.reap_orphaned_sandboxes(runtime="docker")

    assert "rlm-cfork-exited" in swept  # exited husk always swept
    assert "c2" in removed and "c3" not in removed  # over-age reaped, young kept
    assert sorted(swept) == ["rlm-cfork-exited", "rlm-cfork-old"]


def test_hard_lifetime_constants_wired() -> None:
    assert cfb.WARM_CONTAINER_MAX_AGE_S == int(DEFAULT_MAX_AGE_SECS)
    assert cfb._SANDBOX_LABEL.startswith("agent_utilities.rlm.sandbox")


def _iso(epoch: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f000Z")


if __name__ == "__main__":  # script-mode fallback when pytest is unavailable
    import types

    mp = types.SimpleNamespace(setattr=setattr)
    test_registry_age_reaps_busy_non_idle_parent()
    test_registry_keeps_young_busy_parent()
    test_parse_docker_started_at()
    test_hard_lifetime_constants_wired()
    print("zombie-protection core tests passed (pytest skipped)")
