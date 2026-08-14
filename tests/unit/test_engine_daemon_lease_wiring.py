"""Regression tests for the `epistemic-graph-daemon` LEASE wired into the
session-engine fixture (docs/architecture/lane-concurrency.md,
CONCEPT:AU-OS.governance.shared-scope-lease).

History: a lane's 37-minute full-suite run logged 1,234 occurrences of
``ConnectionRefusedError: Cannot connect to epistemic-graph service`` while two
sibling lanes' full-suite runs and an ``--all-files`` pre-commit drove the SAME
externally-provided epistemic-graph daemon concurrently — ``lane_resources.yaml``
classified everything else a lane can collide on but not this daemon. These
tests pin the fix: a second lane's full-suite run against the same shared
daemon must defer (mirroring the CLI's exit code 75) rather than pile on, and
the fixture lanes actually run through must be the one enforcing it — not just
a helper that could be called.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from agent_utilities.governance import lanes


@pytest.fixture(autouse=True)
def _isolate_workspace_arbitration_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route the ``epistemic-graph-daemon`` lease into an isolated ``tmp_path``
    instead of the REAL host-wide arbitration directory (see the identical
    fixture + rationale in ``tests/unit/test_lanes.py``). These tests call
    ``lanes.hold_lease("epistemic-graph-daemon", ...)`` with no ``path=`` at
    all, so without this they take the SAME real, host-wide lease that a
    genuinely concurrent full-suite run elsewhere on this host may be holding
    right now (spurious ``LeaseUnavailable``) -- or briefly hold it themselves
    and defer that real run.
    """
    workspace_dir = tmp_path / "workspace-arbitration"

    def _fake_workspace_arbitration_dir() -> Path:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    monkeypatch.setattr(
        lanes, "workspace_arbitration_dir", _fake_workspace_arbitration_dir
    )


def _conftest_module():
    """Find the loaded tests/conftest.py module regardless of its import name.

    Matched by ``__file__`` (not ``hasattr``) — same pattern as
    ``tests/unit/test_repo_session_guard.py``: the conftest installs MagicMock
    stand-ins for absent heavy deps, and mocks claim every attribute exists.
    """
    conftest_path = str(Path(__file__).resolve().parents[1] / "conftest.py")
    for mod in list(sys.modules.values()):
        if getattr(mod, "__file__", None) == conftest_path:
            return mod
    pytest.skip("session-engine conftest not loaded")


def test_engine_daemon_lease_is_classified_and_workspace_scoped() -> None:
    assert (
        lanes.resource_class("epistemic-graph-daemon") is lanes.ArbitrationClass.LEASE
    )
    assert lanes.resource_scope("epistemic-graph-daemon") == "workspace"


def test_engine_daemon_lease_ttl_has_headroom_over_the_worst_observed_run() -> None:
    """The default hold_lease TTL (30 min) would let a still-running 37-minute
    full-suite lane's lease be reclaimed as dead out from under it."""
    conftest = _conftest_module()
    assert conftest._ENGINE_DAEMON_LEASE_TTL_SECONDS > 37 * 60


def test_acquire_engine_daemon_lease_acquires_and_releases_cleanly() -> None:
    conftest = _conftest_module()

    lease_cm = conftest._acquire_engine_daemon_lease(operation="lane-a full-suite")
    try:
        holder = lanes.lease_status("epistemic-graph-daemon")
        assert holder is not None
        assert holder["operation"] == "lane-a full-suite"
    finally:
        lease_cm.__exit__(None, None, None)
    assert lanes.lease_status("epistemic-graph-daemon") is None


def test_acquire_engine_daemon_lease_defers_when_another_lane_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact incident: a second lane's full-suite run must defer, not pile
    onto the same shared daemon — surfaced as pytest's own exit-75 convention,
    the pytest equivalent of the CLI's ``lane lease`` exit code."""
    conftest = _conftest_module()

    exits: list[tuple[str, int]] = []

    def _fake_exit(msg: str = "", returncode: int = 0) -> None:
        exits.append((msg, returncode))
        raise SystemExit(returncode)

    monkeypatch.setattr(conftest.pytest, "exit", _fake_exit)

    with lanes.hold_lease("epistemic-graph-daemon", operation="lane-a full-suite"):
        with pytest.raises(SystemExit):
            conftest._acquire_engine_daemon_lease(operation="lane-b full-suite")

    assert exits and exits[0][1] == 75
    assert "epistemic-graph-daemon" in exits[0][0]
    # The refusal must not leak a lease of its own, or one deferred lane wedges
    # the daemon for everyone that follows.
    assert lanes.lease_status("epistemic-graph-daemon") is None


def test_session_engine_fixture_actually_calls_the_guard() -> None:
    """Enforcement, not just a helper that could be called: the SAME autouse
    fixture every full-suite run goes through must take the lease before
    reusing an externally-provided engine verbatim."""
    conftest = _conftest_module()
    source = inspect.getsource(conftest._session_engine.__wrapped__)
    assert "_acquire_engine_daemon_lease" in source
    # And it must release on every exit path (normal AND exception), not just
    # the happy path — a lease that leaks on error wedges the next lane forever.
    assert "lease_cm.__exit__" in source
