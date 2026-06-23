"""CONCEPT:ORCH-1.90 — firecracker (forkd microVM) rung: gating + REST request/marshalling logic.

Live microVM forking needs an x86_64+KVM host running forkd (the parked spike), so it cannot run
here. These tests pin the two things that MUST be right without forkd: (1) the rung gates OFF
cleanly when no controller is reachable (so it never wedges the router), and (2) the forkd REST
request-building + result-marshalling for warm/run_forked/branch, verified against a recorded mock
controller.
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.sandboxes.base import ParentHandle, SandboxEnv
from agent_utilities.rlm.sandboxes.firecracker_backend import FirecrackerSandbox
from agent_utilities.rlm.sandboxes.registry import default_sandboxes
from agent_utilities.rlm.telemetry import SandboxFatalError


class _MockForkd:
    """Records requests and returns canned forkd responses (no network)."""

    def __init__(self, snapshots=("pyagent",)):
        self.calls: list[tuple[str, str, dict | None]] = []
        self._snapshots = list(snapshots)

    def healthy(self) -> bool:
        return True

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        self.calls.append((method, path, body))
        if path == "/v1/snapshots":
            return {"snapshots": [{"tag": t} for t in self._snapshots]}
        if path == "/v1/sandboxes" and method == "POST":
            return [{"id": "sb-test-0001", "pid": 4242}]
        if path.endswith("/eval"):
            return {"stdout": "42\n", "error": None}
        if path.endswith("/branch"):
            return {"tag": body["tag"], "status": "ready"}
        if method == "DELETE":
            return {"ok": True}
        return {}


def test_caps_top_isolation_rung():
    c = FirecrackerSandbox().capabilities
    assert c.isolated is True and c.warm_fork is True
    assert c.host_callbacks is False  # v1: no host bridge into the microVM guest
    assert c.preference_rank == 25  # heaviest/last


def test_gates_off_when_controller_unreachable():
    sb = FirecrackerSandbox(base_url="http://127.0.0.1:9", token="")
    assert sb.is_available() is False
    # ...and therefore is not registered in the default chain on a forkd-less host.
    assert "firecracker" not in {b.name for b in default_sandboxes()}


async def test_run_forked_drives_forkd_rest(monkeypatch):
    sb = FirecrackerSandbox(snapshot_tag="pyagent")
    mock = _MockForkd()
    sb._client = mock  # type: ignore[assignment]

    parent = await sb.warm(sb.warm_spec())
    assert parent.ref == {"snapshot": "pyagent"}

    res = await sb.run_forked(parent, "print(6*7)", SandboxEnv(vars={}))
    assert res.error is None and "42" in res.stdout

    methods = [(m, p.split("/")[-1]) for m, p, _ in mock.calls]
    # spawn one child -> eval -> delete (teardown), in order.
    assert ("POST", "sandboxes") in methods
    assert ("POST", "eval") in methods
    assert any(m == "DELETE" for m, _ in methods), "child must be torn down"


async def test_warm_rejects_missing_snapshot(monkeypatch):
    sb = FirecrackerSandbox(snapshot_tag="absent-tag")
    sb._client = _MockForkd(snapshots=("pyagent",))  # type: ignore[assignment]
    with pytest.raises(SandboxFatalError):
        await sb.warm(sb.warm_spec())


async def test_branch_is_microvm_only_verb():
    sb = FirecrackerSandbox()
    mock = _MockForkd()
    sb._client = mock  # type: ignore[assignment]
    parent = await sb.branch("sb-test-0001", tag="checkpoint-1", mode="diff")
    assert isinstance(parent, ParentHandle)
    assert parent.ref == {"snapshot": "checkpoint-1"}
    assert any(p.endswith("/branch") for _, p, _ in mock.calls)
