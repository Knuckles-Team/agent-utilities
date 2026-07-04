"""CONCEPT:AU-ORCH.sandbox.container-fork-sandbox — container_fork warm-pool rung: live-path warm-reuse + bridge.

Gated on a usable docker/podman daemon; skipped otherwise. Proves the same ForkableSandbox
protocol (warm + run_forked + registry-backed execute) generalises from the process substrate
(forkserver) to the container substrate.
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.sandboxes.base import SandboxEnv
from agent_utilities.rlm.sandboxes.container_fork_backend import ContainerForkSandbox
from agent_utilities.runtime.warm_registry import WarmParentRegistry

_SB = ContainerForkSandbox()
pytestmark = pytest.mark.skipif(
    not _SB.is_available(), reason="no docker/podman daemon available"
)


@pytest.fixture
def clean_registry():
    WarmParentRegistry._instance = None  # noqa: SLF001 - test isolation
    yield
    WarmParentRegistry.drain_active()
    WarmParentRegistry._instance = None  # noqa: SLF001


def _env(vars_=None, helpers=None):
    return SandboxEnv(vars=vars_ or {}, helpers=helpers or {})


def test_caps():
    caps = ContainerForkSandbox().capabilities
    assert caps.warm_fork is True and caps.isolated is True
    # warmer than cold docker (20), heavier than forkserver (15)
    assert 15 < caps.preference_rank < 20


async def test_warm_pool_reuse_and_bridge(clean_registry):
    sb = ContainerForkSandbox()
    captured: dict = {}

    def FINAL_VAR(name, value):
        captured[name] = value

    real_warm = sb.warm
    calls = {"n": 0}

    async def counting_warm(spec):
        calls["n"] += 1
        return await real_warm(spec)

    sb.warm = counting_warm  # type: ignore[method-assign]

    code = "r = sum(range(n))\nprint('sum', r)\nFINAL_VAR('r', r)"
    r1 = await sb.execute(code, _env({"n": 5}, {"FINAL_VAR": FINAL_VAR}))
    r2 = await sb.execute(code, _env({"n": 5}, {"FINAL_VAR": FINAL_VAR}))

    assert r1.error is None and "sum 10" in r1.stdout
    assert r2.error is None and "sum 10" in r2.stdout
    assert captured == {"r": 10}
    assert calls["n"] == 1, "warm container paid once; second run reuses it"
    stats = WarmParentRegistry.get().stats()
    assert stats["by_kind"].get("container_fork") == 1
