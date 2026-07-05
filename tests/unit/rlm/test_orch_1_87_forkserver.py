"""CONCEPT:AU-ORCH.sandbox.native-warm-fork-os — forkserver warm-fork rung: live-path warm-reuse + isolation + bridge.

These assert the *invocation*, not just the API (Wire-First): a real forkserver is booted, real
children are forked, and the warm-parent registry (OS-5.58) is exercised end-to-end.
"""

from __future__ import annotations

import os

import pytest

from agent_utilities.rlm.sandboxes.base import SandboxEnv
from agent_utilities.rlm.sandboxes.forkserver_backend import ForkServerSandbox
from agent_utilities.rlm.sandboxes.registry import default_sandboxes
from agent_utilities.rlm.sandboxes.router import SandboxRouter
from agent_utilities.rlm.telemetry import SandboxFatalError
from agent_utilities.runtime.warm_registry import WarmParentRegistry

pytestmark = pytest.mark.skipif(
    "forkserver" not in __import__("multiprocessing").get_all_start_methods(),
    reason="forkserver start method unavailable on this platform",
)


@pytest.fixture
def clean_registry():
    """Isolate the process-singleton registry per test and drain forkservers afterwards."""
    WarmParentRegistry._instance = None  # noqa: SLF001 - test isolation
    yield
    WarmParentRegistry.drain_active()
    WarmParentRegistry._instance = None  # noqa: SLF001


@pytest.fixture
def sandbox():
    # Lean preload (bridge only) so the forkserver boots fast — no numpy import in CI.
    return ForkServerSandbox(preload=())


def test_start_join_serializes_start_under_lock():
    """Regression: ``Process.start()`` (the non-thread-safe forkserver control-channel
    handshake) MUST run while ``_FORK_START_LOCK`` is held, so concurrent fan-out
    starts from executor threads can't race the singleton (which surfaced as an
    intermittent "'NoneType' object cannot be interpreted as an integer" on one
    branch). The lock is released once start() returns; join() stays outside it so
    snippet execution remains concurrent. CONCEPT:AU-ORCH.sandbox.native-warm-fork-os
    """
    from agent_utilities.rlm.sandboxes import forkserver_backend as fb

    observed = {}

    class _Proc:
        def start(self):
            observed["locked_during_start"] = fb._FORK_START_LOCK.locked()

        def join(self, timeout=None):
            observed["locked_during_join"] = fb._FORK_START_LOCK.locked()

        def is_alive(self):
            return False

    killed = fb._start_join(_Proc(), 1.0)

    assert killed is False
    assert observed["locked_during_start"] is True  # start() ran under the lock
    assert observed["locked_during_join"] is False  # join() runs concurrently
    assert fb._FORK_START_LOCK.locked() is False  # released afterwards


def _env(vars_=None, helpers=None):
    return SandboxEnv(vars=vars_ or {}, helpers=helpers or {})


async def test_available_and_caps(sandbox):
    assert sandbox.is_available() is True
    caps = sandbox.capabilities
    assert caps.warm_fork is True
    assert caps.isolated is True and caps.host_callbacks is True
    # Cheaper isolated tier than docker (20), pricier than wasm (10).
    assert 10 < caps.preference_rank < 20


async def test_runs_snippet_and_captures_stdout(clean_registry, sandbox):
    r = await sandbox.execute("z = 6 * 7\nprint('z', z)", _env({"z0": 1}))
    assert r.error is None
    assert "z 42" in r.stdout


async def test_host_callback_round_trips_over_bridge(clean_registry, sandbox):
    captured: dict = {}

    def FINAL_VAR(
        name, value
    ):  # the sync host helper, served host-side over the UDS bridge
        captured[name] = value

    code = "out = sum(range(10))\nFINAL_VAR('out', out)"
    r = await sandbox.execute(code, _env({}, {"FINAL_VAR": FINAL_VAR}))
    assert r.error is None
    assert captured == {"out": 45}


async def test_async_host_helper_awaited_host_side(clean_registry, sandbox):
    seen: dict = {}

    async def rlm_query(
        q,
    ):  # async helper -> async shim in the child, awaited host-side
        seen["q"] = q
        return {"answer": q.upper()}

    code = "r = await rlm_query('hi')\nprint(r['answer'])"
    res = await sandbox.execute(code, _env({}, {"rlm_query": rlm_query}))
    assert res.error is None
    assert seen == {"q": "hi"} and "HI" in res.stdout


async def test_warm_parent_reused_across_executes(clean_registry, sandbox, monkeypatch):
    """The win: only the FIRST execute warms; the rest fork the same registry-pooled parent."""
    real_warm = sandbox.warm
    calls = {"n": 0}

    async def counting_warm(spec):
        calls["n"] += 1
        return await real_warm(spec)

    monkeypatch.setattr(sandbox, "warm", counting_warm)

    await sandbox.execute("a = 1", _env())
    await sandbox.execute("b = 2", _env())
    await sandbox.execute("c = 3", _env())

    assert calls["n"] == 1, (
        "warm() should be paid once; subsequent runs reuse the parent"
    )
    stats = WarmParentRegistry.get().stats()
    assert stats["warm_parents"] == 1
    assert stats["by_kind"].get("forkserver") == 1


async def test_child_is_process_isolated(clean_registry, sandbox):
    """A forked child runs in its own process — a different PID than the orchestrator."""
    captured: dict = {}

    def FINAL_VAR(name, value):
        captured[name] = value

    await sandbox.execute(
        "import os as _os\nFINAL_VAR('child_pid', _os.getpid())",
        _env({}, {"FINAL_VAR": FINAL_VAR}),
    )
    assert captured["child_pid"] != os.getpid()


async def test_in_snippet_error_is_surfaced_not_raised(clean_registry, sandbox):
    """A runtime error inside the snippet comes back as ``error`` (the model retries)."""
    r = await sandbox.execute("1 / 0", _env())
    assert r.error is not None and "division" in r.error.lower()


async def test_router_prefers_forkserver_over_docker(clean_registry):
    """For third-party + host-callback code, forkserver (15) outranks docker (20) in the chain."""
    backends = default_sandboxes()
    names = {b.name for b in backends}
    if "forkserver" not in names:  # pragma: no cover
        pytest.skip("forkserver not registered on this host")
    router = SandboxRouter(backends)
    chain = router.select("import numpy\nx = rlm_query('q')")
    chain_names = [b.name for b in chain]
    if "docker" in chain_names and "forkserver" in chain_names:
        assert chain_names.index("forkserver") < chain_names.index("docker")
    else:
        assert "forkserver" in chain_names


async def test_fatal_when_child_cannot_write_result(
    clean_registry, sandbox, monkeypatch
):
    """If the child dies before writing a result, it's an irreversible failure (fast-fail)."""
    import agent_utilities.rlm.sandboxes._bridge as br

    monkeypatch.setattr(br, "read_result", lambda _d: ("", None, False))
    with pytest.raises(SandboxFatalError):
        await sandbox.execute("x = 1", _env())
