"""CONCEPT:ORCH-1.91 — reward-EMA-aware sandbox routing: route around a failing rung, bounded."""

from __future__ import annotations

from agent_utilities.rlm.sandboxes.base import (
    Sandbox,
    SandboxCapabilities,
    SandboxEnv,
    SandboxResult,
)
from agent_utilities.rlm.sandboxes.reward import SandboxRewardTracker
from agent_utilities.rlm.sandboxes.router import SandboxRouter


def _backend(name: str, rank: int) -> Sandbox:
    # Define the abstract methods in the class body so ABC's __abstractmethods__ is satisfied
    # (post-hoc attribute assignment does not register the overrides).
    class _S(Sandbox):
        capabilities = SandboxCapabilities(
            host_callbacks=True,
            third_party_libs=True,
            classes=True,
            full_stdlib=True,
            network=True,
            isolated=True,
            preference_rank=rank,
        )

        def is_available(self) -> bool:
            return True

        async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:  # noqa: ARG002
            return SandboxResult(updated_vars={}, stdout="")

    _S.name = name
    return _S()


def _chain(router: SandboxRouter) -> list[str]:
    return [b.name for b in router.select("x = 1")]


def test_no_reward_fn_is_pure_rank():
    backends = [
        _backend("local", 30),
        _backend("docker", 20),
        _backend("forkserver", 15),
    ]
    assert _chain(SandboxRouter(backends)) == ["forkserver", "docker", "local"]


def test_neutral_reward_preserves_rank_order():
    backends = [
        _backend("local", 30),
        _backend("docker", 20),
        _backend("forkserver", 15),
    ]
    router = SandboxRouter(backends, reward_fn=lambda _n: 0.5)
    assert _chain(router) == ["forkserver", "docker", "local"]


def test_broken_rung_is_routed_around():
    backends = [
        _backend("local", 30),
        _backend("docker", 20),
        _backend("forkserver", 15),
    ]
    rewards = {"forkserver": 0.0, "docker": 1.0, "local": 0.5}
    router = SandboxRouter(backends, reward_fn=lambda n: rewards.get(n, 0.5))
    # forkserver (rank 15) fully broken, docker (rank 20) healthy -> docker is tried first.
    assert _chain(router)[0] == "docker"


def test_reward_shift_is_bounded_to_about_one_tier():
    """A healthy rung must NOT leapfrog a rung two tiers cheaper — the shift is bounded."""
    backends = [_backend("a", 10), _backend("b", 20)]
    # b fully healthy, a fully broken: a:10+5=15, b:20-5=15 -> tie broken by raw rank (a first).
    router = SandboxRouter(backends, reward_fn=lambda n: {"a": 0.0, "b": 1.0}[n])
    # The bounded shift means even extreme rewards only *tie* a one-tier gap, never invert two.
    assert _chain(router) == ["a", "b"]


def test_tracker_ema_penalises_then_recovers():
    t = SandboxRewardTracker()  # fresh instance (not the singleton) for isolation
    assert t.reward("docker") == 0.5
    for _ in range(10):
        t.record("docker", success=False)
    assert t.reward("docker") < 0.05
    for _ in range(20):
        t.record("docker", success=True)
    assert t.reward("docker") > 0.95
