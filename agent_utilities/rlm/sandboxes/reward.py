"""CONCEPT:ORCH-1.91 — adaptive sandbox-tier selection via a reward-EMA over rungs.

The deterministic router (``router.py``) picks the cheapest *capable* backend by
``preference_rank``. That is right until a rung starts *failing* on this host (a wedged docker
daemon, a forkserver that can't boot) — pure rank order keeps sending work to it, and because a
``SandboxFatalError`` fast-fails the whole run (no escalation), a broken preferred rung is
catastrophic. This tracker records each rung's success/failure as an exponential-moving-average
reward (the same shape as ``CapabilityIndex.record_outcome`` — the sanctioned reward-EMA
mechanism, applied to the sandbox-routing domain), so the router can prefer the *healthiest*
capable rung first. It is a tiny, dependency-free process singleton — deliberately NOT the heavy
KG ``CapabilityIndex`` (which would couple the RLM hot path to the retrieval/HNSW layer); the EMA
math is identical so the two can be unified later if a single reward store is wanted.

The influence on routing is bounded (see ``router._REWARD_WEIGHT``): a fully-broken rung can drop
by at most ~one tier, a healthy one rise by ~one — so steady-state routing is unchanged and only
a persistently failing rung gets routed around.
"""

from __future__ import annotations

import threading

_NEUTRAL = 0.5
_ALPHA = 0.3  # EMA weight; matches CapabilityIndex.record_outcome's default


class SandboxRewardTracker:
    """Process-singleton EMA reward per sandbox rung (keyed by backend ``name``)."""

    _instance: SandboxRewardTracker | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._reward: dict[str, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> SandboxRewardTracker:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record(self, name: str, *, success: bool, alpha: float = _ALPHA) -> float:
        """Fold one outcome into the rung's EMA reward in ``[0, 1]`` (0.5 neutral)."""
        reward = 1.0 if success else 0.0
        with self._lock:
            prev = self._reward.get(name, _NEUTRAL)
            updated = (1.0 - alpha) * prev + alpha * reward
            self._reward[name] = updated
            return updated

    def reward(self, name: str) -> float:
        """Current EMA reward for ``name`` (0.5 if no outcomes recorded yet)."""
        return self._reward.get(name, _NEUTRAL)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._reward)
