"""CONCEPT:ORCH-1.38 — Deterministic sandbox router.

Given a snippet, pick the cheapest backend that can run it and return an *escalation chain*:
the executor tries each in order, advancing on :class:`SandboxRejected` (and stopping on
:class:`~agent_utilities.rlm.telemetry.SandboxFatalError`). The floor backend (``local``,
which can run anything) always anchors the tail, so the chain is never empty and the RLM loop
always has somewhere to run.

Routing is pure capability matching — see :class:`~.base.SandboxCapabilities`. Three hard
filters come from static analysis: third-party imports require ``third_party_libs``, class /
dataclass defs require ``classes``, and any RLM-helper call requires ``host_callbacks``. The
stdlib *subset* of monty is intentionally NOT a hard filter: we can't cheaply enumerate which
stdlib modules monty supports, so monty is allowed to try and the chain escalates if it
rejects an unsupported import — escalation is the safety net, not a static guess.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .analyzer import Analyzer, AstAnalyzer, CodeRequirements
from .base import Sandbox

logger = logging.getLogger(__name__)

# CONCEPT:ORCH-1.91 — how far a rung's reward-EMA may shift its effective rank. Rungs are spaced
# ~5 apart (monty 0, wasm 10, forkserver 15, container_fork 18, docker 20, local 30); a weight of
# 10 maps reward∈[0,1] (centered 0.5) to a ±5 shift — so a *fully broken* rung drops by ~one tier
# and a *fully healthy* one rises by ~one, while steady-state (~0.5) preserves the rank order.
_REWARD_WEIGHT = 10.0


class SandboxRouter:
    """Selects an ordered escalation chain of backends for a snippet."""

    def __init__(
        self,
        backends: list[Sandbox],
        analyzer: Analyzer | None = None,
        *,
        reward_fn: Callable[[str], float] | None = None,
    ):
        if not backends:
            raise ValueError("SandboxRouter needs at least one backend")
        self._backends = backends
        self._analyzer = analyzer or AstAnalyzer()
        # Optional reward-EMA per backend name (CONCEPT:ORCH-1.91): when supplied, a persistently
        # failing rung is routed around. Default None => pure deterministic rank order (unchanged).
        self._reward_fn = reward_fn
        # The floor: lowest-preference (highest rank) backend that runs anything.
        self._floor = max(backends, key=lambda b: b.capabilities.preference_rank)

    def _score(self, backend: Sandbox) -> float:
        """Effective ordering score: rank, nudged by the bounded reward shift (lower = first)."""
        rank = float(backend.capabilities.preference_rank)
        if self._reward_fn is None:
            return rank
        reward = self._reward_fn(backend.name)
        return rank - _REWARD_WEIGHT * (reward - 0.5)

    def select(self, code: str, *, force: str | None = None) -> list[Sandbox]:
        """Return the escalation chain for ``code``.

        ``force`` pins a named backend (the config override). A forced-but-unavailable backend
        degrades to the auto chain rather than dying — isolation preference should never make
        the RLM loop unrunnable.
        """
        if force:
            forced = self._by_name(force)
            if forced is not None and forced.is_available():
                return [forced]
            logger.warning(
                "Forced sandbox %r unavailable/unknown; falling back to auto routing.",
                force,
            )

        req = self._analyzer.analyze(code)
        if not req.syntax_ok:
            # Parses nowhere — surface the SyntaxError cheaply via the floor backend.
            return [self._floor]

        chain = [
            b for b in self._backends if b.is_available() and self._satisfies(b, req)
        ]
        # Primary order is the reward-nudged score; ties break on raw rank for determinism.
        chain.sort(key=lambda b: (self._score(b), b.capabilities.preference_rank))

        # Guarantee the floor anchors the tail even if a capability check excluded it
        # (it shouldn't, since it satisfies everything — but never return an empty chain).
        if self._floor.is_available() and self._floor not in chain:
            chain.append(self._floor)

        logger.debug(
            "Sandbox route: %s -> %s",
            self._req_summary(req),
            [b.name for b in chain],
        )
        return chain

    @staticmethod
    def _satisfies(backend: Sandbox, req: CodeRequirements) -> bool:
        caps = backend.capabilities
        if req.needs_third_party and not caps.third_party_libs:
            return False
        if req.defines_classes and not caps.classes:
            return False
        if req.needs_host_callbacks and not caps.host_callbacks:
            return False
        return True

    def _by_name(self, name: str) -> Sandbox | None:
        return next((b for b in self._backends if b.name == name), None)

    @staticmethod
    def _req_summary(req: CodeRequirements) -> str:
        bits = []
        if req.third_party_imports:
            bits.append(f"3p={sorted(req.third_party_imports)}")
        if req.defines_classes:
            bits.append("classes")
        if req.uses_async:
            bits.append("async")
        if req.helper_calls:
            bits.append(f"helpers={sorted(req.helper_calls)}")
        return ", ".join(bits) or "plain"
