#!/usr/bin/python
from __future__ import annotations

"""Specialization task + machine-verifiable Verifier contract.

CONCEPT:AHE-3.28 — the (task, verifier, target, human-baseline) contract a SAI
specialization run is defined against, and the ``Verifier`` Protocol that turns a
candidate output into a machine-checkable reward.

The SAI factory (AHE-3.29) produces a *certified-superhuman specialist* for a
given important task. "Given a task" means, concretely, a :class:`SpecializationTask`:
a corpus of inputs to specialize over, a :class:`Verifier` that scores any
candidate output with a real reward, a reward target ``tau`` defining "good
enough", and an optional recorded ``human_baseline`` the certifier (SAFE-1.1)
compares against to decide *superhuman*.

The ``Verifier`` Protocol is the single seam every specialization track shares:
the GPU-kernel verifier (compile + correctness + measured speedup), a
classification-accuracy verifier, or a world-model next-state-prediction verifier
all satisfy the same contract, so the controller and the adaptation-speed metric
(AHE-3.27) are task-agnostic. A verifier returns a **real, comparable reward** —
not a pass/fail bit — because the reward *is* the training signal the weight arm
distills (OS-5.34 / AHE-3.25) and the curve the controller optimizes.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VerifierResult:
    """The outcome of verifying one candidate output against a task.

    ``reward`` is a real score (higher is better; not required to be in ``[0, 1]``
    — e.g. a kernel speedup can exceed 1.0). ``passed`` is the boolean "meets the
    correctness/validity bar" gate (a kernel that miscomputes has ``passed=False``
    and ``reward=0.0`` regardless of speed). ``detail`` carries verifier-specific
    evidence (timings, diffs, error text) for provenance and debugging.
    """

    reward: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Verifier(Protocol):
    """Scores a candidate output for a task with a machine-checkable reward.

    Implementations MUST be deterministic given the same candidate and
    environment (so the adaptation curve is reproducible) and MUST NOT raise on a
    malformed candidate — a candidate that fails to compile/parse is a
    ``VerifierResult(reward=0.0, passed=False, detail={...})``, not an exception.
    """

    def verify(self, candidate: str) -> VerifierResult:
        """Return the reward/passed/detail for one candidate output."""
        ...


@dataclass
class SpecializationTask:
    """An important task the SAI factory specializes an agent toward.

    Attributes
    ----------
    task_id:
        Stable identifier; also the signature used to route the resulting
        specialist adapter (Phase 2 ``AdapterLibrary``) and to key its
        adaptation curve.
    prompt_corpus:
        The inputs to specialize over (kernel-op specs, legal documents,
        environment-dynamics episodes, …). Candidates are produced *for* these.
    verifier:
        The machine-verifiable reward function (see :class:`Verifier`).
    target_tau:
        The reward threshold defining "specialized enough" — the target the
        adaptation-speed metric (AHE-3.27) measures time/samples to reach.
    human_baseline:
        Optional recorded human/reference reward on this task; the superhuman
        certifier (SAFE-1.1) certifies only when the specialist's reward
        distribution exceeds it with confidence. ``None`` ⇒ uncertifiable as
        superhuman (only self-improvement is measurable).
    metadata:
        Free-form provenance (domain, source, units).
    """

    task_id: str
    prompt_corpus: list[str]
    verifier: Verifier
    target_tau: float
    human_baseline: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("SpecializationTask.task_id must be non-empty")
        if not isinstance(self.verifier, Verifier):
            raise TypeError(
                "SpecializationTask.verifier must satisfy the Verifier protocol "
                "(a .verify(candidate) -> VerifierResult method)"
            )

    def score(self, candidate: str) -> VerifierResult:
        """Verify one candidate against this task's verifier."""
        return self.verifier.verify(candidate)
