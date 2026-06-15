#!/usr/bin/python
from __future__ import annotations

"""Substrate trainer — the SLOW-loop's corpus builder + training-job emitter.

CONCEPT:ORCH-1.57 — Substrate training-job emission (Fast-Slow SLOW loop)

The real, correctly-factored slow-loop trainer for the Fast-Slow Training
controller (Fast-Slow Training, arXiv:2605.12484). It is the live consumer that
the controller calls in its slow loop where a no-op default would otherwise sit.

Factoring (``docs/architecture/in_house_training_substrate.md``): agent-utilities
owns the *reward spine*, *corpus building*, and *training-job dispatch*; the
actual gradient trainers (torch / PEFT, GRPO / DPO / SFT) live in data-science-mcp
(DSM) and run on GPU. So this class does NOT run gradient descent — that belongs
to the substrate and is GPU-gated. Instead it:

* turns a recurring trace group into a GRPO corpus of
  group-normalized-advantage samples (the reward-spine responsibility), reusing
  the canonical :func:`batch_normalized_advantage`; and
* assembles a :class:`TrainingJobSpec` and emits it to the gradient substrate via
  an injected ``dispatch_fn``.

Dispatch is dependency-injected so the whole flow is testable with no DSM and no
GPU. The default dispatch is *record-only*: it queues the job locally
(``status="recorded"``) so a job is never lost when the substrate is absent — the
correct degradation while DSM / GPU is unavailable (e.g. a hardware fault). A
real injected ``dispatch_fn`` returning ``True`` marks the job ``"dispatched"``;
a dispatch that raises (substrate unreachable) is caught and the job is recorded
as ``"skipped_no_substrate"`` rather than crashing the slow loop.

Everything is deterministic and stdlib-only. The job id is derived from the task
key and corpus size, so re-emitting the same recurring group is idempotent.

Concept: substrate-training
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.graph.training_signals import batch_normalized_advantage

__all__ = [
    "GrpoSample",
    "TrainingJobSpec",
    "DispatchFn",
    "SubstrateTrainer",
]


@dataclass
class GrpoSample:
    """One training sample with its group-normalized advantage.

    Args:
        task_key: The recurring task kind this sample belongs to.
        prompt: The prompt/scaffold text the trace ran (empty when the source
            trace carried none).
        reward: The scalar outcome reward of the originating trace.
        advantage: The GRPO group-normalized advantage of this sample within its
            group, from :func:`batch_normalized_advantage` over the group rewards.
    """

    task_key: str
    prompt: str
    reward: float
    advantage: float


@dataclass
class TrainingJobSpec:
    """A self-contained training-job description emitted to the gradient substrate.

    This is what agent-utilities owns and hands off: it carries the built GRPO
    corpus and the metadata the substrate (DSM / GPU) needs to run the gradient
    step. It never contains a trained model — only the job to produce one.

    Args:
        job_id: Deterministic, idempotent id from ``task_key`` + corpus size.
        task_key: The recurring task kind being trained.
        method: Training method for the substrate to run; ``"grpo"``
            (group-normalized) by default.
        n_samples: Number of samples in ``corpus``.
        mean_advantage: Mean group-normalized advantage over the corpus — the
            GRPO signal strength the substrate would optimize against.
        corpus: The GRPO samples to train on.
        status: Lifecycle of this spec — ``"recorded"`` (queued locally, no
            substrate yet), ``"dispatched"`` (accepted by the substrate), or
            ``"skipped_no_substrate"`` (dispatch failed / substrate absent).
    """

    job_id: str
    task_key: str
    method: str
    n_samples: int
    mean_advantage: float
    corpus: list[GrpoSample] = field(default_factory=list)
    status: str = "recorded"


# Submit a job spec to the gradient substrate (DSM). Returns True when accepted.
# The default is record-only (see ``SubstrateTrainer``): no substrate is called,
# the job is simply queued locally so it is never lost.
DispatchFn = Callable[["TrainingJobSpec"], bool]


class SubstrateTrainer:
    """Real Fast-Slow SLOW-loop trainer: build a GRPO corpus + dispatch a job (CONCEPT:ORCH-1.57).

    Builds a group-normalized-advantage corpus from a recurring trace group (the
    agent-utilities reward-spine responsibility) and emits a
    :class:`TrainingJobSpec` to the gradient substrate (data-science-mcp / GPU).
    The gradient step itself lives in DSM and is GPU-gated — when no
    substrate/dispatch is available the job is RECORDED (queued) rather than lost.

    Args:
        dispatch_fn: Injected job submitter. When ``None`` the trainer uses a
            record-only default that performs no submission (so the job is queued
            locally with ``status="recorded"``). A real ``dispatch_fn`` returning
            ``True`` marks the job ``"dispatched"``; one that raises is caught and
            the job is marked ``"skipped_no_substrate"``.
        min_group: Minimum number of traces required to emit a job. A group below
            this floor is too small to be worth a weight update; its spec is
            returned with ``status="skipped_no_substrate"`` and an empty corpus,
            and ``dispatch_fn`` is not called.
    """

    def __init__(
        self, *, dispatch_fn: DispatchFn | None = None, min_group: int = 2
    ) -> None:
        if min_group < 1:
            raise ValueError("min_group must be >= 1")
        self._dispatch_fn: DispatchFn = dispatch_fn or self._record_only
        self._min_group = min_group
        self._jobs: list[TrainingJobSpec] = []

    @staticmethod
    def _record_only(spec: TrainingJobSpec) -> bool:
        """Default dispatch: no substrate call — the job stays locally queued.

        Returns ``False`` so :meth:`train` leaves the spec at ``status="recorded"``:
        the job is preserved (audit trail / local queue) for a later substrate run
        rather than being submitted or lost. This is the graceful degradation when
        DSM / GPU is unavailable.
        """
        return False

    def build_corpus(self, task_key: str, traces: list[Any]) -> list[GrpoSample]:
        """Turn a recurring trace group into a GRPO group-normalized-advantage corpus.

        Each trace must expose ``.reward``; ``.prompt`` is optional (defaults to
        ``""``). Advantages are computed for the whole group at once via the
        canonical :func:`batch_normalized_advantage`, so each sample's advantage is
        relative to its group (GRPO). Sample order matches trace order, making the
        corpus deterministic for a given trace stream.
        """
        rewards = [float(t.reward) for t in traces]
        advantages = batch_normalized_advantage(rewards)
        return [
            GrpoSample(
                task_key=task_key,
                prompt=str(getattr(t, "prompt", "") or ""),
                reward=reward,
                advantage=advantage,
            )
            for t, reward, advantage in zip(
                traces, rewards, advantages, strict=True
            )
        ]

    def _job_id(self, task_key: str, n: int) -> str:
        """Deterministic, idempotent job id from the task key and corpus size."""
        return f"job-{task_key}-n{n}"

    def train(self, task_key: str, traces: list[Any]) -> TrainingJobSpec:
        """Build the corpus, assemble a job spec, and dispatch it to the substrate.

        Below the ``min_group`` floor, returns a ``skipped_no_substrate`` spec with
        an empty corpus without calling ``dispatch_fn`` (the group is too small to
        be worth a weight update). Otherwise it builds the GRPO corpus, assembles a
        :class:`TrainingJobSpec`, and calls the injected ``dispatch_fn``:

        * default record-only dispatch → ``status="recorded"`` (queued locally);
        * dispatch returns ``True`` → ``status="dispatched"``;
        * dispatch raises (substrate unreachable) → ``status="skipped_no_substrate"``.

        Every produced spec is appended to the local queue (:meth:`jobs`) for audit
        regardless of outcome. The returned spec is the same object that was queued.
        The job id is idempotent in ``task_key`` + corpus size.
        """
        if len(traces) < self._min_group:
            spec = TrainingJobSpec(
                job_id=self._job_id(task_key, len(traces)),
                task_key=task_key,
                method="grpo",
                n_samples=len(traces),
                mean_advantage=0.0,
                corpus=[],
                status="skipped_no_substrate",
            )
            self._jobs.append(spec)
            return spec

        corpus = self.build_corpus(task_key, traces)
        mean_advantage = (
            round(sum(s.advantage for s in corpus) / len(corpus), 6) if corpus else 0.0
        )
        spec = TrainingJobSpec(
            job_id=self._job_id(task_key, len(corpus)),
            task_key=task_key,
            method="grpo",
            n_samples=len(corpus),
            mean_advantage=mean_advantage,
            corpus=corpus,
            status="recorded",
        )
        try:
            accepted = self._dispatch_fn(spec)
        except Exception:
            # The substrate is unreachable (DSM down / GPU fault). Degrade
            # gracefully: keep the job rather than crashing the slow loop.
            spec.status = "skipped_no_substrate"
        else:
            spec.status = "dispatched" if accepted else "recorded"
        self._jobs.append(spec)
        return spec

    def as_trainer_fn(self) -> Callable[[str, list[Any]], None]:
        """Adapt to :class:`FastSlowController`'s ``TrainerFn`` signature.

        Returns a ``(task_key, traces) -> None`` callable the controller can use as
        its slow-loop ``trainer_fn``: it runs :meth:`train` (building + dispatching +
        recording the spec) and discards the return value, since the controller's
        ``TrainerFn`` is side-effecting. Produced specs remain available via
        :meth:`jobs`.
        """

        def trainer_fn(task_key: str, traces: list[Any]) -> None:
            self.train(task_key, traces)

        return trainer_fn

    def jobs(self) -> list[TrainingJobSpec]:
        """All specs produced so far — the local job queue / audit trail."""
        return list(self._jobs)
